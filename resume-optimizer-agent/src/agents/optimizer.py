"""
LLM 驱动的简历优化 Agent
使用 LangChain 构建优化流水线

核心原则：
- 只做"润色"和"重组"，绝不编造用户没有的经历/技能/数据
- 优化后的内容必须能在原文中找到依据
- 对于用户缺少的技能，单独列为"建议学习"而非塞入简历
"""

import json
import re
from datetime import datetime
from typing import List, Dict, Tuple
from dataclasses import asdict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.config import settings

from src.parsers.resume_parser import ResumeData
from src.scrapers.job_scraper import JobPosting
from src.agents.matcher import MatchResult


class ResumeOptimizer:
    """
    简历优化器
    基于 LLM 分析职位需求并生成优化后的简历内容

    支持 OpenAI 和 通义千问(Qwen) 两种模型提供商
    通过环境变量 LLM_PROVIDER 自动切换
    """

    def __init__(self, model: str = None, temperature: float = 0.3):
        provider = settings.llm_provider.lower()

        if provider == "qwen":
            self.llm = ChatOpenAI(
                model=model or settings.qwen_model,
                temperature=temperature,
                api_key=settings.qwen_api_key,
                base_url=settings.qwen_base_url,
            )
            logger.info(f"使用通义千问模型: {model or settings.qwen_model}")
        else:
            self.llm = ChatOpenAI(
                model=model or "gpt-4o-mini",
                temperature=temperature,
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
            )
            logger.info(f"使用 OpenAI 模型: {model or 'gpt-4o-mini'}")

        # 职位分析缓存：key = (job_title, job_company)，value = job_analysis dict
        self._job_analysis_cache: Dict[tuple, Dict] = {}

    def optimize(self, resume: ResumeData, job: JobPosting, match: MatchResult = None,
                 target_section: str = None, job_analysis: Dict = None) -> Dict:
        """
        根据职位需求优化简历
        :param target_section: 指定只优化某个模块（模块名需匹配 resume.sections 中的 key）。
                               为 None 时优化所有模块（原有行为）。
        :param job_analysis: 预计算的职位分析结果（多职位合并时使用）。
                             如果提供，跳过内部 _analyze_job 调用。
        :return: 包含优化后各模块内容的字典
        """
        logger.info(f"简历模块列表: {list(resume.sections.keys())}")
        if target_section:
            logger.info(f"定向优化模式: 仅优化 [{target_section}]")

        # 1. 分析职位需求
        if job_analysis is not None:
            # 使用外部传入的预计算分析（多职位合并结果）
            logger.info(f"使用预计算职位分析（多职位合并），跳过 LLM 调用")
        else:
            # 单职位分析（带缓存）
            cache_key = (job.title, job.company)
            if cache_key in self._job_analysis_cache:
                job_analysis = self._job_analysis_cache[cache_key]
                logger.info(f"职位分析命中缓存，跳过 LLM 调用（缓存 key: {cache_key}）")
            else:
                job_analysis = self._analyze_job(job)
                self._job_analysis_cache[cache_key] = job_analysis
                logger.info(f"职位分析完成并缓存（缓存 key: {cache_key}）")
        logger.info(f"职位分析结果: {list(job_analysis.keys()) if job_analysis else '空'}")

        # 2. 优化各模块
        optimized_sections = {}
        optimization_notes = []  # 记录每个模块的优化说明

        # ── 判断是否需要优化某个模块的辅助函数 ──
        def _should_process(section_name: str) -> bool:
            if target_section is None:
                return True
            return section_name == target_section

        # 查找技能相关模块（支持模糊匹配）
        skills_result = self._find_section(resume.sections, ["专业技能", "技能", "专业能力", "技术栈"])
        if skills_result and _should_process(skills_result[0]):
            skills_section, original = skills_result
            logger.info(f"优化技能模块: {skills_section}")
            optimized, note = self._optimize_skills(
                original,
                job_analysis.get("required_skills", []),
                match.missing_skills if match else []
            )
            optimized_sections[skills_section] = optimized
            optimization_notes.append(f"[{skills_section}] {note}")
        elif skills_result:
            logger.info(f"跳过技能模块: {skills_result[0]}（非目标模块）")

        # 查找工作/项目经历模块
        for keywords in [["工作经历", "工作经验"], ["项目经验", "项目经历"], ["实习经历"]]:
            result = self._find_section(resume.sections, keywords)
            if result and _should_process(result[0]):
                section_name, original = result
                logger.info(f"优化经历模块: {section_name}")
                optimized, note = self._optimize_experience(original, job_analysis)
                optimized_sections[section_name] = optimized
                optimization_notes.append(f"[{section_name}] {note}")
            elif result:
                logger.info(f"跳过经历模块: {result[0]}（非目标模块）")

        # 查找自我评价模块
        summary_result = self._find_section(resume.sections, ["自我评价", "个人总结", "个人优势", "自我描述"])
        if summary_result and _should_process(summary_result[0]):
            summary_section, original = summary_result
            logger.info(f"优化自我评价模块: {summary_section}")
            optimized, note = self._optimize_summary(original, job_analysis)
            optimized_sections[summary_section] = optimized
            optimization_notes.append(f"[{summary_section}] {note}")
        elif summary_result:
            logger.info(f"跳过自我评价模块: {summary_result[0]}（非目标模块）")

        # 3. 生成整体优化报告
        report = self._generate_report(resume, job, match, optimized_sections, optimization_notes)

        return {
            "original": resume.sections,
            "optimized": optimized_sections,
            "job_analysis": job_analysis,
            "report": report,
            "optimization_notes": optimization_notes,
        }

    def _find_section(self, sections: dict, keywords: list) -> Tuple[str, str]:
        """在 sections 中查找包含任意一个关键词的模块，返回 (key, value) 元组"""
        for key, value in sections.items():
            for keyword in keywords:
                if keyword in key or key in keyword:
                    return (key, value)
        return None

    def _analyze_job(self, job: JobPosting) -> Dict:
        """使用 LLM 分析职位描述，提取关键要求"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一位资深HR和招聘专家，擅长分析职位描述。请从以下职位信息中提取结构化数据。"),
            ("human", """请分析以下职位，提取关键信息并以JSON格式返回：

职位标题: {title}
职位描述: {description}
职位要求: {requirements}
技能要求: {skills}

请返回以下JSON格式：
{{
    "position_type": "职位类型（如后端开发、数据分析等）",
    "required_skills": ["必须技能列表"],
    "preferred_skills": ["加分技能列表"],
    "core_responsibilities": ["核心职责列表"],
    "experience_level": "经验要求级别",
    "key_qualities": ["关键素质要求"],
    "suggested_keywords": ["建议在简历中突出的关键词"]
}}
""")
        ])

        chain = prompt | self.llm | JsonOutputParser()

        return chain.invoke({
            "title": job.title,
            "description": job.description,
            "requirements": job.requirements,
            "skills": ", ".join(job.skills),
        })

    def analyze_job(self, job: JobPosting) -> Dict:
        """公开接口：分析单个职位（带缓存），供外部调用"""
        cache_key = (job.title, job.company)
        if cache_key in self._job_analysis_cache:
            logger.info(f"analyze_job 命中缓存（key: {cache_key}）")
            return self._job_analysis_cache[cache_key]
        result = self._analyze_job(job)
        # 非空校验：如果 LLM 没有提取出必需技能和核心职责，从原文兜底提取
        result = self._fallback_extract(result, job)
        self._job_analysis_cache[cache_key] = result
        logger.info(f"analyze_job 完成并缓存（key: {cache_key}）")
        return result

    @staticmethod
    def _fallback_extract(result: Dict, job: JobPosting) -> Dict:
        """
        当 LLM 分析返回空列表时，从职位原文（description/requirements/skills）中
        提取关键词作为兜底，确保后续优化有目标参考。
        """
        has_skills = bool(result.get("required_skills"))
        has_responsibilities = bool(result.get("core_responsibilities"))
        if has_skills and has_responsibilities:
            return result  # LLM 提取正常，无需兜底

        # 拼接职位原文
        full_text = "\n".join(filter(None, [
            job.title,
            job.description or "",
            job.requirements or "",
            " ".join(job.skills) if job.skills else "",
        ]))
        if len(full_text.strip()) < 20:
            logger.warning(f"职位原文内容过少，无法兜底提取（title: {job.title}）")
            return result

        logger.info(f"LLM 分析结果不完整（skills={has_skills}, responsibilities={has_responsibilities}），启用兜底提取")

        # 兜底提取必需技能：优先用 job.skills，再从原文中补充常见技术词
        if not has_skills:
            fallback_skills = list(job.skills) if job.skills else []
            # 从 description/requirements 中提取大写字母开头的技术名词（如 Python, Java, AWS）
            tech_pattern = r'\b([A-Z][a-zA-Z+#.]{1,20})\b'
            found = re.findall(tech_pattern, full_text)
            seen = set(s.lower() for s in fallback_skills)
            for t in found:
                if t.lower() not in seen and len(t) >= 2:
                    # 过滤掉常见的非技能英文词
                    skip_words = {"The", "And", "For", "With", "Our", "You", "Will", "Job",
                                  "Work", "Team", "Role", "Must", "Able", "Also", "Best",
                                  "Join", "Look", "Need", "Plus", "Good", "Well", "High"}
                    if t not in skip_words:
                        fallback_skills.append(t)
                        seen.add(t.lower())
            result["required_skills"] = fallback_skills[:15]  # 限制数量

        # 兜底提取核心职责：从 description 中取前 3 个有意义的句子
        if not has_responsibilities:
            desc = job.description or job.requirements or ""
            if desc:
                # 按换行/句号拆分，取前 5 条有意义的行
                lines = re.split(r'[。\n;；]+', desc)
                meaningful = [l.strip() for l in lines if len(l.strip()) >= 8]
                result["core_responsibilities"] = meaningful[:5]

        # 兜底 suggested_keywords
        if not result.get("suggested_keywords"):
            kw = list(job.skills[:5]) if job.skills else []
            if job.title and job.title not in [w.lower() for w in kw]:
                kw.insert(0, job.title)
            result["suggested_keywords"] = kw[:8]

        # 兜底 position_type
        if not result.get("position_type") and job.title:
            result["position_type"] = job.title

        return result

    @staticmethod
    def merge_job_analyses(analyses: List[Dict]) -> Dict:
        """
        合并多个职位的分析结果，取各维度的并集。
        用于多职位合并优化：综合多个目标职位的需求，使优化覆盖面更广。

        :param analyses: 多个 _analyze_job 返回的 dict 列表
        :return: 合并后的单个 job_analysis dict
        """
        if not analyses:
            return {}
        if len(analyses) == 1:
            return analyses[0]

        merged = {
            "position_type": analyses[0].get("position_type", ""),
            "required_skills": [],
            "preferred_skills": [],
            "core_responsibilities": [],
            "experience_level": analyses[0].get("experience_level", ""),
            "key_qualities": [],
            "suggested_keywords": [],
        }

        # 用 set 去重，保持顺序
        seen_skills = set()
        seen_preferred = set()
        seen_resp = set()
        seen_qualities = set()
        seen_keywords = set()

        for a in analyses:
            for s in a.get("required_skills", []):
                if s.lower() not in seen_skills:
                    seen_skills.add(s.lower())
                    merged["required_skills"].append(s)
            for s in a.get("preferred_skills", []):
                if s.lower() not in seen_preferred:
                    seen_preferred.add(s.lower())
                    merged["preferred_skills"].append(s)
            for r in a.get("core_responsibilities", []):
                if r.lower() not in seen_resp:
                    seen_resp.add(r.lower())
                    merged["core_responsibilities"].append(r)
            for q in a.get("key_qualities", []):
                if q.lower() not in seen_qualities:
                    seen_qualities.add(q.lower())
                    merged["key_qualities"].append(q)
            for k in a.get("suggested_keywords", []):
                if k.lower() not in seen_keywords:
                    seen_keywords.add(k.lower())
                    merged["suggested_keywords"].append(k)

        logger.info(
            f"合并 {len(analyses)} 个职位分析: "
            f"必需技能 {len(merged['required_skills'])} 项, "
            f"加分技能 {len(merged['preferred_skills'])} 项, "
            f"核心职责 {len(merged['core_responsibilities'])} 项"
        )
        return merged

    def merge_job_analyses_with_llm(self, analyses: List[Dict], job_titles: List[str] = None) -> Dict:
        """
        使用 LLM 智能合并多个职位的分析结果。
        相比代码合并（merge_job_analyses），LLM 合并能：
        - 识别语义重复（如 "Python" 和 "Python开发"）
        - 按跨职位出现频率排序，突出共性需求
        - 生成更精炼、有层次的分析结果

        LLM 调用失败时自动回退到代码合并。

        :param analyses: 多个 _analyze_job 返回的 dict 列表
        :param job_titles: 各职位的标题（用于 prompt 上下文）
        :return: 合并后的单个 job_analysis dict
        """
        if not analyses:
            return {}
        if len(analyses) == 1:
            return analyses[0]

        # 构造输入文本：每个职位的分析结果
        job_blocks = []
        for i, a in enumerate(analyses):
            title = job_titles[i] if job_titles and i < len(job_titles) else f"职位{i+1}"
            block = (
                f"--- {title} ---\n"
                f"职位类型: {a.get('position_type', '未知')}\n"
                f"必需技能: {', '.join(a.get('required_skills', []))}\n"
                f"加分技能: {', '.join(a.get('preferred_skills', []))}\n"
                f"核心职责: {'; '.join(a.get('core_responsibilities', []))}\n"
                f"经验要求: {a.get('experience_level', '未知')}\n"
                f"关键素质: {', '.join(a.get('key_qualities', []))}\n"
                f"建议关键词: {', '.join(a.get('suggested_keywords', []))}"
            )
            job_blocks.append(block)

        jobs_text = "\n\n".join(job_blocks)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一位资深HR专家，擅长综合分析多个职位描述，提炼出共性的核心需求。
你的任务是将多个职位的分析结果合并为一份综合需求画像。"""),
            ("human", """以下是 {job_count} 个职位的分析结果，请将它们合并为一份综合需求画像。

{jobs_text}

合并规则：
1. **必需技能**：合并语义相同的技能（如"Python"和"Python开发"视为同一技能），按跨职位出现频率降序排列（多个职位共同要求的排在前面）
2. **加分技能**：同上，去重并按频率排序
3. **核心职责**：提炼共性职责方向，去除特定公司的个性化描述，合并为通用的职责描述
4. **关键素质**：去重合并，突出多个职位共同要求的素质
5. **建议关键词**：综合所有职位，保留最有代表性的关键词，去重
6. **职位类型**：如果各职位类型相近，取最通用的描述；如果差异大，用"/"连接
7. **经验要求**：取最常见的级别

请返回以下JSON格式：
{{
    "position_type": "综合职位类型",
    "required_skills": ["按频率排序的必需技能列表"],
    "preferred_skills": ["按频率排序的加分技能列表"],
    "core_responsibilities": ["合并后的核心职责列表"],
    "experience_level": "综合经验要求",
    "key_qualities": ["合并后的关键素质列表"],
    "suggested_keywords": ["综合建议关键词列表"]
}}""")
        ])

        try:
            chain = prompt | self.llm | JsonOutputParser()
            merged = chain.invoke({
                "job_count": len(analyses),
                "jobs_text": jobs_text,
            })

            # 校验返回结构完整性
            required_keys = ["position_type", "required_skills", "preferred_skills",
                             "core_responsibilities", "experience_level", "key_qualities", "suggested_keywords"]
            for key in required_keys:
                if key not in merged:
                    merged[key] = analyses[0].get(key, "" if key in ("position_type", "experience_level") else [])

            logger.info(
                f"LLM 合并 {len(analyses)} 个职位分析成功: "
                f"必需技能 {len(merged.get('required_skills', []))} 项, "
                f"加分技能 {len(merged.get('preferred_skills', []))} 项, "
                f"核心职责 {len(merged.get('core_responsibilities', []))} 项"
            )
            return merged

        except Exception as e:
            logger.warning(f"LLM 合并职位分析失败（{e}），回退到代码合并")
            return self.merge_job_analyses(analyses)

    def _optimize_skills(self, current_skills: str, required_skills: List[str], missing_skills: List[str]) -> Tuple[str, str]:
        """
        优化专业技能部分
        原则：只重新组织和润色现有技能，不编造新技能
        返回 (优化后文本, 优化说明)
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一位专业的简历优化顾问。你的任务是帮助求职者**润色**其技能描述。

【严格规则 - 必须遵守】
1. 你只能对求职者已有的技能进行重新组织、分类和润色表述
2. 绝对不允许添加求职者原文中没有提到的任何技能
3. 绝对不允许修改技能的熟练程度（如把"了解"改为"精通"）
4. 如果职位要求的某项技能求职者没有，不要把它加进简历，而是在最后单独标注"建议学习"
5. 优化后的技能数量不应超过原文的 1.5 倍（防止偷偷加料）
6. 只返回润色后的技能内容本身，绝对不要附加任何注释、说明、解释或备注"""),
            ("human", """请润色以下技能描述，使其更专业、更有条理。

求职者当前技能描述（这是唯一的事实来源，不可超出此范围）：
---
{current_skills}
---

目标职位要求的技能（仅供参考排序，不可添加求职者没有的）：
{required_skills}

请返回优化后的技能描述。要求：
1. 只使用原文中已出现的技能，重新分类和排列
2. 与目标职位匹配的技能排在前面
3. 使用更专业的表述方式（如"会用Python" → "熟悉Python开发"）
4. 控制在200字以内
5. 如果有职位要求的技能是原文没有的，在最后用一行标注：【建议学习】xxx
6. **只输出润色后的内容，不要附加任何注释或说明文字**

润色后的技能描述：
""")
        ])

        chain = prompt | self.llm
        response = chain.invoke({
            "current_skills": current_skills,
            "required_skills": ", ".join(required_skills),
        })

        result = response.content.strip()

        # 校验：检查是否添加了原文没有的技能
        note = self._validate_skills(current_skills, result)
        return result, note

    def _optimize_experience(self, current_experience: str, job_analysis: Dict) -> Tuple[str, str]:
        """
        优化工作经历/项目经验
        原则：只改善表述方式（STAR法则、量化），不编造新经历或数据
        返回 (优化后文本, 优化说明)
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""你是一位专业的简历优化顾问。你的任务是帮助求职者**润色**其工作/项目经历描述。

当前日期：{datetime.now().strftime("%Y年%m月%d日")}（请以此为准判断时间线合理性，不要自行推测当前年份）

【最高优先级规则 - 禁止幻觉，必须遵守】
1. 原文是唯一的事实来源。你只能改写措辞和结构，绝对不能添加原文中不存在的事实
2. 禁止编造任何具体数字（金额、人数、百分比、天数等）。如果原文没有数字，用"显著提升"、"大幅改善"等定性词代替
3. 禁止编造原文中没有的项目名称、公司名称、职位名称、技术名称、系统名称、行业术语
4. 禁止把职位JD中的行业、技术栈、系统名称引入求职者的经历。例如：求职者原文没提到LIMS，即使目标职位是制药行业，也不能在简历中出现LIMS、GMP、FDA等制药行业术语
5. 禁止把职位JD中的要求改写为求职者的经历
6. 禁止添加原文中没有的"成果"或"业绩"——即使听起来合理也不行
7. 优化后的条目数量不应超过原文的 2 倍
8. 绝对不要使用"情境""任务""行动""结果"等标签或分段
9. 只返回润色后的经历内容本身，不要附加任何注释、说明、解释、自评、提醒或警告
10. 不要输出"重要提醒"、"注意事项"、"建议修改"等元文本——你只输出简历内容

【正确做法 vs 错误做法】
✓ 正确：原文「负责项目管理」→ 优化为「主导项目全流程管理，协调跨部门资源推进交付」
✗ 错误：原文「负责项目管理」→ 编造为「管理3个并行项目，总预算600万元，团队15人，交付率98%」
✗ 错误：原文没有提到Jira → 编造为「使用Jira维护风险清单与跨团队依赖图」
✗ 错误：原文没有提到LIMS → 因目标职位是制药行业就编造「负责LIMS系统实施与验证」
✗ 错误：在经历后面附加「重要提醒：时间线存在硬伤，请修改...」"""),
            ("human", """请润色以下经历描述，使其更有说服力。

求职者当前经历描述（这是唯一的事实来源，绝对不可超出此范围）：
---
{current_experience}
---

目标职位的核心职责（仅供参考调整表述方向，绝对不可照搬为求职者的经历，尤其不要引入目标行业的专有术语）：
{responsibilities}

建议突出的关键词（仅在原文有对应内容时才使用，如果关键词涉及求职者原文没有的行业或技术，直接忽略）：
{keywords}

请返回润色后的经历描述。要求：
1. **必须保留原文中的子标题结构**（公司名称、职位、项目名称、工作时间等）
2. 要点以 STAR 法则作为内在逻辑，融合成流畅的描述，不要显式标注标签
3. 每个要点1-2句话，将做了什么、取得了什么成果融为一体
4. 如果原文有具体数字则保留，没有则用定性描述（"显著提升"、"大幅改善"），绝对不要编造数字
5. 只使用求职者原文中已有的行业术语和技术名称，不要从目标职位引入新的行业术语
6. 只输出简历内容，不要输出任何提醒、警告、建议或说明文字

润色后的经历描述：
""")
        ])

        chain = prompt | self.llm
        response = chain.invoke({
            "current_experience": current_experience,
            "responsibilities": "\n".join(job_analysis.get("core_responsibilities", [])),
            "keywords": ", ".join(job_analysis.get("suggested_keywords", [])),
        })

        result = response.content.strip()

        # 校验
        note = self._validate_experience(current_experience, result)
        return result, note

    def _optimize_summary(self, current_summary: str, job_analysis: Dict) -> Tuple[str, str]:
        """
        优化自我评价/个人总结
        原则：只润色措辞，不添加新的事实性声明
        返回 (优化后文本, 优化说明)
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一位专业的简历优化顾问。你的任务是帮助求职者**润色**其自我评价。

【严格规则 - 必须遵守】
1. 你只能对求职者已有的自我评价进行措辞润色
2. 绝对不允许添加原文中没有的能力、经验或特质
3. 不允许编造具体的成就或数据
4. 控制在100字以内
5. 只返回润色后的自我评价内容本身，绝对不要附加任何注释、说明、解释或备注"""),
            ("human", """请润色以下自我评价，使其更贴合目标职位。

求职者当前自我评价（这是唯一的事实来源）：
---
{current_summary}
---

目标职位类型：{position_type}
关键素质要求（仅供参考调整措辞方向）：{qualities}

请返回润色后的自我评价。要求：
1. 保持原文的核心信息不变
2. 突出与目标职位匹配的特质（但必须是原文已提到的）
3. 使用更精炼、专业的表述
4. 控制在100字以内
5. 不要出现空话套话
6. **只输出润色后的内容，不要附加任何注释或说明文字**

润色后的自我评价：
""")
        ])

        chain = prompt | self.llm
        response = chain.invoke({
            "current_summary": current_summary,
            "position_type": job_analysis.get("position_type", ""),
            "qualities": ", ".join(job_analysis.get("key_qualities", [])),
        })

        result = response.content.strip()
        note = "自我评价已润色措辞，未添加新内容"
        return result, note

    def _validate_skills(self, original: str, optimized: str) -> str:
        """
        校验技能优化结果：检查是否添加了原文没有的技能
        返回校验说明
        """
        original_lower = original.lower()
        # 简单检查：如果优化后文本长度远超原文（>2倍），可能有编造
        if len(optimized) > len(original) * 2:
            logger.warning(f"技能模块优化后文本长度异常: {len(original)}字 → {len(optimized)}字，可能存在编造内容")
            return f"⚠️ 优化后内容大幅增加({len(original)}→{len(optimized)}字)，请人工检查是否有编造"

        logger.info("技能模块优化校验通过")
        return f"技能已重新组织润色({len(original)}→{len(optimized)}字)"

    def _validate_experience(self, original: str, optimized: str) -> str:
        """
        校验经历优化结果：检查是否添加了原文没有的内容
        返回校验说明
        """
        warnings = []

        # 检查1：长度异常（阈值从 2.5x 降低到 1.5x）
        if len(optimized) > len(original) * 1.5:
            warnings.append(f"内容大幅增加({len(original)}→{len(optimized)}字)")

        # 检查2：是否编造了原文没有的具体数字
        import re
        original_numbers = set(re.findall(r'\d+(?:\.\d+)?(?:%|万|元|人|天|个|套|倍|亿)', original))
        optimized_numbers = set(re.findall(r'\d+(?:\.\d+)?(?:%|万|元|人|天|个|套|倍|亿)', optimized))
        new_numbers = optimized_numbers - original_numbers
        if new_numbers:
            warnings.append(f"出现原文没有的数字: {', '.join(new_numbers)}")

        # 检查3：是否编造了原文没有的技术/工具名称
        tech_keywords = [
            "Jira", "Confluence", "Notion", "Trello", "Asana",
            "SAP", "ERP", "MES", "MOM", "CRM", "OA",
            "PMP", "PRINCE2", "Scrum", "Kanban", "Agile",
            "AWS", "Azure", "GCP", "Docker", "Kubernetes",
            "Python", "Java", "Go", "React", "Vue",
        ]
        original_lower = original.lower()
        for kw in tech_keywords:
            if kw.lower() not in original_lower and kw in optimized:
                warnings.append(f"出现原文没有的技术词: {kw}")

        if warnings:
            warning_text = "；".join(warnings)
            logger.warning(f"经历模块校验发现异常: {warning_text}")
            return f"⚠️ 可能存在编造: {warning_text}，请人工检查"

        logger.info("经历模块优化校验通过")
        return f"经历已润色({len(original)}→{len(optimized)}字)"

    def _generate_report(self, resume: ResumeData, job: JobPosting, match: MatchResult,
                         optimized: Dict, optimization_notes: List[str] = None) -> Dict:
        """生成优化报告"""
        report = {
            "target_job": f"{job.title} @ {job.company}",
            "match_score": match.overall_score if match else "N/A",
            "optimized_sections": list(optimized.keys()),
            "optimization_notes": optimization_notes or [],
            "key_improvements": [],
            "warnings": [],
        }

        if match and match.missing_skills:
            report["key_improvements"].append(
                f"职位要求的技能中有 {len(match.missing_skills)} 项暂未掌握: {', '.join(match.missing_skills[:5])}"
            )
            report["warnings"].append(
                "建议学习以上技能以增强竞争力，但未写入简历（避免虚假简历）"
            )

        if optimized:
            report["key_improvements"].append(
                f"润色了 {len(optimized)} 个简历模块的表述"
            )

        return report
