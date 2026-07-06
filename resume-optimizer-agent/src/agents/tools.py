"""
Agent 工具层
将现有模块封装为 LLM 可调用的 LangChain Tools
Agent 可以自主决定何时、以什么参数调用这些工具
"""

import json
from typing import Type, Optional, Dict
from pathlib import Path

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from loguru import logger

from src.parsers.resume_parser import ResumeParser, ResumeData
from src.scrapers.job_scraper import get_scraper, JobPosting
from src.agents.matcher import JobResumeMatcher, MatchResult
from src.agents.optimizer import ResumeOptimizer
from src.generators.resume_generator import ResumeGenerator


# ==================== 工具输入参数定义 ====================

class ParseResumeInput(BaseModel):
    """解析简历工具的输入参数"""
    file_path: str = Field(description="简历文件的路径（支持 PDF/DOCX/TXT）")


class SearchJobsInput(BaseModel):
    """搜索职位工具的输入参数"""
    keyword: str = Field(description="职位搜索关键词，如 'Python后端'、'前端开发'")
    location: str = Field(default="", description="工作地点，如 '北京'、'上海'、'深圳'")
    max_jobs: int = Field(default=10, description="抓取职位数量，全部访问详情页获取完整信息（上限20，默认10）")


class MatchResumeInput(BaseModel):
    """匹配分析工具的输入参数"""
    job_index: int = Field(default=0, description="要分析的职位索引（0=第一个职位）")


class MatchAllJobsInput(BaseModel):
    """匹配所有职位工具的输入参数"""
    top_n: int = Field(default=3, description="保留匹配度最高的前 N 个职位（默认3，用于后续优化参考）")


class OptimizeSectionInput(BaseModel):
    """优化单个模块工具的输入参数"""
    section_name: str = Field(description="要优化的模块名，如 '专业技能'、'工作经历'、'项目经验'、'自我评价'")
    focus: str = Field(default="", description="优化重点，如 '突出项目管理能力'、'强调技术深度'")


class EvaluateResumeInput(BaseModel):
    """评估简历质量工具的输入参数"""
    target_job: str = Field(default="", description="目标职位描述（可选）")


class GenerateResumeInput(BaseModel):
    """生成简历文件工具的输入参数"""
    format: str = Field(default="docx", description="输出格式: docx 或 markdown")


class FetchURLJobsInput(BaseModel):
    """从 URL 抓取职位工具的输入参数"""
    urls: str = Field(description="职位页面 URL 列表，多个 URL 用英文逗号分隔")


class JudgeResumeInput(BaseModel):
    """评估优化效果工具的输入参数（无需参数，使用已缓存的数据）"""
    pass


# ==================== 工具实现 ====================

class ParseResumeTool(BaseTool):
    """解析用户上传的简历文件"""
    name: str = "parse_resume"
    description: str = """解析简历文件（PDF/DOCX/TXT），提取各模块内容。
    输入简历文件路径，返回结构化的简历数据（包括各模块文本、联系方式等）。
    这是第一步操作，必须先解析简历才能进行后续分析和优化。"""
    args_schema: Type[BaseModel] = ParseResumeInput

    parser: Optional[ResumeParser] = None
    resume_data: Optional[ResumeData] = None

    def _run(self, file_path: str) -> str:
        try:
            if self.parser is None:
                self.parser = ResumeParser()
            self.resume_data = self.parser.parse(file_path)

            sections_summary = []
            for name, content in self.resume_data.sections.items():
                preview = content[:100] + "..." if len(content) > 100 else content
                sections_summary.append(f"  [{name}]: {preview}")

            result = f"""简历解析成功！
文件: {file_path}
模块数: {len(self.resume_data.sections)}
联系方式: 邮箱={self.resume_data.email or '未识别'}, 电话={self.resume_data.phone or '未识别'}

各模块内容:
{chr(10).join(sections_summary)}"""
            logger.info(f"[Tool] parse_resume: 解析成功，{len(self.resume_data.sections)} 个模块")
            return result
        except Exception as e:
            logger.error(f"[Tool] parse_resume 失败: {e}")
            return f"简历解析失败: {str(e)}"


class SearchJobsTool(BaseTool):
    """从招聘网站搜索职位信息"""
    name: str = "search_jobs"
    description: str = """从智联招聘搜索目标职位。
    输入关键词和地点，返回职位列表（包含标题、公司、薪资、技能要求等）。
    可以多次调用以搜索不同关键词的职位。"""
    args_schema: Type[BaseModel] = SearchJobsInput

    jobs: list = []

    def _run(self, keyword: str, location: str = "", max_jobs: int = 10) -> str:
        try:
            scraper = get_scraper("zhaopin")
            self.jobs = scraper.fetch(
                keyword=keyword,
                location=location,
                max_jobs=max_jobs,
            )

            if not self.jobs:
                return "未找到相关职位，请尝试更换关键词或地点。"

            # 精简摘要：只保留关键信息，避免过长输出淹没 LLM 上下文
            job_lines = []
            with_detail = 0
            for i, job in enumerate(self.jobs):
                has_detail = job.description and len(job.description) > 50
                if has_detail:
                    with_detail += 1
                job_lines.append(f"  [{i}] {job.title} @ {job.company} | {job.salary}")

            result = f"""职位搜索成功！共找到 {len(self.jobs)} 个职位（{with_detail} 个有完整详情）。

职位列表:
{chr(10).join(job_lines)}

⚠️ 重要：search_jobs 只需调用一次！下一步必须调用 match_all_jobs 对所有职位进行匹配度分析。绝对不要再次调用 search_jobs。"""
            logger.info(f"[Tool] search_jobs: 找到 {len(self.jobs)} 个职位（{with_detail} 个有详情）")
            return result
        except Exception as e:
            logger.error(f"[Tool] search_jobs 失败: {e}")
            return f"职位搜索失败: {str(e)}"


class MatchResumeTool(BaseTool):
    """分析简历与职位的匹配度"""
    name: str = "match_resume"
    description: str = """分析简历与某个职位的匹配程度。
    返回匹配分数、已匹配技能、缺少技能、优化建议。
    必须先调用 parse_resume 和 search_jobs。"""
    args_schema: Type[BaseModel] = MatchResumeInput

    resume_data: Optional[ResumeData] = None
    jobs: list = []
    match_result: Optional[MatchResult] = None

    def _run(self, job_index: int = 0) -> str:
        if not self.resume_data:
            return "错误: 请先调用 parse_resume 解析简历。"
        if not self.jobs:
            return "错误: 请先调用 search_jobs 搜索职位。"
        if job_index >= len(self.jobs):
            return f"错误: 职位索引 {job_index} 超出范围（共 {len(self.jobs)} 个职位）。"

        try:
            matcher = JobResumeMatcher()
            job = self.jobs[job_index]
            matches = matcher.match(self.resume_data, [job])
            self.match_result = matches[0]

            result = f"""匹配分析完成！
目标职位: {self.match_result.job_title} @ {self.match_result.company}
综合匹配度: {self.match_result.overall_score * 100:.1f}%
技能匹配度: {self.match_result.skill_match_score * 100:.1f}%
经验匹配度: {self.match_result.experience_match_score * 100:.1f}%

已匹配技能: {', '.join(self.match_result.matched_skills) if self.match_result.matched_skills else '无'}
缺少技能: {', '.join(self.match_result.missing_skills) if self.match_result.missing_skills else '无'}

优化建议:
{chr(10).join('- ' + s for s in self.match_result.suggestions) if self.match_result.suggestions else '暂无'}

提示: 可以用 optimize_section 工具针对这个职位优化简历模块。"""
            logger.info(f"[Tool] match_resume: 匹配度 {self.match_result.overall_score * 100:.1f}%")
            return result
        except Exception as e:
            logger.error(f"[Tool] match_resume 失败: {e}")
            return f"匹配分析失败: {str(e)}"


class MatchAllJobsTool(BaseTool):
    """匹配所有职位并按匹配度排序，自动选定最佳目标职位"""
    name: str = "match_all_jobs"
    description: str = """对所有搜索到的职位进行匹配度分析，按分数从高到低排序。
    自动将最佳匹配职位设为后续优化和评估的目标。
    必须在 parse_resume 和 search_jobs 之后调用。"""
    args_schema: Type[BaseModel] = MatchAllJobsInput

    resume_data: Optional[ResumeData] = None
    jobs: list = []
    match_result: Optional[MatchResult] = None
    all_match_results: list = []  # 所有职位的匹配结果（已排序）
    top_jobs: list = []  # 匹配度最高的前 N 个职位

    def _run(self, top_n: int = 3) -> str:
        if not self.resume_data:
            return "错误: 请先调用 parse_resume 解析简历。"
        if not self.jobs:
            return "错误: 请先调用 search_jobs 搜索职位。"

        try:
            matcher = JobResumeMatcher()
            # 批量匹配所有职位
            self.all_match_results = matcher.match(self.resume_data, self.jobs)
            # 按匹配度降序排序
            self.all_match_results.sort(key=lambda r: r.overall_score, reverse=True)

            # 保留前 N 个职位
            top_n = min(top_n, len(self.all_match_results))
            self.top_jobs = [self.jobs[self.all_match_results.index(r)] for r in self.all_match_results[:top_n]]

            # 将最佳匹配设为当前 match_result
            self.match_result = self.all_match_results[0]

            # 格式化输出
            lines = [f"全部职位匹配分析完成！（共 {len(self.all_match_results)} 个职位）"]
            lines.append(f"已选定最佳目标职位: {self.match_result.job_title} @ {self.match_result.company}")
            lines.append(f"最佳匹配度: {self.match_result.overall_score * 100:.1f}%\n")
            lines.append(f"匹配度排名（前 {top_n} 个，将用于后续优化参考）:")

            for i, result in enumerate(self.all_match_results[:top_n], 1):
                job = self.jobs[self.all_match_results.index(result)]
                lines.append(
                    f"  [{i}] {result.job_title} @ {result.company} | "
                    f"匹配度: {result.overall_score * 100:.1f}% | "
                    f"技能: {result.skill_match_score * 100:.1f}% | "
                    f"薪资: {job.salary}"
                )

            if len(self.all_match_results) > top_n:
                lines.append(f"\n（其余 {len(self.all_match_results) - top_n} 个职位匹配度较低，已略过）")

            lines.append(f"\n已匹配技能: {', '.join(self.match_result.matched_skills) if self.match_result.matched_skills else '无'}")
            lines.append(f"缺少技能: {', '.join(self.match_result.missing_skills) if self.match_result.missing_skills else '无'}")
            lines.append(f"\n后续优化将综合前 {top_n} 个高匹配职位的需求进行合并优化，覆盖面更广。")

            logger.info(
                f"[Tool] match_all_jobs: {len(self.all_match_results)} 个职位已排序，"
                f"最佳: {self.match_result.job_title} ({self.match_result.overall_score * 100:.1f}%)"
            )
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"[Tool] match_all_jobs 失败: {e}")
            return f"匹配分析失败: {str(e)}"


class OptimizeSectionTool(BaseTool):
    """使用 LLM 优化简历的某个模块"""
    name: str = "optimize_section"
    description: str = """使用大模型优化简历的某个模块（如专业技能、工作经历、项目经验、自我评价）。
    可以指定优化重点，如 '突出技术深度'、'强调管理能力'。
    支持多职位合并优化：自动综合匹配度最高的多个职位需求进行优化。
    必须先调用 parse_resume 和 search_jobs。"""
    args_schema: Type[BaseModel] = OptimizeSectionInput

    resume_data: Optional[ResumeData] = None
    jobs: list = []
    match_result: Optional[MatchResult] = None
    optimized_sections: dict = {}
    top_jobs: list = []  # 匹配度最高的前 N 个职位（来自 match_all_jobs）
    cached_optimizer: Optional[ResumeOptimizer] = None  # 复用实例，使职位分析缓存生效
    merged_cache: dict = {}  # LLM 合并结果缓存：key = 职位标题元组，value = merged dict

    def _get_optimizer(self) -> ResumeOptimizer:
        """获取或创建 optimizer 实例（懒加载，跨调用复用）"""
        if self.cached_optimizer is None:
            self.cached_optimizer = ResumeOptimizer()
            logger.info("[Tool] optimize_section: 创建 ResumeOptimizer 实例（后续调用将复用职位分析缓存）")
        return self.cached_optimizer

    def _build_merged_analysis(self, optimizer: ResumeOptimizer) -> tuple:
        """
        构建多职位合并分析。
        :return: (primary_job, merged_job_analysis_or_None)
                 如果只有 1 个 top_job 或 top_jobs 为空，返回 (job, None) 让 optimize() 走单职位逻辑
        """
        target_jobs = self.top_jobs if self.top_jobs else []

        if len(target_jobs) <= 1:
            # 单职位或无 top_jobs：回退到 match_result 或 jobs[0]
            if self.match_result:
                job = next(
                    (j for j in self.jobs if j.title == self.match_result.job_title and j.company == self.match_result.company),
                    self.jobs[0]
                )
            else:
                job = self.jobs[0]
            return job, None

        # 多职位合并优化（LLM 智能合并，带缓存）
        job_titles = tuple(f"{j.title} @ {j.company}" for j in target_jobs)
        cache_key = job_titles

        if cache_key in self.merged_cache:
            logger.info(f"[Tool] optimize_section: 合并分析命中缓存（{len(target_jobs)} 个职位，跳过 LLM 合并）")
            merged = self.merged_cache[cache_key]
        else:
            logger.info(f"[Tool] optimize_section: 多职位合并优化模式（{len(target_jobs)} 个职位，LLM 智能合并）")
            analyses = []
            for j in target_jobs:
                analysis = optimizer.analyze_job(j)
                analyses.append(analysis)
                logger.info(f"  - 职位分析完成: {j.title} @ {j.company}")

            # 使用 LLM 智能合并（失败时自动回退到代码合并）
            merged = optimizer.merge_job_analyses_with_llm(analyses, job_titles=list(job_titles))
            self.merged_cache[cache_key] = merged
            logger.info(f"[Tool] optimize_section: LLM 合并结果已缓存")

        primary_job = target_jobs[0]
        return primary_job, merged

    def _run(self, section_name: str, focus: str = "") -> str:
        if not self.resume_data:
            return "错误: 请先调用 parse_resume 解析简历。"
        if not self.jobs:
            return "错误: 请先调用 search_jobs 搜索职位。"

        if section_name not in self.resume_data.sections:
            available = ", ".join(self.resume_data.sections.keys())
            return f"错误: 简历中没有模块 '{section_name}'。可用模块: {available}"

        try:
            optimizer = self._get_optimizer()

            # 构建职位分析（多职位合并 or 单职位）
            primary_job, merged_analysis = self._build_merged_analysis(optimizer)

            if merged_analysis:
                logger.info(
                    f"[Tool] optimize_section: 多职位合并优化 → {primary_job.title} @ {primary_job.company} "
                    f"（综合 {len(self.top_jobs)} 个职位需求）"
                )
            else:
                logger.info(f"[Tool] optimize_section: 单职位优化 → {primary_job.title} @ {primary_job.company}")

            # 定向优化
            result_data = optimizer.optimize(
                resume=self.resume_data,
                job=primary_job,
                match=self.match_result,
                target_section=section_name,
                job_analysis=merged_analysis,  # None 时 optimize() 走内部单职位分析
            )

            optimized = result_data.get("optimized", {})
            if section_name in optimized:
                self.optimized_sections[section_name] = optimized[section_name]

                # 输出中标注优化模式
                mode_tag = f"（综合 {len(self.top_jobs)} 个职位需求）" if merged_analysis else ""
                return f"""模块 [{section_name}] 优化完成！{mode_tag}

原始内容:
{self.resume_data.sections[section_name]}

优化后:
{optimized[section_name]}

提示: 可以继续优化其他模块，或用 generate_resume 生成最终简历文件。"""
            else:
                return f"模块 [{section_name}] 未被优化（可能该模块不在优化范围内）。"
        except Exception as e:
            logger.error(f"[Tool] optimize_section 失败: {e}")
            return f"模块优化失败: {str(e)}"


class EvaluateResumeTool(BaseTool):
    """评估简历整体质量"""
    name: str = "evaluate_resume"
    description: str = """评估简历整体质量，给出改进方向。
    可以指定目标职位来评估针对性。
    必须先调用 parse_resume。"""
    args_schema: Type[BaseModel] = EvaluateResumeInput

    resume_data: Optional[ResumeData] = None

    def _run(self, target_job: str = "") -> str:
        if not self.resume_data:
            return "错误: 请先调用 parse_resume 解析简历。"

        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.prompts import ChatPromptTemplate

            optimizer = ResumeOptimizer()
            llm = optimizer.llm

            prompt = ChatPromptTemplate.from_messages([
                ("system", "你是一位资深HR和简历顾问，拥有10年以上招聘经验。请客观评估简历质量并给出专业建议。"),
                ("human", """请评估以下简历，目标职位是: {target_job}

简历内容:
{resume_text}

请从以下维度评估（每项1-10分）并给出改进建议：
1. 内容完整性：是否包含必要模块（教育、工作、项目、技能）
2. 量化程度：是否有具体数据支撑（如提升了X%、处理Y万数据）
3. 关键词覆盖：是否包含行业/岗位常见关键词
4. 表述专业度：用词是否专业、是否有口语化表达
5. 排版逻辑：模块顺序是否合理、重点是否突出

请以JSON格式返回：
{{
    "scores": {{"内容完整性": 0, "量化程度": 0, "关键词覆盖": 0, "表述专业度": 0, "排版逻辑": 0}},
    "total_score": 0,
    "strengths": ["优势1", "优势2"],
    "weaknesses": ["不足1", "不足2"],
    "priority_actions": ["最应该优先改进的事项1", "最应该优先改进的事项2", "最应该优先改进的事项3"]
}}""")
            ])

            chain = prompt | llm
            response = chain.invoke({
                "target_job": target_job or "通用技术岗位",
                "resume_text": self.resume_data.raw_text[:3000],
            })

            return f"""简历评估完成！

{response.content}

提示: 可以用 optimize_section 工具针对薄弱环节进行优化。"""
        except Exception as e:
            logger.error(f"[Tool] evaluate_resume 失败: {e}")
            return f"评估失败: {str(e)}"


class GenerateResumeTool(BaseTool):
    """生成优化后的简历文件"""
    name: str = "generate_resume"
    description: str = """将优化后的内容生成简历文件（DOCX 或 Markdown）。
    必须先完成至少一个模块的优化。"""
    args_schema: Type[BaseModel] = GenerateResumeInput

    resume_data: Optional[ResumeData] = None
    optimized_sections: dict = {}

    def _run(self, format: str = "docx") -> str:
        if not self.resume_data:
            return "错误: 请先调用 parse_resume 解析简历。"
        if not self.optimized_sections:
            return "错误: 请先用 optimize_section 优化至少一个模块。"

        try:
            generator = ResumeGenerator()
            original_info = {
                **self.resume_data.sections,
                "name": self.resume_data.name,
                "email": self.resume_data.email,
                "phone": self.resume_data.phone,
            }

            optimized_data = {"optimized": self.optimized_sections}

            if format == "docx":
                output_path = generator.generate_docx(optimized_data, original_info)
            else:
                md_content = generator.generate_markdown(optimized_data, original_info)
                output_path = str(Path("data/output/optimized_resume.md"))
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                Path(output_path).write_text(md_content, encoding="utf-8")

            return f"""简历文件生成成功！
格式: {format.upper()}
路径: {output_path}

已优化的模块: {', '.join(self.optimized_sections.keys())}

提示: 可以用 evaluate_resume 重新评估优化后的效果。"""
        except Exception as e:
            logger.error(f"[Tool] generate_resume 失败: {e}")
            return f"生成失败: {str(e)}"


class FetchURLJobsTool(BaseTool):
    """从用户提供的 URL 抓取职位信息（Focus 模式专用）"""
    name: str = "fetch_url_jobs"
    description: str = """从用户提供的职位页面 URL 抓取职位详细信息。
    输入一个或多个 URL（逗号分隔），返回每个职位的标题、公司、薪资、描述、技能要求等。
    这是 Focus 模式下获取职位信息的方式，替代 search_jobs。"""
    args_schema: Type[BaseModel] = FetchURLJobsInput

    jobs: list = []

    def _run(self, urls: str) -> str:
        try:
            from src.scrapers.job_scraper import URLJobScraper
            import re
            # 清理 URL：去除 markdown 反引号、空格、换行等
            cleaned = urls.replace("`", "").replace("\n", ",").replace("\r", "")
            url_list = [u.strip() for u in cleaned.split(",") if u.strip()]
            # 进一步提取 http(s):// 开头的 URL（防止 LLM 在 URL 前后加多余文字）
            extracted = []
            for u in url_list:
                # 只匹配合法 URL 字符（ASCII 字母数字和常见 URL 符号），遇到中文/空格/逗号即停止
                match = re.search(r'https?://[A-Za-z0-9\-._~:/?#\[\]@!$&\'()*+,;=%]+', u)
                if match:
                    extracted.append(match.group(0))
                elif u.startswith("http"):
                    extracted.append(u)
            url_list = extracted if extracted else url_list

            if not url_list:
                return "错误: 未提供有效的 URL。请确保输入包含 http:// 或 https:// 开头的链接。"

            scraper = URLJobScraper()
            self.jobs = scraper.fetch(urls=url_list)

            if not self.jobs:
                return "未能从提供的 URL 中抓取到任何职位信息。请检查 URL 是否正确。"

            result_lines = [f"成功抓取 {len(self.jobs)} 个职位：\n"]
            for i, job in enumerate(self.jobs, 1):
                result_lines.append(f"  [{i}] {job.title} @ {job.company}")
                if job.salary:
                    result_lines.append(f"      薪资: {job.salary}")
                if job.location:
                    result_lines.append(f"      地点: {job.location}")
                if job.skills:
                    result_lines.append(f"      技能: {', '.join(job.skills[:8])}")
                desc_preview = job.description[:150] + "..." if len(job.description) > 150 else job.description
                result_lines.append(f"      描述: {desc_preview}")
                result_lines.append("")

            result_lines.append("⚠️ 重要：职位信息已通过 URL 抓取完成！不需要再调用 search_jobs 或 fetch_url_jobs。")
            result_lines.append("下一步必须调用 match_all_jobs 对已抓取的职位进行匹配分析。这是强制指令。")

            logger.info(f"[Tool] fetch_url_jobs: 抓取 {len(self.jobs)} 个职位")
            return "\n".join(result_lines)
        except Exception as e:
            logger.error(f"[Tool] fetch_url_jobs 失败: {e}")
            return f"URL 抓取失败: {str(e)}"


class JudgeResumeTool(BaseTool):
    """LLM-as-Judge 评估优化效果：对比优化前后简历，多维度评分"""
    name: str = "judge_resume"
    description: str = """评估简历优化效果（在 optimize_section 之后调用）。
    使用 LLM-as-Judge 方法，从职位匹配度、技能覆盖度、表述专业度、量化程度、结构清晰度、信息真实性 6 个维度对比评分。
    同时自动计算语义匹配度的前后变化。
    必须先调用 parse_resume 和至少一次 optimize_section。"""
    args_schema: Type[BaseModel] = JudgeResumeInput

    resume_data: Optional[ResumeData] = None
    jobs: list = []
    match_result: Optional[MatchResult] = None
    top_jobs: list = []  # 匹配度最高的前 N 个职位（来自 match_all_jobs）
    optimized_sections: dict = {}

    # 维度名 → 对应的简历模块关键词（用于定位需要重优化的模块）
    _DIMENSION_TO_SECTION = {
        "技能覆盖度": ["专业技能", "技能", "专业能力", "技术栈"],
        "量化程度": ["工作经历", "工作经验", "项目经验", "项目经历"],
        "表述专业度": ["工作经历", "工作经验", "项目经验", "项目经历"],
        "结构清晰度": ["自我评价", "个人总结", "个人优势"],
        "职位匹配度": ["工作经历", "工作经验", "项目经验", "项目经历"],
    }

    # 重优化阈值：after_score 低于此分的维度需要重优化
    _RE_OPTIMIZE_THRESHOLD = 7
    # 每个模块最大重优化次数
    _MAX_RE_OPTIMIZE_PER_SECTION = 2

    def _find_matching_section(self, keywords: list) -> Optional[str]:
        """根据关键词列表在 resume_data.sections 中查找匹配的模块名"""
        if not self.resume_data:
            return None
        for key in self.resume_data.sections.keys():
            for kw in keywords:
                if kw in key or key in kw:
                    return key
        return None

    def _generate_reoptimize_directives(self, judge_result) -> str:
        """
        根据评估结果生成结构化重优化指令。
        返回空字符串表示无需重优化。
        """
        # 统计每个模块已被优化的次数
        optimize_counts: Dict[str, int] = {}
        for section_name in self.optimized_sections.keys():
            optimize_counts[section_name] = optimize_counts.get(section_name, 0) + 1

        directives = []
        dimensions = judge_result.dimensions or []

        for dim in dimensions:
            dim_name = dim.get("name", "")
            after_score = dim.get("after_score", 10)
            analysis = dim.get("analysis", "")

            # 跳过已达标的维度
            if after_score >= self._RE_OPTIMIZE_THRESHOLD:
                continue

            # 查找该维度对应的简历模块
            keywords = self._DIMENSION_TO_SECTION.get(dim_name, [])
            target_section = self._find_matching_section(keywords)
            if not target_section:
                continue

            # 检查是否超过重优化次数上限
            current_count = optimize_counts.get(target_section, 0)
            if current_count >= self._MAX_RE_OPTIMIZE_PER_SECTION:
                logger.info(f"[Tool] judge_resume: 模块 [{target_section}] 已优化 {current_count} 次，跳过重优化")
                continue

            # 根据维度生成优化重点
            focus_map = {
                "技能覆盖度": "重新组织技能排序，将与目标职位匹配的技能排在前面，补充相关技能分类",
                "量化程度": "在经历描述中增加量化表述，用具体数据或定性词（显著提升、大幅改善）替代模糊描述",
                "表述专业度": "使用更专业的行业术语和 STAR 法则改善表述，避免口语化表达",
                "结构清晰度": "改善内容结构，突出重点，使逻辑更清晰",
                "职位匹配度": "调整表述方向，更紧密地贴合目标职位的核心职责和要求",
            }
            focus = focus_map.get(dim_name, f"改善{dim_name}")

            directives.append({
                "section": target_section,
                "dimension": dim_name,
                "score": after_score,
                "focus": focus,
                "analysis": analysis,
            })

        if not directives:
            return ""

        # 格式化输出
        lines = ["\n" + "=" * 50]
        lines.append("  【闭环反馈 — 建议重优化的模块】")
        lines.append("=" * 50)
        for d in directives:
            lines.append(f"\n  ▸ 模块: {d['section']}")
            lines.append(f"    薄弱维度: {d['dimension']}（当前得分 {d['score']}/10）")
            lines.append(f"    评估分析: {d['analysis']}")
            lines.append(f"    优化重点: {d['focus']}")
            lines.append(f"    → 请调用 optimize_section(section_name=\"{d['section']}\", focus=\"{d['focus']}\")")

        lines.append(f"\n  ⚠️ 强制指令：你必须先对上述模块调用 optimize_section 进行重优化，然后再次调用 judge_resume。")
        lines.append(f"  在所有薄弱维度达标或达到重优化上限之前，禁止调用 generate_resume。")
        lines.append("=" * 50)
        return "\n".join(lines)

    def _format_job_reference(self) -> str:
        """
        格式化参考职位信息区块，追加到评估报告末尾。
        展示搜索到的职位基本信息和链接，方便用户查阅。
        """
        # 优先展示 top_jobs（参与优化的职位），否则展示全部 jobs
        display_jobs = self.top_jobs if self.top_jobs else self.jobs
        if not display_jobs:
            return ""

        # 构建 top_jobs 集合用于标记
        top_set = set()
        for j in (self.top_jobs or []):
            top_set.add((j.title, j.company))

        lines = ["\n" + "=" * 50]
        lines.append("  【参考职位信息】")
        if self.top_jobs:
            lines.append(f"  （以下 {len(display_jobs)} 个职位参与了合并优化）")
        lines.append("=" * 50)

        for i, job in enumerate(display_jobs, 1):
            is_top = (job.title, job.company) in top_set
            marker = " ★" if is_top and self.top_jobs and len(self.top_jobs) < len(self.jobs) else ""
            lines.append(f"\n  [{i}] {job.title} @ {job.company}{marker}")
            if job.salary:
                lines.append(f"      薪资: {job.salary}")
            if job.location:
                lines.append(f"      地点: {job.location}")
            if job.experience:
                lines.append(f"      经验: {job.experience}")
            if job.education:
                lines.append(f"      学历: {job.education}")
            if job.skills:
                lines.append(f"      技能: {', '.join(job.skills[:8])}")
            if job.url:
                lines.append(f"      链接: {job.url}")

        if not self.top_jobs and len(self.jobs) > len(display_jobs):
            lines.append(f"\n  （仅展示前 {len(display_jobs)} 个，共 {len(self.jobs)} 个职位）")

        lines.append("\n" + "=" * 50)
        return "\n".join(lines)

    def _run(self) -> str:
        if not self.resume_data:
            return "错误: 请先调用 parse_resume 解析简历。"
        if not self.optimized_sections:
            return "错误: 请先用 optimize_section 优化至少一个模块。"
        if not self.jobs:
            return "错误: 请先用 search_jobs 或 fetch_url_jobs 获取职位信息。"

        try:
            from src.agents.evaluator import ResumeEvaluator

            evaluator = ResumeEvaluator()
            # 优先使用匹配分析选定的最佳职位
            if self.match_result:
                target_job = next(
                    (j for j in self.jobs if j.title == self.match_result.job_title and j.company == self.match_result.company),
                    self.jobs[0]
                )
            else:
                target_job = self.jobs[0]
            logger.info(f"[Tool] judge_resume: 目标职位 {target_job.title} @ {target_job.company}")

            # 1. LLM-as-Judge 评估
            logger.info("[Tool] judge_resume: 开始 LLM-as-Judge 评估...")
            judge_result = evaluator.judge(
                original_resume=self.resume_data,
                optimized_sections=self.optimized_sections,
                job=target_job,
            )

            # 2. 语义匹配度前后对比
            logger.info("[Tool] judge_resume: 计算匹配度前后对比...")
            compare_result = evaluator.compare_match_scores(
                original_resume=self.resume_data,
                optimized_sections=self.optimized_sections,
                job=target_job,
            )

            # 3. 格式化报告
            report = evaluator.format_judge_report(judge_result, compare_result)
            logger.info(f"[Tool] judge_resume: 评估完成，综合提升 {judge_result.improvement:+.1f} 分")

            # 3.5 追加参考职位信息
            job_ref = self._format_job_reference()
            if job_ref:
                report = report + "\n" + job_ref

            # 4. 保存报告到文件（固定文件名，每次评估覆盖上一次，只保留最新报告）
            output_dir = Path(__file__).resolve().parent.parent.parent / "data" / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            report_path = output_dir / "evaluation_report_latest.md"
            report_path.write_text(report, encoding="utf-8")
            logger.info(f"[Tool] judge_resume: 报告已保存: {report_path}")

            # 5. 生成闭环反馈：重优化指令
            reoptimize_directives = self._generate_reoptimize_directives(judge_result)
            if reoptimize_directives:
                logger.info(f"[Tool] judge_resume: 检测到薄弱维度，已生成重优化建议")
            else:
                logger.info(f"[Tool] judge_resume: 所有维度达标，无需重优化")

            return f"{report}\n\n报告已保存到: {report_path}{reoptimize_directives}"

        except Exception as e:
            logger.error(f"[Tool] judge_resume 失败: {e}")
            return f"评估失败: {str(e)}"


# ==================== 工具集合（供 Agent 使用） ====================

def create_agent_tools() -> list:
    """创建所有 Agent 工具实例"""
    # 共享状态：各工具之间通过引用共享数据
    shared_state = {
        "resume_data": None,
        "jobs": [],
        "match_result": None,
        "optimized_sections": {},
    }

    parse_tool = ParseResumeTool()
    search_tool = SearchJobsTool()
    fetch_url_tool = FetchURLJobsTool()
    match_tool = MatchResumeTool()
    optimize_tool = OptimizeSectionTool()
    evaluate_tool = EvaluateResumeTool()
    generate_tool = GenerateResumeTool()
    judge_tool = JudgeResumeTool()

    # 绑定共享状态
    parse_tool.resume_data = shared_state.get("resume_data")
    search_tool.jobs = shared_state.get("jobs", [])
    fetch_url_tool.jobs = shared_state.get("jobs", [])
    match_tool.resume_data = shared_state.get("resume_data")
    match_tool.jobs = shared_state.get("jobs", [])
    optimize_tool.resume_data = shared_state.get("resume_data")
    optimize_tool.jobs = shared_state.get("jobs", [])
    optimize_tool.match_result = shared_state.get("match_result")
    evaluate_tool.resume_data = shared_state.get("resume_data")
    generate_tool.resume_data = shared_state.get("resume_data")
    generate_tool.optimized_sections = shared_state.get("optimized_sections", {})
    judge_tool.resume_data = shared_state.get("resume_data")
    judge_tool.jobs = shared_state.get("jobs", [])
    judge_tool.optimized_sections = shared_state.get("optimized_sections", {})

    return [parse_tool, search_tool, fetch_url_tool, match_tool, optimize_tool, evaluate_tool, judge_tool, generate_tool]
