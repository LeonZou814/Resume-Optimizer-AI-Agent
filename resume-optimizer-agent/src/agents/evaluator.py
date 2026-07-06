"""
简历优化效果评估器

提供两种评估能力：
1. LLM-as-Judge：用 LLM 对优化前后的简历进行多维度对比评分
2. 前后对比：自动计算优化前后的匹配度变化
"""

import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

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
from src.agents.matcher import JobResumeMatcher, MatchResult


@dataclass
class JudgeDimension:
    """单个评估维度"""
    name: str
    before_score: int   # 1-10
    after_score: int    # 1-10
    analysis: str       # 分析说明


@dataclass
class JudgeResult:
    """LLM-as-Judge 评估结果"""
    overall_before: float         # 优化前综合分 (1-10)
    overall_after: float          # 优化后综合分 (1-10)
    improvement: float            # 提升幅度
    dimensions: List[Dict]        # 各维度评分
    verdict: str                  # 总结性评语
    strengths: List[str]          # 优化后的亮点
    remaining_issues: List[str]   # 仍可改进的方向


class ResumeEvaluator:
    """
    简历优化效果评估器

    使用方式：
    1. judge() — LLM-as-Judge 多维度对比评估
    2. compare_match_scores() — 匹配分数前后对比
    """

    def __init__(self):
        provider = settings.llm_provider.lower()
        if provider == "qwen":
            self.llm = ChatOpenAI(
                model=settings.qwen_model,
                temperature=0.2,
                api_key=settings.qwen_api_key,
                base_url=settings.qwen_base_url,
            )
        else:
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.2,
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
            )
        self.matcher = JobResumeMatcher()

    def judge(
        self,
        original_resume: ResumeData,
        optimized_sections: Dict[str, str],
        job: JobPosting,
    ) -> JudgeResult:
        """
        LLM-as-Judge 评估

        将原始简历和 optimised 内容同时呈现给 LLM，让它在 6 个维度上对比评分。

        :param original_resume: 原始简历数据
        :param optimized_sections: 优化后的模块内容 {模块名: 优化后文本}
        :param job: 目标职位
        :return: JudgeResult
        """
        # 构建优化后的简历全文（将优化模块替换原文）
        optimized_full_parts = []
        for section_name, section_content in original_resume.sections.items():
            if section_name in optimized_sections:
                optimized_full_parts.append(f"【{section_name}】\n{optimized_sections[section_name]}")
            else:
                optimized_full_parts.append(f"【{section_name}】\n{section_content}")
        optimized_text = "\n\n".join(optimized_full_parts)

        # 原始简历全文
        original_parts = [f"【{k}】\n{v}" for k, v in original_resume.sections.items()]
        original_text = "\n\n".join(original_parts)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一位拥有 15 年经验的资深 HR 总监和简历评审专家。
你的任务是对比同一份简历的「优化前」和「优化后」版本，结合目标职位要求，进行客观、专业的评估。

评估规则：
1. 每个维度打分 1-10 分，必须给出具体分数，不能含糊
2. 对比分析必须具体到文本内容，引用原文中的例子
3. 重点关注：优化是否提升了简历与目标职位的匹配度
4. 诚实评估：如果某个维度没有改善甚至变差，如实指出
5. 检查优化是否引入了原文不存在的信息（编造风险）"""),
            ("human", """请对比评估以下简历的优化前后版本。

## 目标职位
职位：{job_title} @ {company}
职位描述：{job_description}
职位要求：{job_requirements}

## 优化前简历
{original_resume}

## 优化后简历
{optimized_resume}

请从以下 6 个维度对比评分（1-10 分），并以 JSON 格式返回：

{{
    "dimensions": [
        {{
            "name": "职位匹配度",
            "before_score": 分数,
            "after_score": 分数,
            "analysis": "简要分析（1-2句话）"
        }},
        {{
            "name": "技能覆盖度",
            "before_score": 分数,
            "after_score": 分数,
            "analysis": "..."
        }},
        {{
            "name": "表述专业度",
            "before_score": 分数,
            "after_score": 分数,
            "analysis": "..."
        }},
        {{
            "name": "量化程度",
            "before_score": 分数,
            "after_score": 分数,
            "analysis": "..."
        }},
        {{
            "name": "结构清晰度",
            "before_score": 分数,
            "after_score": 分数,
            "analysis": "..."
        }},
        {{
            "name": "信息真实性",
            "before_score": 分数,
            "after_score": 分数,
            "analysis": "是否发现优化后出现了原文没有的信息"
        }}
    ],
    "verdict": "总结性评语（2-3句话，概括优化效果）",
    "strengths": ["优化后的亮点1", "亮点2"],
    "remaining_issues": ["仍可改进的方向1", "方向2"]
}}""")
        ])

        chain = prompt | self.llm | JsonOutputParser()

        try:
            result = chain.invoke({
                "job_title": job.title,
                "company": job.company,
                "job_description": job.description[:500],
                "job_requirements": job.requirements[:500],
                "original_resume": original_text[:3000],
                "optimized_resume": optimized_text[:3000],
            })

            # 解析结果
            dimensions = result.get("dimensions", [])
            before_scores = [d.get("before_score", 5) for d in dimensions]
            after_scores = [d.get("after_score", 5) for d in dimensions]

            overall_before = sum(before_scores) / len(before_scores) if before_scores else 5.0
            overall_after = sum(after_scores) / len(after_scores) if after_scores else 5.0
            improvement = overall_after - overall_before

            judge_result = JudgeResult(
                overall_before=round(overall_before, 1),
                overall_after=round(overall_after, 1),
                improvement=round(improvement, 1),
                dimensions=dimensions,
                verdict=result.get("verdict", ""),
                strengths=result.get("strengths", []),
                remaining_issues=result.get("remaining_issues", []),
            )

            logger.info(f"[Judge] 评估完成: {overall_before:.1f} → {overall_after:.1f} ({'+' if improvement >= 0 else ''}{improvement:.1f})")
            return judge_result

        except Exception as e:
            logger.error(f"[Judge] LLM 评估失败: {e}")
            return JudgeResult(
                overall_before=0,
                overall_after=0,
                improvement=0,
                dimensions=[],
                verdict=f"评估失败: {e}",
                strengths=[],
                remaining_issues=[],
            )

    def compare_match_scores(
        self,
        original_resume: ResumeData,
        optimized_sections: Dict[str, str],
        job: JobPosting,
    ) -> Dict:
        """
        匹配分数前后对比

        用语义匹配器分别计算原始简历和优化后简历与目标职位的匹配度。

        :return: 包含 before/after/diff 的字典
        """
        # 优化前匹配
        before_match = self.matcher.match(original_resume, [job])
        before_score = before_match[0].overall_score if before_match else 0
        before_detail = {
            "overall": round(before_score * 100, 1),
            "skill": round(before_match[0].skill_match_score * 100, 1) if before_match else 0,
            "semantic": round(before_match[0].experience_match_score * 100, 1) if before_match else 0,
            "matched_skills": before_match[0].matched_skills if before_match else [],
            "missing_skills": before_match[0].missing_skills if before_match else [],
        }

        # 构建优化后的 ResumeData（替换优化过的模块）
        optimized_resume = ResumeData(
            raw_text=original_resume.raw_text,
            sections={},
            name=original_resume.name,
            email=original_resume.email,
            phone=original_resume.phone,
        )
        for section_name, content in original_resume.sections.items():
            if section_name in optimized_sections:
                optimized_resume.sections[section_name] = optimized_sections[section_name]
            else:
                optimized_resume.sections[section_name] = content

        # 更新 raw_text 以反映优化后的内容
        optimized_resume.raw_text = " ".join(optimized_resume.sections.values())

        # 优化后匹配
        after_match = self.matcher.match(optimized_resume, [job])
        after_score = after_match[0].overall_score if after_match else 0
        after_detail = {
            "overall": round(after_score * 100, 1),
            "skill": round(after_match[0].skill_match_score * 100, 1) if after_match else 0,
            "semantic": round(after_match[0].experience_match_score * 100, 1) if after_match else 0,
            "matched_skills": after_match[0].matched_skills if after_match else [],
            "missing_skills": after_match[0].missing_skills if after_match else [],
        }

        # 计算变化
        diff = round((after_score - before_score) * 100, 1)
        new_matched = set(after_detail["matched_skills"]) - set(before_detail["matched_skills"])
        still_missing = set(after_detail["missing_skills"]) & set(before_detail["missing_skills"])

        result = {
            "before": before_detail,
            "after": after_detail,
            "diff": diff,
            "new_matched_skills": list(new_matched),
            "still_missing_skills": list(still_missing),
        }

        logger.info(f"[Compare] 匹配度对比: {before_detail['overall']}% → {after_detail['overall']}% ({'+' if diff >= 0 else ''}{diff}%)")
        return result

    def format_judge_report(self, judge_result: JudgeResult, compare_result: Dict = None) -> str:
        """
        将评估结果格式化为可读的文本报告
        """
        lines = []
        lines.append("=" * 50)
        lines.append("  简历优化效果评估报告")
        lines.append("=" * 50)

        # LLM-as-Judge 评分
        lines.append(f"\n【LLM-as-Judge 综合评分】")
        lines.append(f"  优化前: {judge_result.overall_before:.1f}/10")
        lines.append(f"  优化后: {judge_result.overall_after:.1f}/10")
        sign = "+" if judge_result.improvement >= 0 else ""
        lines.append(f"  提升幅度: {sign}{judge_result.improvement:.1f}")

        if judge_result.dimensions:
            lines.append(f"\n【各维度评分】")
            for d in judge_result.dimensions:
                name = d.get("name", "")
                # 强制转 int，防止 LLM 返回浮点数导致 "█" * 7.5 报错
                before = int(round(float(d.get("before_score", 0))))
                after = int(round(float(d.get("after_score", 0))))
                before = max(0, min(10, before))
                after = max(0, min(10, after))
                diff = after - before
                diff_str = f"+{diff}" if diff > 0 else str(diff)
                bar_before = "█" * before + "░" * (10 - before)
                bar_after = "█" * after + "░" * (10 - after)
                lines.append(f"  {name}:")
                lines.append(f"    前: {bar_before} {before}/10")
                lines.append(f"    后: {bar_after} {after}/10 ({diff_str})")
                if d.get("analysis"):
                    lines.append(f"    → {d['analysis']}")

        # 匹配度对比
        if compare_result:
            lines.append(f"\n【语义匹配度对比】")
            before = compare_result["before"]
            after = compare_result["after"]
            diff = compare_result["diff"]
            diff_str = f"+{diff}" if diff > 0 else str(diff)
            lines.append(f"  综合匹配度: {before['overall']}% → {after['overall']}% ({diff_str}%)")
            lines.append(f"  技能匹配度: {before['skill']}% → {after['skill']}%")

            new_matched = compare_result.get("new_matched_skills", [])
            if new_matched:
                lines.append(f"  新增匹配技能: {', '.join(new_matched)}")
            still_missing = compare_result.get("still_missing_skills", [])
            if still_missing:
                lines.append(f"  仍缺少技能: {', '.join(list(still_missing)[:5])}")

        # 总结
        if judge_result.verdict:
            lines.append(f"\n【评审总结】")
            lines.append(f"  {judge_result.verdict}")

        if judge_result.strengths:
            lines.append(f"\n【优化亮点】")
            for s in judge_result.strengths:
                lines.append(f"  ✓ {s}")

        if judge_result.remaining_issues:
            lines.append(f"\n【仍可改进】")
            for r in judge_result.remaining_issues:
                lines.append(f"  → {r}")

        lines.append("\n" + "=" * 50)
        return "\n".join(lines)
