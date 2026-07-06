"""
Resume Optimizer AI Agent - 主入口

支持两种运行模式：
1. Pipeline 模式（固定流水线）：python -m src.main --resume xxx.pdf --keyword Python
2. Agent 模式（AI Agent 自主决策）：python -m src.main --agent
3. 交互式 Agent 模式：python -m src.main --interactive
"""

import os
import sys
import argparse
from pathlib import Path

# 强制 HuggingFace 离线模式（避免模型加载时联网超时）
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 将项目根目录加入路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

from config.config import settings
from src.parsers.resume_parser import ResumeParser
from src.scrapers.job_scraper import get_scraper, JobPosting
from src.agents.matcher import JobResumeMatcher
from src.agents.optimizer import ResumeOptimizer
from src.agents.evaluator import ResumeEvaluator
from src.generators.resume_generator import ResumeGenerator


def setup_logger():
    """配置日志"""
    logger.remove()
    logger.add(
        sys.stdout,
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logger.add(log_dir / "app.log", rotation="10 MB", retention="7 days", level="DEBUG")


class ResumeOptimizerPipeline:
    """简历优化流水线（Pipeline 模式）"""

    def __init__(self):
        setup_logger()
        self.parser = ResumeParser()
        self.matcher = JobResumeMatcher()
        self.optimizer = ResumeOptimizer()
        self.generator = ResumeGenerator()
        self.evaluator = ResumeEvaluator()
        logger.info("Resume Optimizer Pipeline 初始化完成")

    def run(
        self,
        resume_path: str,
        job_keyword: str,
        job_location: str = "",
        min_salary: int = 0,
        industry: str = "",
        scraper_source: str = "mock",
        output_format: str = "docx",
        max_jobs: int = 10,
    ) -> dict:
        """执行完整优化流程"""
        results = {}

        # ========== 步骤 1: 解析简历 ==========
        logger.info(f"步骤 1: 解析简历 [{resume_path}]")
        resume = self.parser.parse(resume_path)
        logger.success(f"简历解析完成，共 {len(resume.sections)} 个模块")
        results["resume"] = resume

        # ========== 步骤 2: 抓取职位 ==========
        filter_info = []
        if job_location:
            filter_info.append(f"地点: {job_location}")
        if min_salary > 0:
            filter_info.append(f"最低薪资: {min_salary}元/月")
        if industry:
            filter_info.append(f"行业: {industry}")
        filter_str = ", ".join(filter_info) if filter_info else "不限"
        logger.info(f"步骤 2: 抓取职位 [关键词: {job_keyword}, 筛选: {filter_str}]")
        scraper = get_scraper(scraper_source)
        jobs = scraper.fetch(
            keyword=job_keyword,
            location=job_location,
            min_salary=min_salary,
            industry=industry,
            max_jobs=max_jobs,
        )
        logger.success(f"抓取到 {len(jobs)} 个职位")
        results["jobs"] = jobs

        # ========== 步骤 3: 匹配分析 ==========
        logger.info("步骤 3: 简历与职位匹配分析")
        matches = self.matcher.match(resume, jobs)
        best_match = matches[0] if matches else None

        if best_match:
            logger.success(
                f"最佳匹配职位: {best_match.job_title} @ {best_match.company} "
                f"(匹配度: {best_match.overall_score * 100:.1f}%)"
            )
            logger.info(f"已匹配技能: {', '.join(best_match.matched_skills)}")
            if best_match.missing_skills:
                logger.warning(f"缺少技能: {', '.join(best_match.missing_skills)}")
        else:
            logger.warning("未找到匹配职位")
        results["matches"] = matches

        # ========== 步骤 4: LLM 优化 ==========
        logger.info("步骤 4: LLM 智能优化")
        target_job = jobs[0] if jobs else None

        if target_job:
            optimized_data = self.optimizer.optimize(
                resume=resume,
                job=target_job,
                match=best_match,
            )
            logger.success(f"优化完成，共优化 {len(optimized_data.get('optimized', {}))} 个模块")
            results["optimized"] = optimized_data
        else:
            logger.error("没有可用职位数据，无法进行优化")
            return results

        # ========== 步骤 4.5: 优化效果评估 ==========
        logger.info("步骤 4.5: 优化效果评估（LLM-as-Judge + 匹配度对比）")
        try:
            optimized_sections = optimized_data.get("optimized", {})
            if optimized_sections and target_job:
                # LLM-as-Judge 评估
                judge_result = self.evaluator.judge(resume, optimized_sections, target_job)
                # 匹配度前后对比
                compare_result = self.evaluator.compare_match_scores(resume, optimized_sections, target_job)
                # 格式化报告
                judge_report = self.evaluator.format_judge_report(judge_result, compare_result)
                logger.info(f"评估完成: {judge_result.overall_before:.1f} → {judge_result.overall_after:.1f}/10")
                results["judge_result"] = judge_result
                results["compare_result"] = compare_result
                results["judge_report"] = judge_report
                print(judge_report)
            else:
                logger.warning("无可用的优化内容，跳过评估")
        except Exception as e:
            logger.warning(f"优化效果评估失败（不影响后续流程）: {e}")

        # ========== 步骤 5: 生成输出 ==========
        logger.info(f"步骤 5: 生成输出文件 [格式: {output_format}]")

        original_info = {
            **resume.sections,
            "name": resume.name,
            "email": resume.email,
            "phone": resume.phone,
        }

        if output_format == "docx":
            output_path = self.generator.generate_docx(optimized_data, original_info)
        else:
            output_path = self.generator.generate_markdown(optimized_data, original_info)
            md_path = Path(settings.output_dir) / "optimized_resume.md"
            md_path.write_text(output_path, encoding="utf-8")
            output_path = str(md_path)

        logger.success(f"优化简历已生成: {output_path}")
        results["output_resume"] = output_path

        report_md = self.generator.generate_report(optimized_data, best_match, jobs=jobs)
        report_path = Path(settings.output_dir) / "optimization_report.md"
        report_path.write_text(report_md, encoding="utf-8")
        logger.success(f"优化报告已生成: {report_path}")
        results["output_report"] = str(report_path)

        logger.info("=" * 50)
        logger.info("简历优化流程全部完成!")
        logger.info("=" * 50)
        self._print_summary(results)

        return results

    def _print_summary(self, results: dict):
        """打印执行摘要"""
        print("\n" + "=" * 60)
        print("  简历优化执行摘要")
        print("=" * 60)

        if "matches" in results and results["matches"]:
            best = results["matches"][0]
            print(f"\n目标职位: {best.job_title}")
            print(f"公司名称: {best.company}")
            print(f"综合匹配度: {best.overall_score * 100:.1f}%")
            print(f"技能匹配度: {best.skill_match_score * 100:.1f}%")
            print(f"已匹配技能: {', '.join(best.matched_skills)}")
            if best.missing_skills:
                print(f"缺少技能: {', '.join(best.missing_skills)}")

        # 打印优化说明（含校验结果）
        if "optimized" in results:
            opt = results["optimized"]
            notes = opt.get("optimization_notes", [])
            if notes:
                print("\n优化详情:")
                for note in notes:
                    print(f"  {note}")

            report = opt.get("report", {})
            warnings = report.get("warnings", [])
            if warnings:
                print("\n注意事项:")
                for w in warnings:
                    print(f"  ⚠️ {w}")

        if "output_resume" in results:
            print(f"\n优化简历: {results['output_resume']}")
        if "output_report" in results:
            print(f"优化报告: {results['output_report']}")
        print("=" * 60 + "\n")


def run_agent_mode(resume_path: str = None, keyword: str = "", location: str = "", max_steps: int = 15):
    """
    Agent 模式：单次任务执行
    Agent 自主规划并执行任务
    """
    from src.agents.resume_agent import ResumeAgent

    setup_logger()
    agent = ResumeAgent(max_steps=max_steps)

    # 如果命令行没有提供关键词和地点，交互式询问用户
    if not keyword:
        keyword = input("请输入目标职位关键词（如：项目经理、Python后端）：").strip()
    if not location:
        location = input("请输入目标工作地点（如：成都、北京，留空表示不限）：").strip()

    # 构建用户输入
    parts = [f"请帮我优化简历，文件路径是 {resume_path}。"]
    if keyword:
        parts.append(f"目标职位是{keyword}。")
    if location:
        parts.append(f"地点在{location}。")
    user_input = "".join(parts)

    print("\n" + "=" * 60)
    print("  Agent 模式 - 单次任务执行")
    print("=" * 60)
    print(f"\n用户输入: {user_input}\n")

    response = agent.run(user_input)

    print("\n" + "-" * 60)
    print("Agent 回复:")
    print("-" * 60)
    print(response)
    print("-" * 60)

    return response


def run_interactive_mode(max_steps: int = 15):
    """
    交互式 Agent 模式：多轮对话
    """
    from src.agents.interactive import InteractiveAgent

    setup_logger()
    interactive = InteractiveAgent(max_steps=max_steps)
    interactive.start()


def create_demo_resume():
    """创建一个示例简历用于测试"""
    demo_dir = Path("data/resumes")
    demo_dir.mkdir(parents=True, exist_ok=True)
    demo_path = demo_dir / "demo_resume.txt"

    content = """张三
邮箱: zhangsan@example.com
电话: 13800138000

求职意向
Python后端开发工程师

教育背景
XX大学 计算机科学与技术 本科 2018-2022

工作经历
某科技公司 Python开发工程师 2022-至今
- 负责公司内部管理系统的后端开发
- 使用Flask框架搭建RESTful API
- 维护MySQL数据库，编写复杂查询

项目经验
电商订单系统
- 使用Python和Flask开发订单管理模块
- 实现用户下单、支付、退款等核心流程
- 使用Redis缓存热点数据

专业技能
- 熟悉Python编程，了解Java
- 掌握Flask、Django等Web框架
- 熟悉MySQL、Redis数据库
- 了解Linux基本操作

自我评价
工作认真负责，有较强的学习能力，喜欢钻研新技术。
"""
    demo_path.write_text(content, encoding="utf-8")
    return str(demo_path)


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="简历智能优化 Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
运行模式:
  1. Pipeline 模式（固定流水线）:
     python -m src.main --resume ./简历.pdf --keyword Python --location 北京

  2. Agent 模式（AI Agent 自主决策）:
     python -m src.main --agent --resume ./简历.pdf

  3. 交互式 Agent 模式（多轮对话）:
     python -m src.main --interactive

  4. Demo 模式（快速体验）:
     python -m src.main --demo
        """
    )

    # 模式选择
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--agent", action="store_true", help="Agent 模式：AI 自主规划执行")
    mode_group.add_argument("--interactive", action="store_true", help="交互式 Agent 模式：多轮对话")
    mode_group.add_argument("--demo", action="store_true", help="Demo 模式：使用示例数据快速体验")

    # Pipeline 模式参数
    parser.add_argument("--resume", "-r", type=str, help="简历文件路径")
    parser.add_argument("--keyword", "-k", type=str, default=None, help="职位搜索关键词（Agent模式会交互询问）")
    parser.add_argument("--location", "-l", type=str, default="", help="工作地点")
    parser.add_argument("--min-salary", type=int, default=0, help="最低薪资要求（元/月，0表示不限）")
    parser.add_argument("--industry", type=str, default="", help="行业筛选（如：互联网、金融、房地产，空表示不限）")
    parser.add_argument("--source", "-s", type=str, default="zhaopin", choices=["mock", "boss", "zhipin", "zhaopin", "zhilian"], help="数据来源（默认智联招聘）")
    parser.add_argument("--format", "-f", type=str, default="docx", choices=["docx", "markdown"], help="输出格式")
    parser.add_argument("--max-jobs", type=int, default=10, help="抓取职位数量（全部访问详情页，上限20）")
    parser.add_argument("--max-steps", type=int, default=15, help="Agent 最大执行步数")

    args = parser.parse_args()

    # 根据模式执行
    if args.interactive:
        run_interactive_mode(max_steps=args.max_steps)
    elif args.agent:
        resume_path = args.resume
        if not resume_path:
            resume_path = create_demo_resume()
            print(f"[INFO] 未指定简历，使用示例简历: {resume_path}")
        run_agent_mode(
            resume_path=resume_path,
            keyword=args.keyword or "",
            location=args.location or "",
            max_steps=args.max_steps,
        )
    elif args.demo:
        # Demo 模式使用 Pipeline + Mock 数据
        resume_path = create_demo_resume()
        print(f"[INFO] Demo 模式，使用示例简历: {resume_path}")
        pipeline = ResumeOptimizerPipeline()
        pipeline.run(
            resume_path=resume_path,
            job_keyword=args.keyword or "Python",
            job_location=args.location,
            min_salary=args.min_salary,
            industry=args.industry,
            scraper_source="mock",
            output_format=args.format,
            max_jobs=0,
        )
    else:
        # 默认 Pipeline 模式
        if not args.resume:
            print("[INFO] 未指定简历文件，使用示例简历。如需指定请使用 --resume 参数")
            resume_path = create_demo_resume()
            source = "mock"
        else:
            resume_path = args.resume
            source = args.source

        pipeline = ResumeOptimizerPipeline()
        pipeline.run(
            resume_path=resume_path,
            job_keyword=args.keyword or "Python",
            job_location=args.location,
            min_salary=args.min_salary,
            industry=args.industry,
            scraper_source=source,
            output_format=args.format,
            max_jobs=args.max_jobs,
        )


if __name__ == "__main__":
    main()
