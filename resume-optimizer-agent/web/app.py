"""
简历优化 Agent - Web 前端
基于 FastAPI 构建，支持 Pipeline / Agent / Interactive 三种模式
"""

# ── 必须在所有 import 之前设置 HuggingFace 离线模式 ──
# 否则 transformers/huggingface_hub 被间接导入后，再设置环境变量就无效了
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# ──────────────────────────────────────────────────────

import sys
import uuid
import json
import threading
import time
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

# 将项目根目录加入路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel

from loguru import logger

# ── 任务存储 ──────────────────────────────────────────────

tasks: Dict[str, dict] = {}
# 每个 task 的结构:
# {
#     "id": str,
#     "mode": str,  # pipeline / agent / interactive
#     "status": str,  # pending / running / completed / failed
#     "created_at": str,
#     "resume_path": str,
#     "params": dict,
#     "logs": list[str],
#     "result": {
#         "output_resume": str,      # pipeline
#         "output_report": str,      # pipeline
#         "summary": dict,           # pipeline
#         "agent_response": str,     # agent
#         "output_files": list,      # agent (新生成的文件)
#     },
#     "error": str,
#     "chat_history": list,  # interactive 模式的对话历史
# }

# 存储交互式 Agent 实例（需要跨请求保持状态）
interactive_agents: Dict[str, object] = {}


def create_task(mode: str, resume_path: str, params: dict) -> str:
    task_id = str(uuid.uuid4())[:8]
    tasks[task_id] = {
        "id": task_id,
        "mode": mode,
        "status": "pending",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "resume_path": resume_path,
        "params": params,
        "logs": [],
        "result": {},
        "error": "",
        "chat_history": [],
        "step_info": {
            "current_step": 0,
            "max_steps": 0,
            "step_mode": "fixed",  # "fixed" = 固定步数(Pipeline/Focus), "dynamic" = 动态步数(Agent)
            "phase": "",
            "tool_name": "",
            "tool_args": {},
            "last_activity": time.time(),
            "started_at": 0,
        },
    }
    return task_id


def update_step_info(task_id: str, **kwargs):
    """更新任务的步骤信息"""
    if task_id in tasks:
        info = tasks[task_id]["step_info"]
        info.update(kwargs)
        info["last_activity"] = time.time()


def add_log(task_id: str, message: str):
    if task_id in tasks:
        timestamp = datetime.now().strftime("%H:%M:%S")
        tasks[task_id]["logs"].append(f"[{timestamp}] {message}")


# ── 日志捕获（将 loguru 输出重定向到任务日志） ────────────

_task_context = threading.local()


def _task_log_sink(message):
    """loguru sink：将日志转发到当前线程对应的任务"""
    task_id = getattr(_task_context, 'task_id', None)
    if task_id and task_id in tasks:
        text = str(message).strip()
        if text:
            add_log(task_id, text)


_log_sink_id = logger.add(_task_log_sink, level="INFO", format="{time:HH:mm:ss} | {level: <8} | {message}")


# ── 后台任务执行 ──────────────────────────────────────────

def run_pipeline_task(task_id: str):
    """后台执行 Pipeline 模式"""
    from src.main import ResumeOptimizerPipeline

    _task_context.task_id = task_id
    task = tasks[task_id]
    task["status"] = "running"
    params = task["params"]

    update_step_info(task_id, started_at=time.time(), max_steps=5)

    pipeline_steps = [
        "解析简历",
        "搜索职位",
        "匹配分析",
        "LLM 优化简历",
        "生成输出文件",
    ]

    try:
        add_log(task_id, "启动 Pipeline 模式...")
        add_log(task_id, f"简历: {task['resume_path']}")
        add_log(task_id, f"关键词: {params.get('keyword', '')}, 地点: {params.get('location', '')}")

        pipeline = ResumeOptimizerPipeline()

        # 步骤 1: 解析简历
        update_step_info(task_id, current_step=1, phase=pipeline_steps[0])
        add_log(task_id, "步骤 1: 解析简历...")
        from src.parsers.resume_parser import ResumeParser
        parser = ResumeParser()
        resume_data = parser.parse(task["resume_path"])
        add_log(task_id, f"  解析成功: {len(resume_data.sections)} 个模块")

        # 步骤 2: 搜索职位
        update_step_info(task_id, current_step=2, phase=pipeline_steps[1])
        add_log(task_id, f"步骤 2: 搜索职位（{params.get('source', 'zhaopin')}）...")
        from src.scrapers.job_scraper import get_scraper
        scraper = get_scraper(params.get("source", "zhaopin"))
        jobs = scraper.fetch(
            keyword=params.get("keyword", "Python"),
            location=params.get("location", ""),
            min_salary=int(params.get("min_salary", 0)),
            industry=params.get("industry", ""),
            max_jobs=int(params.get("max_jobs", 10)),
        )
        add_log(task_id, f"  搜索到 {len(jobs)} 个职位")

        # 步骤 3: 匹配分析
        update_step_info(task_id, current_step=3, phase=pipeline_steps[2])
        add_log(task_id, "步骤 3: 匹配分析...")
        from src.agents.matcher import JobResumeMatcher
        matcher = JobResumeMatcher()
        matches = matcher.match(resume_data, jobs)
        best_match = matches[0] if matches else None
        if best_match:
            add_log(task_id, f"  最佳匹配: {best_match.job_title} @ {best_match.company} ({best_match.overall_score * 100:.1f}%)")
        else:
            add_log(task_id, "  未找到匹配职位")

        # 步骤 4: LLM 优化
        update_step_info(task_id, current_step=4, phase=pipeline_steps[3])
        add_log(task_id, "步骤 4: LLM 智能优化...")
        from src.agents.optimizer import ResumeOptimizer
        optimizer = ResumeOptimizer()
        target_job = jobs[0] if jobs else None
        optimized_data = {}
        if target_job and best_match:
            optimized_data = optimizer.optimize(
                resume=resume_data,
                job=target_job,
                match=best_match,
            )
            add_log(task_id, f"  优化完成: {len(optimized_data.get('optimized', {}))} 个模块")
        else:
            add_log(task_id, "  跳过优化（无职位数据）")

        # 步骤 5: 生成输出文件
        update_step_info(task_id, current_step=5, phase=pipeline_steps[4])
        add_log(task_id, "步骤 5: 生成输出文件...")
        from src.generators.resume_generator import ResumeGenerator
        from pathlib import Path
        from datetime import datetime
        generator = ResumeGenerator()
        output_format = params.get("format", "docx")
        output_dir = Path(__file__).resolve().parent.parent / "data" / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        results = {"output_resume": "", "output_report": "", "matches": matches}

        if optimized_data.get("optimized"):
            optimized_sections = optimized_data.get("optimized", {})
            original_info = {
                **resume_data.sections,
                "name": resume_data.name,
                "email": resume_data.email,
                "phone": resume_data.phone,
            }

            if output_format == "docx":
                resume_path = generator.generate_docx(
                    optimized_sections, original_info,
                    filename=f"optimized_resume_{timestamp}",
                )
            else:
                md_content = generator.generate_markdown(optimized_sections, original_info)
                resume_path = str(output_dir / f"optimized_resume_{timestamp}.md")
                Path(resume_path).write_text(md_content, encoding="utf-8")
            results["output_resume"] = resume_path
            add_log(task_id, f"  简历已生成: {resume_path}")

            # 生成评估报告
            if target_job:
                from src.agents.evaluator import ResumeEvaluator
                evaluator = ResumeEvaluator()
                judge_result = evaluator.judge(resume_data, optimized_sections, target_job)
                compare_result = evaluator.compare_match_scores(resume_data, optimized_sections, target_job)
                judge_report = evaluator.format_judge_report(judge_result, compare_result)
                report_path = output_dir / f"evaluation_report_latest.md"
                report_path.write_text(judge_report, encoding="utf-8")
                results["output_report"] = str(report_path)
                add_log(task_id, f"  评估报告: {judge_result.overall_before:.1f} → {judge_result.overall_after:.1f}/10")

        # 收集结果
        task["result"] = {
            "output_resume": results.get("output_resume", ""),
            "output_report": results.get("output_report", ""),
        }

        if results.get("matches"):
            best = results["matches"][0]
            task["result"]["summary"] = {
                "job_title": best.job_title,
                "company": best.company,
                "overall_score": f"{best.overall_score * 100:.1f}%",
                "skill_score": f"{best.skill_match_score * 100:.1f}%",
                "matched_skills": best.matched_skills,
                "missing_skills": best.missing_skills,
            }

        task["status"] = "completed"
        update_step_info(task_id, current_step=5, phase="已完成")
        add_log(task_id, "Pipeline 执行完成!")

    except Exception as e:
        task["status"] = "failed"
        task["error"] = str(e)
        update_step_info(task_id, phase="执行失败")
        add_log(task_id, f"执行失败: {e}")
        logger.exception(f"Pipeline task {task_id} failed")
    finally:
        _task_context.task_id = None


def run_agent_task(task_id: str):
    """后台执行 Agent 模式"""
    from src.agents.resume_agent import ResumeAgent

    _task_context.task_id = task_id
    task = tasks[task_id]
    task["status"] = "running"
    params = task["params"]

    # 初始化步骤信息（Agent 步数不固定，15 只是安全上限）
    update_step_info(task_id, started_at=time.time(), max_steps=15, step_mode="dynamic")

    def step_callback(**kwargs):
        """Agent 步骤回调：将内部状态同步到前端 + 写入日志"""
        update_step_info(task_id, **kwargs)
        # 同时将关键步骤写入日志（不再依赖 loguru sink 的 threading.local 转发）
        phase = kwargs.get("phase", "")
        tool_name = kwargs.get("tool_name", "")
        tool_args = kwargs.get("tool_args", {})
        if tool_name and phase.startswith("执行工具"):
            args_summary = ", ".join(f"{k}={v}" for k, v in tool_args.items()) if tool_args else "无参数"
            add_log(task_id, f"调用工具: {tool_name}({args_summary})")
        elif tool_name and "返回完成" in phase:
            add_log(task_id, f"  ✓ {tool_name} 完成")
        elif phase and not tool_name:
            add_log(task_id, phase)

    try:
        keyword = params.get("keyword", "")
        location = params.get("location", "")

        add_log(task_id, "启动 Agent 模式...")
        add_log(task_id, f"简历: {task['resume_path']}")
        add_log(task_id, f"目标: {keyword} @ {location}")

        agent = ResumeAgent(max_steps=15)

        # 构建用户输入
        parts = [f"请帮我优化简历，文件路径是 {task['resume_path']}。"]
        if keyword:
            parts.append(f"目标职位是「{keyword}」")
        if location:
            parts.append(f"工作地点在「{location}」")
        max_jobs_val = params.get("max_jobs", 10)
        parts.append(f"搜索职位数量：{max_jobs_val} 个（全部访问详情页）。使用默认值即可，不需要再询问。")
        user_input = "".join(parts)

        # 记录执行前的输出目录文件
        output_dir = project_root / "data" / "output"
        files_before = set()
        if output_dir.exists():
            files_before = {f.name for f in output_dir.iterdir() if f.is_file()}

        response = agent.run(user_input, step_callback=step_callback)

        # 检查新生成的文件
        files_after = set()
        if output_dir.exists():
            files_after = {f.name for f in output_dir.iterdir() if f.is_file()}
        new_files = list(files_after - files_before)

        # 固定文件名的报告可能已存在于 files_before 中，但内容已更新，需强制加入
        report_name = "evaluation_report_latest.md"
        if report_name in files_after and report_name not in new_files:
            new_files.append(report_name)

        task["result"] = {
            "agent_response": response,
            "output_files": new_files,
        }
        task["status"] = "completed"
        update_step_info(task_id, phase="已完成", current_step=task["step_info"]["current_step"])
        add_log(task_id, "Agent 执行完成!")

    except Exception as e:
        task["status"] = "failed"
        task["error"] = str(e)
        update_step_info(task_id, phase="执行失败")
        add_log(task_id, f"执行失败: {e}")
        logger.exception(f"Agent task {task_id} failed")
    finally:
        _task_context.task_id = None


def run_task(task_id: str):
    """根据模式分发任务"""
    task = tasks[task_id]
    mode = task["mode"]

    if mode == "pipeline":
        run_pipeline_task(task_id)
    elif mode == "agent":
        run_agent_task(task_id)
    elif mode == "focus":
        run_focus_task(task_id)
    elif mode == "interactive":
        # 交互式模式：任务执行完全由 /api/chat 驱动，这里只需保持 running 状态
        task["status"] = "running"
        add_log(task_id, "交互式 Agent 已就绪，等待用户消息...")
        # 线程保持存活，等待 chat endpoint 完成所有对话
        # 当 chat 中 agent 生成文件后，由 chat endpoint 更新 task 状态
    else:
        task["status"] = "failed"
        task["error"] = f"不支持的模式: {mode}"


def run_focus_task(task_id: str):
    """后台执行 Focus 模式（从 URL 抓取职位 → 全职位合并分析 → 优化 → 评估）"""
    from src.parsers.resume_parser import ResumeParser
    from src.scrapers.job_scraper import URLJobScraper
    from src.agents.matcher import JobResumeMatcher
    from src.agents.optimizer import ResumeOptimizer
    from src.agents.evaluator import ResumeEvaluator
    from src.generators.resume_generator import ResumeGenerator
    from pathlib import Path as PathLib

    _task_context.task_id = task_id
    task = tasks[task_id]
    task["status"] = "running"
    params = task["params"]

    update_step_info(task_id, started_at=time.time(), max_steps=7, step_mode="dynamic")

    try:
        urls = params.get("urls", "")
        url_list = [u.strip() for u in urls.split(",") if u.strip()]

        add_log(task_id, "启动 Focus 模式...")
        add_log(task_id, f"简历: {task['resume_path']}")
        add_log(task_id, f"目标 URL 数量: {len(url_list)}")

        # 1. 解析简历
        update_step_info(task_id, current_step=1, phase="解析简历")
        add_log(task_id, "步骤 1: 解析简历...")
        parser = ResumeParser()
        resume_data = parser.parse(task["resume_path"])
        add_log(task_id, f"  解析成功: {len(resume_data.sections)} 个模块")

        # 2. 从 URL 抓取职位
        update_step_info(task_id, current_step=2, phase="抓取职位信息")
        add_log(task_id, "步骤 2: 从 URL 抓取职位信息...")
        scraper = URLJobScraper()
        jobs = scraper.fetch(urls=url_list)
        add_log(task_id, f"  抓取成功: {len(jobs)} 个职位")

        if not jobs:
            raise ValueError("未能从提供的 URL 中抓取到任何职位信息，请检查 URL 是否正确")

        # 3. 全职位匹配分析（排序选最佳）
        update_step_info(task_id, current_step=3, phase="全职位匹配分析")
        add_log(task_id, f"步骤 3: 匹配全部 {len(jobs)} 个职位...")
        matcher = JobResumeMatcher()
        all_matches = matcher.match(resume_data, jobs)
        all_matches.sort(key=lambda r: r.overall_score, reverse=True)
        best_match = all_matches[0]
        add_log(task_id, f"  最佳匹配: {best_match.job_title} @ {best_match.company} ({best_match.overall_score * 100:.1f}%)")
        for i, m in enumerate(all_matches[:5], 1):
            add_log(task_id, f"  [{i}] {m.job_title} @ {m.company} | {m.overall_score * 100:.1f}%")

        # 4. 全职位 LLM 合并分析 + 优化
        update_step_info(task_id, current_step=4, phase="LLM 合并分析 + 优化")
        add_log(task_id, f"步骤 4: 全职位 LLM 合并分析（{len(jobs)} 个职位）...")
        optimizer = ResumeOptimizer()

        # 分析所有职位
        analyses = []
        job_titles = []
        for j in jobs:
            analysis = optimizer.analyze_job(j)
            analyses.append(analysis)
            job_titles.append(f"{j.title} @ {j.company}")
        add_log(task_id, f"  {len(jobs)} 个职位分析完成，开始 LLM 合并...")

        # LLM 合并
        merged_analysis = optimizer.merge_job_analyses_with_llm(analyses, job_titles=job_titles)
        add_log(task_id, f"  合并完成: 必需技能 {len(merged_analysis.get('required_skills', []))} 项, "
                         f"核心职责 {len(merged_analysis.get('core_responsibilities', []))} 项")

        # 逐模块优化（使用合并分析）
        optimized_sections = {}
        target_modules = ["专业技能", "工作经历", "项目经历", "自我评价", "个人优势"]
        for module_name in list(resume_data.sections.keys()):
            # 只优化目标模块
            is_target = any(kw in module_name for kw in ["技能", "工作", "项目", "评价", "优势", "经历", "实习"])
            if not is_target:
                continue
            add_log(task_id, f"  优化模块: {module_name}...")
            result = optimizer.optimize(
                resume_data, jobs[0], best_match,
                target_section=module_name,
                job_analysis=merged_analysis,
            )
            opt = result.get("optimized", {})
            if module_name in opt:
                optimized_sections[module_name] = opt[module_name]
                add_log(task_id, f"    ✓ {module_name} 优化完成")

        add_log(task_id, f"  共优化 {len(optimized_sections)} 个模块")

        # 5. LLM-as-Judge 评估
        update_step_info(task_id, current_step=5, phase="LLM 评估优化效果")
        add_log(task_id, "步骤 5: LLM-as-Judge 评估...")
        evaluator = ResumeEvaluator()
        target_job = jobs[0]  # 用第一个职位作为 judge 目标
        judge_result = evaluator.judge(resume_data, optimized_sections, target_job)
        compare_result = evaluator.compare_match_scores(resume_data, optimized_sections, target_job)
        judge_report = evaluator.format_judge_report(judge_result, compare_result)
        add_log(task_id, f"  评估结果: {judge_result.overall_before:.1f} → {judge_result.overall_after:.1f}/10 "
                         f"(+{judge_result.improvement:.1f})")

        # 6. 闭环反馈（如有薄弱维度则重优化一次）
        update_step_info(task_id, current_step=6, phase="闭环反馈检查")
        weak_directives = evaluator._generate_reoptimize_directives(judge_result) if hasattr(evaluator, '_generate_reoptimize_directives') else []
        if not weak_directives:
            # 尝试从 judge_result 中提取
            weak_directives = []
            if hasattr(judge_result, 'dimensions') and judge_result.dimensions:
                for d in judge_result.dimensions:
                    if d.get("after_score", 0) < 7:
                        weak_directives.append(d)

        if weak_directives:
            add_log(task_id, f"步骤 6: 检测到 {len(weak_directives)} 个薄弱维度，执行闭环重优化...")
            # 找到最薄弱的维度对应的模块进行重优化
            dimension_to_section = {
                "技能覆盖度": ["专业技能", "技能", "专业能力"],
                "量化程度": ["工作经历", "项目经历", "实习经历"],
                "表述专业度": ["工作经历", "项目经历"],
                "结构清晰度": ["工作经历", "项目经历"],
                "职位匹配度": ["个人优势", "自我评价"],
            }
            reoptimized_count = 0
            for d in weak_directives[:2]:  # 最多重优化 2 个模块
                dim_name = d.get("name", "")
                section_keywords = dimension_to_section.get(dim_name, ["工作经历"])
                for module_name in resume_data.sections.keys():
                    if any(kw in module_name for kw in section_keywords):
                        add_log(task_id, f"  重优化: {module_name}（针对 {dim_name}）...")
                        result = optimizer.optimize(
                            resume_data, jobs[0], best_match,
                            target_section=module_name,
                            job_analysis=merged_analysis,
                        )
                        opt = result.get("optimized", {})
                        if module_name in opt:
                            optimized_sections[module_name] = opt[module_name]
                            reoptimized_count += 1
                        break
            if reoptimized_count > 0:
                add_log(task_id, f"  闭环重优化完成: {reoptimized_count} 个模块")
                # 重新评估
                judge_result = evaluator.judge(resume_data, optimized_sections, target_job)
                compare_result = evaluator.compare_match_scores(resume_data, optimized_sections, target_job)
                judge_report = evaluator.format_judge_report(judge_result, compare_result)
                add_log(task_id, f"  重评估: {judge_result.overall_before:.1f} → {judge_result.overall_after:.1f}/10")
            else:
                add_log(task_id, "步骤 6: 无需重优化")
        else:
            add_log(task_id, "步骤 6: 所有维度达标，无需重优化")

        # 7. 生成输出文件
        update_step_info(task_id, current_step=7, phase="生成输出文件")
        add_log(task_id, "步骤 7: 生成输出文件...")
        generator = ResumeGenerator()
        original_info = {
            **resume_data.sections,
            "name": resume_data.name,
            "email": resume_data.email,
            "phone": resume_data.phone,
        }
        output_format = params.get("format", "docx")
        timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PathLib(__file__).resolve().parent.parent / "data" / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        if output_format == "docx":
            output_path = generator.generate_docx(optimized_sections, original_info)
        else:
            md_content = generator.generate_markdown(optimized_sections, original_info)
            output_path = str(output_dir / f"optimized_resume_{timestamp}.md")
            PathLib(output_path).write_text(md_content, encoding="utf-8")
        add_log(task_id, f"  简历已生成: {output_path}")

        # 保存评估报告（含职位信息）
        report_lines = [judge_report]
        report_lines.append("\n" + "=" * 50)
        report_lines.append("  【参考职位信息】")
        report_lines.append(f"  （以下 {len(jobs)} 个职位参与了合并优化）")
        report_lines.append("=" * 50)
        for i, job in enumerate(jobs, 1):
            report_lines.append(f"\n  [{i}] {job.title} @ {job.company}")
            if job.salary:
                report_lines.append(f"      薪资: {job.salary}")
            if job.location:
                report_lines.append(f"      地点: {job.location}")
            if job.skills:
                report_lines.append(f"      技能: {', '.join(job.skills[:8])}")
            if job.url:
                report_lines.append(f"      链接: {job.url}")
        report_lines.append("\n" + "=" * 50)

        report_path = str(output_dir / "evaluation_report_latest.md")
        PathLib(report_path).write_text("\n".join(report_lines), encoding="utf-8")
        add_log(task_id, f"  评估报告已保存")

        task["result"] = {
            "output_resume": output_path,
            "output_report": report_path,
        }
        if best_match:
            task["result"]["summary"] = {
                "job_title": best_match.job_title,
                "company": best_match.company,
                "overall_score": f"{best_match.overall_score * 100:.1f}%",
                "skill_score": f"{best_match.skill_match_score * 100:.1f}%",
                "matched_skills": best_match.matched_skills,
                "missing_skills": best_match.missing_skills,
            }

        task["status"] = "completed"
        update_step_info(task_id, current_step=7, phase="已完成")
        add_log(task_id, "Focus 模式执行完成!")

    except Exception as e:
        task["status"] = "failed"
        task["error"] = str(e)
        update_step_info(task_id, phase="执行失败")
        add_log(task_id, f"执行失败: {e}")
        logger.exception(f"Focus task {task_id} failed")
    finally:
        _task_context.task_id = None


# ── FastAPI 应用 ──────────────────────────────────────────

app = FastAPI(title="简历优化 Agent", version="1.0")

# 目录设置
upload_dir = Path(__file__).parent / "uploads"
upload_dir.mkdir(exist_ok=True)
output_dir = project_root / "data" / "output"
output_dir.mkdir(parents=True, exist_ok=True)
templates_dir = Path(__file__).parent / "templates"
templates_dir.mkdir(exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def index():
    """返回前端页面"""
    html_path = templates_dir / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/api/upload")
async def upload_resume(file: UploadFile = File(...)):
    """上传简历文件"""
    safe_name = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    save_path = upload_dir / safe_name
    content = await file.read()
    save_path.write_bytes(content)
    return {"path": str(save_path), "filename": file.filename}


@app.post("/api/run")
async def start_task(
    mode: str = Form(...),
    resume_path: str = Form(...),
    keyword: str = Form(default=""),
    location: str = Form(default=""),
    min_salary: int = Form(default=0),
    industry: str = Form(default=""),
    source: str = Form(default="zhaopin"),
    format: str = Form(default="docx"),
    max_jobs: int = Form(default=10),
    urls: str = Form(default=""),
):
    """启动优化任务"""
    params = {
        "keyword": keyword,
        "location": location,
        "min_salary": min_salary,
        "industry": industry,
        "source": source,
        "format": format,
        "max_jobs": max_jobs,
        "urls": urls,
    }

    task_id = create_task(mode, resume_path, params)

    # 后台线程执行
    thread = threading.Thread(target=run_task, args=(task_id,), daemon=True)
    thread.start()

    return {"task_id": task_id, "status": "pending"}


@app.get("/api/status/{task_id}")
async def get_status(task_id: str):
    """查询任务状态"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    return tasks[task_id]


@app.get("/api/tasks")
async def list_tasks():
    """列出所有任务"""
    return list(tasks.values())


@app.get("/api/download/{filename}")
async def download_file(filename: str):
    """下载输出文件"""
    # 依次在 output、uploads、项目根目录查找
    for directory in [output_dir, upload_dir, project_root]:
        file_path = directory / filename
        if file_path.exists() and file_path.is_file():
            return FileResponse(
                path=str(file_path),
                filename=filename,
                media_type="application/octet-stream",
            )
    raise HTTPException(status_code=404, detail="文件不存在")


@app.post("/api/chat/{task_id}")
async def chat(task_id: str, message: str = Form(...)):
    """Interactive 模式：发送消息"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    task = tasks[task_id]

    def step_callback(**kwargs):
        update_step_info(task_id, **kwargs)
        # 将思考步骤写入 chat_history，前端轮询时实时渲染
        phase = kwargs.get("phase", "")
        tool_name = kwargs.get("tool_name", "")
        tool_args = kwargs.get("tool_args", {})
        if not phase:
            return
        if tool_name and phase.startswith("执行工具"):
            args_summary = ", ".join(f"{k}={v}" for k, v in tool_args.items()) if tool_args else "无参数"
            # 截断过长的参数值
            if len(args_summary) > 120:
                args_summary = args_summary[:120] + "..."
            task["chat_history"].append({"role": "step", "content": f"调用工具: {tool_name}({args_summary})"})
        elif tool_name and "返回完成" in phase:
            task["chat_history"].append({"role": "step", "content": f"✓ {tool_name} 完成"})
        elif not tool_name:
            # 跳过"生成最终回复"等过渡性提示（最终回复本身会作为 assistant 消息展示）
            if "生成最终回复" not in phase:
                task["chat_history"].append({"role": "step", "content": phase})

    # 首次消息时创建 Agent 实例
    if task_id not in interactive_agents:
        _task_context.task_id = task_id
        from src.agents.resume_agent import ResumeAgent
        interactive_agents[task_id] = ResumeAgent(max_steps=30)
        task["status"] = "running"
        update_step_info(task_id, started_at=time.time(), max_steps=30, step_mode="dynamic")
        add_log(task_id, "交互式 Agent 已启动")

    task["chat_history"].append({"role": "user", "content": message})

    try:
        agent = interactive_agents[task_id]

        # 首次消息时自动注入简历路径，让 Agent 知道文件位置
        resume_path = task.get("resume_path", "")
        if resume_path and resume_path != "interactive" and len(task["chat_history"]) <= 2:
            message = f"[系统信息：用户上传的简历文件路径为 {resume_path}]\n\n{message}"

        # 在线程池中执行同步的 agent.run()，避免阻塞 async 事件循环
        # 同时在线程内设置 _task_context，确保 loguru sink 能正确转发日志
        def _run_in_thread():
            _task_context.task_id = task_id
            try:
                return agent.run(message, step_callback=step_callback)
            finally:
                _task_context.task_id = None

        import asyncio
        response_text = await asyncio.to_thread(_run_in_thread)

        task["chat_history"].append({"role": "assistant", "content": response_text})
        update_step_info(task_id, phase="等待用户输入")

        # 检查输出目录，将生成的文件加入 task result（供前端下载）
        output_dir = project_root / "data" / "output"
        if output_dir.exists():
            output_files = []
            for f in output_dir.iterdir():
                if f.is_file():
                    # 检查文件是否是本次会话中生成/更新的（修改时间在任务创建之后）
                    created_at = datetime.strptime(task["created_at"], "%Y-%m-%d %H:%M:%S")
                    if datetime.fromtimestamp(f.stat().st_mtime) >= created_at:
                        output_files.append(f.name)
            if output_files:
                task["result"]["output_files"] = output_files
                # 设置 output_report 供前端"查看报告"按钮使用
                if "evaluation_report_latest.md" in output_files:
                    task["result"]["output_report"] = str(output_dir / "evaluation_report_latest.md")

        return {"response": response_text, "output_files": task["result"].get("output_files", []), "output_report": task["result"].get("output_report", "")}
    except Exception as e:
        error_msg = f"执行出错: {e}"
        task["chat_history"].append({"role": "assistant", "content": error_msg})
        update_step_info(task_id, phase="执行出错")
        return {"response": error_msg, "error": True}


# ── 启动入口 ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("  简历优化 Agent - Web 界面")
    print("  访问地址: http://localhost:8080")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8080)
