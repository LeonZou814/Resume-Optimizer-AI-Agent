"""
Resume Agent - 真正的 AI Agent 核心

实现 ReAct (Reasoning + Acting) 循环：
1. 思考 (Thought)：分析当前状态，决定下一步
2. 行动 (Action)：调用工具执行操作
3. 观察 (Observation)：获取工具返回结果
4. 循环：直到任务完成或达到最大步数

与固定流水线的区别：
- Agent 自主决定调用什么工具、什么顺序、调用几次
- 能根据中间结果动态调整策略
- 支持自我评估和迭代优化
"""

import json
import re
from typing import List, Dict, Optional
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.config import settings

from src.agents.tools import (
    ParseResumeTool, SearchJobsTool, FetchURLJobsTool, MatchResumeTool, MatchAllJobsTool,
    OptimizeSectionTool, EvaluateResumeTool, JudgeResumeTool, GenerateResumeTool,
)
from src.agents.memory import AgentMemory


# ==================== Agent 系统提示词 ====================

AGENT_SYSTEM_PROMPT = """你是一个专业的简历优化 AI Agent。你的任务是帮助用户优化简历，使其更好地匹配目标职位。

## 你可以使用的工具

{tool_descriptions}

## 工具调用格式（必须严格遵守）

当你需要调用工具时，**必须**使用以下格式输出，不要使用 markdown 代码块：

TOOL_CALL: 工具名称
ARGS: {{"参数名": "参数值"}}

示例：
TOOL_CALL: parse_resume
ARGS: {{"file_path": "data/resumes/test.pdf"}}

**注意**：
- 每次回复只调用一个工具
- TOOL_CALL 和 ARGS 各占一行
- ARGS 必须是合法的 JSON 格式
- 不要使用 markdown 代码块（```）包裹工具调用
- 如果你已经完成了所有任务，直接输出最终回复文字即可，不需要调用工具

## 工作流程（必须严格按此顺序执行）

以下是强制的工具调用顺序，**不允许跳步或乱序**：

第 1 步：parse_resume — 解析简历
第 2 步：获取职位信息（二选一，只允许调用一次！）
  - 如果用户提供了职位页面 URL → 使用 fetch_url_jobs（传入逗号分隔的 URL）
  - 如果用户没有提供 URL → 使用 search_jobs（按关键词搜索）
第 3 步：match_all_jobs — 匹配所有职位，自动排序选定最佳目标
第 4 步：optimize_section — 逐模块优化（每个模块调用一次）
第 5 步：judge_resume — 评估优化效果
第 6 步：（如需要）针对薄弱模块重优化 → 再次 judge_resume
第 7 步：generate_resume — 生成最终简历文件

**关键约束**：
- match_all_jobs 必须在 optimize_section 之前调用！没有匹配结果就无法进行有针对性的优化
- search_jobs 和 fetch_url_jobs 都只允许调用一次！系统会直接拦截重复调用
- 获取职位成功后，立即进入 match_all_jobs
- 当用户消息中包含 URL（如 zhaopin.com、zhipin.com 等链接），必须使用 fetch_url_jobs 而非 search_jobs

## 搜索参数引导

调用 search_jobs 之前，如果用户没有在消息中明确指定搜索数量参数（max_jobs），你可以先询问用户。
默认 max_jobs=10（上限20）。如果用户未指定或回复使用默认值，直接使用 max_jobs=10 即可，不需要反复询问。

## 重要原则

1. **一次性完成全部任务**：你必须自主完成从简历解析到生成最终简历文件的全部步骤，不要中途停下来询问用户。优化完所有模块后，必须调用 generate_resume 生成最终文件。**唯一的例外**：在调用 search_jobs 之前，如果用户没有指定搜索数量参数，可以先询问一次
2. **严禁重复搜索职位（最高优先级）**：search_jobs 和 fetch_url_jobs 都只允许调用一次！不管你认为结果是否完美、不管你想换什么关键词或 URL，都绝对不允许再次调用。系统会直接拦截并浪费你的步数。职位获取成功后，下一步必须是 match_all_jobs
3. **工具调用顺序不可打乱**：必须先 match_all_jobs，再 optimize_section。不允许在没有匹配结果的情况下直接优化
4. **工具报错时先检查参数**：如果工具返回参数错误，检查参数名是否正确并重试，不要误以为是数据问题而重复搜索
5. **闭环反馈（必须执行，严禁跳过）**：调用 judge_resume 后，如果返回了「建议重优化的模块」，必须按建议调用 optimize_section 重优化，然后再次 judge_resume。每个模块最多重优化 1 次。只有当 judge_resume 不再返回重优化建议时，才可以调用 generate_resume
6. **诚实反馈**：如果某些技能用户确实没有，不要编造
7. **search_jobs 参数说明**：参数名是 keyword（不是 keywords）和 location（不是 city）

## 当前用户偏好

{user_preference}

## 对话历史

{conversation_history}

请根据用户的请求，先简要说明你的思考，然后调用合适的工具。

重要：你的最终输出必须包含调用 generate_resume 生成简历文件这一步。不要在中间步骤停下来询问用户，自主完成全部流程。
"""


class ResumeAgent:
    """
    简历优化 AI Agent

    核心特性：
    - ReAct 循环：思考 → 行动 → 观察 → 循环
    - 自主规划：LLM 决定调用什么工具、什么顺序
    - 记忆系统：短期对话记忆 + 长期用户偏好
    - 迭代优化：可以多次优化、评估、调整
    """

    def __init__(self, max_steps: int = 15):
        """
        初始化 Agent
        :param max_steps: 单次任务最大执行步数（防止无限循环）
        """
        self.max_steps = max_steps
        self.memory = AgentMemory()

        # 初始化工具
        self.tools = self._init_tools()
        self.tool_map = {tool.name: tool for tool in self.tools}

        # 初始化 LLM
        self.llm = self._init_llm()

        logger.info(f"Resume Agent 初始化完成，可用工具: {list(self.tool_map.keys())}")

    def _init_tools(self) -> list:
        """初始化工具实例"""
        parse_tool = ParseResumeTool()
        search_tool = SearchJobsTool()
        fetch_url_tool = FetchURLJobsTool()
        match_tool = MatchResumeTool()
        match_all_tool = MatchAllJobsTool()
        optimize_tool = OptimizeSectionTool()
        evaluate_tool = EvaluateResumeTool()
        judge_tool = JudgeResumeTool()
        generate_tool = GenerateResumeTool()

        # 直接赋初始值（不使用 list 包装），状态同步由 _sync_shared_state 负责
        match_tool.resume_data = None
        match_tool.jobs = []
        match_all_tool.resume_data = None
        match_all_tool.jobs = []
        fetch_url_tool.jobs = []
        optimize_tool.resume_data = None
        optimize_tool.jobs = []
        optimize_tool.match_result = None
        optimize_tool.top_jobs = []
        evaluate_tool.resume_data = None
        judge_tool.resume_data = None
        judge_tool.jobs = []
        judge_tool.optimized_sections = {}
        generate_tool.resume_data = None

        return [parse_tool, search_tool, fetch_url_tool, match_tool, match_all_tool, optimize_tool, evaluate_tool, judge_tool, generate_tool]

    def _init_llm(self) -> ChatOpenAI:
        """初始化 LLM"""
        provider = settings.llm_provider.lower()
        if provider == "qwen":
            return ChatOpenAI(
                model=settings.qwen_model,
                temperature=0.1,  # Agent 需要较低温度以保证稳定决策
                api_key=settings.qwen_api_key,
                base_url=settings.qwen_base_url,
            )
        else:
            return ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.1,
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
            )

    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        tool_descs = []
        for tool in self.tools:
            tool_descs.append(f"- **{tool.name}**: {tool.description}")

        return AGENT_SYSTEM_PROMPT.format(
            tool_descriptions="\n".join(tool_descs),
            user_preference=self.memory.get_preference_context(),
            conversation_history=self.memory.get_recent_context(max_turns=8) or "（新会话，无历史）",
        )

    def _parse_tool_call(self, llm_output: str) -> Optional[Dict]:
        """
        从 LLM 输出中解析工具调用指令
        支持多种 LLM 可能输出的格式:
          1. TOOL_CALL: tool_name + ARGS: {"key": "value"}
          2. 纯 JSON: {"tool": "tool_name", "args": {"key": "value"}}
          3. Markdown 代码块包裹的 JSON: ```json { ... } ```
          4. 参数键名兼容: "args" / "tool_input" / "parameters" / "input"
        """
        # ── 格式 1: TOOL_CALL 格式 ──
        tool_match = re.search(r'TOOL_CALL:\s*(\w+)', llm_output)
        args_match = re.search(r'ARGS:\s*(\{.*?\})', llm_output, re.DOTALL)

        if tool_match:
            tool_name = tool_match.group(1)
            args = {}
            if args_match:
                try:
                    args = json.loads(args_match.group(1))
                except json.JSONDecodeError:
                    pass
            return {"tool": tool_name, "args": args}

        # ── 提取 JSON 文本（去除 markdown 代码块包裹） ──
        json_text = llm_output.strip()

        # 尝试从 ```json ... ``` 或 ``` ... ``` 中提取
        code_block = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', llm_output, re.DOTALL)
        if code_block:
            json_text = code_block.group(1).strip()

        # ── 格式 2/3: JSON 格式 ──
        # 如果 json_text 不是以 { 开头，尝试找到第一个 { 到最后一个 } 之间的内容
        brace_start = json_text.find('{')
        brace_end = json_text.rfind('}')
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            json_text = json_text[brace_start:brace_end + 1]

        try:
            data = json.loads(json_text)
        except json.JSONDecodeError:
            # 最后尝试从原始输出中找 JSON
            try:
                data = json.loads(llm_output.strip())
            except json.JSONDecodeError:
                return None

        # 标准化键名
        if "tool" in data:
            # 兼容多种参数键名
            args = data.get("args") or data.get("tool_input") or data.get("parameters") or data.get("input") or {}
            return {"tool": data["tool"], "args": args}

        # 有些模型输出 {"name": "tool_name", "arguments": {...}} 格式
        if "name" in data and isinstance(data.get("arguments"), dict):
            return {"tool": data["name"], "args": data["arguments"]}

        return None

    # 参数别名映射：LLM 经常用错的参数名 → 工具实际接受的参数名
    _PARAM_ALIASES = {
        "keywords": "keyword",
        "query": "keyword",
        "position": "keyword",
        "job_title": "target_job",
        "target_position": "target_job",
        "position_title": "target_job",
        "city": "location",
        "area": "location",
        "place": "location",
        "optimization_focus": "focus",
        "emphasis": "focus",
        "highlight": "focus",
        "index": "job_index",
        "job_idx": "job_index",
        "path": "file_path",
        "resume_path": "file_path",
        "output_format": "format",
        "section": "section_name",
        "module": "section_name",
    }

    def _normalize_args(self, tool_name: str, args: Dict) -> Dict:
        """将 LLM 传入的参数名映射到工具实际接受的参数名"""
        normalized = {}
        for key, value in args.items():
            actual_key = self._PARAM_ALIASES.get(key, key)
            normalized[actual_key] = value
        return normalized

    # 每个工具拥有哪些共享字段（工具名 → 字段集合）
    _TOOL_FIELDS = {
        "parse_resume":     {"resume_data"},
        "search_jobs":      {"jobs"},
        "fetch_url_jobs":   {"jobs"},
        "match_resume":     {"resume_data", "jobs", "match_result"},
        "match_all_jobs":   {"resume_data", "jobs", "match_result", "top_jobs"},
        "optimize_section": {"resume_data", "jobs", "match_result", "optimized_sections", "top_jobs"},
        "evaluate_resume":  {"resume_data"},
        "judge_resume":     {"resume_data", "jobs", "match_result", "optimized_sections", "top_jobs"},
        "generate_resume":  {"resume_data", "optimized_sections"},
    }

    def _sync_shared_state(self):
        """
        工具执行后，将各工具的最新状态同步到所有拥有对应字段的工具。
        """
        # 收集最新状态
        latest = {}
        for tool in self.tools:
            rd = getattr(tool, 'resume_data', None)
            if rd is not None and not isinstance(rd, (list, dict)):
                latest['resume_data'] = rd
            jb = getattr(tool, 'jobs', None)
            if jb and isinstance(jb, list) and len(jb) > 0:
                latest['jobs'] = jb
            mr = getattr(tool, 'match_result', None)
            if mr is not None and not isinstance(mr, (list, dict)):
                latest['match_result'] = mr
            os_val = getattr(tool, 'optimized_sections', None)
            if os_val and isinstance(os_val, dict) and len(os_val) > 0:
                latest['optimized_sections'] = os_val
            tj = getattr(tool, 'top_jobs', None)
            if tj and isinstance(tj, list) and len(tj) > 0:
                latest['top_jobs'] = tj

        # 同步到拥有对应字段的工具
        for tool in self.tools:
            allowed = self._TOOL_FIELDS.get(tool.name, set())
            for field_name, value in latest.items():
                if field_name in allowed:
                    setattr(tool, field_name, value)

    def _filter_tool_args(self, tool, args: Dict) -> Dict:
        """过滤掉工具 _run 方法不接受的多余参数"""
        import inspect
        sig = inspect.signature(tool._run)
        valid_params = set(sig.parameters.keys()) - {'self'}
        # 如果 _run 有 **kwargs，则不过滤
        for p in sig.parameters.values():
            if p.kind == inspect.Parameter.VAR_KEYWORD:
                return args
        return {k: v for k, v in args.items() if k in valid_params}

    def _execute_tool(self, tool_name: str, args: Dict) -> str:
        """执行工具调用"""
        if tool_name not in self.tool_map:
            return f"错误: 未知工具 '{tool_name}'。可用工具: {list(self.tool_map.keys())}"

        tool = self.tool_map[tool_name]

        # 参数名标准化（兼容 LLM 常见的命名变体）
        args = self._normalize_args(tool_name, args)

        # 过滤掉工具不接受的多余参数
        args = self._filter_tool_args(tool, args)

        try:
            result = tool._run(**args)
            self.memory.add_tool_result(tool_name, result)

            # 执行后同步共享状态
            self._sync_shared_state()

            return result
        except TypeError as e:
            if "unexpected keyword argument" in str(e):
                error_msg = f"工具 {tool_name} 参数错误: {e}。请检查参数名称是否正确。"
            else:
                error_msg = f"工具 {tool_name} 执行失败: {e}"
            logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"工具 {tool_name} 执行失败: {e}"
            logger.error(error_msg)
            return error_msg

    def run(self, user_input: str, step_callback=None) -> str:
        """
        Agent 主循环（ReAct）

        流程：
        1. 接收用户输入
        2. LLM 思考并决定行动
        3. 如果需要调用工具 → 执行工具 → 获取结果 → 回到 2
        4. 如果不需要工具 → 直接回复用户
        5. 循环直到任务完成或达到最大步数

        :param step_callback: 可选回调函数，用于向前端报告当前步骤状态
        """
        self.memory.add_user_message(user_input)
        logger.info(f"[Agent] 用户输入: {user_input[:100]}...")

        if step_callback:
            step_callback(phase="正在思考...", tool_name="", tool_args={})

        messages = [
            SystemMessage(content=self._build_system_prompt()),
        ]

        # 添加对话历史
        for turn in self.memory.conversation_history[-8:]:
            if turn.role == "user":
                messages.append(HumanMessage(content=turn.content))
            elif turn.role == "assistant":
                messages.append(AIMessage(content=turn.content))

        step = 0
        final_response = ""
        # 追踪工具调用历史，用于检测重复调用
        tool_call_history = []  # [(tool_name, args_key), ...]

        while step < self.max_steps:
            step += 1
            logger.info(f"[Agent] 第 {step} 步")

            if step_callback:
                step_callback(
                    current_step=step,
                    phase=f"LLM 思考中（第 {step} 步）...",
                    tool_name="",
                    tool_args={},
                )

            # LLM 思考（带计时）
            import time as _time
            _llm_t0 = _time.time()
            response = self.llm.invoke(messages)
            _llm_elapsed = _time.time() - _llm_t0
            llm_output = response.content
            logger.info(f"[Agent] LLM 推理耗时: {_llm_elapsed:.1f}s")

            # 检查是否有工具调用
            tool_call = self._parse_tool_call(llm_output)

            if tool_call:
                # 有工具调用 → 执行工具
                tool_name = tool_call["tool"]
                args = tool_call["args"]
                logger.info(f"[Agent] 调用工具: {tool_name}, 参数: {args}")

                # 检测重复调用
                args_key = json.dumps(args, sort_keys=True, ensure_ascii=False)
                call_sig = (tool_name, args_key)
                repeat_count = tool_call_history.count(call_sig)

                # 对 search_jobs / fetch_url_jobs 特殊处理：按工具名拦截（不管参数是否变化）
                _once_tools = {"search_jobs", "fetch_url_jobs"}
                once_tool_calls = {t: [c for c in tool_call_history if c[0] == t] for t in _once_tools}
                is_once_repeat = (tool_name in _once_tools and len(once_tool_calls.get(tool_name, [])) >= 1)

                if repeat_count >= 1 or is_once_repeat:
                    # 重复调用！注入警告，跳过执行
                    if tool_name == "search_jobs":
                        n = len(once_tool_calls.get("search_jobs", []))
                        warning = (
                            f"⚠️ 严重警告：你已经调用过 search_jobs {n} 次了！"
                            f"换关键词搜索不会带来更好的结果，已有的职位数据完全够用。"
                            f"你必须立即停止搜索，下一步调用 match_all_jobs 对已有职位进行匹配分析。"
                            f"这是强制指令，不允许再调用 search_jobs。"
                        )
                    elif tool_name == "fetch_url_jobs":
                        n = len(once_tool_calls.get("fetch_url_jobs", []))
                        warning = (
                            f"⚠️ 严重警告：你已经调用过 fetch_url_jobs {n} 次了！"
                            f"URL 职位已抓取完成，下一步调用 match_all_jobs 对已有职位进行匹配分析。"
                            f"这是强制指令，不允许再调用 fetch_url_jobs。"
                        )
                    else:
                        warning = (
                            f"⚠️ 警告：你已经调用过 {tool_name}（参数相同）{repeat_count} 次了！"
                            f"重复调用不会带来新结果，纯属浪费时间。"
                            f"请立即使用已有结果，调用下一个工具继续流程。"
                        )
                    logger.warning(f"[Agent] 检测到重复调用: {tool_name} (args={args})")
                    messages.append(AIMessage(content=llm_output))
                    messages.append(HumanMessage(content=warning))
                    tool_call_history.append(call_sig)

                    if step_callback:
                        step_callback(
                            current_step=step,
                            phase=f"⚠️ 拦截重复调用: {tool_name}",
                            tool_name=tool_name,
                            tool_args=args,
                        )
                    continue

                tool_call_history.append(call_sig)

                if step_callback:
                    # 构造可读的参数描述
                    args_summary = ", ".join(f"{k}={v}" for k, v in args.items()) or "无参数"
                    step_callback(
                        current_step=step,
                        phase=f"执行工具: {tool_name}",
                        tool_name=tool_name,
                        tool_args=args,
                    )

                # 把 LLM 的思考加入消息历史
                messages.append(AIMessage(content=llm_output))

                # 执行工具
                tool_result = self._execute_tool(tool_name, args)
                logger.info(f"[Agent] 工具结果: {tool_result[:200]}...")

                if step_callback:
                    result_preview = tool_result[:100].replace("\n", " ")
                    step_callback(
                        current_step=step,
                        phase=f"工具 {tool_name} 返回完成",
                        tool_name=tool_name,
                        tool_args={},
                    )

                # 把工具结果加入消息历史
                messages.append(HumanMessage(content=f"工具 {tool_name} 返回结果:\n{tool_result}\n\n请根据结果决定下一步。"))

            else:
                # 没有工具调用 → 这是最终回复
                final_response = llm_output
                if step_callback:
                    step_callback(phase="Agent 生成最终回复...", tool_name="", tool_args={})
                break

        if step >= self.max_steps:
            final_response = final_response or "已达到最大执行步数，任务暂停。请告诉我接下来需要做什么。"
            if step_callback:
                step_callback(phase=f"已达安全上限 ({self.max_steps} 步)，自动暂停", tool_name="", tool_args={})

        self.memory.add_assistant_message(final_response)
        logger.info(f"[Agent] 任务完成，共 {step} 步")

        return final_response

    def chat(self, user_input: str) -> str:
        """
        多轮对话接口
        与 run() 的区别：chat 会保持对话上下文，适合交互式使用
        """
        return self.run(user_input)

    def reset(self):
        """重置 Agent（清空短期记忆，保留用户偏好）"""
        self.memory.clear_short_term()
        # 重新初始化工具状态
        for tool in self.tools:
            if hasattr(tool, 'resume_data'):
                tool.resume_data = None
            if hasattr(tool, 'jobs'):
                tool.jobs = []
            if hasattr(tool, 'match_result'):
                tool.match_result = None
            if hasattr(tool, 'optimized_sections'):
                tool.optimized_sections = {}
            if hasattr(tool, 'top_jobs'):
                tool.top_jobs = []
        logger.info("Agent 已重置")

    def get_status(self) -> str:
        """获取 Agent 当前状态"""
        return self.memory.get_session_summary()
