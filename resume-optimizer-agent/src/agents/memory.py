"""
Agent 记忆模块
- 短期记忆：当前会话的对话历史和工具调用结果
- 长期记忆：用户偏好、历史优化记录（持久化到 JSON 文件）
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict

from loguru import logger


@dataclass
class ConversationTurn:
    """单轮对话"""
    role: str           # "user" | "assistant" | "tool"
    content: str
    timestamp: str = ""
    tool_name: str = ""  # 如果是工具调用，记录工具名

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class UserPreference:
    """用户偏好（长期记忆）"""
    target_industry: str = ""         # 目标行业
    target_role: str = ""             # 目标岗位
    preferred_keywords: List[str] = field(default_factory=list)  # 偏好的关键词
    style_preference: str = ""        # 风格偏好（如"简洁"、"详细"）
    past_optimizations: List[Dict] = field(default_factory=list)  # 历史优化记录


class AgentMemory:
    """
    Agent 记忆管理器

    短期记忆：当前会话的对话历史（内存中）
    长期记忆：用户偏好和历史记录（JSON 文件持久化）
    """

    def __init__(self, memory_dir: str = "data/memory"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        # 短期记忆
        self.conversation_history: List[ConversationTurn] = []
        self.tool_results: Dict[str, str] = {}  # 最近一次工具调用结果

        # 长期记忆
        self.user_preference = self._load_preference()

    # ==================== 短期记忆 ====================

    def add_user_message(self, content: str):
        """记录用户消息"""
        self.conversation_history.append(ConversationTurn(role="user", content=content))

    def add_assistant_message(self, content: str):
        """记录助手回复"""
        self.conversation_history.append(ConversationTurn(role="assistant", content=content))

    def add_tool_result(self, tool_name: str, result: str):
        """记录工具调用结果"""
        self.tool_results[tool_name] = result
        self.conversation_history.append(
            ConversationTurn(role="tool", content=result, tool_name=tool_name)
        )

    def get_recent_context(self, max_turns: int = 10) -> str:
        """获取最近的对话上下文（用于 LLM prompt）"""
        recent = self.conversation_history[-max_turns:]
        lines = []
        for turn in recent:
            prefix = {"user": "用户", "assistant": "助手", "tool": f"工具[{turn.tool_name}]"}
            lines.append(f"[{prefix.get(turn.role, turn.role)}]: {turn.content[:500]}")
        return "\n".join(lines)

    def get_last_tool_result(self, tool_name: str) -> Optional[str]:
        """获取某个工具最近一次的调用结果"""
        return self.tool_results.get(tool_name)

    def clear_short_term(self):
        """清空短期记忆（新会话）"""
        self.conversation_history.clear()
        self.tool_results.clear()

    # ==================== 长期记忆 ====================

    def _load_preference(self) -> UserPreference:
        """从文件加载用户偏好"""
        pref_path = self.memory_dir / "user_preference.json"
        if pref_path.exists():
            try:
                data = json.loads(pref_path.read_text(encoding="utf-8"))
                return UserPreference(**data)
            except Exception as e:
                logger.warning(f"加载用户偏好失败: {e}")
        return UserPreference()

    def save_preference(self):
        """保存用户偏好到文件"""
        pref_path = self.memory_dir / "user_preference.json"
        data = asdict(self.user_preference)
        pref_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("用户偏好已保存")

    def update_preference(self, **kwargs):
        """更新用户偏好"""
        for key, value in kwargs.items():
            if hasattr(self.user_preference, key):
                setattr(self.user_preference, key, value)
        self.save_preference()

    def record_optimization(self, target_job: str, match_score: float, optimized_sections: List[str]):
        """记录一次优化历史"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "target_job": target_job,
            "match_score": match_score,
            "optimized_sections": optimized_sections,
        }
        self.user_preference.past_optimizations.append(record)
        # 只保留最近 20 条
        if len(self.user_preference.past_optimizations) > 20:
            self.user_preference.past_optimizations = self.user_preference.past_optimizations[-20:]
        self.save_preference()

    def get_preference_context(self) -> str:
        """获取用户偏好上下文（用于 LLM prompt）"""
        pref = self.user_preference
        parts = []
        if pref.target_industry:
            parts.append(f"目标行业: {pref.target_industry}")
        if pref.target_role:
            parts.append(f"目标岗位: {pref.target_role}")
        if pref.preferred_keywords:
            parts.append(f"偏好关键词: {', '.join(pref.preferred_keywords)}")
        if pref.style_preference:
            parts.append(f"风格偏好: {pref.style_preference}")
        if pref.past_optimizations:
            recent = pref.past_optimizations[-3:]
            parts.append(f"最近优化记录: {len(recent)} 次")
            for r in recent:
                parts.append(f"  - {r['target_job']} (匹配度: {r['match_score']:.1f}%)")
        return "\n".join(parts) if parts else "暂无用户偏好记录"

    # ==================== 会话摘要 ====================

    def get_session_summary(self) -> str:
        """获取当前会话摘要"""
        tool_calls = [t for t in self.conversation_history if t.role == "tool"]
        return f"""会话摘要:
对话轮数: {len(self.conversation_history)}
工具调用次数: {len(tool_calls)}
已调用工具: {', '.join(set(t.tool_name for t in tool_calls)) if tool_calls else '无'}
用户偏好: {self.get_preference_context()[:200]}"""
