"""
Agent 交互层
提供命令行多轮对话界面，用户可以与 Agent 持续交互
"""

import sys
from pathlib import Path

from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.agents.resume_agent import ResumeAgent


class InteractiveAgent:
    """
    交互式 Agent 界面

    支持：
    - 多轮对话（保持上下文）
    - 特殊命令（/status, /reset, /quit 等）
    - 流式输出（逐步显示 Agent 思考过程）
    """

    COMMANDS = {
        "/help": "显示帮助信息",
        "/status": "查看 Agent 当前状态",
        "/reset": "重置 Agent（清空短期记忆）",
        "/history": "查看对话历史摘要",
        "/quit": "退出程序",
        "/exit": "退出程序",
    }

    def __init__(self, max_steps: int = 15):
        self.agent = ResumeAgent(max_steps=max_steps)
        self.running = False

    def start(self):
        """启动交互界面"""
        self.running = True
        self._print_welcome()

        while self.running:
            try:
                user_input = input("\n你: ").strip()

                if not user_input:
                    continue

                # 处理特殊命令
                if user_input.startswith("/"):
                    self._handle_command(user_input)
                    continue

                # 正常对话
                print("\n" + "=" * 60)
                print("Agent 思考中...")
                print("=" * 60 + "\n")

                response = self.agent.chat(user_input)

                print("\n" + "-" * 60)
                print("Agent:")
                print("-" * 60)
                print(response)
                print("-" * 60)

            except KeyboardInterrupt:
                print("\n\n已中断，输入 /quit 退出或继续对话。")
            except EOFError:
                break

        print("\n再见！")

    def _handle_command(self, command: str):
        """处理特殊命令"""
        cmd = command.lower().strip()

        if cmd in ("/quit", "/exit"):
            self.running = False
        elif cmd == "/help":
            self._print_help()
        elif cmd == "/status":
            print("\n" + self.agent.get_status())
        elif cmd == "/reset":
            self.agent.reset()
            print("\nAgent 已重置，短期记忆已清空。")
        elif cmd == "/history":
            history = self.agent.memory.get_recent_context(max_turns=20)
            if history:
                print("\n对话历史:")
                print(history)
            else:
                print("\n暂无对话历史。")
        else:
            print(f"\n未知命令: {command}")
            self._print_help()

    def _print_welcome(self):
        """打印欢迎信息"""
        print("""
╔══════════════════════════════════════════════════════════════╗
║                  Resume Optimizer AI Agent                   ║
║                      简历优化智能助手                         ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  我是一个真正的 AI Agent，可以自主规划和执行简历优化任务。     ║
║                                                              ║
║  我能做什么：                                                ║
║  • 解析你的简历（PDF/DOCX/TXT）                              ║
║  • 从智联招聘搜索目标职位                                    ║
║  • 分析简历与职位的匹配度                                    ║
║  • 针对性优化简历各模块                                      ║
║  • 生成优化后的简历文件                                      ║
║  • 多轮对话，持续改进                                        ║
║                                                              ║
║  示例对话：                                                  ║
║  > 帮我优化简历，文件在 data/resumes/my_resume.pdf           ║
║  > 搜索北京的 Python 后端职位                                ║
║  > 分析一下我的简历和第一个职位的匹配度                      ║
║  > 帮我优化专业技能部分，突出分布式系统经验                  ║
║  > 生成优化后的简历                                          ║
║                                                              ║
║  输入 /help 查看所有命令                                     ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

    def _print_help(self):
        """打印帮助信息"""
        print("\n可用命令:")
        for cmd, desc in self.COMMANDS.items():
            print(f"  {cmd:12s} - {desc}")
        print()


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description="Resume Optimizer AI Agent - 交互式界面")
    parser.add_argument("--max-steps", type=int, default=15, help="单次任务最大执行步数（默认15）")
    args = parser.parse_args()

    interactive = InteractiveAgent(max_steps=args.max_steps)
    interactive.start()


if __name__ == "__main__":
    main()
