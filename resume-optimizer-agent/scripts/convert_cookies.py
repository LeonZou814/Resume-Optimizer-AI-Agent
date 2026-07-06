"""
将 Cookie Editor 导出的 JSON 转换为 Playwright storage_state 格式

使用方法：
1. 用 Cookie Editor 导出 Boss直聘的 Cookie（JSON 格式）
2. 把导出的 JSON 保存到 data/cookies_raw.json
3. 运行此脚本：python scripts/convert_cookies.py
4. 生成 data/boss_cookies.json，可直接被爬虫使用
"""

import json
from pathlib import Path
from datetime import datetime


def convert_cookies(input_path: str = "data/cookies_raw.json",
                    output_path: str = "data/boss_cookies.json"):
    """转换 Cookie 格式"""

    input_file = Path(input_path)
    if not input_file.exists():
        print(f"错误: 找不到文件 {input_path}")
        print("请先用 Cookie Editor 导出 Boss直聘的 Cookie，保存为 data/cookies_raw.json")
        return

    # 读取原始 Cookie
    raw_cookies = json.loads(input_file.read_text(encoding="utf-8"))

    # 如果导出的是 {"url": ..., "cookies": [...]} 格式
    if isinstance(raw_cookies, dict) and "cookies" in raw_cookies:
        cookies = raw_cookies["cookies"]
    elif isinstance(raw_cookies, list):
        cookies = raw_cookies
    else:
        print("错误: 无法识别的 Cookie 格式")
        return

    # 转换为 Playwright storage_state 格式
    playwright_cookies = []
    for cookie in cookies:
        pc = {
            "name": cookie.get("name", ""),
            "value": cookie.get("value", ""),
            "domain": cookie.get("domain", ".zhipin.com"),
            "path": cookie.get("path", "/"),
            "httpOnly": cookie.get("httpOnly", False),
            "secure": cookie.get("secure", True),
            "sameSite": cookie.get("sameSite", "Lax"),
        }

        # 处理过期时间
        if "expirationDate" in cookie:
            pc["expires"] = cookie["expirationDate"]
        elif "expiry" in cookie:
            pc["expires"] = cookie["expiry"]
        else:
            pc["expires"] = -1  # session cookie

        playwright_cookies.append(pc)

    # 构建 storage_state 格式
    storage_state = {
        "cookies": playwright_cookies,
        "origins": []
    }

    # 保存
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(storage_state, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"转换成功！")
    print(f"输入: {input_path} ({len(cookies)} 个 Cookie)")
    print(f"输出: {output_path}")
    print(f"\n现在可以运行爬虫了：")
    print(f"  python -m src.main --resume data/resumes/你的简历.pdf --keyword 'Python后端' --location 北京")


if __name__ == "__main__":
    convert_cookies()
