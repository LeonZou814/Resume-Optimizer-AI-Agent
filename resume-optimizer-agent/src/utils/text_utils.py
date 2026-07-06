"""
文本处理工具函数
"""

import re
from typing import List


def clean_text(text: str) -> str:
    """清理文本：去除多余空白、特殊字符"""
    if not text:
        return ""
    # 去除多余空白
    text = re.sub(r'\s+', ' ', text)
    # 去除不可见字符
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    return text.strip()


def extract_keywords(text: str, min_length: int = 2) -> List[str]:
    """从文本中提取候选关键词（简单分词）"""
    import jieba
    words = jieba.cut(text)
    keywords = []
    for word in words:
        word = word.strip()
        if len(word) >= min_length and not word.isdigit():
            keywords.append(word)
    return list(set(keywords))


def split_by_sections(text: str) -> dict:
    """
    将简历文本按常见模块拆分
    返回 {"模块名": "内容"}

    支持三种格式：
    1. 标题独占一行：  "工作经历\n..."
    2. 标题+冒号：      "工作经历：\n..."
    3. 标题+空格+内容： "工作经历 xxx公司 项目经理..."（PDF提取常见）
    """
    # 常见简历标题关键词（按优先级排列，长的在前避免被短的先匹配）
    section_keywords = [
        "教育背景", "教育经历", "工作经历", "工作经验", "项目经验", "项目经历",
        "专业技能", "技能", "自我评价", "个人优势", "求职意向",
        "资格证书", "证书", "获奖", "语言能力", "兴趣爱好",
        "实习经历", "校园经历", "学历",
    ]

    # 按行分割，逐行检测模块标题
    lines = text.split('\n')
    sections = {}
    current_title = "基本信息"
    current_content = []

    def _match_section_keyword(line_text: str):
        """
        检测一行文本是否以某个模块关键词开头。
        返回 (keyword, remaining_text) 或 (None, None)
        """
        for keyword in section_keywords:
            if line_text == keyword:
                return keyword, ""
            if line_text.startswith(keyword + "："):
                return keyword, line_text.split("：", 1)[1]
            if line_text.startswith(keyword + ":"):
                return keyword, line_text.split(":", 1)[1]
            # 标题后跟空格再跟内容（PDF提取的常见格式）
            if line_text.startswith(keyword + " ") or line_text.startswith(keyword + "\t"):
                remaining = line_text[len(keyword):].strip()
                return keyword, remaining
        return None, None

    for line in lines:
        line_stripped = line.strip()

        # 检查是否是模块标题
        keyword, remaining = _match_section_keyword(line_stripped)

        if keyword is not None:
            # 保存上一个模块
            if current_content:
                sections[current_title] = clean_text('\n'.join(current_content))
            # 开始新模块
            current_title = keyword
            current_content = []
            # 如果标题后面有内容，提取出来
            if remaining:
                current_content.append(remaining)
        else:
            if line_stripped:  # 跳过空行
                current_content.append(line_stripped)

    # 保存最后一个模块
    if current_content:
        sections[current_title] = clean_text('\n'.join(current_content))

    return sections


def calculate_similarity(text1: str, text2: str) -> float:
    """计算两段文本的简易相似度（基于共有词汇）"""
    set1 = set(clean_text(text1).lower().split())
    set2 = set(clean_text(text2).lower().split())
    if not set1 or not set2:
        return 0.0
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union)
