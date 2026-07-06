"""
简历与职位需求匹配模块
使用向量相似度 + 关键词匹配
"""

import os
import json
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

# 在加载 SentenceTransformer 前设置 HuggingFace 镜像（国内加速）
# 先尝试从 .env 加载配置
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)  # override=False 不覆盖已有的环境变量
except ImportError:
    pass

# 如果 .env 中没有设置 HF_ENDPOINT，使用默认镜像
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 强制离线模式：避免 SentenceTransformer 联网检查模型更新导致长时间超时
# 模型已缓存在本地 ~/.cache/huggingface/hub/ 中，无需联网
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer
import numpy as np

from loguru import logger
from src.scrapers.job_scraper import JobPosting
from src.parsers.resume_parser import ResumeData


@dataclass
class MatchResult:
    """匹配结果"""
    job_title: str
    company: str
    overall_score: float          # 综合匹配分数 0-1
    skill_match_score: float      # 技能匹配分数
    experience_match_score: float # 经验匹配分数
    missing_skills: List[str]     # 简历中缺少的技能
    matched_skills: List[str]     # 已匹配的技能
    suggestions: List[str]        # 优化建议


# 模块级模型缓存（避免重复加载）
_model_cache: Dict[str, SentenceTransformer] = {}


def _get_model(model_name: str) -> SentenceTransformer:
    """获取或加载 SentenceTransformer 模型（带缓存）"""
    if model_name not in _model_cache:
        logger.info(f"正在加载语义匹配模型: {model_name} ...（离线模式，从本地缓存加载）")
        try:
            _model_cache[model_name] = SentenceTransformer(
                model_name,
                local_files_only=True,  # 强制仅使用本地缓存，不联网
            )
        except Exception as e:
            logger.warning(f"本地缓存加载失败: {e}，尝试默认加载...")
            _model_cache[model_name] = SentenceTransformer(model_name)
        logger.info(f"模型加载完成 ✓")
    return _model_cache[model_name]


class JobResumeMatcher:
    """职位简历匹配器"""

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """
        初始化匹配器
        :param model_name: sentence-transformers 模型名称
        """
        self.model = _get_model(model_name)
        self.skill_embedding_cache: Dict[str, np.ndarray] = {}

    def match(self, resume: ResumeData, jobs: List[JobPosting]) -> List[MatchResult]:
        """
        将简历与多个职位进行匹配，返回按匹配度排序的结果
        """
        results = []
        resume_text = self._build_resume_text(resume)
        resume_embedding = self.model.encode(resume_text)

        for job in jobs:
            result = self._match_single(resume, resume_embedding, job)
            results.append(result)

        # 按综合匹配分数降序排列
        results.sort(key=lambda x: x.overall_score, reverse=True)
        return results

    def _match_single(self, resume: ResumeData, resume_embedding: np.ndarray, job: JobPosting) -> MatchResult:
        """单个职位匹配逻辑"""
        # 1. 技能匹配
        skill_score, matched, missing = self._skill_match(resume, job)

        # 2. 语义相似度匹配（向量）
        job_text = f"{job.title} {job.description} {job.requirements}"
        job_embedding = self.model.encode(job_text)
        semantic_score = self._cosine_similarity(resume_embedding, job_embedding)

        # 3. 经验匹配（简单规则）
        exp_score = self._experience_match(resume, job)

        # 4. 综合分数（加权）
        overall = skill_score * 0.5 + semantic_score * 0.3 + exp_score * 0.2

        # 5. 生成建议
        suggestions = self._generate_suggestions(missing, job, semantic_score)

        return MatchResult(
            job_title=job.title,
            company=job.company,
            overall_score=round(overall, 3),
            skill_match_score=round(skill_score, 3),
            experience_match_score=round(exp_score, 3),
            missing_skills=missing,
            matched_skills=matched,
            suggestions=suggestions,
        )

    def _skill_match(self, resume: ResumeData, job: JobPosting) -> tuple:
        """
        技能匹配：计算职位要求技能在简历中的覆盖度
        如果 job.skills 为空，自动从 description + requirements 中提取技能关键词
        返回 (分数, 已匹配技能列表, 缺少技能列表)
        """
        skills = job.skills or self._extract_skills_from_text(
            f"{job.description} {job.requirements}"
        )

        if not skills:
            return 1.0, [], []

        resume_text_lower = resume.raw_text.lower()
        matched = []
        missing = []

        for skill in skills:
            skill_lower = skill.lower()
            # 简单的字符串包含匹配（实际可优化为同义词匹配）
            if skill_lower in resume_text_lower:
                matched.append(skill)
            else:
                missing.append(skill)

        coverage = len(matched) / len(skills) if skills else 1.0
        return coverage, matched, missing

    @staticmethod
    def _extract_skills_from_text(text: str) -> List[str]:
        """
        从职位描述文本中提取候选技能关键词
        基于常见技术栈关键词库进行匹配
        """
        if not text:
            return []

        # 常见技术技能关键词库（可按需扩展）
        skill_keywords = [
            # 编程语言
            "Python", "Java", "Go", "Golang", "C++", "C", "C#", "Rust", "Ruby", "PHP",
            "JavaScript", "TypeScript", "Node.js", "HTML", "CSS", "SQL", "Shell", "Lua",
            # 前端框架
            "Vue", "Vue.js", "React", "React.js", "Angular", "Next.js", "Nuxt.js",
            "jQuery", "Bootstrap", "Tailwind", "ElementUI", "Ant Design",
            # 后端框架
            "Django", "Flask", "FastAPI", "Tornado", "Spring", "Spring Boot",
            "Spring Cloud", "Express", "Koa", "NestJS", "ThinkPHP", "Laravel",
            # 数据库
            "MySQL", "PostgreSQL", "Oracle", "SQL Server", "MongoDB", "Redis",
            "Elasticsearch", "ClickHouse", "TiDB", "SQLite", "DynamoDB", "Neo4j",
            # 大数据/中间件
            "Kafka", "RabbitMQ", "RocketMQ", "Zookeeper", "Hadoop", "Spark",
            "Flink", "Hive", "HBase", "Storm", "Pulsar", "NATS",
            # 云原生/DevOps
            "Docker", "Kubernetes", "K8s", "Jenkins", "GitLab CI", "GitHub Actions",
            "Terraform", "Ansible", "Prometheus", "Grafana", "ELK", "Istio",
            # 云服务
            "AWS", "阿里云", "腾讯云", "华为云", "Azure", "GCP",
            "ECS", "OSS", "RDS", "Lambda", "Serverless", "CDN",
            # AI/数据科学
            "PyTorch", "TensorFlow", "Keras", "Scikit-learn", "Pandas", "NumPy",
            "OpenCV", "NLP", "LLM", "机器学习", "深度学习", "计算机视觉",
            # 工具/其他
            "Git", "Linux", "Nginx", "Apache", "Tomcat", "Vim", "Markdown",
            "RESTful", "GraphQL", "gRPC", "WebSocket", "OAuth", "JWT",
            "微服务", "分布式", "高并发", "高可用", "负载均衡", "CI/CD",
            "敏捷开发", "Scrum", "TDD", "DDD", "领域驱动设计",
        ]

        text_lower = text.lower()
        found = []
        for skill in skill_keywords:
            # 使用单词边界匹配，避免部分匹配（如 "Java" 匹配 "JavaScript"）
            pattern = r'(?<![a-zA-Z#+])' + re.escape(skill.lower()) + r'(?![a-zA-Z#+])'
            if re.search(pattern, text_lower):
                found.append(skill)

        return found

    def _experience_match(self, resume: ResumeData, job: JobPosting) -> float:
        """经验匹配（简化版）"""
        resume_text = resume.raw_text

        # 检查是否有相关项目/工作关键词
        has_project = "项目" in resume_text
        has_work = "工作" in resume_text or "实习" in resume_text

        score = 0.5
        if has_project:
            score += 0.25
        if has_work:
            score += 0.25

        return min(score, 1.0)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度"""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    @staticmethod
    def _build_resume_text(resume: ResumeData) -> str:
        """构建用于向量化的简历文本"""
        parts = [resume.raw_text]
        # 加权专业技能部分
        if "专业技能" in resume.sections:
            parts.append(resume.sections["专业技能"] * 2)
        if "项目经验" in resume.sections:
            parts.append(resume.sections["项目经验"])
        return " ".join(parts)

    @staticmethod
    def _generate_suggestions(missing_skills: List[str], job: JobPosting, semantic_score: float) -> List[str]:
        """基于匹配结果生成优化建议"""
        suggestions = []

        if missing_skills:
            suggestions.append(f"建议补充以下技能关键词：{', '.join(missing_skills)}")

        if semantic_score < 0.5:
            suggestions.append("简历整体描述与职位要求差异较大，建议根据目标岗位调整表述方向")
        elif semantic_score < 0.7:
            suggestions.append("可在项目描述中更多地使用与职位相关的技术术语")

        if job.experience and "年" in job.experience:
            suggestions.append(f"该职位要求{job.experience}，请确保工作经历部分体现相应年限")

        return suggestions
