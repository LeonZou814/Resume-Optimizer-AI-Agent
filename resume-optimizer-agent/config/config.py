"""
项目全局配置文件
使用 pydantic-settings 从环境变量读取配置
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """应用配置类"""

    # 项目根目录
    BASE_DIR: Path = Path(__file__).resolve().parent.parent

    # 环境
    environment: str = Field(default="development", alias="ENVIRONMENT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # LLM API
    llm_provider: str = Field(default="openai", alias="LLM_PROVIDER")  # openai | qwen
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default="https://api.openai.com/v1", alias="OPENAI_BASE_URL")

    # 通义千问 API（阿里云百炼）
    qwen_api_key: str = Field(default="", alias="QWEN_API_KEY")
    qwen_base_url: str = Field(default="https://dashscope.aliyuncs.com/compatible-mode/v1", alias="QWEN_BASE_URL")
    qwen_model: str = Field(default="qwen3.7-plus", alias="QWEN_MODEL")

    # 数据库
    database_url: str = Field(default="sqlite:///./data/resume_optimizer.db", alias="DATABASE_URL")

    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")

    # 向量数据库
    chroma_persist_dir: str = Field(default="./data/chroma_db", alias="CHROMA_PERSIST_DIR")
    chroma_collection_name: str = Field(default="job_descriptions", alias="CHROMA_COLLECTION_NAME")

    # 抓取配置
    request_delay: int = Field(default=2, alias="REQUEST_DELAY")
    max_retries: int = Field(default=3, alias="MAX_RETRIES")
    http_proxy: str | None = Field(default=None, alias="HTTP_PROXY")
    https_proxy: str | None = Field(default=None, alias="HTTPS_PROXY")

    # 输出目录
    output_dir: str = Field(default="./data/output", alias="OUTPUT_DIR")

    # Celery
    celery_broker_url: str = Field(default="redis://localhost:6379/1", alias="CELERY_BROKER_URL")
    celery_result_backend: str = Field(default="redis://localhost:6379/2", alias="CELERY_RESULT_BACKEND")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# 全局配置实例
settings = Settings()
