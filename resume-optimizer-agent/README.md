# Resume Optimizer AI Agent

基于招聘需求检索的简历智能优化系统。通过抓取真实职位信息，利用语义匹配和大语言模型对简历进行针对性优化，支持 Pipeline 流水线、Agent 自主决策、交互式对话、Focus 定向优化四种运行模式，并提供 Web 可视化界面。

## 功能特性

- **四种运行模式**：Pipeline（固定流水线）、Agent（AI 自主规划执行）、Interactive（多轮对话逐步优化）、Focus（根据指定 URL 定向优化）
- **智能职位检索**：自动从智联招聘抓取目标职位，支持分页抓取、薪资/行业二次过滤、详情页补充
- **语义匹配分析**：基于 SentenceTransformer 多语言模型计算简历与职位的语义相似度，结合技能关键词匹配和经验规则，加权综合评分
- **LLM 驱动优化**：分模块优化简历内容（专业技能、工作经历、项目经验、自我评价），严格遵循"只润色不编造"原则
- **反幻觉机制**：三层防御（Prompt 约束 + 后置校验 + 输出清洗），防止 LLM 编造数字、技术词和时间线
- **Agent 记忆系统**：短期对话记忆 + 长期用户偏好持久化，支持跨会话保持上下文
- **Web 可视化界面**：实时进度追踪（步骤指示器、阶段描述、耗时统计、停滞警告）、日志查看器、结果下载
- **多格式支持**：输入支持 PDF/DOCX/TXT，输出支持 DOCX（专业排版）和 Markdown
- **多数据源**：智联招聘（默认）、Boss 直聘、任意 URL 抓取

## 系统架构

```
用户简历 (PDF/DOCX/TXT)
        │
        ▼
┌─────────────────┐
│   简历解析器     │  提取模块内容、联系方式
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   职位检索       │  智联招聘 / Boss直聘 / 自定义URL
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   语义匹配分析   │  SentenceTransformer + 技能匹配 + 经验规则
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   LLM 简历优化   │  分模块优化 + 反幻觉校验
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   文件生成       │  DOCX（专业排版）/ Markdown + 优化报告
└─────────────────┘
```

Agent 模式下，上述流程由 AI Agent 通过 ReAct 循环自主规划执行顺序，可动态调整策略、迭代优化。

## 快速开始

### 环境要求

- Python 3.10+
- 通义千问 API Key（默认）或 OpenAI API Key

### 安装

```bash
# 克隆项目
git clone https://github.com/your-username/resume-optimizer-agent.git
cd resume-optimizer-agent

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 安装 Playwright 浏览器
playwright install chromium
```

### 配置

复制环境变量模板并填写 API Key：

```bash
cp .env.example .env
```

编辑 `.env`，至少配置 LLM API Key：

```env
# 使用通义千问（推荐）
LLM_PROVIDER=qwen
QWEN_API_KEY=your-api-key
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen-plus

# 或使用 OpenAI
# LLM_PROVIDER=openai
# OPENAI_API_KEY=your-api-key
```

### 运行

**Web 界面（推荐）**

```bash
python -m web.app
# 访问 http://localhost:8080
```

**命令行模式**

```bash
# Pipeline 模式（固定流水线）
python -m src.main --resume data/resumes/your_resume.pdf --keyword "Python后端" --location "北京"

# Agent 模式（AI 自主决策）
python -m src.main --agent --resume data/resumes/your_resume.pdf

# 交互式 Agent（多轮对话）
python -m src.main --interactive --resume data/resumes/your_resume.pdf

# 带筛选条件
python -m src.main --resume data/resumes/your_resume.pdf \
    --keyword "产品经理" \
    --location "上海" \
    --min-salary 15000 \
    --industry "互联网" \
    --max-detail-jobs 5 \
    --max-jobs 20 \
    --format docx
```

## 运行模式说明

### Pipeline 模式

固定五步流水线：解析简历 → 搜索职位 → 匹配分析 → LLM 优化 → 生成文件。流程确定，适合批量处理。

### Agent 模式

AI Agent 通过 ReAct 循环（思考 → 行动 → 观察 → 循环）自主规划执行。可调用 7 个工具：

| 工具 | 功能 |
|------|------|
| `parse_resume` | 解析简历文件 |
| `search_jobs` | 从智联招聘搜索职位 |
| `fetch_url_jobs` | 从指定 URL 抓取职位 |
| `match_resume` | 语义匹配分析 |
| `optimize_section` | 优化单个模块 |
| `evaluate_resume` | 评估简历质量 |
| `generate_resume` | 生成优化后的文件 |

### Interactive 模式

多轮对话模式，与 Agent 持续交互，逐步优化简历。支持 `/status`、`/history`、`/reset`、`/quit` 等命令。

### Focus 模式

用户提供目标职位页面的 URL，系统直接从指定 URL 抓取职位信息并优化简历，适合有明确目标岗位的场景。

## 项目结构

```
resume-optimizer-agent/
├── config/
│   └── config.py                # 全局配置（pydantic-settings）
├── src/
│   ├── main.py                  # CLI 入口 + Pipeline 流水线
│   ├── agents/
│   │   ├── resume_agent.py      # Agent 核心（ReAct 循环）
│   │   ├── tools.py             # Agent 工具层（7 个 LangChain Tools）
│   │   ├── matcher.py           # 语义匹配（SentenceTransformer）
│   │   ├── optimizer.py         # LLM 简历优化 + 反幻觉校验
│   │   ├── memory.py            # 记忆系统（短期 + 长期）
│   │   └── interactive.py       # 命令行交互式 Agent
│   ├── parsers/
│   │   └── resume_parser.py     # 简历解析（PDF/DOCX/TXT）
│   ├── scrapers/
│   │   └── job_scraper.py       # 职位抓取（智联/Boss/URL）
│   ├── generators/
│   │   └── resume_generator.py  # 简历生成（DOCX/Markdown）
│   └── utils/
│       └── text_utils.py        # 文本处理工具
├── web/
│   ├── app.py                   # FastAPI Web 后端
│   └── templates/
│       └── index.html           # 前端单页面
├── data/
│   ├── resumes/                 # 输入简历
│   ├── output/                  # 优化输出
│   └── memory/                  # Agent 记忆持久化
├── docker/
│   └── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## 技术栈

| 类别 | 技术 |
|------|------|
| LLM / Agent | LangChain, 通义千问 (Qwen) / OpenAI |
| 语义匹配 | SentenceTransformer (paraphrase-multilingual-MiniLM-L12-v2) |
| 数据抓取 | Playwright (Chromium) |
| 简历解析 | pdfplumber, python-docx |
| 文件生成 | python-docx (DOCX), Markdown |
| Web 框架 | FastAPI, Uvicorn |
| 配置管理 | pydantic-settings |
| 中文分词 | jieba |

## Docker 部署

```bash
docker-compose up -d
```

包含 4 个服务：应用主程序、Redis（缓存 + 消息队列）、ChromaDB（向量数据库）、Celery（异步任务 Worker）。

## 注意事项

- 语义匹配模型首次运行时会从 HuggingFace 镜像站下载（约 458MB），后续使用本地缓存
- 智联招聘抓取频率已做限速处理（默认 2 秒间隔），请勿调低以避免触发反爬
- LLM 优化遵循"只润色不编造"原则，不会凭空添加原文中不存在的技能、数字或经历
- Agent 模式的最大执行步数默认为 15 步，可在代码中调整

## License

MIT
