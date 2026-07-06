# Resume Optimizer Agent

基于 LLM + ReAct Agent 架构的智能简历优化系统。通过语义匹配引擎 + LLM-as-Judge 评估 + 反幻觉三层防御，从智联招聘真实职位驱动简历定向优化，支持 Pipeline / Agent / Interactive / Focus 四种运行模式。

## 核心特性

- **ReAct Agent 架构** — 手写 ReAct 循环，LLM 自主决定工具调用顺序和参数，支持动态策略调整
- **反幻觉三层防御** — Prompt 约束 + 后置校验（长度/数字/技术词三重检查）+ 输出清洗（30+ 正则模式），确保不编造任何经历
- **语义匹配引擎** — SentenceTransformer 多语言模型，技能匹配(50%) + 语义相似度(30%) + 经验匹配(20%) 三维加权评分
- **LLM-as-Judge 评估** — 6 维度对比评分 + 闭环反馈（薄弱维度自动重优化），量化优化效果
- **多职位合并优化** — 自动综合多个高匹配职位需求，LLM 智能合并后定向优化
- **四种运行模式** — Pipeline（一键执行）、Agent（AI 自主决策）、Interactive（多轮对话）、Focus（URL 定向）
- **Web 可视化界面** — 实时步骤追踪、工具调用监控、终端风格日志、文件上传下载
- **共享状态同步** — 声明式字段权限表 + 发布-订阅模式，工具间数据自动流转

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    Web 前端 (index.html)                         │
│    纯原生 HTML/CSS/JS · 4 Tab · 500ms 轮询 · 实时追踪           │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP (FastAPI)
┌────────────────────────────▼────────────────────────────────────┐
│                    Web 后端 (web/app.py)                         │
│    FastAPI · 内存任务管理 · 后台线程 · loguru 日志隔离            │
│                                                                  │
│    ┌──────────┐ ┌──────────┐ ┌────────────┐ ┌───────────────┐   │
│    │ Pipeline │ │  Agent   │ │ Interactive│ │    Focus      │   │
│    └────┬─────┘ └────┬─────┘ └─────┬──────┘ └──────┬────────┘   │
└─────────┼────────────┼─────────────┼───────────────┼────────────┘
          │            │             │               │
          │   ┌────────▼─────────────▼───────────────▼──────┐
          │   │          ResumeAgent (ReAct 核心)            │
          │   │  手写循环 · 文本解析工具调用 · 共享状态同步    │
          │   └────────────────────┬────────────────────────┘
          │                        │
     ┌────▼────────────────────────▼──────────────────────────┐
     │                    工具层 (tools.py)                     │
     │  ParseResume · SearchJobs · FetchURLJobs · MatchAllJobs │
     │  OptimizeSection · EvaluateResume · JudgeResume         │
     │  GenerateResume                                         │
     └──────┬──────────┬──────────┬──────────┬────────────────┘
            │          │          │          │
     ┌──────▼───┐ ┌───▼────┐ ┌──▼───────┐ ┌▼──────────────┐
     │ 简历解析 │ │职位抓取│ │匹配引擎  │ │ 优化 + 评估   │
     │ parser   │ │scraper │ │matcher   │ │ optimizer     │
     │          │ │        │ │          │ │ evaluator     │
     │pdfplumber│ │Playwright│ │Sentence │ │ LLM (Qwen)   │
     │python-docx│ │BS4     │ │Transformer│ │ LCEL chain   │
     └──────────┘ └────────┘ └──────────┘ └───────────────┘
```

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
# macOS 需要清除可能冲突的环境变量
env -u PYTHONHOME -u PYTHONPATH python -m web.app

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
```

## 运行模式

| 模式 | 执行方式 | 工具决策 | 适用场景 |
|------|---------|---------|---------|
| **Pipeline** | 固定 5 步串行 | 代码硬编码 | 一键执行，无需交互 |
| **Agent** | ReAct 循环，最多 15 步 | LLM 自主决策 | 灵活优化策略 |
| **Interactive** | 多轮对话，每轮 ReAct | LLM 自主决策 | 逐步调整优化方向 |
| **Focus** | 固定 7 步串行 | 代码硬编码 | 定向 URL 精准优化 |

### Agent 工具列表

Agent 通过文本格式（`TOOL_CALL` + `ARGS`）调用以下 9 个工具：

| 工具 | 功能 | 关键约束 |
|------|------|---------|
| `parse_resume` | 解析 PDF/DOCX/TXT 简历 | 必须第一步调用 |
| `search_jobs` | 从智联招聘搜索职位 | 只允许调用一次 |
| `fetch_url_jobs` | 从指定 URL 抓取职位详情 | 只允许调用一次，与 search_jobs 二选一 |
| `match_all_jobs` | 全职位匹配排序 | 必须在 optimize 之前调用 |
| `optimize_section` | 定向优化单个模块 | 支持多职位合并优化 |
| `judge_resume` | LLM-as-Judge 6 维度评估 | 触发闭环反馈 |
| `generate_resume` | 生成 DOCX/Markdown 文件 | 最后一步调用 |

### Agent 执行流程

```
parse_resume → search_jobs / fetch_url_jobs → match_all_jobs
    → optimize_section (逐模块) → judge_resume
    → [薄弱维度重优化 → judge_resume] (闭环反馈)
    → generate_resume
```

## 核心设计

### 反幻觉三层防御

```
第一层: Prompt 约束
  └─ 系统提示词严格限制"只润色不编造"，含正确/错误示例对比

第二层: 后置校验
  ├─ 长度检查: 优化后 > 原文 1.5 倍 → 标记异常
  ├─ 数字检查: 正则提取带单位数字，对比原文
  └─ 技术词检查: 30+ 行业术语逐一检查 (LIMS, GMP, SAP...)

第三层: 输出清洗
  └─ 30+ 正则模式去除 LLM 元文本 ("重要提醒", "⚠️", "📌"...)
```

### 语义匹配引擎

```
综合分数 = 技能匹配 × 0.5 + 语义相似度 × 0.3 + 经验匹配 × 0.2

技能匹配 (50%):  100+ 技术关键词库 + 字符串包含匹配
语义相似度 (30%): SentenceTransformer 余弦相似度，专业技能加权 2x
经验匹配 (20%):  规则匹配 ("项目", "工作", "实习" 关键词)
```

### 共享状态同步

工具间通过声明式字段权限表实现数据自动流转：

```python
_TOOL_FIELDS = {
    "parse_resume":     {"resume_data"},
    "search_jobs":      {"jobs"},
    "match_all_jobs":   {"resume_data", "jobs", "match_result", "top_jobs"},
    "optimize_section": {"resume_data", "jobs", "match_result", "optimized_sections", "top_jobs"},
    "judge_resume":     {"resume_data", "jobs", "match_result", "optimized_sections", "top_jobs"},
    "generate_resume":  {"resume_data", "optimized_sections"},
}
```

每次工具执行后，`_sync_shared_state()` 自动收集最新状态并按权限分发给所有声明了对应字段的工具。

### 记忆系统

- **短期记忆**（内存）：对话历史 + 工具调用结果，支持最近 N 轮上下文注入
- **长期记忆**（JSON 持久化）：用户偏好（目标行业/岗位/关键词）+ 历史优化记录（最近 20 条）

## 项目结构

```
resume-optimizer-agent/
├── config/
│   └── config.py                # pydantic-settings 全局配置
├── src/
│   ├── main.py                  # CLI 入口 + Pipeline 编排
│   ├── agents/
│   │   ├── resume_agent.py      # ReAct Agent 核心（手写循环）
│   │   ├── tools.py             # 9 个 LangChain BaseTool 定义
│   │   ├── matcher.py           # 语义匹配引擎（SentenceTransformer）
│   │   ├── optimizer.py         # LLM 简历优化器（含反幻觉校验）
│   │   ├── evaluator.py         # LLM-as-Judge 评估器
│   │   ├── memory.py            # 短期 + 长期记忆系统
│   │   └── interactive.py       # 交互式 Agent 封装
│   ├── parsers/
│   │   └── resume_parser.py     # 多格式简历解析器
│   ├── scrapers/
│   │   └── job_scraper.py       # 职位抓取（智联/Boss/URL）
│   ├── generators/
│   │   └── resume_generator.py  # DOCX/Markdown 简历生成
│   └── utils/
│       └── text_utils.py        # 文本清洗工具
├── web/
│   ├── app.py                   # FastAPI 后端（7 个 API 端点）
│   └── templates/
│       └── index.html           # 前端单文件（4 Tab + 实时追踪）
├── data/
│   ├── resumes/                 # 用户上传的简历
│   ├── output/                  # 优化输出文件
│   ├── jobs/                    # 职位数据缓存
│   └── memory/                  # 用户偏好 JSON 持久化
├── tests/
├── .env.example
├── requirements.txt
└── docker/
```

## 技术栈

| 类别 | 技术 |
|------|------|
| LLM / Agent | LangChain + 通义千问 (Qwen) / OpenAI |
| Agent 架构 | 手写 ReAct 循环，文本解析工具调用 |
| 语义匹配 | SentenceTransformer (paraphrase-multilingual-MiniLM-L12-v2) |
| 数据抓取 | Playwright (Chromium) + BeautifulSoup4 |
| 简历解析 | pdfplumber (PDF) + python-docx (DOCX) |
| 简历生成 | python-docx (A4 排版 + 专业配色) |
| Web 框架 | FastAPI + Uvicorn |
| 前端 | 纯原生 HTML/CSS/JS（零框架依赖） |
| 日志 | loguru (自定义 sink + threading.local 线程隔离) |
| 配置 | pydantic-settings |

## API 端点

| 方法 | 路径 | 功能 |
|------|------|------|
| GET | `/` | 前端页面 |
| POST | `/api/upload` | 上传简历文件 |
| POST | `/api/run` | 启动优化任务 |
| GET | `/api/status/{task_id}` | 查询任务状态（500ms 轮询） |
| GET | `/api/download/{filename}` | 下载输出文件 |
| POST | `/api/chat/{task_id}` | 交互式 Agent 对话 |

## 性能参考

| 操作 | 耗时 |
|------|------|
| Qwen API 单次调用 | ~10.8 秒 |
| Agent 完整流程 | 2-3 分钟 |
| 智联招聘抓取 | ~50 秒 |
| SentenceTransformer 加载 | ~1.5 秒（离线模式） |
| 定向优化 + 缓存 | 节省 ~66% LLM 调用 |

## 注意事项

- 语义匹配模型首次运行需从 HuggingFace 镜像站下载（~458MB），后续使用本地缓存。国内环境已配置 `HF_ENDPOINT=https://hf-mirror.com`
- 模型强制离线加载（`TRANSFORMERS_OFFLINE=1`），在 `web/app.py` 最顶部设置，早于所有 import
- 智联招聘抓取已做限速处理（默认 2 秒间隔），请勿调低以避免触发反爬
- Boss 直聘反爬机制严格，默认使用智联招聘
- macOS 启动 uvicorn 时需使用 `env -u PYTHONHOME -u PYTHONPATH` 避免环境冲突

## License

MIT
