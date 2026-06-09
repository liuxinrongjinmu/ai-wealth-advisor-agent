# 智能投顾AI助手系统

基于 LangChain + LangGraph 构建的多层级、多模式金融AI助手系统，覆盖私募基金问答、投研分析、财富顾问三大场景。

---

## 项目架构

```
┌─────────────────────────────────────┐
│  混合式智能体（03）                │  智能路由 + 深度分析
│  反应式 + 深思熟虑 + 对话记忆      │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│  深思熟虑智能体（02）              │  五阶段流程 + 状态管理
│  感知→建模→推理→决策→报告         │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│  反应式智能体（01）                │  规则匹配 + 快速检索
│  关键词加权 + 语义理解 + 缓存      │
└─────────────────────────────────────┘
```

---

## 目录结构

```
智能投顾AI助手系统/
├── README.md                                              # 项目说明文档
├── requirements.txt                                       # 依赖清单
├── main.py                                                # 统一入口脚本（CLI）
├── web_api.py                                             # Web API服务（FastAPI）
├── 01-私募基金运作指引问答助手（反应式）/
│   ├── fund_qa_langgraph_v2.py                            # 核心代码：22条规则 + 多级匹配
│   ├── test_qa_assistant.py                               # 快速测试（7用例）
│   └── test_comprehensive.py                              # 综合测试（67用例）
├── 02-智能投研助手（深思熟虑）/
│   ├── deliberative_research_langgraph.py                 # 核心代码：五阶段投研流程
│   ├── test_research_assistant.py                         # 测试脚本（5项测试）
│   └── research_report_20260127_143626.txt                # 生成报告样本
└── 03-投顾AI助手（混合式）/
    ├── hybrid_wealth_advisor_langgraph.py                 # 核心代码：混合路由 + 客户画像
    └── test_wealth_advisor.py                             # 测试脚本（7项测试）
```

---

## 三大子系统

### 1. 私募基金运作指引问答助手（反应式）

**适用场景**：合规查询、规则咨询、知识问答

- 内置22条私募基金核心规则，覆盖8大类别（设立募集、监管规定、信息披露、投资范围、费用结构、退出清算、风险管理、合规要求）
- 多级智能匹配：22个特殊处理器 → 关键词加权匹配(权重1-100) → 语义相似度(SequenceMatcher) → LLM兜底增强
- 综合评分：关键词归一化(70%) + 语义相似度(30%) 加权合并
- 查询缓存：FIFO策略，默认128条，相同查询直接返回
- 支持标准问法、关键词查询、口语化表达、场景化提问

```
用户输入 → 关键词加权匹配 → 语义匹配 → 7:3综合评分 → 规则检索 → 答案返回
                ↓
          特殊处理器（优先）
```

### 2. 智能投研助手（深思熟虑）

**适用场景**：行业研究、投资报告生成、策略分析

- 五阶段工作流（LangGraph StateGraph）：

| 阶段 | 功能 | 输出格式 |
|------|------|---------|
| 感知(Perception) | 收集市场数据、新闻、指标 | JSON |
| 建模(Modeling) | 构建市场内部模型、风险评估 | JSON |
| 推理(Reasoning) | 生成3个候选投资方案 | JSON数组 |
| 决策(Decision) | 评估方案并选择最优策略 | JSON |
| 报告(Report) | 生成完整投研报告 | 纯文本 |

- 每阶段独立重试机制（最多3次）
- 完整状态管理（TypedDict）+ 依赖关系检查
- 报告自动保存为文件
- 提供同步 `run_research_agent()` 和异步 `arun_research_agent()` 接口

### 3. 投顾AI助手（混合式）

**适用场景**：财富管理、投资规划、行情查询

- 三层架构：
  - **协调层**：LLM评估查询类型（emergency/informational/analytical），动态选择处理模式
  - **反应式层**：快速响应 + 工具调用（上证指数/深证成指/沪深300实时行情）
  - **深思熟虑层**：数据收集 → 深度分析 → 建议生成

```
用户查询
    ↓
评估查询类型（assess）
    ↓
    ├── [反应式] → 工具调用 → 快速响应 → respond → END
    │
    └── [深思熟虑] → 数据收集 → 深度分析 → 建议生成 → respond → END
```

- 真实行情数据：新浪财经API获取实时指数行情，失败时自动降级为模拟数据
- 客户画像系统：内置平衡型/进取型两种画像，支持扩展
- 对话记忆：LangGraph MemorySaver checkpoint，通过 `thread_id` 实现多轮对话
- 提供同步 `run_wealth_advisor()` 和异步 `arun_wealth_advisor()` 接口

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置API密钥

```bash
# Windows
set DASHSCOPE_API_KEY=your-api-key

# Linux/Mac
export DASHSCOPE_API_KEY=your-api-key
```

### 3. 启动方式

**方式一：统一入口（CLI）**

```bash
python main.py
```

进入交互式菜单，选择要启动的子系统。

**方式二：Web API服务**

```bash
python web_api.py
```

启动后访问：
- API文档：http://localhost:8002/docs
- 健康检查：http://localhost:8002/

**方式三：单独运行子系统**

```bash
# 私募基金问答
python "01-私募基金运作指引问答助手（反应式）/fund_qa_langgraph_v2.py"

# 智能投研
python "02-智能投研助手（深思熟虑）/deliberative_research_langgraph.py"

# 财富顾问
python "03-投顾AI助手（混合式）/hybrid_wealth_advisor_langgraph.py"
```

### 4. 运行测试

```bash
python "01-私募基金运作指引问答助手（反应式）/test_qa_assistant.py"
python "01-私募基金运作指引问答助手（反应式）/test_comprehensive.py"
python "02-智能投研助手（深思熟虑）/test_research_assistant.py"
python "03-投顾AI助手（混合式）/test_wealth_advisor.py"
```

---

## Web API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 健康检查 |
| `/api/fund-qa` | POST | 私募基金问答 |
| `/api/research` | POST | 投研分析 |
| `/api/wealth-advisor` | POST | 财富顾问 |

### 请求示例

**私募基金问答**

```bash
curl -X POST http://localhost:8002/api/fund-qa \
  -H "Content-Type: application/json" \
  -d '{"query": "私募基金的合格投资者标准是什么？"}'
```

**投研分析**

```bash
curl -X POST http://localhost:8002/api/research \
  -H "Content-Type: application/json" \
  -d '{"topic": "新能源汽车行业投资机会", "industry": "电动汽车制造", "horizon": "长期"}'
```

**财富顾问**

```bash
curl -X POST http://localhost:8002/api/wealth-advisor \
  -H "Content-Type: application/json" \
  -d '{"query": "今天上证指数怎么样？", "customer_id": "customer1"}'
```

---

## 代码调用

```python
# 私募基金问答
from fund_qa_langgraph_v2 import FundQAAssistant
assistant = FundQAAssistant(cache_size=128)
result = assistant.process_query("合格投资者标准是什么")

# 投研分析
from deliberative_research_langgraph import run_research_agent
result = run_research_agent("AI行业投资机会", "大语言模型", "中期")
print(result["final_report"])

# 财富顾问（支持多轮对话）
from hybrid_wealth_advisor_langgraph import run_wealth_advisor
result = run_wealth_advisor("如何调整投资组合？", "customer1", thread_id="session-001")
print(result["final_response"])
```

---

## 技术栈

| 技术 | 版本 | 用途 |
|------|------|------|
| Python | 3.8+ | 开发语言 |
| LangChain | 1.2.0+ | LLM应用框架 |
| LangGraph | 0.2.0+ | 状态图工作流 |
| dashscope | 1.14.0+ | 通义千问API |
| pydantic | 2.x | 数据验证 |
| FastAPI | 0.100.0+ | Web API服务 |
| uvicorn | 0.23.0+ | ASGI服务器 |

---

## 性能指标

| 场景 | 响应时间 | 说明 |
|------|---------|------|
| 规则问答（01） | < 2秒 | 缓存命中更快 |
| 快速查询（03反应式） | 2-5秒 | 含行情API调用 |
| 深度分析（02/03深思熟虑） | 10-25秒 | 4-5次LLM调用 |

---

## 适用人群

- **金融从业者**：基金公司员工、合规人员、风控人员
- **投资分析师**：研究员、投资经理、策略师
- **财富顾问**：理财顾问、私人银行家、投资顾问
- **金融科技从业者**：产品经理、算法工程师、AI开发者
- **个人投资者**：有理财需求的专业投资者
- **AI/NLP学习者**：学习LangChain/LangGraph实战应用

---

## 扩展指南

- **新增规则**：在 `fund_qa_langgraph_v2.py` 的 `rules_db` 中添加，并在 `keyword_weights` 中配置权重
- **新增行情工具**：在 `hybrid_wealth_advisor_langgraph.py` 的 `reactive_processing` 工具字典中添加
- **新增客户画像**：在 `SAMPLE_CUSTOMER_PROFILES` 字典中添加新配置
- **自定义Prompt**：各模块顶部的 Prompt 常量可直接修改
- **接入新数据源**：参考 `_fetch_realtime_index` 实现模式，替换或新增数据获取函数

---

## 项目信息

- **技术栈**：Python 3.8+ / LangChain / LangGraph / 通义千问 / FastAPI
- **代码规模**：约2500+ 行Python代码
- **LLM模型**：Qwen-Turbo-2025-04-28
- **最后更新**：2026年6月
