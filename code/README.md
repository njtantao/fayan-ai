# 法眼AI 法律案件分析系统 v3.0

基于 MiniMax 大模型 + 案例库的法律问答工具，支持命令行和 Web 界面两种使用方式。

**案例库规模：** 10,241 条（来自 `data/all_cases_perfect.csv`）

---

## 快速开始

### 方式一：命令行

```bash
cd /Users/tt/Desktop/hermes/项目开发/法眼ai代码

# 设置 API Key
export MINIMAX_API_KEY=your-minimax-api-key

# 交互模式
python fayan_main.py

# 单次问答
python fayan_main.py ask "工伤赔偿标准"
python fayan_main.py ask "民间借贷纠纷 借款合同"
```

### 方式二：Web 服务

```bash
python fayan_main.py server
```

启动后访问：**http://localhost:5099**

---

## API 接口

Web 服务同时暴露 REST API，可供其他程序调用。

### POST /api/analyze

案件分析（调用 LLM）

```bash
curl -X POST http://localhost:5099/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "case_text": "甲向乙借款50万元，出具借条，约定一年后还款。借款到期后甲无力还款，乙多次催要无果。",
    "amount": 500000,
    "party_count": 2,
    "has_evidence_gap": false,
    "has_criminal_cross": false
  }'
```

响应示例：
```json
{
  "trace_id": "a3f1b2c3d4e5",
  "complexity": "medium",
  "lawyer_referral": false,
  "retrieved_cases": [
    {
      "case_number": "[2019] 039_余某发诉张某合民间借贷案",
      "title": "民间借贷纠纷",
      "cause_of_action": "民间借贷纠纷",
      "ruling_points": "隐名投资人未经公司其他股东半数以上同意...",
      "score": 0.784
    }
  ],
  "conclusions": [
    {
      "content": "根据类案参考，本案属于典型的民间借贷纠纷...",
      "citations": [{"type": "case", "id": "[2019] 039", "text": "..."}]
    }
  ]
}
```

### POST /api/retrieve

纯检索（不调用 LLM，只返回相似案例）

```bash
curl -X POST http://localhost:5099/api/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "借款合同纠纷", "top_k": 5}'
```

### GET /status

服务状态检查

```bash
curl http://localhost:5099/status
```

---

## 配置说明

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `MINIMAX_API_KEY` | MiniMax API 密钥 | 必填 |
| `PORT` | Web 服务端口 | 5099 |

配置方式：
```bash
export MINIMAX_API_KEY=your-key-here
export PORT=5099
```

或创建 `.env` 文件（与 `fayan_main.py` 同目录）：
```
MINIMAX_API_KEY=your-key-here
PORT=5099
```

---

## 系统架构

```
用户输入 → 案件分类（民事/刑事/刑民交叉）→ BM25+TF-IDF+MMR 检索 → 构建Prompt → MiniMax LLM → 规则校验 → 输出
```

### 核心模块

**LegalRetriever（检索层）**
- BM25（35%）+ TF-IDF（45%）+ 法律术语bonus（20%）三路融合
- MMR（最大边际相关）多样性重排，取 Top-5

**RuleEngine（规则引擎）**
- 禁用词检测（胜诉率、一定赢、法院会判等）
- 复杂度判定（低/中/高/超高）
- 律师介入触发（高/超高复杂度自动提示）

**LegalLLM（生成层）**
- MiniMax M2.7 模型，temperature=0.2
- 强制引用：每条结论必须绑定案例编号
- 结构化 JSON 输出，防幻觉机制

---

## 目录结构

```
法眼ai代码/
├── fayan_main.py           # 主程序（CLI + Web服务）
├── requirements.txt        # Python 依赖
├── data/
│   ├── all_cases_perfect.csv   # 案例库（10,241条）
│   └── 最终.csv文件.zip        # 原始数据备份
├── 旧代码/                  # 旧版本代码（参考）
│   ├── rag_qa.py
│   ├── fayan_legal_rag.py
│   └── web_app/
└── 说明文件/                # 项目文档
```

---

## 使用说明

### 案情描述建议

- 建议 100 字以上，越详细检索越准确
- 包含：当事人关系、事实经过、诉求、涉及金额
- 明确标注是否涉及刑事（诈骗、盗窃等关键词会自动识别）

### 案件复杂度说明

系统根据以下因素自动判定复杂度：

| 因素 | 阈值 |
|------|------|
| 涉案金额 > 50万 | +1分 |
| 涉案金额 > 200万 | +1分 |
| 当事人数 > 5 | +1分 |
| 当事人数 > 10 | +1分 |
| 证据存在缺口 | +2分 |
| 涉及刑民交叉 | 直接超高 |

- 低/中复杂度：正常分析
- 高/超高复杂度：触发律师介入提示

### 检索测试（不调用 LLM）

```bash
python fayan_main.py ask "工伤赔偿 劳动纠纷"
```

输出 Top-3 相似案例，可验证检索效果。

---

## 依赖安装

```bash
pip install -r requirements.txt
```

依赖清单：
- `jieba` — 中文分词
- `rank-bm25` — BM25 检索算法
- `scikit-learn` — TF-IDF 向量化
- `scipy` — 稀疏矩阵运算
- `langchain-openai` — LLM 调用封装
- `fastapi` — Web 框架
- `uvicorn` — ASGI 服务器
- `pandas` — 数据处理
- `python-dotenv` — 环境变量

---

## 数据说明

### 案例库字段（all_cases_perfect.csv）

| 字段 | 说明 |
|------|------|
| 文件名 | 案例编号和来源路径 |
| 案件描述 | 案情正文（经去标识化处理） |
| 原告诉求 | 原告的诉讼请求 |
| 判别标准 | 裁判要点 / 争议焦点 |
| 判决结果 | 一审/二审判决结论 |
| 关键词_01~10 | 10 个分类标签（案由、法规等） |

### 案例分类

系统根据关键词自动区分民事/刑事/刑民交叉：

- **刑事关键词**：`盗窃`、`诈骗`、`故意伤害`、`受贿`、`贩毒` 等
- **民刑交叉关键词**：`合同诈骗罪`、`非法吸收公众存款罪` 等
- 无以上关键词则默认为民事

---

## 注意事项

- 分析结果仅供参考，不构成法律意见
- 重要案件请咨询执业律师
- 系统基于有限案例库，结论受案例覆盖度限制
- 禁用判断性表达（胜诉率、一定赢、会赢等），违规则自动过滤
- 所有结论必须附带案例引用，无引用则结论被删除