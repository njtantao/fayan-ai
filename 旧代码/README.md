# 法眼AI - 法律案件分析系统

基于 MiniMax 大模型 + 裁判案例库的法律问答工具，支持命令行和 Web 界面两种使用方式。

**案例库规模：** 12,267 条（民事 4,026 + 刑事 8,241）

---

## 快速开始

### 方式一：Web 界面（推荐）

```bash
cd /Users/tt/Desktop/hermes/项目开发/法眼ai/web_app
python app.py
```

启动后访问：**http://localhost:5099**

界面功能：
- 案情描述输入（建议100字以上）
- 涉案金额（选填）
- 当事人数量（默认2）
- 证据缺口标记
- 刑民交叉标记
- 返回分析结论 + 类案参考

---

### 方式二：命令行问答

```bash
cd /Users/tt/Desktop/hermes/项目开发/法眼ai

# 单次问答
python3 rag_qa.py "工伤赔偿标准"

# 进入交互模式
python3 rag_qa.py
```

---

## API 接口

Web 服务同时暴露 REST API，可供其他程序调用。

### POST /api/analyze

案件分析（调用 LLM）

```bash
curl -X POST http://localhost:5099/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "case_text": "甲向乙借款50万元，出具借条，约定一年后还款。借款到期后甲无力还款，乙多次催要无果。乙持有借条和转账记录。",
    "amount": 500000,
    "party_count": 2,
    "has_evidence_gap": false,
    "has_criminal_cross": false
  }'
```

响应示例：
```json
{
  "complexity": "低复杂度",
  "conclusions": [
    {
      "content": "本案属于典型的民间借贷纠纷...",
      "citations": [...],
      "has_forbidden": false
    }
  ],
  "retrieved_cases": [
    {
      "case_number": "2023-16-2-104-001",
      "title": "方某诉上海某集团有限公司保证合同纠纷案",
      "court": "上海市第一中级人民法院",
      "score": 0.77
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

## 目录结构

```
法眼ai/
├── rag_qa.py                 # 命令行版 RAG 问答
├── web_app/
│   ├── app.py                # Flask Web 服务入口
│   ├── fayan_api.py          # 核心 API 层（LLM调用、规则引擎）
│   └── templates/index.html  # 前端页面
├── extracted_cases/
│   ├── all_cases.json        # 民事案例库
│   └── criminal_cases.json   # 刑事案例库
└── *.py                      # 数据提取脚本（一般不需要修改）
```

---

## 配置说明

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `MINIMAX_API_KEY` | MiniMax API 密钥 | 必填 |
| `PORT` | Web 服务端口 | 5000 |

配置文件：`web_app/.env`

```
MINIMAX_API_KEY=your-api-key-here
PORT=5099
```

### 模型配置（一般不需要修改）

| 配置项 | 值 | 说明 |
|--------|-----|------|
| API 地址 | `https://api.minimax.chat/v1` | MiniMax API |
| 模型 | `MiniMax-M2.7` | 当前使用模型 |
| 案例库 | `extracted_cases/all_cases.json` | 民事 |
| 刑事库 | `extracted_cases/criminal_cases.json` | 刑事 |

---

## 案例库更新

案例数据从原始文档提取，保存在 `extracted_cases/` 目录。

如需重新提取或更新案例，运行以下脚本：

```bash
# 提取2025年度案例
python3 extract_annual_2025.py

# 提取指导性案例
python3 extract_guidance_cases.py

# 提取刑事案例
python3 extract_criminal.py

# 合并所有案例
python3 extract_all_cases.py
```

---

## 系统架构

```
用户输入 → 规则判断（民事/刑事）→ BM25检索 → 构建Prompt → MiniMax LLM → 返回结果
                                    ↓
                              匹配相关判例 + 引用机制（防幻觉）
```

1. **分类**：根据关键词规则 + LLM辅助判断民事/刑事
2. **检索**：BM25 全文检索，取 Top-5 最相似判例
3. **生成**：将判例作为上下文，Prompt 引导大模型回答
4. **规则引擎**：过滤禁用词（胜诉率、一定赢等），判断案件复杂度

---

## 注意事项

- 分析结果仅供参考，不构成法律意见
- 重要案件请咨询执业律师
- 系统基于有限案例库，结论受案例覆盖度限制
- 案例库默认读取路径为 `extracted_cases/`，确保该目录存在且包含 `all_cases.json`