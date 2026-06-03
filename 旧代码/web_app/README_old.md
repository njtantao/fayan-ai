# 法眼AI - 法律案件分析系统

基于 Minimax 大模型 + 356 条执行案例库的法律辅助分析工具。

---

## 快速启动

```bash
cd /Users/tt/Desktop/hermes/项目开发/法眼ai/web_app

export MINIMAX_API_KEY="sk-cp-ky-cbLKQr5w4-KHrV9wKzxU-hTtYhNqZyyB6Q8If3N2YwMipNx4WsYzWszA7jzpAcUuCEERm_l98RaLSfZPCmpzVxOy8256mh07SP6qkeJMbUlZQcwrnIyg"
export PORT=5099

python app.py
```

浏览器打开：`http://localhost:5099`

---

## 目录结构

```
web_app/
├── app.py              # Flask 服务，API 路由
├── fayan_api.py        # 核心模块（检索 + LLM + 规则引擎）
└── templates/
    └── index.html      # 前端页面

../extracted_cases/
└── cases.json          # 356 条执行案例知识库
```

---

## 接口说明

### 1. 案件分析（POST /api/analyze）

**请求**
```json
{
  "case_text": "案情描述（100字以上）",
  "amount": 500000,          // 可选，涉案金额（元）
  "party_count": 3,          // 可选，当事人数，默认2
  "has_evidence_gap": false, // 可选，证据是否有缺口
  "has_criminal_cross": false // 可选，是否刑民交叉
}
```

**响应**
```json
{
  "trace_id": "c9d502aba578",
  "complexity": "low",
  "lawyer_referral": false,
  "retrieved_cases": [
    {
      "case_number": "2025-17-5-201-003",
      "title": "0023长子县某工贸公司与山西某工贸公司执行异议案",
      "court": "长治市中级人民法院",
      "cause_of_action": "执行异议",
      "ruling_points": "执行中，对被执行一般银行账户中的资金，应当按照银行记载的情况及账户的名称作为权属判断的基础...",
      "score": 0.875
    }
  ],
  "conclusions": [
    {
      "content": "甲的请求不予支持。根据类案参考，执行中对于银行账户内资金的权属判断，应当以银行记载情况和账户名称作为基础与依据...",
      "citations": [
        {"type": "case", "id": "2025-17-5-201-003", "text": "执行中，对被执行一般银行账户中的资金，应当按照银行记载的情况及账户的名称作为权属判断的基础与依据..."}
      ]
    }
  ]
}
```

### 2. 类案检索（POST /api/retrieve）

**请求**
```json
{
  "query": "借用账户 排除执行",
  "top_k": 5
}
```

**响应**
```json
{
  "query": "借用账户 排除执行",
  "total": 5,
  "results": [...]
}
```

### 3. 服务状态（GET /status）

```json
{"ok": true, "cases_count": 356, "model": "MiniMax-M2.7", "error": null}
```

---

## 核心流程

```
用户输入案情
       │
       ▼
┌─────────────────────┐
│  规则引擎·复杂度判定  │  金额>50万/当事>5人/证据缺口 → complexity=high
└─────────┬───────────┘
          │
   complexity=high?
          │
    ┌─────┴─────┐
    │  是        │  否
    ▼            ▼
 律师引导页   RAG 检索
 停止分析     jieba 分词 + BM25 + TF-IDF + MMR 多样性重排
             返回 top-5 相关案例
                    │
                    ▼
             Minimax LLM 生成结论
             （仅引用知识库案例，禁止判断性措辞）
                    │
                    ▼
             规则引擎·禁用词扫描 + 引用校验
                    │
                    ▼
              结构化 JSON → 前端渲染
```

---

## 技术栈

| 组件 | 技术 |
|------|------|
| 前端 | 原生 HTML/CSS/JS（无框架） |
| 后端 | Flask + flask-cors |
| 检索 | jieba 分词 + BM25 + TF-IDF + MMR 多样性重排 |
| LLM | Minimax-M2.7（ChatOpenAI 兼容接口）|
| 知识库 | 356 条执行案例（JSON）|

---

## 规则引擎约束

**禁用词**（命中则删除该条结论）：
```
胜诉率 / 一定赢 / 会赢 / 会输 / 法院会支持 /
应当赔偿 / 必须赔偿 / 建议你去 / 胜算很大 / ...
```

**复杂度判定**：

| 条件 | 得分 | 等级 |
|------|------|------|
| 金额 > 50万 | +1 | |
| 金额 > 200万 | +1 | |
| 当事 > 5人 | +1 | |
| 当事 > 10人 | +1 | |
| 证据缺口 | +2 | |
| 刑民交叉 | 直接ultra | |
| 总分 ≥ 4 | | ultra |
| 总分 ≥ 2 | | high |
| 总分 ≥ 1 | | medium |
| 其他 | | low |

**律师介入触发**：complexity=high/ultra，或存在致命证据缺口。

---

## 检索调优参数

| 参数 | 值 | 说明 |
|------|----|------|
| BM25权重 | 35% | 关键词精确匹配 |
| TF-IDF权重 | 45% | 语义相关度 |
| 法条bonus | 20% | 命中"第X条"加分 |
| BM25 k1 | 1.5 | 词频饱和度 |
| BM25 b | 0.75 | 文档长度归一化 |
| MMR α | 0.7 | 70%相关性 + 30%多样性 |
| top_k | 5 | 返回案例数 |

---

## 案例库统计

| 指标 | 数值 |
|------|------|
| 总案例 | 356 条 |
| 有裁判要旨 | 287 条（80.6%）|
| 有关键词 | 355 条（99.7%）|
| 有关联法规 | 337 条（94.7%）|
| 最高法院案例 | 49 条（13.8%）|
| 时间跨度 | 1997–2025 |
| 主体案由 | 执行监督、执行复议、执行异议 |

---

## 启动失败排查

**`MINIMAX_API_KEY` 未设置**
```bash
echo $MINIMAX_API_KEY   # 应输出 key
```

**端口被占用**
```bash
lsof -ti:5099 | xargs kill -9
python app.py
```

**案例库加载失败**
检查 `../extracted_cases/cases.json` 是否存在且格式正确。

**API Key 无效（401）**
确认 key 为 `sk-cp-ky-...` 格式，可通过 `/status` 接口查看 error 字段。



   数据结构清晰，共13个字段。大模型问答的实现路径如下：
    
    
    
    三种实现方案
    
    方案1：轻量级（RAG检索，直接调用LLM）
    
    
    用户问题 → 向量检索(top-K相似案例) → 构建prompt → 调用LLM → 返回答案
    
    
    优点：快速搭建，灵活切换模型  
    缺点：单次请求，无多轮对话
    
    
    python
    核心逻辑（伪代码）
    query_embedding = embed(user_question)           # 1. 编码问题
    top_cases = vector_search(query_embedding, k=5)  # 2. 检索相似案例
    prompt = build_prompt(user_question, top_cases)  # 3. 构建上下文
    answer = llm(prompt)                              # 4. 调用LLM
    
    
    Embedding推荐：text-embedding-3-small（OpenAI）或 bge-m3（本地）  
    向量数据库：Chroma（轻量）、Milvus（生产级）
    
    
    
    方案2：本地知识库对话（LangChain + Ollama）
    
    如果你想完全本地运行，用Ollama部署：
    
    python
    用LangChain串联
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.chat_models import ChatOllama
    
    检索 → 构建RAG链
    rag_chain = RetrievalQA.from_chain_type(
        llm=ChatOllama(model="qwen3"),
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5})
    )
    answer = rag_chain.run("工伤赔偿标准是什么？")
    
    
    
    
    方案3：生产级系统（Milvus + FastAPI + 前端）
    
    如果你想做成完整产品，建议架构：
    
    
    ┌─────────────┐    ┌──────────────┐    ┌─────────────┐
    │  前端界面   │───▶│  FastAPI后端  │───▶│  Milvus向量库│
    └─────────────┘    └──────────────┘    └─────────────┘
                            │                      │
                            ▼                      ▼
                      ┌──────────────┐      ┌─────────────┐
                      │  LLM推理服务  │◀────│  Embedding  │
                      │  (vLLM/Ollama)│     │   服务      │
                      └──────────────┘      └─────────────┘
    
    
    
    
    快速起步建议
    
    你想用哪种方案？
    - 快速验证 → 方案1，几十行代码出结果
    - 完全本地 → 方案2，Ollama + qwen3
    - 做成产品 → 方案3，我来帮你搭完整流水线
    
    告诉我你的选择，我直接帮你实现。