#!/usr/bin/env python3
"""
法眼AI 法律案件分析系统 v3.0（重新设计）
基于新数据格式：data/all_cases_perfect.csv（10,241条案例）

核心架构：
  用户输入 → 案情分类（民事/刑事）→ BM25+TF-IDF+MMR 检索 → 构建Prompt → MiniMax LLM → 规则校验 → 输出

依赖安装：
  pip install fastapi uvicorn pandas python-dotenv jieba rank-bm25 scikit-learn scipy langchain-openai

用法：
  Web服务: python fayan_main.py server
  CLI问答: python fayan_main.py ask "工伤赔偿标准"
  交互模式: python fayan_main.py
"""

import os
import re
import json
import hashlib
import traceback
from dataclasses import dataclass, field, asdict
from typing import Optional, Literal

# ============================================================
# 配置
# ============================================================
from dotenv import load_dotenv
load_dotenv()

MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", os.environ.get("MINIMAX_API_KEY_2", ""))
MINIMAX_BASE_URL = "https://api.minimax.chat/v1"
LLM_MODEL = "MiniMax-M2.7"

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CASES_CSV = os.path.join(DATA_DIR, "data", "all_cases_perfect.csv")

DEFAULT_TOP_K = 5
MAX_CONTEXT_CHARS = 4500

# ============================================================
# 数据结构
# ============================================================
@dataclass
class Citation:
    type: str  # "case" | "statute"
    id: str
    text: str

@dataclass
class LegalConclusion:
    content: str
    citations: list[Citation] = field(default_factory=list)
    has_forbidden: bool = False
    is_valid: bool = True

@dataclass
class RetrievedCase:
    case_number: str       # 文件名作为编号
    title: str             # 关键词_01（案由）作为标题
    court: str             # 从案件描述中提取，或"N/A"
    cause_of_action: str   # 关键词_01
    ruling_points: str     # 判别标准
    judgment_result: str   # 判决结果
    keywords: list[str]    # 关键词_01~10
    score: float

@dataclass
class AnalysisResult:
    conclusions: list[LegalConclusion]
    complexity: str         # "low" | "medium" | "high" | "ultra"
    lawyer_referral: bool = False
    lawyer_message: str = ""
    raw_output: str = ""
    trace_id: str = ""
    retrieved_cases: list[RetrievedCase] = field(default_factory=list)

# ============================================================
# 规则引擎
# ============================================================
class RuleEngine:
    FORBIDDEN = [
        "胜诉率", "一定赢", "会赢", "会输", "法院会支持", "法院会判",
        "应当赔偿", "必须赔偿", "肯定胜诉", "绝对胜诉",
        "建议你去", "建议你找", "你应该请", "胜算很大", "胜算较高",
        "法院大概率会", "通常会判", "一般会认定",
    ]

    def __init__(self):
        self.pat = re.compile("|".join(re.escape(w) for w in self.FORBIDDEN), re.I)

    def check(self, text: str) -> tuple[bool, list]:
        hits = self.pat.findall(text)
        return len(hits) > 0, hits

    def judge_complexity(self, case_text: str, amount: float = 0,
                         party_count: int = 2,
                         has_evidence_gap: bool = False,
                         has_criminal_cross: bool = False) -> str:
        score = 0
        if amount > 500000: score += 1
        if amount > 2000000: score += 1
        if party_count > 5: score += 1
        if party_count > 10: score += 1
        if has_evidence_gap: score += 2
        if has_criminal_cross: return "ultra"
        if score >= 4: return "ultra"
        if score >= 2: return "high"
        if score >= 1: return "medium"
        return "low"

    def should_refer_lawyer(self, complexity: str,
                             has_fatal_gap: bool = False) -> tuple[bool, str]:
        if complexity in ["high", "ultra"] or has_fatal_gap:
            return True, (
                "基于目前风险评估，该案件已超出系统智能辅助范围，"
                "建议您考虑专业律师支持。如需了解公共法律服务资源，可通过12348法律服务热线获取帮助。"
            )
        return False, ""

# ============================================================
# 案件分类器
# ============================================================
class CaseClassifier:
    """根据案情文本自动判断案件类型（民事/刑事/刑民交叉）"""

    CRIMINAL_KEYWORDS = [
        "盗窃", "抢劫", "抢夺", "故意伤害", "故意杀人", "过失致人死亡",
        "诈骗", "合同诈骗", "集资诈骗", "贷款诈骗", "票据诈骗",
        "敲诈勒索", "绑架", "非法拘禁",
        "贩卖毒品", "运输毒品", "持有毒品", "走私毒品",
        "贪污", "贿赂", "受贿", "行贿", "挪用公款",
        "职务侵占", "挪用资金",
        "强奸", "强制猥亵", "猥亵儿童",
        "危险驾驶", "交通肇事", "肇事逃逸",
        "非法吸收公众存款", "组织、领导传销活动",
        "开设赌场", "聚众赌博", "网络赌博",
        "寻衅滋事", "聚众斗殴", "非法持有枪支",
        "走私", "逃税", "骗取出口退税",
        "伪证", "妨害作证", "帮助毁灭证据",
        "重婚", "破坏军婚",
        "盗伐林木", "非法采矿", "污染环境",
        "偷渡", "组织偷渡",
        "传播淫秽物品", "制作、复制、传播淫秽物品",
        "拐卖妇女", "拐卖儿童", "收买被拐卖妇女",
        "虐待", "遗弃", "暴力干涉婚姻自由",
        "高空抛物", "妨害公务", "袭警",
        "罪", "刑事", "犯罪",
    ]

    CROSS_KEYWORDS = [
        "先刑后民", "刑民交叉", "刑事附带民事",
        "合同诈骗罪", "诈骗罪", "非法吸收公众存款罪",
        "职务侵占罪", "挪用资金罪", "合同欺诈",
        "涉嫌犯罪", "刑事立案", "刑事责任",
    ]

    @classmethod
    def classify(cls, text: str) -> tuple[str, float]:
        criminal_score = sum(1 for kw in cls.CRIMINAL_KEYWORDS if kw in text)
        cross_score = sum(1 for kw in cls.CROSS_KEYWORDS if kw in text)

        if cross_score >= 1:
            return "刑民交叉", 0.85
        if criminal_score >= 2:
            return "刑事", 0.9
        if criminal_score >= 1:
            return "刑事", 0.75
        return "民事", 0.8

    @classmethod
    def is_criminal(cls, cause_of_action: str) -> bool:
        """根据案由关键词判断是否刑事"""
        criminal_causes = ["罪", "盗窃", "抢劫", "诈骗", "伤害", "贩毒", "贪污", "贿赂", "挪用", "开设赌场", "交通肇事", "重婚", "遗弃", "虐待", "拐卖", "走私", "逃税"]
        return any(kw in cause_of_action for kw in criminal_causes)

# ============================================================
# RAG 检索（BM25 + TF-IDF + MMR）
# ============================================================
class LegalRetriever:
    """三路检索融合：BM25 + TF-IDF + 法律术语bonus，MMR多样性重排"""

    def __init__(self, csv_path: str):
        import pandas as pd
        import jieba
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        from rank_bm25 import BM25Okapi

        print(f"加载案例库: {csv_path}")
        df = pd.read_csv(csv_path)

        # 数据清洗
        df["案件描述"] = df["案件描述"].fillna("")
        df["原告诉求"] = df["原告诉求"].fillna("")
        df["判别标准"] = df["判别标准"].fillna("")
        df["判决结果"] = df["判决结果"].fillna("")

        # 构建案例列表
        self.cases = []
        for idx, row in df.iterrows():
            case = {
                "id": idx,
                "file_name": str(row["文件名"]) if pd.notna(row["文件名"]) else "",
                "description": str(row["案件描述"]) if row["案件描述"] else "",
                "claim": str(row["原告诉求"]) if row["原告诉求"] else "",
                "ruling_points": str(row["判别标准"]) if row["判别标准"] else "",
                "judgment_result": str(row["判决结果"]) if row["判决结果"] else "",
                "cause": str(row["关键词_01"]) if pd.notna(row["关键词_01"]) else "",
            }
            # 收集所有关键词
            keywords = []
            for i in range(1, 11):
                col = f"关键词_{i:02d}"
                if col in row and pd.notna(row[col]):
                    keywords.append(str(row[col]))
            case["keywords"] = keywords

            # 构建搜索文本（字段加权：关键字段重复多次以提升匹配权重）
            def w(text: str, n: int) -> str:
                """将文本重复n次（用于字段加权）"""
                return " ".join([text] * n) if text else ""

            search_parts = [
                w(case["cause"], 3),                              # 案由权重最高
                w(" ".join(keywords), 2),                          # 关键词次之
                w(case["ruling_points"], 2),                      # 判别标准
                w(case["claim"], 1),                             # 原告诉求
                w(case["description"], 1),                         # 案件描述
                w(case["judgment_result"], 1),                    # 判决结果
            ]
            search_text = " ".join(p.strip() for p in search_parts if p.strip() and p != "nan")
            search_text = re.sub(r'[\s\n\r\t]+', ' ', search_text).strip()
            case["search_text"] = search_text

            self.cases.append(case)

        print(f"  加载案例: {len(self.cases)} 条")

        # 分词
        self.tokenized = []
        for case in self.cases:
            tokens = [t for t in jieba.cut(case["search_text"]) if len(t) > 1]
            self.tokenized.append(tokens)

        # TF-IDF
        token_strs = [" ".join(tokens) for tokens in self.tokenized]
        self.vectorizer = TfidfVectorizer(max_features=2048, token_pattern=r'(?u)\b\w+\b')
        self.tfidf_matrix = self.vectorizer.fit_transform(token_strs)
        self.cosine_sim = cosine_similarity

        # BM25
        self.bm25 = BM25Okapi(self.tokenized)
        self.bm25.param_b = 0.75
        self.bm25.param_k1 = 1.5

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        import jieba
        import numpy as np

        query_tokens = [t for t in jieba.cut(query) if len(t) > 1]
        query_str = " ".join(query_tokens)

        # BM25
        bm25_scores_raw = np.array(self.bm25.get_scores(query_tokens), dtype=float)

        # TF-IDF
        query_vec = self.vectorizer.transform([query_str])
        tfidf_scores = self.cosine_sim(query_vec, self.tfidf_matrix).flatten()

        # 法律术语精确匹配 bonus
        legal_terms = [t for t in query_tokens if len(t) >= 2]
        term_bonus = np.zeros(len(self.cases))
        if legal_terms:
            for idx, tokens in enumerate(self.tokenized):
                term_hits = sum(1 for t in legal_terms if t in tokens)
                term_bonus[idx] = term_hits / max(len(legal_terms), 1)

        # 归一化
        def normalize(scores):
            mn, mx = scores.min(), scores.max()
            if mx <= mn:
                return np.zeros_like(scores)
            return (scores - mn) / (mx - mn)

        bm25_norm = normalize(bm25_scores_raw)
        tfidf_norm = normalize(tfidf_scores)
        bonus_norm = normalize(term_bonus)

        # 加权融合
        fused = 0.35 * bm25_norm + 0.45 * tfidf_norm + 0.20 * bonus_norm

        # MMR 多样性重排
        top_candidates = np.argsort(fused)[::-1][:k * 3]
        selected_vectors = []
        mmr_ranked = []

        cand_dense = self.tfidf_matrix[top_candidates].toarray().astype(float)
        norms = np.linalg.norm(cand_dense, axis=1, keepdims=True)
        norms[norms == 0] = 1
        cand_dense_norm = cand_dense / norms

        for local_rank, global_idx in enumerate(top_candidates):
            if len(mmr_ranked) >= k:
                break
            case_vec_norm = cand_dense_norm[local_rank]
            if mmr_ranked:
                sims = np.dot(cand_dense_norm, case_vec_norm)
                max_sim = float(np.max([s for i, s in enumerate(sims) if i != local_rank]))
                mmr_score = 0.7 * fused[global_idx] - 0.3 * max_sim
            else:
                mmr_score = fused[global_idx]
            selected_vectors.append(case_vec_norm)
            mmr_ranked.append((global_idx, mmr_score))

        mmr_ranked.sort(key=lambda x: x[1], reverse=True)

        results = []
        for global_idx, mmr_score in mmr_ranked:
            case = self.cases[global_idx]
            results.append({
                "case": case,
                "score": float(mmr_score),
                "bm25_score": float(bm25_norm[global_idx]),
                "tfidf_score": float(tfidf_norm[global_idx]),
            })
        return results

# ============================================================
# LLM 调用
# ============================================================
class LegalLLM:
    def __init__(self, api_key: str, base_url: str, model: str):
        from langchain_openai import ChatOpenAI
        self.llm = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=0.2,
            request_timeout=120
        )

    def build_prompt(self, case_text: str, retrieved_cases: list[dict]) -> str:
        cases_text = []
        for i, item in enumerate(retrieved_cases, 1):
            case = item["case"]
            case_id = case["file_name"].split("/")[-1].replace(".txt", "").replace(".TXT", "")
            cases_text.append(
                f"""[案例{i}] {case_id}
案由: {case['cause']}
判别标准: {case['ruling_points'][:300]}
原告诉求: {case['claim'][:200]}
判决结果: {case['judgment_result'][:200]}
"""
            )
        cases_block = "\n".join(cases_text)

        prompt = (
            "你是一个法律分析助手，基于以下类案参考进行分析。\n"
            "\n"
            "【硬性规则 - 违反直接终止输出】\n"
            "1. 只允许引用下方「类案参考」中列出的案例，禁止自行编造、推测、拼接案例编号\n"
            "2. 每条结论的 citations 数组中，id 字段必须严格等于下面某个案例的文件名字符串\n"
            "3. 禁止使用判断性表达（胜诉率、一定赢、会赢、会输、应当赔偿等）\n"
            "4. 如果类案参考中没有与用户案情相关的案例，结论 content 必须以「现有类案参考不足」开头\n"
            "5. 只输出结构化JSON，不要输出其他内容\n"
            "\n"
            "---\n"
            "【类案参考】（共 " + str(len(retrieved_cases)) + " 个案例，全部真实存在）\n"
            "\n"
            + cases_block + "\n"
            "---\n"
            "【用户案情】\n"
            + case_text + "\n"
            "---\n"
            "【输出格式（严格按照此JSON结构）】\n"
            "{\n"
            '  "conclusions": [\n'
            '    {\n'
            '      "content": "结论内容（引用来源时必须附上案例编号）",\n'
            '      "citations": [{"type": "case", "id": "案例编号（文件名字符串）", "text": "引用原文片段"}]\n'
            "    }\n"
            "  ],\n"
            '  "summary": "简要总结（100字内）"\n'
            "}\n"
        )
        return prompt

    def call(self, case_text: str, retrieved_cases: list[dict]) -> str:
        prompt = self.build_prompt(case_text, retrieved_cases)
        resp = self.llm.invoke(prompt)
        return resp.content

    def parse_response(self, raw: str) -> dict:
        json_match = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = raw.strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            try:
                start = raw.index('{')
                end = raw.rindex('}') + 1
                return json.loads(raw[start:end])
            except:
                return {"conclusions": [], "summary": raw[:200], "parse_error": True}

# ============================================================
# 主系统
# ============================================================
class FaYanLegal:
    def __init__(self, api_key: str, base_url: str, model: str, csv_path: str):
        self.rule_engine = RuleEngine()
        self.llm = LegalLLM(api_key, base_url, model)
        self.retriever = LegalRetriever(csv_path)
        self.trace_counter = 0

    def analyze(self, case_text: str,
                amount: float = 0,
                party_count: int = 2,
                has_evidence_gap: bool = False,
                has_criminal_cross: bool = False) -> AnalysisResult:
        import time
        start = time.time()
        self.trace_counter += 1
        trace_id = hashlib.md5(
            f"{case_text}{self.trace_counter}{start}".encode()
        ).hexdigest()[:12]

        # 案件类型判断
        case_type_raw, _ = CaseClassifier.classify(case_text)
        is_criminal = CaseClassifier.is_criminal(case_type_raw)
        if case_type_raw == "刑民交叉" and has_criminal_cross:
            is_criminal = True

        # 复杂度判定
        complexity = self.rule_engine.judge_complexity(
            case_text, amount, party_count, has_evidence_gap, has_criminal_cross
        )

        # 律师介入检查
        lawyer_referral, lawyer_msg = self.rule_engine.should_refer_lawyer(
            complexity, has_evidence_gap
        )
        if lawyer_referral:
            return AnalysisResult([], complexity, True, lawyer_msg, "", trace_id, [])

        # RAG 检索
        retrieved = self.retriever.retrieve(case_text, k=5)

        # LLM 生成
        raw = self.llm.call(case_text, retrieved)

        # 解析响应
        parsed = self.llm.parse_response(raw)

        # 规则校验
        conclusions = []
        for conc in parsed.get("conclusions", []):
            content = conc.get("content", "")
            has_forbidden, _ = self.rule_engine.check(content)
            if has_forbidden:
                continue
            if content:
                citations = [
                    Citation(
                        type=c.get("type", "case"),
                        id=c.get("id", ""),
                        text=c.get("text", "")
                    ) for c in conc.get("citations", [])
                ]
                conclusions.append(LegalConclusion(content, citations))

        # 包装检索到的案例
        retrieved_cases_info = []
        for r in retrieved:
            case = r["case"]
            case_id = case["file_name"].split("/")[-1].replace(".txt", "").replace(".TXT", "")
            retrieved_cases_info.append(RetrievedCase(
                case_number=case_id,
                title=case["cause"],
                court="N/A",
                cause_of_action=case["cause"],
                ruling_points=case["ruling_points"][:300] if case["ruling_points"] else "",
                judgment_result=case["judgment_result"][:200] if case["judgment_result"] else "",
                keywords=case["keywords"],
                score=round(r["score"], 3)
            ))

        if not conclusions:
            return AnalysisResult(
                [], complexity, False,
                "依据现有材料与知识库，无法形成具有依据的分析结论。建议您咨询专业律师。",
                raw, trace_id, retrieved_cases_info
            )

        return AnalysisResult(conclusions, complexity, False, "", raw, trace_id, retrieved_cases_info)

    def to_dict(self, result: AnalysisResult) -> dict:
        def _clean(v):
            if v is None:
                return None
            if isinstance(v, float):
                return round(v, 6)
            s = str(v)
            s = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', s)
            return s

        return {
            "trace_id": result.trace_id,
            "complexity": result.complexity,
            "lawyer_referral": result.lawyer_referral,
            "lawyer_message": result.lawyer_message,
            "retrieved_cases": [
                {
                    "case_number": rc.case_number,
                    "title": rc.title,
                    "court": rc.court,
                    "cause_of_action": rc.cause_of_action,
                    "ruling_points": rc.ruling_points[:200],
                    "judgment_result": rc.judgment_result[:200] if rc.judgment_result else "",
                    "keywords": rc.keywords,
                    "score": rc.score,
                }
                for rc in result.retrieved_cases
            ],
            "conclusions": [
                {
                    "content": _clean(c.content),
                    "citations": [
                        {"type": cit.type, "id": _clean(cit.id), "text": _clean(cit.text[:80])}
                        for cit in c.citations
                    ]
                }
                for c in result.conclusions
            ]
        }

    def ask(self, query: str, top_k: int = 5) -> dict:
        """纯检索模式（不调用LLM）"""
        retrieved = self.retriever.retrieve(query, k=top_k)
        return {
            "query": query,
            "total": len(retrieved),
            "results": [
                {
                    "case_number": r["case"]["file_name"].split("/")[-1].replace(".txt", "").replace(".TXT", ""),
                    "title": r["case"]["cause"],
                    "cause_of_action": r["case"]["cause"],
                    "ruling_points": r["case"]["ruling_points"][:300] if r["case"]["ruling_points"] else "",
                    "judgment_result": r["case"]["judgment_result"][:200] if r["case"]["judgment_result"] else "",
                    "keywords": r["case"]["keywords"],
                    "score": round(r["score"], 3),
                }
                for r in retrieved
            ]
        }

# ============================================================
# FastAPI Web 服务
# ============================================================
def create_app(fayan: FaYanLegal):
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel

    app = FastAPI(title="法眼AI API", version="3.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 挂载静态文件（如果存在）
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.isdir(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    # 分析请求模型
    class AnalyzeRequest(BaseModel):
        case_text: str
        amount: float = 0
        party_count: int = 2
        has_evidence_gap: bool = False
        has_criminal_cross: bool = False

    # 检索请求模型
    class RetrieveRequest(BaseModel):
        query: str
        top_k: int = 5

    @app.get("/", response_class=HTMLResponse)
    def root():
        template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
        if os.path.exists(template_path):
            with open(template_path, "r", encoding="utf-8") as f:
                return f.read()
        return "<html><body><h1>法眼AI v3.0</h1><p>templates/index.html not found</p></body></html>"

    @app.get("/status")
    def status():
        return {
            "ok": True,
            "cases_loaded": len(fayan.retriever.cases),
            "model": LLM_MODEL,
        }

    @app.post("/api/analyze")
    def analyze(req: AnalyzeRequest):
        if not req.case_text.strip():
            raise HTTPException(status_code=400, detail="case_text 不能为空")
        if len(req.case_text) < 10:
            raise HTTPException(status_code=400, detail="案情描述过短")
        if len(req.case_text) > 5000:
            raise HTTPException(status_code=400, detail="案情描述过长，请控制在5000字以内")

        result = fayan.analyze(
            case_text=req.case_text,
            amount=req.amount,
            party_count=req.party_count,
            has_evidence_gap=req.has_evidence_gap,
            has_criminal_cross=req.has_criminal_cross,
        )
        return fayan.to_dict(result)

    @app.post("/api/retrieve")
    def retrieve(req: RetrieveRequest):
        if not req.query.strip():
            raise HTTPException(status_code=400, detail="query 不能为空")
        top_k = min(req.top_k, 10)
        return fayan.ask(req.query, top_k=top_k)

    @app.get("/health")
    def health():
        return "ok"

    # ============================================================
    # 案件自动分类接口（供前端智能识别使用）
    # ============================================================
    class ClassifyRequest(BaseModel):
        case_text: str

    @app.post("/api/classify")
    def classify(req: ClassifyRequest):
        if not req.case_text.strip():
            raise HTTPException(status_code=400, detail="case_text 不能为空")
        if len(req.case_text) < 5:
            raise HTTPException(status_code=400, detail="案情描述过短")
        case_type, type_confidence = CaseClassifier.classify(req.case_text)
        return {
            "case_type": case_type,
            "amount": None,
            "party_count": None,
            "amount_reason": "",
            "party_count_reason": "",
            "confidence": type_confidence,
        }

    return app

# ============================================================
# CLI 工具
# ============================================================
def run_cli(fayan: FaYanLegal):
    print("=" * 60)
    print("法眼AI 法律案件分析系统 v3.0（交互模式）")
    print("=" * 60)
    print(f"案例库: {len(fayan.retriever.cases)} 条")
    print("输入 exit 退出\n")

    while True:
        try:
            q = input("请输入法律问题> ").strip()
            if q.lower() in ("exit", "quit", "q"):
                break
            if not q:
                continue

            print("\n检索中...")
            retrieved = fayan.ask(q, top_k=3)

            print(f"\n找到 {retrieved['total']} 条相关案例：")
            for i, r in enumerate(retrieved["results"], 1):
                print(f"  [{i}] {r['case_number']} | {r['title']}")
                print(f"      判别标准: {r['ruling_points'][:80]}...")

            # 检查是否需要律师介入
            case_type, _ = CaseClassifier.classify(q)
            complexity = fayan.rule_engine.judge_complexity(q)

            if complexity in ["high", "ultra"]:
                print(f"\n⚠️  该案件复杂度: {complexity}，建议咨询专业律师")
            else:
                if MINIMAX_API_KEY:
                    print("\n生成分析中...")
                    result = fayan.analyze(q)
                    if result.lawyer_referral:
                        print(f"\n⚠️ {result.lawyer_message}")
                    else:
                        print("\n分析结论:")
                        for i, c in enumerate(result.conclusions, 1):
                            print(f"  {i}. {c.content}")
                        for cit in c.citations:
                            print(f"     引用: [{cit.type}] {cit.id}")
                else:
                    print("\n(MINIMAX_API_KEY 未设置，仅显示检索结果)")

            print()
        except EOFError:
            break

# ============================================================
# 入口
# ============================================================
def main():
    import sys

    if not MINIMAX_API_KEY:
        print("⚠️ 未设置 MINIMAX_API_KEY，部分功能受限")
        print("  export MINIMAX_API_KEY=your-key\n")

    fayan = FaYanLegal(
        api_key=MINIMAX_API_KEY,
        base_url=MINIMAX_BASE_URL,
        model=LLM_MODEL,
        csv_path=CASES_CSV
    )

    if len(sys.argv) < 2:
        run_cli(fayan)
        return

    cmd = sys.argv[1]

    if cmd == "server":
        import uvicorn
        port = int(os.environ.get("PORT", 5099))
        app = create_app(fayan)
        print(f"法眼AI 启动中... 访问 http://localhost:{port}")
        uvicorn.run(app, host="0.0.0.0", port=port)

    elif cmd == "ask":
        if len(sys.argv) < 3:
            print("用法: python fayan_main.py ask \"你的法律问题\"")
            return
        query = " ".join(sys.argv[2:])
        print(f"\n问题: {query}")

        print("\n检索中...")
        retrieved = fayan.ask(query, top_k=3)
        print(f"找到 {retrieved['total']} 条相关案例：")
        for i, r in enumerate(retrieved["results"], 1):
            print(f"  [{i}] {r['case_number']} | {r['title']}")
            print(f"      判别标准: {r['ruling_points'][:100]}...")

        if MINIMAX_API_KEY:
            print("\n生成分析中...")
            result = fayan.analyze(query)
            if result.lawyer_referral:
                print(f"\n⚠️ {result.lawyer_message}")
            else:
                print("\n分析结论:")
                for i, c in enumerate(result.conclusions, 1):
                    print(f"  {i}. {c.content}")
        else:
            print("\n(未设置 MINIMAX_API_KEY，仅显示检索结果)")

    else:
        print(f"未知命令: {cmd}")
        print("用法:")
        print("  python fayan_main.py         # 交互模式")
        print("  python fayan_main.py server  # 启动Web服务")
        print("  python fayan_main.py ask \"问题\"  # 单次问答")


if __name__ == "__main__":
    main()