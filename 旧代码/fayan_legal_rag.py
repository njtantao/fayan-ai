"""
法眼AI - 完整RAG法律问答系统
集成：执行案例库(356条) + Minimax LLM + 规则引擎校验
用法: python fayan_legal_rag.py
"""

import os
import json
import re
import hashlib
import sqlite3
from dataclasses import dataclass, field
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============================================================
# 配置
# ============================================================
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "your-api-key")
MINIMAX_BASE_URL = "https://api.minimax.chat/v1"
LLM_MODEL = "abab6.5s-chat"  # 修改为你的模型名

CASES_JSON = "./extracted_cases/cases.json"
CASE_DB = "./extracted_cases/cases.db"

# ============================================================
# 数据结构
# ============================================================
@dataclass
class Citation:
    type: str   # "statute" | "case"
    id: str
    text: str

@dataclass
class LegalConclusion:
    content: str
    citations: list[Citation] = field(default_factory=list)
    has_forbidden: bool = False
    is_valid: bool = True

@dataclass
class AnalysisResult:
    conclusions: list[LegalConclusion]
    complexity: str
    lawyer_referral: bool = False
    lawyer_message: str = ""
    raw_output: str = ""
    trace_id: str = ""

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
# RAG检索（优化版：jieba分词 + 字段加权 + MMR多样性重排）
# ============================================================
import jieba
import numpy as np

class LegalRetriever:
    def __init__(self, cases_json: str):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        from rank_bm25 import BM25Okapi

        with open(cases_json, "r", encoding="utf-8") as f:
            self.cases = json.load(f)

        # ---- 1. 丰富每个案例的搜索文本（字段加权）----
        # title ×3, keywords ×2, ruling_points ×2, basic_facts ×1
        def build_search_text(case):
            mp = case.get("metadata", {})
            parts = []
            parts.append(case.get("title", "") * 3)
            parts.append(" ".join(mp.get("keywords", [])) * 2)
            parts.append(mp.get("ruling_points", "") * 2)
            parts.append(mp.get("related_laws", "") * 1)
            # content 是最完整的文本
            parts.append(case.get("content", "") * 1)
            combined = " ".join(p.strip() for p in parts if p.strip())
            combined = re.sub(r'[\s\n\r\t]+', ' ', combined).strip()
            return combined

        self.search_texts = [build_search_text(c) for c in self.cases]
        self.case_ids = [c["id"] for c in self.cases]

        # ---- 2. jieba 分词后的 token 列表（用于 BM25 和 TF-IDF）----
        self.tokenized = [list(jieba.cut(text)) for text in self.search_texts]
        # 去掉单字符词（BM25 噪音过滤）
        self.tokenized = [[t for t in tokens if len(t) > 1] for tokens in self.tokenized]

        # ---- 3. TF-IDF（jieba 分词）----
        # 用空格连接 jieba 分词结果，模拟词袋输入
        token_strs = [" ".join(tokens) for tokens in self.tokenized]
        self.vectorizer = TfidfVectorizer(max_features=2048, token_pattern=r'(?u)\b\w+\b')
        self.tfidf_matrix = self.vectorizer.fit_transform(token_strs)
        self.cosine_sim = cosine_similarity

        # ---- 4. BM25（jieba 分词）----
        self.bm25 = BM25Okapi(self.tokenized)

        # BM25 参数调优（中文法律文本）
        self.bm25.param_b = 0.75  # 文档长度归一化
        self.bm25.param_k1 = 1.5  # 词频饱和度

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        """
        三路检索 + 字段加权 + MMR 多样性重排：
        1. BM25 (jieba) — 关键词精确匹配
        2. TF-IDF (jieba) — 语义相关度
        3. 精确法律术语匹配（bonus）— 法律概念命中加分
        """
        import numpy as np

        # jieba 分词查询
        query_tokens = [t for t in jieba.cut(query) if len(t) > 1]
        query_str = " ".join(query_tokens)

        # ---- BM25 ----
        bm25_scores_raw = np.array(self.bm25.get_scores(query_tokens), dtype=float)

        # ---- TF-IDF ----
        query_vec = self.vectorizer.transform([query_str])
        tfidf_scores = self.cosine_sim(query_vec, self.tfidf_matrix).flatten()

        # ---- 精确法律术语 bonus ----
        # 识别法律术语（3字以上词）作为精确匹配加分
        legal_terms = [t for t in query_tokens if len(t) >= 3]
        term_bonus = np.zeros(len(self.cases))
        if legal_terms:
            for idx, tokens in enumerate(self.tokenized):
                term_hits = sum(1 for t in legal_terms if t in tokens)
                term_bonus[idx] = term_hits / max(len(legal_terms), 1)

        # ---- 归一化 ----
        def normalize(scores):
            mn, mx = scores.min(), scores.max()
            if mx <= mn:
                return np.zeros_like(scores)
            return (scores - mn) / (mx - mn)

        bm25_norm   = normalize(bm25_scores_raw)
        tfidf_norm  = normalize(tfidf_scores)
        bonus_norm  = normalize(term_bonus)

        # ---- 加权融合 ----
        # BM25 关键词权重高（法律术语精确匹配），TF-IDF 负责语义泛化
        fused = 0.35 * bm25_norm + 0.45 * tfidf_norm + 0.20 * bonus_norm

        # ---- MMR 多样性重排 ----
        from scipy import sparse

        top_candidates = np.argsort(fused)[::-1][:k * 3]  # 全局索引
        selected_vectors = []
        mmr_ranked = []

        # 预计算候选的密集向量（归一化，用于快速余弦相似度）
        cand_dense = self.tfidf_matrix[top_candidates].toarray().astype(float)
        # 行归一化：余弦相似度 = 点积
        norms = np.linalg.norm(cand_dense, axis=1, keepdims=True)
        norms[norms == 0] = 1
        cand_dense_norm = cand_dense / norms

        for local_rank, global_idx in enumerate(top_candidates):
            if len(mmr_ranked) >= k:
                break
            case_vec_norm = cand_dense_norm[local_rank]
            if mmr_ranked:
                # 最大相似度（余弦）
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
                "case":       case,
                "score":      float(mmr_score),
                "bm25_score": float(bm25_norm[global_idx]),
                "tfidf_score":float(tfidf_norm[global_idx]),
                "term_score": float(bonus_norm[global_idx]),
                "source":     "mmr_fused"
            })

        return results

# ============================================================
# LLM 调用
# ============================================================
class LegalLLM:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.llm = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=0.2,
            request_timeout=60
        )

    def build_prompt(self, case_text: str, retrieved_cases: list[dict]) -> str:
        # 格式化检索结果
        cases_text = []
        for item in retrieved_cases:
            case = item["case"]
            cases_text.append(
                f"""[案例{case['case_number']}] {case['title']}
法院: {case.get('court', 'N/A')} | 案由: {case.get('cause_of_action', 'N/A')}
裁判要旨: {case.get('metadata', {}).get('ruling_points', 'N/A')[:200]}
关联法规: {case.get('metadata', {}).get('related_laws', 'N/A')[:150]}
"""
            )
        cases_block = "\n".join(cases_text)

        prompt = (
            "你是一个法律分析助手，基于以下类案参考进行分析。\n"
            "\n"
            "【硬性规则 - 违反直接终止输出】\n"
            "1. 只允许引用下方「类案参考」中列出的案例，禁止自行编造、推测、拼接案例编号\n"
            "2. 每条结论的 citations 数组中，id 字段必须严格等于下面某个案例的 case_number\n"
            "3. 禁止使用判断性表达（胜诉率、一定赢、会赢、会输、应当赔偿等）\n"
            "4. 如果类案参考中没有与用户案情相关的案例，结论 content 必须以「现有类案参考不足」开头\n"
            "5. 只输出结构化JSON，不要输出其他内容\n"
            "\n"
            "---\n"
            "【类案参考】（共 " + str(len(retrieved_cases)) + " 个案例，全部真实存在）\n"
            "\n"
            + cases_block +
            "\n"
            "---\n"
            "【用户案情】\n"
            + case_text +
            "\n"
            "---\n"
            "【输出格式（严格按照此JSON结构，不要添加任何额外字段）】\n"
            "{\n"
            '  "conclusions": [\n'
            '    {\n'
            '      "content": "结论内容（引用来源时必须附上 case_number）",\n'
            '      "citations": [{"type": "case", "id": "必须填入类案参考中的 case_number", "text": "引用原文片段"}]\n'
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
            return {"conclusions": [], "summary": raw}

# ============================================================
# 主系统
# ============================================================
class FaYanLegal:
    def __init__(self, api_key: str, base_url: str, model: str,
                 cases_json: str):
        self.rule_engine = RuleEngine()
        self.llm = LegalLLM(api_key, base_url, model)
        self.retriever = LegalRetriever(cases_json)
        self.trace_counter = 0

    def analyze(self, case_text: str,
                amount: float = 0,
                party_count: int = 2,
                has_evidence_gap: bool = False,
                has_criminal_cross: bool = False) -> AnalysisResult:
        self.trace_counter += 1
        trace_id = hashlib.md5(
            f"{case_text}{self.trace_counter}".encode()
        ).hexdigest()[:12]

        # Step 1: 复杂度判定
        complexity = self.rule_engine.judge_complexity(
            case_text, amount, party_count, has_evidence_gap, has_criminal_cross
        )

        # Step 2: 律师介入检查
        lawyer_referral, lawyer_msg = self.rule_engine.should_refer_lawyer(
            complexity, has_evidence_gap
        )
        if lawyer_referral:
            return AnalysisResult([], complexity, True, lawyer_msg, "", trace_id)

        # Step 3: RAG检索
        retrieved = self.retriever.retrieve(case_text, k=5)

        # Step 4: LLM生成
        raw = self.llm.call(case_text, retrieved)

        # Step 5: 解析响应
        parsed = self.llm.parse_response(raw)

        # Step 6: 规则校验
        conclusions = []
        for conc in parsed.get("conclusions", []):
            conclusion = LegalConclusion(
                content=conc.get("content", ""),
                citations=[
                    Citation(
                        type=c.get("type", "case"),
                        id=c.get("id", ""),
                        text=c.get("text", "")
                    ) for c in conc.get("citations", [])
                ]
            )

            has_forbidden, _ = self.rule_engine.check(conclusion.content)
            if has_forbidden:
                conclusion.has_forbidden = True
                conclusion.is_valid = False
                continue

            if conclusion.content:
                conclusions.append(conclusion)

        if not conclusions:
            return AnalysisResult(
                [], complexity, False,
                "依据现有材料与知识库，无法形成具有依据的分析结论。建议您咨询专业律师。",
                raw, trace_id
            )

        return AnalysisResult(conclusions, complexity, False, "", raw, trace_id)

    def to_json(self, result: AnalysisResult) -> dict:
        return {
            "trace_id": result.trace_id,
            "complexity": result.complexity,
            "lawyer_referral": result.lawyer_referral,
            "lawyer_message": result.lawyer_message,
            "conclusions": [
                {
                    "content": c.content,
                    "citations": [
                        {"type": cit.type, "id": cit.id, "text": cit.text[:50]}
                        for cit in c.citations
                    ]
                }
                for c in result.conclusions
            ]
        }

# ============================================================
# 示例
# ============================================================
if __name__ == "__main__":
    # 初始化
    fayan = FaYanLegal(
        api_key=MINIMAX_API_KEY,
        base_url=MINIMAX_BASE_URL,
        model=LLM_MODEL,
        cases_json=CASES_JSON
    )

    # 测试RAG检索（不调用LLM）
    print("=" * 60)
    print("RAG检索测试")
    print("=" * 60)

    test_queries = [
        "执行异议中，如何认定案外人借用被执行人银行账户的资金所有权？",
        "网络司法拍卖中房屋存在非正常死亡事件是否需要披露？",
        "被执行人账户资金混同，能否主张专款专用排除执行？",
    ]

    for q in test_queries:
        print(f"\n查询: {q}")
        results = fayan.retriever.retrieve(q, k=2)
        for r in results:
            case = r["case"]
            print(f"  [{case['case_number']}] {case['title']}")
            print(f"    案由: {case.get('cause_of_action')} | 法院: {case.get('court', 'N/A')}")
            print(f"    要旨: {case.get('metadata', {}).get('ruling_points', 'N/A')[:100]}...")

    # 完整问答示例（需要真实API Key）
    print("\n" + "=" * 60)
    print("完整问答测试（需要MINIMAX_API_KEY）")
    print("=" * 60)

    case_text = (
        "甲借用乙的银行账户收取经营款项，后乙涉及债务纠纷被强制执行，"
        "甲主张账户内资金属于自己所有，请求排除强制执行，是否支持？"
    )

    if MINIMAX_API_KEY and MINIMAX_API_KEY != "your-api-key":
        result = fayan.analyze(
            case_text=case_text,
            amount=500000,
            party_count=3,
            has_evidence_gap=True,
        )

        print(f"Trace ID: {result.trace_id}")
        print(f"复杂度: {result.complexity}")
        print(f"律师介入: {'是' if result.lawyer_referral else '否'}")
        if result.lawyer_message:
            print(f"提示: {result.lawyer_message}")
        print(f"\n分析结论:")
        for i, c in enumerate(result.conclusions, 1):
            print(f"  {i}. {c.content}")
            for cit in c.citations:
                print(f"     引用: [{cit.type}] {cit.id}")
        print(f"\nJSON输出:")
        print(json.dumps(fayan.to_json(result), ensure_ascii=False, indent=2))
    else:
        print("未设置MINIMAX_API_KEY，跳过LLM调用")
        print(f"\n查询语句: {case_text}")
        print("设置 export MINIMAX_API_KEY=your-key 后可运行完整测试")
