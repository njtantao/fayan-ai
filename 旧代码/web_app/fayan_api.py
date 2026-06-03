"""
法眼AI - Web API 层
包装 FaYanLegal 核心，提供 REST 接口
"""

import os
import json
import re
import hashlib
import traceback
from dataclasses import dataclass, field, asdict
from typing import Optional
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# ============================================================
# 配置
# ============================================================
MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")
MINIMAX_BASE_URL = "https://api.minimax.chat/v1"
LLM_MODEL = "MiniMax-M2.7"

CASES_JSON = os.path.join(os.path.dirname(__file__), "..", "extracted_cases", "merged_cases.csv")
CIVIL_CASES_JSON = os.path.join(os.path.dirname(__file__), "..", "extracted_cases", "civil_cases.csv")
CRIMINAL_CASES_JSON = os.path.join(os.path.dirname(__file__), "..", "extracted_cases", "criminal_cases_merged.csv")

# ============================================================
# 数据结构
# ============================================================
@dataclass
class Citation:
    type: str
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
    case_number: str
    title: str
    court: str
    cause_of_action: str
    ruling_points: str
    related_laws: str
    score: float

@dataclass
class AnalysisResult:
    conclusions: list[LegalConclusion]
    complexity: str
    lawyer_referral: bool = False
    lawyer_message: str = ""
    raw_output: str = ""
    trace_id: str = ""
    retrieved_cases: list[RetrievedCase] = field(default_factory=list)

import pandas as pd

def _safe_meta(m):
    if isinstance(m, dict):
        return m
    if isinstance(m, str):
        try:
            return json.loads(m)
        except:
            return {}
    return {}

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

    def check(self, text: str):
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
                             has_fatal_gap: bool = False):
        if complexity in ["high", "ultra"] or has_fatal_gap:
            return True, (
                "基于目前风险评估，该案件已超出系统智能辅助范围，"
                "建议您考虑专业律师支持。如需了解公共法律服务资源，可通过12348法律服务热线获取帮助。"
            )
        return False, ""

# ============================================================
# RAG 检索（优化版：jieba + BM25 + TF-IDF + MMR）
# ============================================================
import jieba
import numpy as np

class LegalRetriever:
    def __init__(self, cases_json: str):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        from rank_bm25 import BM25Okapi
        from scipy import sparse

        import pandas as pd
        ext = os.path.splitext(cases_json)[1].lower()
        if ext == '.csv':
            df = pd.read_csv(cases_json)
            self.cases = df.to_dict('records')
        else:
            with open(cases_json, "r", encoding="utf-8") as f:
                self.cases = json.load(f)

        self._cases_loaded = len(self.cases)

        def build_search_text(case):
            try:
                mp_raw = case.get("metadata") or "{}"
                if isinstance(mp_raw, str):
                    mp = json.loads(mp_raw)
                elif isinstance(mp_raw, dict):
                    mp = mp_raw
                else:
                    mp = {}
            except (json.JSONDecodeError, TypeError, AttributeError):
                mp = {}
            title = case.get("title") if case.get("title") and not (isinstance(case.get("title"), float) and pd.isna(case.get("title"))) else ""
            content = case.get("content") if case.get("content") and not (isinstance(case.get("content"), float) and pd.isna(case.get("content"))) else ""
            parts = [
                str(title or "") * 3,
                " ".join(mp.get("keywords", [])) * 2 if isinstance(mp.get("keywords"), list) else str(mp.get("keywords") or "") * 2,
                str(mp.get("ruling_points") or "") * 2,
                str(mp.get("related_laws") or "") * 1,
                str(content) * 1,
            ]
            combined = " ".join(p.strip() for p in parts if p.strip() and p.lower() != "nan")
            combined = re.sub(r'[\s\n\r\t]+', ' ', combined).strip()
            return combined

        self.search_texts = [build_search_text(c) for c in self.cases]
        self.case_ids = [c.get("case_number", c.get("id", "")) for c in self.cases]

        # jieba 分词
        self.tokenized = [[t for t in jieba.cut(text) if len(t) > 1] for text in self.search_texts]

        # TF-IDF
        token_strs = [" ".join(tokens) for tokens in self.tokenized]
        self.vectorizer = TfidfVectorizer(max_features=2048, token_pattern=r'(?u)\b\w+\b')
        self.tfidf_matrix = self.vectorizer.fit_transform(token_strs)
        self.cosine_sim = cosine_similarity
        self.sparse = sparse

        # BM25
        self.bm25 = BM25Okapi(self.tokenized)
        self.bm25.param_b = 0.75
        self.bm25.param_k1 = 1.5

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        query_tokens = [t for t in jieba.cut(query) if len(t) > 1]
        query_str = " ".join(query_tokens)

        bm25_scores_raw = np.array(self.bm25.get_scores(query_tokens), dtype=float)
        query_vec = self.vectorizer.transform([query_str])
        tfidf_scores = self.cosine_sim(query_vec, self.tfidf_matrix).flatten()

        legal_terms = [t for t in query_tokens if len(t) >= 3]
        term_bonus = np.zeros(len(self.cases))
        if legal_terms:
            for idx, tokens in enumerate(self.tokenized):
                term_hits = sum(1 for t in legal_terms if t in tokens)
                term_bonus[idx] = term_hits / max(len(legal_terms), 1)

        def normalize(scores):
            mn, mx = scores.min(), scores.max()
            if mx <= mn:
                return np.zeros_like(scores)
            return (scores - mn) / (mx - mn)

        bm25_norm = normalize(bm25_scores_raw)
        tfidf_norm = normalize(tfidf_scores)
        bonus_norm = normalize(term_bonus)

        fused = 0.35 * bm25_norm + 0.45 * tfidf_norm + 0.20 * bonus_norm

        # MMR
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
                "term_score": float(bonus_norm[global_idx]),
                "source": "mmr_fused"
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
        for item in retrieved_cases:
            case = item["case"]
            cases_text.append(
                f"""[案例{case.get('case_number', case.get('id', ''))}] {case.get('title', 'N/A')}
法院: {case.get('court', 'N/A')} | 案由: {case.get('cause_of_action', 'N/A')}
裁判要旨: {_safe_meta(case.get('metadata', {})).get('ruling_points', 'N/A')[:300]}
关联法规: {_safe_meta(case.get('metadata', {})).get('related_laws', 'N/A')[:200]}
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
            # 尝试找 JSON 块
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
    def __init__(self, api_key: str, base_url: str, model: str, cases_json: str = None, civil_cases_json: str = None, criminal_cases_json: str = None):
        self.rule_engine = RuleEngine()
        self.llm = LegalLLM(api_key, base_url, model)
        self.trace_counter = 0

        # 支持单一或双数据库
        self.civil_retriever = None
        self.criminal_retriever = None
        self.single_retriever = None

        if civil_cases_json and criminal_cases_json:
            self.civil_retriever = LegalRetriever(civil_cases_json)
            self.criminal_retriever = LegalRetriever(criminal_cases_json)
            print(f"  民事案例库: {self.civil_retriever._cases_loaded} 条")
            print(f"  刑事案例库: {self.criminal_retriever._cases_loaded} 条")
        elif cases_json:
            self.single_retriever = LegalRetriever(cases_json)
            print(f"  案例库: {self.single_retriever._cases_loaded} 条")

    def _get_retriever(self, case_type: str = "民事"):
        if self.single_retriever is not None:
            return self.single_retriever
        if case_type == "刑事":
            return self.criminal_retriever
        return self.civil_retriever

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

        case_type_raw, _ = CaseClassifier._classify_type(case_text)
        if case_type_raw == "刑事":
            target_db = "刑事"
        elif case_type_raw == "刑民交叉" and has_criminal_cross:
            target_db = "刑事"
        else:
            target_db = "民事"

        # Step 1: 复杂度判定
        complexity = self.rule_engine.judge_complexity(
            case_text, amount, party_count, has_evidence_gap, has_criminal_cross
        )

        # Step 2: 律师介入
        lawyer_referral, lawyer_msg = self.rule_engine.should_refer_lawyer(
            complexity, has_evidence_gap
        )
        if lawyer_referral:
            return AnalysisResult([], complexity, True, lawyer_msg, "", trace_id, [])

        # Step 3: RAG 检索（根据案件类型选择数据库）
        retriever = self._get_retriever(target_db)
        retrieved = retriever.retrieve(case_text, k=5)

        # Step 4: LLM 生成
        raw = self.llm.call(case_text, retrieved)

        # Step 5: 解析
        parsed = self.llm.parse_response(raw)

        # Step 6: 规则校验
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

        # 包装检索到的案例（给前端展示用）
        retrieved_cases_info = [
            RetrievedCase(
                case_number=r["case"].get("case_number", r["case"].get("id", "")),
                title=r["case"].get("title", ""),
                court=r["case"].get("court", "N/A"),
                cause_of_action=r["case"].get("cause_of_action", "N/A"),
                ruling_points=_safe_meta(r["case"].get("metadata", {})).get("ruling_points", "")[:200],
                related_laws=_safe_meta(r["case"].get("metadata", {})).get("related_laws", "")[:200],
                score=round(r["score"], 3)
            ) for r in retrieved
        ]

        if not conclusions:
            return AnalysisResult(
                [], complexity, False,
                "依据现有材料与知识库，无法形成具有依据的分析结论。建议您咨询专业律师。",
                raw, trace_id, retrieved_cases_info
            )

        return AnalysisResult(conclusions, complexity, False, "", raw, trace_id, retrieved_cases_info)

    def to_dict(self, result: AnalysisResult) -> dict:
        def _clean(v):
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return None
            if isinstance(v, float):
                return round(v, 6)
            s = str(v)
            s = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', s)
            return s
        def _clean_case(rc):
            d = asdict(rc)
            for k, v in d.items():
                if isinstance(v, float) and pd.isna(v):
                    d[k] = None
                elif isinstance(v, str) and v.lower() == 'nan':
                    d[k] = None
            return d
        return {
            "trace_id": result.trace_id,
            "complexity": result.complexity,
            "lawyer_referral": result.lawyer_referral,
            "lawyer_message": result.lawyer_message,
            "retrieved_cases": [_clean_case(rc) for rc in result.retrieved_cases],
            "conclusions": [
                {
                    "content": _clean(c.content),
                    "citations": [{"type": cit.type, "id": _clean(cit.id), "text": _clean(cit.text[:80])} for cit in c.citations]
                } for c in result.conclusions
            ]
        }


# ============================================================
# 案件自动分类器（规则 + LLM 辅助）
# ============================================================
class CaseClassifier:
    """根据案情文本自动判断案件类型、涉案金额、当事人数"""

    # 刑事关键词（优先）
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
    ]

    # 刑民交叉关键词
    CROSS_KEYWORDS = [
        "先刑后民", "刑民交叉", "刑事附带民事", "刑事与民事",
        "合同诈骗罪", "诈骗罪", "非法吸收公众存款罪",
        "职务侵占罪", "挪用资金罪", "合同欺诈",
        "涉嫌犯罪", "刑事立案", "刑事责任",
    ]

    # 金额提取正则
    AMOUNT_PATTERNS = [
        (r"(\d[\d,，.]*)\s*(?:万|万元)", 10000),
        (r"(\d[\d,，.]*)\s*(?:千|千元)", 1000),
        (r"(\d[\d,，.]*)\s*(?:亿|亿元)", 100000000),
        (r"人民币\s*(\d[\d,，.]+)", 1),
        (r"¥\s*(\d[\d,，.]+)", 1),
        (r"\$\s*(\d[\d,，.]+)", 1),
    ]

    # 当事人数量（人名出现次数估算）
    PARTY_PATTERNS = [
        (r"甲[某又]?(?:某)?(?:某)?(?:等)?", 1),
        (r"乙[某又]?(?:某)?(?:某)?(?:等)?", 1),
        (r"丙[某又]?(?:某)?(?:等)?", 1),
        (r"被告人[^，。,]{0,6}?(?:某|\\S)[^，。,]{0,4}", 1),
        (r"犯罪嫌疑人", 1),
        (r"原告[^，。,]{0,10}?(?:称|诉|称道)", 1),
        (r"被告[^，。,]{0,10}?(?:称|辩|应诉)", 1),
        (r"当事人[^，。,]{0,8}?(?:各方|双方|多方|三方|多方)", 1),
    ]

    @classmethod
    def classify(cls, text: str) -> dict:
        """
        返回结构: {
            case_type: "民事" | "刑事" | "刑民交叉",
            amount: float | None,
            party_count: int | None,
            amount_reason: str,
            party_count_reason: str,
            confidence: float (0-1),
        }
        """
        case_type, type_confidence = cls._classify_type(text)
        amount, amount_reason = cls._extract_amount(text)
        party_count, party_reason = cls._extract_party_count(text)
        confidence = round((type_confidence + amount_confidence(amount) + party_confidence(party_count)) / 3, 2)

        return {
            "case_type": case_type,
            "amount": amount,
            "party_count": party_count,
            "amount_reason": amount_reason,
            "party_count_reason": party_reason,
            "confidence": confidence,
        }

    @classmethod
    def _classify_type(cls, text: str) -> tuple:
        """判断案件类型"""
        criminal_score = sum(1 for kw in cls.CRIMINAL_KEYWORDS if kw in text)
        cross_score = sum(1 for kw in cls.CROSS_KEYWORDS if kw in text)

        # 刑民交叉优先
        if cross_score >= 1 or (criminal_score >= 1 and cross_score >= 1):
            return "刑民交叉", 0.85
        if criminal_score >= 2:
            return "刑事", 0.9
        if criminal_score == 1:
            return "刑事", 0.75
        return "民事", 0.8

    @classmethod
    def _extract_amount(cls, text: str) -> tuple:
        """提取涉案金额"""
        candidates = []
        for pattern, multiplier in cls.AMOUNT_PATTERNS:
            matches = re.findall(pattern, text)
            for m in matches:
                try:
                    num_str = m.replace(",", "").replace("，", "").replace(".", "")
                    val = float(num_str) * multiplier
                    if val > 0:
                        candidates.append(val)
                except ValueError:
                    pass

        if not candidates:
            return None, "未识别到具体金额"
        # 取最大金额（通常是最核心的涉案金额）
        amount = max(candidates)
        if amount >= 100000000:
            return amount, f"识别金额约 {amount/100000000:.1f} 亿元"
        if amount >= 10000:
            return amount, f"识别金额约 {amount/10000:.1f} 万元"
        return amount, f"识别金额约 {amount:.0f} 元"

    @classmethod
    def _extract_party_count(cls, text: str) -> tuple:
        """估算当事人数"""
        parties = set()

        # 提取"甲、乙、丙、丁"等标记
        party_letters = re.findall(r"[甲乙丙丁戊己庚辛壬癸](?:某|[A-Za-z0-9])?(?:等)?", text)
        parties.update(party_letters)

        # 提取"原告XXX"、"被告XXX"、"第三人XXX"
        for role in ["原告", "被告", "第三人", "上诉人", "被上诉人", "申请人", "被申请人"]:
            matches = re.findall(f"{role}[^，。,，、：:]{1,8}", text)
            parties.update(matches)

        # 提取"被告人XXX"
        matches = re.findall(r"被告人[^，。,，、：:]{1,8}", text)
        parties.update(matches)

        # 提取"犯罪嫌疑人XXX"
        matches = re.findall(r"犯罪嫌疑人[^，。,，、：:]{1,8}", text)
        parties.update(matches)

        # 提取"当事人"
        parties.add("当事人")

        # 提取"双方"、"各方"、"多方"、"三方"
        multi = re.findall(r"[各多三]方", text)
        parties.update(multi)

        count = len(parties)
        # 结合明确的人数表述
        explicit = re.findall(r"(\d+)\s*(?:名|人|位|个)\s*(?:当事人|被告人|犯罪嫌疑人|原告|被告|当事人)", text)
        if explicit:
            try:
                explicit_count = int(explicit[0])
                if explicit_count > count:
                    count = explicit_count
            except ValueError:
                pass

        if count == 0:
            return None, "无法识别当事人数"
        if count > 20:
            count = 20  # 上限
        return count, f"识别到约 {count} 方当事人"

    @classmethod
    def to_dict(cls, result: dict) -> dict:
        return result


def amount_confidence(amount) -> float:
    if amount is None: return 0.3
    if amount > 0: return 0.8
    return 0.3

def party_confidence(count) -> float:
    if count is None: return 0.3
    if count >= 2: return 0.8
    return 0.5
