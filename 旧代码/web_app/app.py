"""
法眼AI - Flask Web 服务
启动: python app.py
访问: http://localhost:5000
"""

import os
import json
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS

# 导入核心模块
from fayan_api import FaYanLegal, MINIMAX_BASE_URL, LLM_MODEL, CASES_JSON, CIVIL_CASES_JSON, CRIMINAL_CASES_JSON

# ============================================================
# Flask App
# ============================================================
app = Flask(__name__,
            template_folder="templates",
            static_folder="static")
CORS(app)

# 全局实例（启动时初始化一次）
_fayan: FaYanLegal = None
_init_error: str = None

def get_fayan():
    global _fayan, _init_error
    if _fayan is None:
        api_key = os.environ.get("MINIMAX_API_KEY", "")
        if not api_key or api_key == "your-api-key":
            _init_error = "MINIMAX_API_KEY 未设置或无效"
            return None
        try:
            _fayan = FaYanLegal(
                api_key=api_key,
                base_url=MINIMAX_BASE_URL,
                model=LLM_MODEL,
                cases_json=None,
                civil_cases_json=CIVIL_CASES_JSON,
                criminal_cases_json=CRIMINAL_CASES_JSON
            )
            _init_error = None
        except Exception as e:
            _init_error = str(e)
            return None
    return _fayan

# ============================================================
# 页面路由
# ============================================================
@app.route("/")
def index():
    """主页"""
    return render_template("index.html")

@app.route("/status")
def status():
    """服务状态检查"""
    fayan = get_fayan()
    civil = fayan.civil_retriever._cases_loaded if fayan and fayan.civil_retriever else 0
    criminal = fayan.criminal_retriever._cases_loaded if fayan and fayan.criminal_retriever else 0
    total = fayan.single_retriever._cases_loaded if fayan and fayan.single_retriever else 0
    return jsonify({
        "ok": fayan is None if _init_error else (fayan is not None),
        "error": _init_error,
        "civil_count": civil,
        "criminal_count": criminal,
        "total_count": civil + criminal or total,
        "model": LLM_MODEL,
    })

# ============================================================
# 分析 API
# ============================================================
@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    POST /api/analyze
    Body (JSON):
    {
        "case_text": "案情描述",
        "amount": 500000,          // 可选，涉案金额（元）
        "party_count": 3,           // 可选，当事人数
        "has_evidence_gap": false,  // 可选，证据是否有缺口
        "has_criminal_cross": false // 可选，是否刑民交叉
    }
    """
    if _init_error:
        return jsonify({"error": _init_error}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "请求体为空，需要 JSON"}), 400

    case_text = data.get("case_text", "").strip()
    if not case_text:
        return jsonify({"error": "case_text 不能为空"}), 400
    if len(case_text) < 10:
        return jsonify({"error": "案情描述过短，请提供更完整的描述"}), 400
    if len(case_text) > 5000:
        return jsonify({"error": "案情描述过长，请控制在5000字以内"}), 400

    try:
        result = get_fayan().analyze(
            case_text=case_text,
            amount=float(data.get("amount", 0) or 0),
            party_count=int(data.get("party_count", 2) or 2),
            has_evidence_gap=bool(data.get("has_evidence_gap", False)),
            has_criminal_cross=bool(data.get("has_criminal_cross", False)),
        )
        return jsonify(get_fayan().to_dict(result))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"分析失败: {str(e)}"}), 500

# ============================================================
# 检索 API（仅 RAG，不调用 LLM）
# ============================================================
@app.route("/api/retrieve", methods=["POST"])
def retrieve():
    """
    POST /api/retrieve
    Body: { "query": "案情关键词", "top_k": 5 }
    """
    if _init_error:
        return jsonify({"error": _init_error}), 500

    data = request.get_json()
    query = data.get("query", "").strip()
    top_k = min(int(data.get("top_k", 5) or 5), 10)

    if not query:
        return jsonify({"error": "query 不能为空"}), 400

    try:
        fayan = get_fayan()
        results = fayan.retriever.retrieve(query, k=top_k)
        return jsonify({
            "query": query,
            "total": len(results),
            "results": [
                {
                    "case_number": r["case"].get("case_number", r["case"].get("id", "")),
                    "title": r["case"].get("title", ""),
                    "court": r["case"].get("court", "N/A"),
                    "cause_of_action": r["case"].get("cause_of_action", "N/A"),
                    "ruling_points": r["case"].get("metadata", {}).get("ruling_points", "")[:300],
                    "score": round(r["score"], 3),
                }
                for r in results
            ]
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ============================================================
# 案件自动分类接口
# ============================================================
@app.route("/api/classify", methods=["POST"])
def classify():
    """
    POST /api/classify
    根据案情文本自动判断：案件类型（民事/刑事/刑民交叉）、涉案金额、当事人数
    Body: { "case_text": "案情描述" }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "请求体为空"}), 400
    case_text = data.get("case_text", "").strip()
    if len(case_text) < 5:
        return jsonify({"error": "案情描述过短"}), 400

    from fayan_api import CaseClassifier
    try:
        result = CaseClassifier.classify(case_text)
        return jsonify(result)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ============================================================
# 健康检查
# ============================================================
@app.route("/health")
def health():
    return Response("ok", mimetype="text/plain")

# ============================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"法眼AI 启动中...")
    print(f"API Key: {'已设置 ✓' if os.environ.get('MINIMAX_API_KEY') else '未设置 ✗'}")
    print(f"案例库: {CASES_JSON}")
    print(f"访问地址: http://localhost:{port}")
    print(f"按 Ctrl+C 停止服务")

    app.run(host="0.0.0.0", port=port, debug=False)
