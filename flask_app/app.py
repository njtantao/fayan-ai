#!/usr/bin/env python3
"""
法眼AI - Flask Web 服务
启动: python app.py
访问: http://localhost:5099
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()

# Environment variables (set in Render dashboard or .env locally)
MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "all_cases_perfect.csv")
PORT = int(os.environ.get("PORT", 5099))
DATA_URL = os.environ.get("DATA_URL", "https://github.com/njtantao/fayan-ai/releases/download/v1.0.0/all_cases_perfect.csv")  # Optional: download case CSV at startup

# 导入核心模块（从项目根目录导入）
from fayan_main import FaYanLegal, MINIMAX_BASE_URL, LLM_MODEL

# ============================================================
# Flask App
# ============================================================
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# 全局实例（启动时初始化一次）
_fayan: FaYanLegal = None
_init_error: str = None

def get_fayan():
    global _fayan, _init_error
    if _fayan is None:
        # 启动时：如果CSV不存在但有DATA_URL，自动下载
        if DATA_URL and not os.path.exists(CSV_PATH):
            print(f"案例库不存在，从 {DATA_URL} 下载...")
            try:
                import urllib.request
                os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
                urllib.request.urlretrieve(DATA_URL, CSV_PATH + ".tmp")
                os.rename(CSV_PATH + ".tmp", CSV_PATH)
                print(f"案例库下载完成: {CSV_PATH}")
            except Exception as e:
                print(f"下载案例库失败: {e}")

        api_key = os.environ.get("MINIMAX_API_KEY", "")
        if not api_key:
            _init_error = "MINIMAX_API_KEY 未设置或无效"
            return None
        try:
            _fayan = FaYanLegal(
                api_key=api_key,
                base_url=MINIMAX_BASE_URL,
                model=LLM_MODEL,
                csv_path=CSV_PATH
            )
            _init_error = None
            print(f"法眼AI 加载案例库: {_fayan.retriever.cases.__len__()} 条")
        except Exception as e:
            _init_error = str(e)
            import traceback; traceback.print_exc()
            return None
    return _fayan

# ============================================================
# 页面路由
# ============================================================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/status")
def status():
    fayan = get_fayan()
    return jsonify({
        "ok": fayan is not None,
        "error": _init_error,
        "cases_loaded": len(fayan.retriever.cases) if fayan else 0,
        "model": LLM_MODEL,
    })

# ============================================================
# 分析 API
# ============================================================
@app.route("/api/analyze", methods=["POST"])
def analyze():
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
        import traceback; traceback.print_exc()
        return jsonify({"error": f"分析失败: {str(e)}"}), 500

# ============================================================
# 检索 API（仅 RAG，不调用 LLM）
# ============================================================
@app.route("/api/retrieve", methods=["POST"])
def retrieve():
    if _init_error:
        return jsonify({"error": _init_error}), 500

    data = request.get_json()
    query = data.get("query", "").strip()
    top_k = min(int(data.get("top_k", 5) or 5), 10)

    if not query:
        return jsonify({"error": "query 不能为空"}), 400

    try:
        fayan = get_fayan()
        return jsonify(fayan.ask(query, top_k=top_k))
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ============================================================
# 案件自动分类接口
# ============================================================
@app.route("/api/classify", methods=["POST"])
def classify():
    data = request.get_json()
    if not data:
        return jsonify({"error": "请求体为空"}), 400
    case_text = data.get("case_text", "").strip()
    if len(case_text) < 5:
        return jsonify({"error": "案情描述过短"}), 400

    from fayan_main import CaseClassifier
    try:
        case_type, type_confidence = CaseClassifier.classify(case_text)
        amount, amount_reason = CaseClassifier.extract_amount(case_text)
        party_count, party_reason = CaseClassifier.extract_party_count(case_text)
        domain = CaseClassifier.extract_domain(case_text)

        # 复杂度粗估
        score = 0
        if amount > 2000000: score += 2
        elif amount > 500000: score += 1
        if party_count > 10: score += 2
        elif party_count > 5: score += 1
        if case_type == '刑民交叉': score += 3
        complexity = 'ultra' if score >= 5 else 'high' if score >= 3 else 'medium' if score >= 1 else 'low'

        return jsonify({
            "case_type": case_type,
            "amount": amount if amount > 0 else None,
            "amount_reason": amount_reason,
            "party_count": party_count if party_count > 0 else None,
            "party_count_reason": party_reason,
            "complexity": complexity,
            "domain": domain,
            "confidence": type_confidence,
            "has_evidence_gap": False,
        })
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
    print(f"法眼AI 启动中...")
    print(f"API Key: {'已设置 ✓' if MINIMAX_API_KEY else '未设置 ✗'}")
    print(f"案例库: {CSV_PATH}")
    print(f"访问地址: http://localhost:{PORT}")
    print(f"按 Ctrl+C 停止服务")
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)