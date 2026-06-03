#!/usr/bin/env python3
"""
法眼AI - 轻量级RAG问答系统 v2
民事/刑事分类检索 + BM25 + MiniMax LLM
"""

import json, os, sys, time
import re
import jieba  # 预加载，避免重复构建词典
from rank_bm25 import BM25Okapi
jieba.initialize()  # 预热缓存

# ========== 配置 ==========
MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY",
    "sk-cp-ky-cbLKQr5w4-KHrV9wKzxU-hTtYhNqZyyB6Q8If3N2YwMipNx4WsYzWszA7jzpAcUuCEERm_l98RaLSfZPCmpzVxOy8256mh07SP6qkeJMbUlZQcwrnIyg")
MINIMAX_BASE_URL = "https://api.minimax.chat/v1"
MODEL = "MiniMax-M2.7"
TOP_K = 3
MAX_CONTEXT_CHARS = 5000  # 控制输入长度

# ========== 加载案例（分类） ==========
def load_cases():
    """加载案例并按民事/刑事分类"""
    base = "/Users/tt/Desktop/hermes/项目开发/法眼ai/extracted_cases"
    civil_cases = []
    criminal_cases = []

    civil_path = os.path.join(base, "all_cases.json")
    if os.path.exists(civil_path):
        cases = json.load(open(civil_path, encoding="utf-8"))
        print(f"加载民事案例: {len(cases)} 条")
        civil_cases.extend(cases)

    criminal_path = os.path.join(base, "criminal_cases.json")
    if os.path.exists(criminal_path):
        cases = json.load(open(criminal_path, encoding="utf-8"))
        print(f"加载刑事案例: {len(cases)} 条")
        criminal_cases.extend(cases)

    print(f"总计: {len(civil_cases) + len(criminal_cases)} 条案例")
    return civil_cases, criminal_cases


def simple_tokenizer(text):
    """中文分词器——使用jieba（已全局预加载）"""
    text = re.sub(r'[^\w\u4e00-\u9fff]', ' ', text)
    return list(jieba.cut(text.strip()))


def build_index(cases):
    """构建BM25索引"""
    corpus_texts = []
    for case in cases:
        search_text = f"{case.get('title','')} {case.get('cause_of_action','')} {case.get('content','')[:500]}"
        corpus_texts.append(search_text)
    return BM25Okapi(corpus_texts, tokenizer=simple_tokenizer)


# ========== 强制刑事关键词 ==========
CRIMINAL_KEYWORDS = [
    "罪", "盗窃", "抢劫", "杀人", "伤害", "诈骗", "强奸", "猥亵",
    "走私", "贩毒", "吸毒", "赌博", "开设赌场", "组织卖淫",
    "贪污", "贿赂", "受贿", "行贿", "挪用", "滥用职权", "玩忽职守",
    "盗窃罪", "抢劫罪", "故意杀人", "过失致人死亡", "交通肇事",
    "非法拘禁", "绑架", "敲诈勒索", "抢夺", "侵占", "职务侵占",
    "寻衅滋事", "聚众斗殴", "黑社会", "组织领导参加黑社会",
    "毒品", "贩卖毒品", "制造毒品", "持有毒品", "容留他人吸毒",
    "偷税", "逃税", "骗税", "虚开", "非法经营", "生产销售伪劣产品",
    "拐卖", "收买被拐卖", "绑架儿童", "拐骗儿童",
    "重婚", "遗弃", "虐待", "暴力干涉婚姻自由",
    "故意毁坏财物", "破坏生产经营",
    "伪造", "变造", "冒充", "买卖国家机关证件", "公文", "印章",
    "拒不支付劳动报酬", "重大安全事故", "消防责任事故",
    "危险驾驶", "醉驾", "飙车",
    "传播淫秽物品", "组织传播", "开设赌场",
    "非法吸收公众存款", "集资诈骗", "洗钱",
    "窃取", "盗取", "入户盗窃", "扒窃",
]

CIVIL_KEYWORDS = [
    "合同", "纠纷", "侵权", "赔偿", "借贷", "借款", "离婚", "继承",
    "婚姻", "房产", "房屋", "土地", "建设工程", "施工", "租赁",
    "劳动", "工伤", "仲裁", "劳动合同", "工资", "社保",
    "公司", "股权", "股东", "合伙", "投资",
    "交通事故", "保险", "医疗", "教育", "服务",
    "物权", "所有权", "抵押", "质押", "担保", "债权",
    "执行", "异议", "查封", "冻结", "拍卖",
]


def classify_query(query):
    """判断问题是民事还是刑事——规则优先，LLM辅助"""
    q = query
    criminal_score = sum(1 for kw in CRIMINAL_KEYWORDS if kw in q)
    civil_score = sum(1 for kw in CIVIL_KEYWORDS if kw in q)

    if criminal_score > 0 and criminal_score >= civil_score:
        return "刑事"
    if civil_score > 0 and civil_score > criminal_score:
        return "民事"

    # 规则无法判断时，用LLM
    import requests
    prompt = f"""判断以下法律问题的类别，只需回答"民事"或"刑事"：

问题：{query}

回答："""

    data = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 10,
        "temperature": 0
    }
    headers = {
        "Authorization": f"Bearer {MINIMAX_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(f"{MINIMAX_BASE_URL}/chat/completions",
                             headers=headers, json=data, timeout=15)
        resp.raise_for_status()
        result = resp.json()
        label = result["choices"][0]["message"]["content"].strip()
        if "刑" in label:
            return "刑事"
        return "民事"
    except:
        return "民事"  # 出错默认民事


def retrieve(query, bm25, cases, top_k=TOP_K):
    """检索top-k最相似案例"""
    scores = bm25.get_scores(simple_tokenizer(query))
    indices = scores.argsort()[-top_k:][::-1]
    return [cases[i] for i in indices]


def build_prompt(question, retrieved_cases, case_type):
    """构建带上下文的prompt"""
    context_parts = []
    total_chars = 0

    for i, case in enumerate(retrieved_cases, 1):
        title = case.get("title", "未知标题")
        cause = case.get("cause_of_action", "")
        court = case.get("court", "")
        date = case.get("judgment_date", "")
        content = case.get("content", "")[:800]  # 限制每条长度

        part = f"""【案例{i}】{title}
案由: {cause} | 法院: {court} | 判决日期: {date}
{content}"""
        part_chars = len(part)
        if total_chars + part_chars > MAX_CONTEXT_CHARS:
            break
        context_parts.append(part)
        total_chars += part_chars

    context = "\n\n".join(context_parts)

    prompt = f"""你是一位专业的法律AI助手。根据以下{case_type}相关判例，回答用户的问题。

【相关判例】
{context}

【用户问题】
{question}

请结合判例内容给出专业回答。如果判例中有类似案例，应引用相关内容。回答应当：
1. 先给出结论
2. 结合判例说明法律依据和实践
3. 如有不同观点或特殊情况应予说明
"""
    return prompt


def call_llm(prompt):
    """调用MiniMax LLM"""
    import requests
    data = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}]
    }
    headers = {
        "Authorization": f"Bearer {MINIMAX_API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(f"{MINIMAX_BASE_URL}/chat/completions",
                            headers=headers, json=data, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        return result["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as e:
        return f"API错误 ({e.response.status_code}): {e.response.text[:500]}"
    except Exception as e:
        return f"请求失败: {e}"


def ask(question, civil_bm25=None, criminal_bm25=None,
         civil_cases=None, criminal_cases=None, interactive=False):
    """主问答函数（分类检索版）"""
    if civil_bm25 is None:
        print("加载索引中...")
        civil_cases, criminal_cases = load_cases()
        civil_bm25 = build_index(civil_cases)
        criminal_bm25 = build_index(criminal_cases)

    print(f"\n问题: {question}")
    print("判断问题类型...")
    case_type = classify_query(question)
    print(f"判定为: {case_type}")

    # 选择对应索引
    if case_type == "刑事":
        bm25, cases = criminal_bm25, criminal_cases
    else:
        bm25, cases = civil_bm25, civil_cases

    print(f"在{case_type}库中检索...")
    retrieved = retrieve(question, bm25, cases)

    if not retrieved:
        # fallback：跨库搜
        print("本库无结果，跨库搜索...")
        retrieved = retrieve(question, civil_bm25, civil_cases)

    print(f"找到 {len(retrieved)} 条相关案例")

    print("生成回答中...")
    prompt = build_prompt(question, retrieved, case_type)
    answer = call_llm(prompt)

    print(f"\n{'='*60}")
    print(f"问题: {question}")
    print(f"类型: {case_type}")
    print(f"\n回答:\n{answer}")

    if interactive:
        print(f"\n参考案例:")
        for i, c in enumerate(retrieved, 1):
            print(f"  [{i}] {c.get('title','未知')} ({c.get('case_type','')})")

    return answer, retrieved, case_type


if __name__ == "__main__":
    print("=" * 60)
    print("法眼AI - 法律案例问答系统 v2（分类检索）")
    print("=" * 60)

    civil_cases, criminal_cases = load_cases()
    civil_bm25 = build_index(civil_cases)
    criminal_bm25 = build_index(criminal_cases)
    print("索引构建完成")

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        ask(question, civil_bm25, criminal_bm25, civil_cases, criminal_cases)
    else:
        print("\n请输入法律问题（输入 exit 退出）：")
        while True:
            try:
                q = input("\n> ").strip()
                if q.lower() in ("exit", "quit", "q"):
                    break
                if q:
                    ask(q, civil_bm25, criminal_bm25, civil_cases, criminal_cases)
            except EOFError:
                break
