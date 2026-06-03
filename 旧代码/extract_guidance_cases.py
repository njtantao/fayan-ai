"""
提取（共49批）最高人民法院指导性案例汇编 PDF
输出: extracted_cases/guidance_cases.json
"""

import json, re, time
from pathlib import Path
from pdfminer.high_level import extract_text
import pdfplumber

BASE_DIR = Path("/Users/tt/Desktop/hermes/项目开发/法眼ai")
PDF_PATH = BASE_DIR / "执行案例356" / "（共49批）最高人民法院批指导性案例汇编（2026年2月）.pdf"
OUT_JSON = BASE_DIR / "extracted_cases" / "guidance_cases.json"


def parse_guidance_case(chunk_text, case_num, idx):
    """从案例块中提取字段"""
    text = chunk_text

    # 标题：第一行通常是"指导案例X号\n标题"
    lines = text.split('\n')
    title = ""
    for i, line in enumerate(lines[:5]):
        clean = line.strip()
        if clean and clean != f"指导案例 {case_num} 号" and '关键词' not in clean and '裁判要点' not in clean:
            title = clean.strip().rstrip('案').strip() + '案'
            break
    if not title:
        title = f"指导案例{case_num}号"

    # 关键词
    kw_m = re.search(r'关键词[：:]\s*(.*?)(?=裁判要点|相关法条|基本案情|裁判结果)', text, re.DOTALL)
    keywords_raw = kw_m.group(1).strip() if kw_m else ""
    keywords = [k.strip() for k in re.split(r'[,，、\n]', keywords_raw) if k.strip()]

    # 裁判要点
    ruling_m = re.search(r'裁判要点[：:]\s*(.*?)(?=相关法条|基本案情|裁判结果|裁判理由)', text, re.DOTALL)
    ruling_points = ruling_m.group(1).strip() if ruling_m else ""

    # 相关法条
    law_m = re.search(r'相关法条[：:]\s*(.*?)(?=基本案情|裁判结果|关键词|裁判理由)', text, re.DOTALL)
    related_laws = law_m.group(1).strip() if law_m else ""

    # 基本案情
    facts_m = re.search(r'基本案情[：:]\s*\n?(.*?)(?=裁判结果|裁判理由|裁判要旨|裁判要点)', text, re.DOTALL)
    basic_facts = facts_m.group(1).strip() if facts_m else ""

    # 裁判结果
    result_m = re.search(r'裁判结果[：:]\s*\n?(.*?)(?=裁判理由|裁判要旨|$)', text, re.DOTALL)
    ruling_result = result_m.group(1).strip() if result_m else ""

    # 裁判理由
    reason_m = re.search(r'裁判理由[：:]\s*\n?(.*?)(?=裁判结果|基本案情|$)', text, re.DOTALL)
    ruling_reason = reason_m.group(1).strip() if reason_m else ""

    # 审理法院（从正文中提取）
    court_m = re.search(r'(?:省|市|自治区|自治区)?.{0,10}?(?:中级人民法院|高级人民法院|最高人民法院|人民法院)', text)
    court = court_m.group(0).strip() if court_m else ""

    # 案号
    case_id_m = re.search(r'[（(](\d{4})[^0-9]*?(\d+)[^0-9]*?号[)）]', text)
    if case_id_m:
        case_id = f"{case_id_m.group(1)}-18-2-{case_id_m.group(2).zfill(3)}"
    else:
        case_id = f"GUID-{case_num}"

    content_body = "\n".join([
        f"【基本案情】{basic_facts[:1500]}",
        f"【裁判理由】{ruling_reason[:500]}",
        f"【裁判要点】{ruling_points[:500]}",
        f"【裁判结果】{ruling_result[:300]}",
    ]).strip()

    if not ruling_points and not basic_facts:
        return None

    return {
        "id": case_id,
        "case_number": case_id,
        "title": title[:200],
        "court": court[:80],
        "judgment_date": "",
        "case_type": "指导性案例",
        "cause_of_action": keywords[0] if keywords else "其他",
        "trial_level": "",
        "province": "",
        "case_id": f"指导案例{case_num}号",
        "content": content_body,
        "metadata": {
            "keywords": keywords[:10],
            "ruling_points": ruling_points[:500],
            "basic_facts": basic_facts[:1000],
            "ruling_reason": ruling_reason[:500],
            "related_laws": related_laws[:200],
            "source": "指导性案例汇编",
        }
    }


def extract_guidance_cases():
    t0 = time.time()

    # 提取全部文本（只做一次）
    print("提取PDF全文...")
    full_text = extract_text(str(PDF_PATH), maxpages=0)
    print(f"全文长度: {len(full_text)}")

    # 找到所有"指导案例 X 号"的位置
    # 排除目录（目录中也有但后面不跟关键词）
    all_markers = [(m.start(), m.group()) for m in re.finditer(r'指导案例\s*(\d+)\s*号', full_text)]
    print(f"总标记数: {len(all_markers)}")

    # 对于每个marker，检查是否是真正的案例（后面跟"关键词："）
    valid_markers = []
    for pos, name in all_markers:
        chunk = full_text[pos:pos+2000]
        if '关键词：' in chunk or '关键词:' in chunk:
            case_num = re.search(r'(\d+)', name).group(1)
            valid_markers.append((pos, case_num))

    print(f"有效案例数: {len(valid_markers)}")

    # 按位置排序
    valid_markers.sort(key=lambda x: x[0])

    cases = []
    for idx, (pos, case_num) in enumerate(valid_markers):
        end_pos = valid_markers[idx+1][0] if idx+1 < len(valid_markers) else len(full_text)
        chunk = full_text[pos:end_pos]

        parsed = parse_guidance_case(chunk, case_num, idx)
        if parsed:
            cases.append(parsed)

    print(f"提取完成: {len(cases)} 条，耗时 {time.time()-t0:.1f}s")
    return cases


def main():
    cases = extract_guidance_cases()

    # 保存
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    print(f"已保存: {OUT_JSON}")

    # 统计
    from collections import Counter
    cats = Counter(c.get('cause_of_action', '未知') for c in cases)
    print("\n案由TOP10:")
    for cat, cnt in sorted(cats.items(), key=lambda x: -x[1])[:10]:
        print(f"  {cat}: {cnt}")

    return cases


if __name__ == "__main__":
    main()