"""
提取【Z110】中国法院2025年度案例（全23册）PDF
处理超清对照PDF，每个子目录一个PDF
输出: extracted_cases/annual_2025.json
"""

import json, re, time
from pathlib import Path
from pdfminer.high_level import extract_text
from concurrent.futures import ProcessPoolExecutor, as_completed

BASE_DIR = Path("/Users/tt/Desktop/hermes/项目开发/法眼ai")
PDF_DIR = BASE_DIR / "执行案例356" / "【Z110】中国法院2025年度案例（全23册）"
OUT_JSON = BASE_DIR / "extracted_cases" / "annual_2025.json"


def extract_one_pdf(args):
    """提取单个PDF，返回案例列表"""
    pdf_path, category = args
    try:
        text = extract_text(str(pdf_path), maxpages=0)
    except Exception as e:
        return [], f"{category}: 提取失败 {e}"

    if not text or len(text) < 500:
        return [], f"{category}: 文本过短"

    # 按【案件基本信息】拆分
    marker_positions = [m.start() for m in re.finditer(r'【案件基本信息】', text)]

    cases = []
    for idx, start_pos in enumerate(marker_positions):
        end_pos = marker_positions[idx + 1] if idx + 1 < len(marker_positions) else len(text)
        # 往前多取300字用于提取标题
        case_text = text[max(0, start_pos-300):end_pos]

        parsed = parse_case_text(case_text, category, idx, start_pos)
        if parsed:
            cases.append(parsed)

    return cases, f"{category}: {len(cases)} 条"


def parse_case_text(text, category, idx, start_pos=0):
    """从案例文本块中提取关键字段"""
# 标题：从【案件基本信息】往前找"——XXX案"或"XXX诉XXX案"
    pos = text.find('【案件基本信息】')
    chunk_before_raw = text[max(0, pos-1500):pos]
    title = category

    # 优先找最后一个'——'或'— —'之后的内容（两种格式都处理）
    last_dash = chunk_before_raw.rfind('——')
    if last_dash < 0:
        last_dash = chunk_before_raw.rfind('— —')
    if last_dash < 0:
        last_dash = chunk_before_raw.rfind('—')
    if last_dash >= 0:
        title_text = chunk_before_raw[last_dash+1:pos].strip().lstrip('—－-— ‑').strip()
        title = title_text.split('\n')[0].strip().rstrip('案').strip() + '案'
    else:
        # 没有"——"，清理OCR空格后匹配"XXX诉XXX案"格式
        chunk_clean = re.sub(r'\s+', '', chunk_before_raw)
        su_m = re.search(r'([^\n]{2,30}?诉[^\n]{1,20}?案)', chunk_clean)
        if su_m:
            title = su_m.group(1).strip()
        else:
            title = category  # 保底用分类名

    # 入库编号
    id_m = re.search(r'裁判书字号\s*[:：]?\s*([^\n]{10,80})', text)
    case_number_raw = id_m.group(1).strip() if id_m else ""
    # 从裁判书字号提取案号
    # 从裁判书字号提取案号（清理空格OCR噪音）
    case_num_m = re.search(r'(\d{4}).*?(\d+)\s*号', re.sub(r'\s+', '', case_number_raw))
    if case_num_m:
        case_number = f"{case_num_m.group(1)}-18-2-{case_num_m.group(2).zfill(3)}"
    else:
        case_number = f"2025-18-2-{idx:04d}"
    # 加入category后缀避免不同PDF间案号重复
    case_number = f"{case_number}-{category[:2]}"

    # 案由
    cause_m = re.search(r'案由[：:]\s*([^\n【】]{2,40})', text)
    cause = cause_m.group(1).strip() if cause_m else category

    # 审理法院
    court_m = re.search(r'审理法院[：:]\s*([^\n]{2,50})', text)
    court = court_m.group(1).strip() if court_m else ""

    # 裁判日期
    date_m = re.search(r'裁判日期[：:]\s*(\d{4}[年/-]\d{1,2}[月/-]\d{1,2})', text)
    judgment_date = date_m.group(1).strip() if date_m else ""

    # 基本案情
    facts_m = re.search(r'【基本案情】\n?(.*?)(?=【案件焦点】|【裁判要旨】|【法院裁判要旨】|【法官后语】|$)', text, re.DOTALL)
    facts = facts_m.group(1).strip() if facts_m else ""

    # 案件焦点
    focus_m = re.search(r'【案件焦点】\n?(.*?)(?=【裁判要旨】|【法院裁判要旨】|【法官后语】|【基本案情】|$)', text, re.DOTALL)
    focus = focus_m.group(1).strip() if focus_m else ""

    # 裁判要旨
    ruling_m = re.search(r'【法院裁判要旨】\n?(.*?)(?=【法官后语】|【案件基本信息】|【基本案情】|$)', text, re.DOTALL)
    if not ruling_m:
        ruling_m = re.search(r'【裁判要旨】\n?(.*?)(?=【法官后语】|【案件基本信息】|【基本案情】|$)', text, re.DOTALL)
    ruling_points = ruling_m.group(1).strip() if ruling_m else ""

    # 关联法规
    law_refs = re.findall(r'《([^》]{2,30}?)》', ruling_points[:1000])
    related_laws = "、".join(law_refs[:5])

    if not facts and not ruling_points:
        return None

    return {
        "id": case_number,
        "case_number": case_number,
        "title": title[:200],
        "court": court[:80],
        "judgment_date": judgment_date,
        "case_type": category,
        "cause_of_action": cause,
        "trial_level": "",
        "province": "",
        "case_id": case_number_raw[:100],
        "content": "\n".join([
            f"【基本案情】{facts[:1500]}",
            f"【案件焦点】{focus[:300]}",
            f"【裁判要旨】{ruling_points[:800]}",
        ]).strip(),
        "metadata": {
            "keywords": [category],
            "ruling_points": ruling_points[:500],
            "basic_facts": facts[:1000],
            "ruling_reason": "",
            "related_laws": related_laws,
            "source": "2025年度案例",
        }
    }


def main():
    t0 = time.time()

    # 收集所有PDF（优先"超清对照"版本）
    pdf_args = []
    for subdir in sorted(PDF_DIR.iterdir()):
        if not subdir.is_dir():
            continue
        # 优先选"超清对照"PDF
        pdfs = list(subdir.glob("*超清对照*.pdf"))
        if not pdfs:
            pdfs = list(subdir.glob("*.pdf"))
        if not pdfs:
            continue
        category = subdir.name.split(" ", 1)[-1] if " " in subdir.name else subdir.name
        pdf_args.append((str(pdfs[0]), category))

    print(f"找到 {len(pdf_args)} 个PDF")
    print("=" * 50)

    all_cases = []
    results = []

    # 并行处理（最多6个并发）
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(extract_one_pdf, args): args for args in pdf_args}
        for future in as_completed(futures):
            cases, msg = future.result()
            all_cases.extend(cases)
            results.append(msg)
            print(f"  {msg}")

    # 去重（按id）
    seen = set()
    unique = []
    for c in all_cases:
        if c['id'] not in seen:
            seen.add(c['id'])
            unique.append(c)

    print(f"\n提取完成: {len(all_cases)} 条，去重后 {len(unique)} 条，耗时 {time.time()-t0:.1f}s")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(unique, f, ensure_ascii=False, indent=2)
    print(f"已保存: {OUT_JSON}")

    # 统计
    from collections import Counter
    cats = Counter(c.get('case_type', '未知') for c in unique)
    print("\n分类统计:")
    for cat, cnt in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {cnt}")

    return unique


if __name__ == "__main__":
    main()