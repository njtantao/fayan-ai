"""
提取【中国法院2014年度案例18册】PDF（图片扫描件）
使用 tesseract OCR + pypdfium2 渲染
输出: extracted_cases/annual_2014.json
"""

import json, re, time, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pytesseract
import pypdfium2 as pdfium
from PIL import Image
import numpy as np

BASE_DIR = Path("/Users/tt/Desktop/hermes/项目开发/法眼ai")
PDF_DIR = BASE_DIR / "执行案例356" / "1.中国法院2014年度案例18册"
OUT_JSON = BASE_DIR / "项目开发" / "法眼ai代码" / "extracted_cases" / "annual_2014.json"

# tesseract 路径
TESSERACT_CMD = "/opt/homebrew/bin/tesseract"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


def ocr_page(img):
    """对一张图片OCR，返回文字"""
    try:
        text = pytesseract.image_to_string(img, lang='chi_sim+eng', timeout=20)
        return text
    except Exception:
        return ""


def render_page(pdf_doc, page_idx, scale=1.0):
    """渲染PDF某一页为PIL图片"""
    page = pdf_doc[page_idx]
    return page.render(scale=scale).to_pil()


def extract_one_pdf(args):
    """提取单个PDF的所有案例页OCR文本"""
    pdf_path, category = args
    try:
        pdf = pdfium.PdfDocument(pdf_path)
    except Exception as e:
        return [], f"{category}: 打开失败 {e}"

    n_pages = len(pdf)
    print(f"  [{category}] 共{n_pages}页，开始OCR...")

    all_text = []
    start_t = time.time()

    # 跳过前14页（封面、目录、前言等），从第15页开始
    start_page = 14
    # 也跳过最后几页空白
    end_page = n_pages - 2

    for pg in range(start_page, end_page):
        img = render_page(pdf, pg, scale=1.5)
        # 检查是否几乎是白页（跳过空白页）
        arr = np.array(img)
        white_ratio = np.mean(np.all(arr > 250, axis=2))
        if white_ratio > 0.98:
            continue  # 跳过几乎全白的页

        text = ocr_page(img)
        if text and len(text.strip()) > 50:
            all_text.append(text)

        if (pg - start_page + 1) % 50 == 0:
            elapsed = time.time() - start_t
            print(f"    [{category}] 已完成 {pg - start_page + 1}/{end_page - start_page} 页，{elapsed:.0f}s")

    elapsed = time.time() - start_t
    print(f"  [{category}] OCR完成，{len(all_text)}段文字，耗时{elapsed:.0f}s")

    # 按【案件基本信息】拆分
    cases = []
    combined = "\n".join(all_text)
    marker_positions = [m.start() for m in re.finditer(r'【案件基本信息】', combined)]

    for idx, start_pos in enumerate(marker_positions):
        end_pos = marker_positions[idx + 1] if idx + 1 < len(marker_positions) else len(combined)
        case_text = combined[max(0, start_pos - 2000):end_pos]

        parsed = parse_case_text(case_text, category, idx)
        if parsed:
            cases.append(parsed)

    return cases, f"{category}: {len(cases)}条案例"


def parse_case_text(text, category, idx):
    """解析单条案例文本"""
    # 清理OCR噪音（去除多余空白）
    text = re.sub(r'[ \u00a0]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    pos = text.find('【案件基本信息】')
    chunk_before = text[max(0, pos - 1500):pos]

    # 提取标题
    title = category
    last_dash = chunk_before.rfind('——')
    if last_dash >= 0:
        title_text = chunk_before[last_dash + 1:pos].strip().lstrip('—－-— ‑').strip()
        title = title_text.split('\n')[0].strip().rstrip('案').strip() + '案'
    else:
        # 找"XXX诉XXX案"格式
        chunk_clean = re.sub(r'\s+', '', chunk_before)
        su_m = re.search(r'([^\n]{2,30}?诉[^\n]{1,20}?案)', chunk_clean)
        if su_m:
            title = su_m.group(1).strip()

    # 提取案号
    case_number_raw = ""
    id_m = re.search(r'裁判书字号\s*[:：]?\s*([^\n]{8,80})', text)
    if id_m:
        case_number_raw = id_m.group(1).strip()

    case_num_m = re.search(r'(\d{4}).*?(\d+)\s*号', re.sub(r'\s+', '', case_number_raw))
    if case_num_m:
        case_number = f"{case_num_m.group(1)}-18-2-{case_num_m.group(2).zfill(3)}-{category[:4]}"
    else:
        case_number = f"2014-18-2-{idx:04d}-{category[:4]}"

    # 案由
    cause = ""
    cause_m = re.search(r'案由\s*[:：]\s*([^\n【】]{2,40})', text)
    if cause_m:
        cause = cause_m.group(1).strip()
    if not cause:
        cause = category

    # 审理法院
    court = ""
    court_m = re.search(r'审理法院\s*[:：]\s*([^\n]{2,50})', text)
    if court_m:
        court = court_m.group(1).strip()

    # 裁判日期
    judgment_date = ""
    date_m = re.search(r'裁判日期\s*[:：]\s*(\d{4}[年/-]\d{1,2}[月/-]\d{1,2})', text)
    if date_m:
        judgment_date = date_m.group(1).strip()

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
            "related_laws": "",
            "source": "2014年度案例",
        }
    }


def main():
    t0 = time.time()

    # 收集所有PDF
    pdf_args = []
    for pdf_file in sorted(PDF_DIR.glob("*.pdf")):
        fname = pdf_file.name
        # 去掉扩展名和"中国法院2014年度案例_"前缀
        category = fname.replace("中国法院2014年度案例_", "").replace(".pdf", "").strip()
        pdf_args.append((str(pdf_file), category))

    print(f"找到 {len(pdf_args)} 个PDF:")
    for args in pdf_args:
        print(f"  {args[1]}")

    all_cases = []
    results = []

    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(extract_one_pdf, args): args for args in pdf_args}
        for future in as_completed(futures):
            cases, msg = future.result()
            all_cases.extend(cases)
            results.append(msg)
            print(f"  {msg}")

    # 去重
    seen = set()
    unique = []
    for c in all_cases:
        if c['id'] not in seen:
            seen.add(c['id'])
            unique.append(c)

    elapsed = time.time() - t0
    print(f"\n提取完成: {len(all_cases)} 条，去重后 {len(unique)} 条，耗时 {elapsed:.1f}s")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(unique, f, ensure_ascii=False, indent=2)
    print(f"已保存: {OUT_JSON}")

    # 分类统计
    from collections import Counter
    cats = Counter(c.get('case_type', '未知') for c in unique)
    print("\n分类统计:")
    for cat, cnt in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {cnt}")

    return unique


if __name__ == "__main__":
    main()
