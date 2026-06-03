"""
批量提取【中国法院2014年度案例18册】所有PDF
OCR + 宽松正则匹配（兼容扫描件OCR错误）
输出: extracted_cases/all_cases.json
"""
import json, re, time, os
from pathlib import Path
import pytesseract
import pypdfium2 as pdfium
from PIL import Image
import numpy as np

TESSERACT_CMD = "/opt/homebrew/bin/tesseract"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

BASE_DIR = Path("/Users/tt/Desktop/hermes/项目开发/法眼ai/执行案例356/1.中国法院2014年度案例18册")
OUT_JSON = Path("/Users/tt/Desktop/hermes/项目开发/法眼ai代码/extracted_cases/all_cases.json")

# 类别映射（从文件名提取）
CATEGORIES = {
    "保险纠纷": "保险纠纷",
    "道路交通纠纷": "道路交通纠纷",
    "房屋买卖合同纠纷": "房屋买卖合同纠纷",
    "雇员受害赔偿纠纷": "雇员受害赔偿纠纷",
    "合同纠纷": "合同纠纷",
    "婚姻家庭与继承纠纷": "婚姻家庭与继承纠纷",
    "借款担保纠纷": "借款担保纠纷",
    "金融纠纷": "金融纠纷",
    "劳动纠纷": "劳动纠纷",
    "民间借贷纠纷": "民间借贷纠纷",
    "侵权赔偿纠纷": "侵权赔偿纠纷",
    "人格权纠纷": "人格权纠纷",
    "土地纠纷": "土地纠纷",
    "物权纠纷": "物权纠纷",
    "刑事案例": "刑事案例",
    "行政纠纷": "行政纠纷",
    "公司纠纷": "公司纠纷",
}

PDF_FILES = [
    ("中国法院2014年度案例_保险纠纷_扫描版.pdf", "保险纠纷"),
    ("中国法院2014年度案例_道路交通纠纷_扫描版.pdf", "道路交通纠纷"),
    ("中国法院2014年度案例_房屋买卖合同纠纷_扫描版.pdf", "房屋买卖合同纠纷"),
    ("中国法院2014年度案例_雇员受害赔偿纠纷_含帮工损害赔偿纠纷_扫描版.pdf", "雇员受害赔偿纠纷"),
    ("中国法院2014年度案例_合同纠纷_扫描版.pdf", "合同纠纷"),
    ("中国法院2014年度案例_婚姻家庭与继承纠纷_扫描版.pdf", "婚姻家庭与继承纠纷"),
    ("中国法院2014年度案例_借款担保纠纷_扫描版.pdf", "借款担保纠纷"),
    ("中国法院2014年度案例_金融纠纷_扫描版.pdf", "金融纠纷"),
    ("中国法院2014年度案例_劳动纠纷_扫描版.pdf", "劳动纠纷"),
    ("中国法院2014年度案例_民间借贷纠纷_扫描版.pdf", "民间借贷纠纷"),
    ("中国法院2014年度案例_侵权赔偿纠纷_扫描版.pdf", "侵权赔偿纠纷"),
    ("中国法院2014年度案例_人格权纠纷_扫描版.pdf", "人格权纠纷"),
    ("中国法院2014年度案例_土地纠纷_含林地纠纷_扫描版.pdf", "土地纠纷"),
    ("中国法院2014年度案例_物权纠纷_扫描版.pdf", "物权纠纷"),
    ("中国法院2014年度案例_刑事案例_扫描版.pdf", "刑事案例"),
    ("中国法院2014年度案例_行政纠纷_扫描版.pdf", "行政纠纷"),
    ("中国法院2014年度案例公司纠纷_扫描版.pdf", "公司纠纷"),
]


def ocr_pdf(pdf_path, start_page=14, end_page_offset=2, scale=1.0):
    """OCR PDF，返回合并文本列表"""
    try:
        pdf = pdfium.PdfDocument(pdf_path)
    except Exception as e:
        print(f"  打开失败: {e}")
        return []
    
    n_pages = len(pdf)
    end_page = n_pages - end_page_offset
    all_text = []
    
    for pg in range(start_page, end_page):
        page = pdf[pg]
        img = page.render(scale=scale).to_pil()
        arr = np.array(img)
        if np.mean(np.all(arr > 250, axis=2)) > 0.98:
            continue
        try:
            text = pytesseract.image_to_string(img, lang='chi_sim', timeout=15)
        except Exception:
            text = ""
        if text and len(text.strip()) > 50:
            all_text.append(text)
        if (pg - start_page + 1) % 50 == 0:
            print(f"    {pg - start_page + 1}/{end_page - start_page}页")
    
    return all_text


def find_section_starts(text):
    """用宽松正则找所有章节标记"""
    markers = []
    for m in re.finditer(r'【([^】]+)】', text):
        raw = m.group(1)
        if any(k in raw for k in ['罕必', '案件基本信息', '于件天本信息', '计件基本信息', 'A件基本信息']):
            markers.append(('case_info', m.start()))
        elif any(k in raw for k in ['至本案情', '基本案情', '本案情']):
            markers.append(('case_facts', m.start()))
        elif any(k in raw for k in ['罕件信点', '琳件点', '于件人点', '于件全点', '案件焦点', '守件人点', '罕见点']):
            markers.append(('case_focus', m.start()))
        elif any(k in raw for k in ['法院才判要司', '法过才要局', '法院才到要悍', '裁判要旨', '法院才列要晤', '法院相到要悍', '法院才关要悍', '法院才判要']):
            markers.append(('ruling', m.start()))
        elif '法官后语' in raw or '法官辐语' in raw:
            markers.append(('judge_note', m.start()))
    return markers


def parse_one_case(case_text, category, idx, all_markers):
    """解析单条案例"""
    text = re.sub(r'[ \u00a0]+', ' ', case_text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 标题
    first_bracket = text.find('【')
    chunk_before = text[:first_bracket] if first_bracket >= 0 else text[:1500]
    title = category
    last_dash = chunk_before.rfind('——')
    if last_dash >= 0:
        title_text = chunk_before[last_dash + 1:].strip().lstrip('—－-— ‑').strip()
        title = title_text.split('\n')[0].strip().rstrip('案').strip() + '案'
    else:
        chunk_c = re.sub(r'\s+', '', chunk_before)
        su_m = re.search(r'([^\n]{2,30}?诉[^\n]{1,20}?案)', chunk_c)
        if su_m:
            title = su_m.group(1).strip()
    
    # 案号
    id_m = re.search(r'裁判书字号\s*[:：]?\s*([^\n]{8,80})', text)
    case_number_raw = id_m.group(1).strip() if id_m else ''
    case_num_m = re.search(r'(\d{4}).*?(\d+)\s*号', re.sub(r'\s+', '', case_number_raw))
    if case_num_m:
        case_number = f"{case_num_m.group(1)}-18-2-{case_num_m.group(2).zfill(3)}-{category[:4]}"
    else:
        case_number = f"2014-18-2-{idx:04d}-{category[:4]}"
    
    # 案由
    cause = ''
    for pat in [r'案由\s*[:：]\s*([^\n【】]{2,40})', r'案由[:：]\s*([^\n]{2,30})']:
        m = re.search(pat, text)
        if m:
            cause = m.group(1).strip()
            break
    if not cause:
        cause = category
    
    # 法院
    court = ''
    court_m = re.search(r'审理法院\s*[:：]\s*([^\n]{2,50})', text)
    if court_m:
        court = court_m.group(1).strip()
    
    # 裁判日期
    judgment_date = ''
    date_m = re.search(r'裁判日期\s*[:：]\s*(\d{4}[年/-]\d{1,2}[月/-]\d{1,2})', text)
    if date_m:
        judgment_date = date_m.group(1).strip()
    
    # 基本案情
    facts = ''
    for pat in [r'【至本案情】\n?(.*?)(?=【)', r'【基本案情】\n?(.*?)(?=【)', r'【本案情】\n?(.*?)(?=【)']:
        m = re.search(pat, text, re.DOTALL)
        if m:
            facts = m.group(1).strip()
            break
    
    # 案件焦点
    focus = ''
    for pat in [r'【罕件信点】\n?(.*?)(?=【)', r'【案件焦点】\n?(.*?)(?=【)', r'【守件人点】\n?(.*?)(?=【)', r'【罕见点】\n?(.*?)(?=【)']:
        m = re.search(pat, text, re.DOTALL)
        if m:
            focus = m.group(1).strip()
            break
    
    # 裁判要旨
    ruling = ''
    for pat in [r'【法院才判要司】\n?(.*?)(?=【)', r'【法过才要局】\n?(.*?)(?=【)',
                r'【法院相到要悍】\n?(.*?)(?=【)', r'【法院才列要晤】\n?(.*?)(?=【)',
                r'【法院才关要悍】\n?(.*?)(?=【)', r'【裁判要旨】\n?(.*?)(?=【)',
                r'【法院才判要】\n?(.*?)(?=【)']:
        m = re.search(pat, text, re.DOTALL)
        if m:
            ruling = m.group(1).strip()
            break
    
    return {
        'case_number': case_number,
        'title': title,
        'cause': cause,
        'court': court,
        'judgment_date': judgment_date,
        'facts': facts,
        'focus': focus,
        'ruling_points': ruling,
        'full_text': text[:8000],
        'category': category,
    }


def process_pdf(pdf_path, category):
    """处理单个PDF，返回案例列表"""
    print(f"\n[{category}] 开始处理...")
    t0 = time.time()
    
    all_text = ocr_pdf(pdf_path)
    if not all_text:
        print(f"  [{category}] 无OCR内容")
        return []
    
    combined = '\n'.join(all_text)
    markers = find_section_starts(combined)
    info_starts = [(t, p) for t, p in markers if t == 'case_info']
    
    print(f"  [{category}] {len(all_text)}页OCR, {len(info_starts)}个案例, 耗时{time.time()-t0:.0f}s")
    
    cases = []
    for idx, (seg_type, start_pos) in enumerate(info_starts):
        end_pos = info_starts[idx + 1][1] if idx + 1 < len(info_starts) else len(combined)
        case_text = combined[max(0, start_pos - 2000):end_pos]
        case = parse_one_case(case_text, category, idx, markers)
        cases.append(case)
    
    return cases


def main():
    # 加载现有库
    if OUT_JSON.exists():
        with open(OUT_JSON, 'r', encoding='utf-8') as f:
            existing = json.load(f)
    else:
        existing = []
    print(f"现有 {len(existing)} 条案例")
    
    total_new = 0
    grand_start = time.time()
    
    for filename, category in PDF_FILES:
        pdf_path = BASE_DIR / filename
        if not pdf_path.exists():
            print(f"[{category}] 文件不存在: {pdf_path}")
            continue
        
        cases = process_pdf(str(pdf_path), category)
        
        added = 0
        skipped = 0
        for case in cases:
            # 查重
            is_dup = any(
                c.get('case_number') == case['case_number'] or
                (c.get('title') == case['title'] and c.get('court') == case['court'] and case['court'])
                for c in existing
            )
            if is_dup:
                skipped += 1
                continue
            existing.append(case)
            added += 1
        
        total_new += added
        print(f"  [{category}] 新增 {added}, 跳过 {skipped} (已有)")
    
    # 保存
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    
    elapsed = time.time() - grand_start
    print(f"\n=== 完成 ===")
    print(f"新增 {total_new} 条, 总计 {len(existing)} 条")
    print(f"总耗时 {elapsed/60:.0f} 分钟")
    print(f"保存到: {OUT_JSON}")


if __name__ == '__main__':
    main()
