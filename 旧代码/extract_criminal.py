"""
提取刑事PDF案例（人民法院案例库 刑事）
使用pdfminer批量提取，速度快
"""

import json, re, os, sys
from pathlib import Path
from pdfminer.high_level import extract_text
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

BASE_DIR = Path("/Users/tt/Desktop/hermes/项目开发/法眼ai")
CRIMINAL_DIR = BASE_DIR / "执行案例356" / "刑事 pdf"
OUT_JSON = BASE_DIR / "extracted_cases" / "criminal_cases.json"


def split_cases_by_marker(full_text):
    """按【基本案情】【裁判要旨】等标记拆分案例"""
    # 找所有案例起始位置：标题后跟案件信息
    # 案例以 "案件信息" 或 "入库编号" 开头
    pattern = re.compile(r'(?<=\n)([\u4e00-\u9fa5]{2,30}?(?:罪|纠纷|案)\s*\n+案件信息)', re.MULTILINE)
    starts = [m.start() for m in pattern.finditer(full_text)]
    
    cases_text = []
    for i, start in enumerate(starts):
        end = starts[i+1] if i+1 < len(starts) else len(full_text)
        cases_text.append(full_text[start:end])
    
    return cases_text


def parse_criminal_case(text, case_num_prefix="2025", full_text=None):
    """解析单个刑事案例文本"""
    # 入库编号
    id_m = re.search(r'入库编号[：:]\s*(\S+)', text)
    case_number = id_m.group(1).strip() if id_m else ""

    # 标题：从入库编号往前找 "XXX案" 形式的行
    pos = id_m.start() if id_m else 100
    chunk_before = text[max(0, pos-200):pos]
    title_m = re.search(r'\n([^\n]{4,60}?案)\n', chunk_before)
    title = title_m.group(1).strip() if title_m else case_number
    
    # 基本案情
    facts_m = re.search(r'基本案情[：:]\s*\n?(.*?)(?=\n裁判理由|\n裁判要旨|\n关联索引|\Z)', text, re.DOTALL)
    basic_facts = facts_m.group(1).strip() if facts_m else ""
    
    # 裁判理由
    ruling_m = re.search(r'裁判理由[：:]\s*\n?(.*?)(?=\n裁判要旨|\n关联索引|\Z)', text, re.DOTALL)
    ruling_reason = ruling_m.group(1).strip() if ruling_m else ""
    
    # 裁判要旨
    rp_m = re.search(r'裁判要旨[：:]\s*\n?(.*?)(?=\n关联索引|\Z)', text, re.DOTALL)
    ruling_points = rp_m.group(1).strip() if rp_m else ""
    
    # 案件信息字段
    court_m = re.search(r'审理法院[：:]\s*([^\n]+)', text)
    court = court_m.group(1).strip() if court_m else ""
    
    date_m = re.search(r'裁判日期[：:]\s*([^\n]+)', text)
    judgment_date = date_m.group(1).strip()[:10] if date_m else ""
    
    case_id_m = re.search(r'案件证号[：:]\s*([^\n]+)', text)
    case_id = case_id_m.group(1).strip() if case_id_m else ""
    
    cause_m = re.search(r'二级分类[：:]\s*([^\n]+)', text)
    cause = cause_m.group(1).strip() if cause_m else ""
    
    trial_m = re.search(r'庭审[：:]\s*([^\n]+)', text)
    trial = trial_m.group(1).strip() if trial_m else ""
    
    province_m = re.search(r'省份[：:]\s*([^\n]+)', text)
    province = province_m.group(1).strip() if province_m else ""
    
    kw_m = re.search(r'关键词[：:]\s*\n?([^\n]+)', text)
    keywords_raw = kw_m.group(1).strip() if kw_m else ""
    keywords = [k.strip() for k in re.split(r'[,，、\t]', keywords_raw) if k.strip()]
    
    content_body = "\n".join([
        f"【基本案情】{basic_facts}",
        f"【裁判理由】{ruling_reason}",
        f"【裁判要旨】{ruling_points}",
    ]).strip()
    
    if not ruling_points and not basic_facts:
        return None
    
    return {
        "id": case_number or title,
        "case_number": case_number or title,
        "title": title[:200],
        "court": court,
        "judgment_date": judgment_date,
        "case_type": "刑事",
        "cause_of_action": cause,
        "trial_level": trial,
        "province": province,
        "case_id": case_id,
        "content": content_body[:3000],
        "metadata": {
            "keywords": keywords,
            "ruling_points": ruling_points[:500],
            "basic_facts": basic_facts[:1000],
            "ruling_reason": ruling_reason[:500],
            "related_laws": "",
            "source": "刑事案例库",
        }
    }


def extract_pdf_fast(pdf_path):
    """快速提取单个PDF的所有案例"""
    try:
        text = extract_text(pdf_path, maxpages=0)  # 0 = 全部页面
    except Exception as e:
        print(f"  提取失败: {e}", file=sys.stderr)
        return []
    
    if not text or len(text) < 100:
        print(f"  文本过短，跳过")
        return []
    
    # 按"基本案情"拆分
    # 每个案例从 "关键词" 或 "入库编号" 开始，到下一个 "关键词" 前结束
    # 更可靠的方式：按 "案件信息" 分割
    
    # 收集入库编号所在位置（全局位置）
    positions = [(m.start(), m.group(1)) for m in re.finditer(r'入库编号[：:]\s*(\S+)', text)]
    print(f"  入库编号数: {len(positions)}")

    cases = []
    for idx, (start, case_num) in enumerate(positions):
        # 往前找"案件信息"或"入库编号"前的标题
        # 每个案例：从上一个"案件信息"之后 到 下一个"案件信息"之前
        # 但入库编号在案件信息段内，所以我们从"入库编号"往前300字开始（包含标题）
        chunk_start = max(0, start - 300)
        chunk_end = positions[idx+1][0] if idx+1 < len(positions) else len(text)
        case_text = text[chunk_start:chunk_end]

        parsed = parse_criminal_case(case_text)
        if parsed:
            cases.append(parsed)

    return cases


def main():
    t0 = time.time()
    
    pdf_files = sorted(CRIMINAL_DIR.glob("*.pdf"))
    print(f"找到 {len(pdf_files)} 个刑事PDF")
    
    all_cases = []
    for pdf_path in pdf_files:
        print(f"\n处理: {pdf_path.name}")
        cases = extract_pdf_fast(str(pdf_path))
        print(f"  -> {len(cases)} 条")
        all_cases.extend(cases)
    
    print(f"\n刑事案例总提取: {len(all_cases)} 条，耗时 {time.time()-t0:.1f}s")
    
    # 保存
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(all_cases, f, ensure_ascii=False, indent=2)
    print(f"已保存: {OUT_JSON}")
    
    return all_cases


if __name__ == "__main__":
    main()
