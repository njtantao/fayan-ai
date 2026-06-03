"""
从所有来源提取案例：
1. 民事 txt zip (2183条) — 已结构化
2. 已有 cases.json (356条) — 执行案例
3. 2025年度案例 PDF — 多分类，需要逐一解析

输出: extracted_cases/all_cases.json
"""

import json, re, os, zipfile, pdfplumber
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = Path("/Users/tt/Desktop/hermes/项目开发/法眼ai")
CASES_DIR = BASE_DIR / "extracted_cases"
SRC_DIR   = BASE_DIR / "执行案例356"
OUT_JSON  = CASES_DIR / "all_cases.json"

# ============================================================
# 1. 民事 txt (已有)
# ============================================================
def extract_civil_txt():
    zip_path = SRC_DIR / "民事类案例/民事 pdf txt md/人民法院案例库 (民事) (2184).zip"
    cases = []

    with zipfile.ZipFile(zip_path, 'r') as z:
        txt_files = sorted([n for n in z.namelist()
                           if n.startswith('txt/') and n.endswith('.txt')])

        for fname in txt_files:
            content = z.read(fname).decode('utf-8', errors='replace')

            id_match = re.search(r'入库编号[：:]\s*(\S+)', content)
            case_number = id_match.group(1).strip() if id_match else os.path.basename(fname)

            info_match = re.search(r'## 案件信息\s*\n(.*?)(?=\n##|\Z)', content, re.DOTALL)
            info_text = info_match.group(1) if info_match else ""

            court_m = re.search(r'审理法院[：:]\s*([^\n]+)', info_text)
            court = court_m.group(1).strip() if court_m else "未知"

            type_m = re.search(r'一级分类[：:]\s*([^\n]+)', info_text)
            case_type = type_m.group(1).strip() if type_m else "民事"

            cause_m = re.search(r'二级分类[：:]\s*([^\n]+)', info_text)
            cause = cause_m.group(1).strip() if cause_m else "其他"

            trial_m = re.search(r'庭审[：:]\s*([^\n]+)', info_text)
            trial = trial_m.group(1).strip() if trial_m else ""

            prov_m = re.search(r'省份[：:]\s*([^\n]+)', info_text)
            province = prov_m.group(1).strip() if prov_m else ""

            date_m = re.search(r'裁判日期[：:]\s*([^\n]+)', info_text)
            judgment_date = date_m.group(1).strip()[:10] if date_m else ""

            case_id_m = re.search(r'案件证号[：:]\s*([^\n]+)', info_text)
            case_id = case_id_m.group(1).strip() if case_id_m else ""

            kw_match = re.search(r'## 关键词：\s*\n\t?(.*?)(?=\n##)', content, re.DOTALL)
            keywords_raw = kw_match.group(1).strip() if kw_match else ""
            keywords = [k.strip() for k in re.split(r'[,，、\t]', keywords_raw) if k.strip()]

            facts_match = re.search(r'## 基本案情：\s*\n\t?(.*?)(?=\n##)', content, re.DOTALL)
            basic_facts = facts_match.group(1).strip() if facts_match else ""

            ruling_match = re.search(r'## 裁判理由：\s*\n\t?(.*?)(?=\n##)', content, re.DOTALL)
            ruling_reason = ruling_match.group(1).strip() if ruling_match else ""

            rp_match = re.search(r'## 裁判要旨：\s*\n\t?(.*?)(?=\n##)', content, re.DOTALL)
            ruling_points = rp_match.group(1).strip() if rp_match else ""

            law_match = re.search(r'## 关联索引：\s*\n\t?(.*?)(?=\n##)', content, re.DOTALL)
            related_laws = law_match.group(1).strip() if law_match else ""

            title_m = re.search(r'^#\s+(.+)', content, re.MULTILINE)
            title = title_m.group(1).strip() if title_m else case_number

            content_body = "\n".join([
                f"【基本案情】{basic_facts}",
                f"【裁判理由】{ruling_reason}",
                f"【裁判要旨】{ruling_points}",
            ]).strip()

            if not ruling_points and not basic_facts:
                continue

            cases.append({
                "id": case_number,
                "case_number": case_number,
                "title": title,
                "court": court,
                "judgment_date": judgment_date,
                "case_type": case_type,
                "cause_of_action": cause,
                "trial_level": trial,
                "province": province,
                "case_id": case_id,
                "content": content_body,
                "metadata": {
                    "keywords": keywords,
                    "ruling_points": ruling_points,
                    "basic_facts": basic_facts,
                    "ruling_reason": ruling_reason,
                    "related_laws": related_laws,
                }
            })

    print(f"民事txt: {len(cases)} 条")
    return cases


# ============================================================
# 2. 已有 cases.json
# ============================================================
def load_existing():
    path = CASES_DIR / "cases.json"
    with open(path, 'r', encoding='utf-8') as f:
        cases = json.load(f)
    print(f"已有案例: {len(cases)} 条")
    return cases


# ============================================================
# 3. 2025年度案例 PDF
# ============================================================
def extract_2025_pdf(pdf_path, category):
    """从2025年度案例PDF中提取所有案例"""
    cases = []
    with pdfplumber.open(pdf_path) as pdf:
        pages = pdf.pages

        # 收集所有文本，按页拆分
        page_texts = [p.extract_text() or "" for p in pages]

        # 找到所有案例起始位置（【案件基本信息】出现的位置）
        case_starts = []
        for i, text in enumerate(page_texts):
            if '【案件基本信息】' in text and len(text) > 50:
                case_starts.append(i)

        print(f"  {category}: 发现 {len(case_starts)} 个案例")

        for idx, start in enumerate(case_starts):
            # 合并从 start 到下一个案例（或末尾）的文本
            end = case_starts[idx + 1] if idx + 1 < len(case_starts) else len(page_texts)
            raw = "\n".join(page_texts[start:end])

            # ---- 标题（通常在"一、对不动产的执行"那段之后）----
            # 找 === 开头的大标题行
            title_m = re.search(r'([^《\n【】]{10,60}?)[\n　 ]*(?:——|—)', raw)
            title = title_m.group(1).strip() if title_m else category

            # ---- 裁判书字号 ----
            case_id_m = re.search(r'裁判书字号\n?(.{10,80})', raw)
            case_id_str = case_id_m.group(1).strip() if case_id_m else ""

            # ---- 案由 ----
            cause_m = re.search(r'案由[：:]\s*([^【\n]+)', raw)
            cause = cause_m.group(1).strip() if cause_m else category

            # ---- 当事人 ----
            party_m = re.search(r'当事人\n?(.*?)(?:【基本案情】|【案件焦点】|【裁判要旨】|【法院裁判要旨】)', raw, re.DOTALL)
            parties = party_m.group(1).strip()[:200] if party_m else ""

            # ---- 基本案情 ----
            facts_m = re.search(r'【基本案情】\n?(.*?)(?:【案件焦点】|【裁判要旨】|【法院裁判要旨】|【法官后语】)', raw, re.DOTALL)
            facts = facts_m.group(1).strip() if facts_m else ""

            # ---- 裁判要旨 ----
            ruling_m = re.search(r'【法院裁判要旨】\n?(.*?)(?:【法官后语】|【案件基本信息】|$)', raw, re.DOTALL)
            if not ruling_m:
                ruling_m = re.search(r'【裁判要旨】\n?(.*?)(?:【法官后语】|【案件基本信息】|$)', raw, re.DOTALL)
            ruling_points = ruling_m.group(1).strip() if ruling_m else ""

            # ---- 案件焦点 ----
            focus_m = re.search(r'【案件焦点】\n?(.*?)(?:【法院裁判要旨】|【裁判要旨】|【法官后语】|【基本案情】)', raw, re.DOTALL)
            focus = focus_m.group(1).strip() if focus_m else ""

            # ---- 关联法规（从要旨中提取）----
            law_refs = re.findall(r'《([^》]+)》', ruling_points[:500])
            related_laws = "、".join(law_refs[:5])

            # ---- 生成 case_number ----
            # 从裁判书字号中提取年月案号
            case_num_m = re.search(r'[(（](\d{4})[^0-9]*?(\d+)[^0-9]*?号[)）]?', case_id_str)
            if case_num_m:
                case_number = f"{case_num_m.group(1)}-18-2-{case_num_m.group(2).zfill(3)}"
            else:
                case_number = f"2025-18-2-{idx:04d}"

            if not facts and not ruling_points:
                continue

            cases.append({
                "id": case_number,
                "case_number": case_number,
                "title": title[:200],
                "court": case_id_str[:80],
                "judgment_date": "",
                "case_type": category,
                "cause_of_action": cause,
                "trial_level": "",
                "province": "",
                "case_id": case_id_str[:100],
                "content": "\n".join([
                    f"【基本案情】{facts[:1000]}",
                    f"【案件焦点】{focus[:300]}",
                    f"【裁判要旨】{ruling_points[:500]}",
                ]).strip(),
                "metadata": {
                    "keywords": [category],
                    "ruling_points": ruling_points[:500],
                    "basic_facts": facts[:1000],
                    "ruling_reason": "",
                    "related_laws": related_laws,
                    "parties": parties,
                    "source": "2025年度案例",
                }
            })

    return cases


def extract_pdf_dir(base_dir, label):
    """扫描目录下的所有PDF（每个子目录一个PDF）"""
    base = Path(base_dir)
    if not base.exists():
        print(f"未找到: {label}")
        return []

    all_cases = []
    for subdir in sorted(base.iterdir()):
        if not subdir.is_dir():
            continue
        pdfs = list(subdir.glob("*.pdf")) + list(subdir.glob("*.PDF"))
        if not pdfs:
            continue
        pdf_path = pdfs[0]
        category = subdir.name.split(" ", 1)[-1] if " " in subdir.name else subdir.name
        print(f"处理: {label} - {category}")
        cases = extract_2025_pdf(str(pdf_path), category)
        all_cases.extend(cases)

    print(f"{label}: 共 {len(all_cases)} 条")
    return all_cases


def extract_2025_pdf_dir(base_dir, label):
    """扫描目录下所有PDF（直接放PDF文件的目录）"""
    base = Path(base_dir)
    if not base.exists():
        print(f"未找到: {label}")
        return []

    all_cases = []
    for pdf_path in sorted(base.glob("*.pdf")) + sorted(base.glob("*.PDF")):
        print(f"处理: {label} - {pdf_path.name}")
        cases = extract_2025_pdf(str(pdf_path), label)
        all_cases.extend(cases)

    print(f"{label}: 共 {len(all_cases)} 条")
    return all_cases


def extract_all_2025():
    """扫描所有2025年度案例PDF"""
    return extract_pdf_dir(SRC_DIR / "中国法院2025年度案例", "2025年度案例")


def extract_case_2026():
    """处理案例2026目录"""
    return extract_pdf_dir(SRC_DIR / "案例2026", "案例2026")


def extract_rmsf_2025():
    """处理人民司法案例2025"""
    return extract_2025_pdf_dir(SRC_DIR / "人民司法案例2025", "人民司法案例2025")


def load_criminal_cases():
    """加载刑事案例JSON（由extract_criminal.py生成）"""
    path = CASES_DIR / "criminal_cases.json"
    if not path.exists():
        print("未找到刑事案例JSON，跳过")
        return []
    with open(path, 'r', encoding='utf-8') as f:
        cases = json.load(f)
    print(f"刑事案例: {len(cases)} 条")
    return cases


def extract_case_2026_pdf(pdf_path, source_label="案例2026"):
    """解析案例2026 PDF（合并全文后按【裁判要旨】分块）"""
    cases = []
    with pdfplumber.open(pdf_path) as pdf:
        texts = [p.extract_text() or "" for p in pdf.pages]

    full_text = "\n".join(texts)
    positions = [(m.start(), m.end()) for m in re.finditer(r"【裁判要旨】", full_text)]

    for idx, (start, end) in enumerate(positions):
        chunk = full_text[max(0, start - 400):start + 800]

        # 标题
        title_m = re.search(r"案例\s*\n(.+?)\n\s*文/", chunk, re.DOTALL)
        title = re.sub(r"\s+", " ", title_m.group(1).strip()) if title_m else ""
        if not title:
            title_m = re.search(r"案例\s*\n(.+)", chunk)
            title = re.sub(r"\s+", " ", title_m.group(1).strip())[:80] if title_m else source_label

        # 案号
        case_id_m = re.search(r"[（(](\d{4})[^0-9]*?(\d+)[^0-9]*?号[)）]", chunk)
        case_number = f"{case_id_m.group(1)}-999-{case_id_m.group(2).zfill(3)}" if case_id_m else f"2026-999-{idx:04d}"

        # 案由
        cause_m = re.search(r"案由[：:]\s*([^【\n]+)", chunk)
        cause = cause_m.group(1).strip() if cause_m else ""

        # 基本案情
        facts_m = re.search(r"【案情】\n?(.*?)(?=【裁判要旨】|$)", chunk, re.DOTALL)
        facts = facts_m.group(1).strip() if facts_m else ""

        # 裁判要旨（chunk已从裁判要旨之后开始，取到【案情】或□案号为止）
        end_idx = re.search(r"\n\s*□\s*案号|【案情】", full_text[end:end + 600])
        ruling = full_text[end:end + 600][:end_idx.start()].strip() if end_idx else full_text[end:end + 600].strip()
        ruling = ruling.split("\n□")[0].strip()

        if not ruling:
            continue

        cases.append({
            "id": case_number,
            "case_number": case_number,
            "title": title[:200],
            "court": "",
            "judgment_date": "",
            "case_type": source_label,
            "cause_of_action": cause,
            "trial_level": "",
            "province": "",
            "case_id": "",
            "content": "\n".join([
                f"【基本案情】{facts[:1000]}",
                f"【裁判要旨】{ruling[:500]}",
            ]).strip(),
            "metadata": {
                "keywords": [source_label],
                "ruling_points": ruling[:500],
                "basic_facts": facts[:1000],
                "ruling_reason": "",
                "related_laws": "",
                "source": source_label,
            }
        })

    return cases


def extract_case_2026():
    """处理案例2026目录"""
    base = SRC_DIR / "案例2026"
    if not base.exists():
        print("未找到案例2026目录")
        return []

    all_cases = []
    for pdf_path in sorted(base.glob("*.pdf")) + sorted(base.glob("*.PDF")):
        label = pdf_path.stem
        print(f"处理: {label}")
        cases = extract_case_2026_pdf(str(pdf_path), label)
        print(f"  -> {len(cases)} 条")
        all_cases.extend(cases)

    print(f"案例2026: 共 {len(all_cases)} 条")
    return all_cases


# ============================================================
# 合并去重
# ============================================================
def merge():
    civil = extract_civil_txt()
    existing = load_existing()
    cases_2025 = extract_all_2025()
    cases_2026 = extract_case_2026()
    cases_rmsf = extract_rmsf_2025()
    criminal = load_criminal_cases()

    seen = {}
    merged = []

    for c in civil:
        key = c['id']
        if key not in seen:
            seen[key] = True
            merged.append(c)

    for c in existing:
        key = c.get('case_number') or c.get('id', '')
        if key and key not in seen:
            seen[key] = True
            merged.append(c)

    for c in cases_2025:
        key = c.get('case_number') or c.get('id', '')
        if key and key not in seen:
            seen[key] = True
            merged.append(c)

    for c in cases_2026:
        key = c.get('case_number') or c.get('id', '')
        if key and key not in seen:
            seen[key] = True
            merged.append(c)

    for c in cases_rmsf:
        key = c.get('case_number') or c.get('id', '')
        if key and key not in seen:
            seen[key] = True
            merged.append(c)

    for c in criminal:
        key = c.get('case_number') or c.get('id', '')
        if key and key not in seen:
            seen[key] = True
            merged.append(c)

    # 按年份排序（从 case_number 或 judgment_date 提取）
    def year_key(c):
        d = c.get('judgment_date', '')
        # 尝试从 case_number 提取
        cn = c.get('case_number', '')
        yr_m = re.search(r'(\d{4})-', cn)
        yr = yr_m.group(1) if yr_m else (d[:4] if len(d) >= 4 else '0')
        try:
            return -int(yr)
        except:
            return 0

    merged.sort(key=year_key, reverse=True)

    # 统计
    from collections import Counter
    types = Counter(c.get('case_type', '未知') for c in merged)
    causes = Counter(c.get('cause_of_action', '未知') for c in merged)

    print(f"\n总计: {len(merged)} 条")
    print(f"类型: {dict(sorted(types.items(), key=lambda x: -x[1])[:6])}")
    print(f"案由TOP5: {dict(sorted(causes.items(), key=lambda x: -x[1])[:5])}")

    # 保存
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"已保存: {OUT_JSON}")
    return merged


if __name__ == "__main__":
    merge()
