"""
法眼AI - 执行案例数据提取 v3
修复：竖排截断 + 跨行section合并
"""

import os
import re
import json
import zipfile
import io
import pdfplumber
from dataclasses import dataclass
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

# ============================================================
# 数据结构
# ============================================================
@dataclass
class ExtractedCase:
    id: str
    case_number: str
    title: str
    court: str
    judgment_date: str
    case_type: str
    cause_of_action: str
    keywords: str
    basic_facts: str
    reasoning: str
    ruling_points: str
    related_laws: str
    full_text: str

    def to_rag_dict(self) -> dict:
        return {
            "id": self.id,
            "type": "case",
            "title": self.title,
            "case_number": self.case_number,
            "court": self.court,
            "judgment_date": self.judgment_date,
            "case_type": self.case_type,
            "cause_of_action": self.cause_of_action,
            "content": self.build_search_content(),
            "metadata": {
                "keywords": self.keywords,
                "ruling_points": self.ruling_points,
                "related_laws": self.related_laws,
            }
        }

    def build_search_content(self) -> str:
        parts = [
            f"【案号】{self.case_number}",
            f"【案由】{self.cause_of_action}",
            f"【关键词】{self.keywords}",
            f"【基本案情】{self.basic_facts}",
            f"【裁判要点】{self.ruling_points}",
        ]
        return "\n".join(parts)


# ============================================================
# 文本清理
# ============================================================
WATERFALL_CHARS = set('库例案院法人民民人人')

def is_noise_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    # 纯水印短行
    if len(stripped) <= 12:
        chars = set(stripped.replace(' ', ''))
        if chars.issubset(WATERFALL_CHARS) and len(chars) <= 3:
            return True
    # 页码
    if re.match(r'^第\s*\d+\s*页$', stripped):
        return True
    return False

def clean_line(line: str) -> str:
    cleaned = ''.join(c for c in line if c not in WATERFALL_CHARS)
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    return cleaned.strip()

def extract_clean_pages(pdf_bytes: bytes) -> list[str]:
    """返回清理后的行列表"""
    all_lines = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
            for line in text.split('\n'):
                if is_noise_line(line):
                    continue
                cleaned = clean_line(line)
                if cleaned:
                    all_lines.append(cleaned)
    return all_lines


# ============================================================
# Section 解析 v2（支持竖排截断 + 跨行合并）
# ============================================================

# 段标题及其常见截断变体
SECTION_PATTERNS = {
    'keywords': ['关键词', '关键 词', '关键 语'],
    'facts': ['基本案情', '基本 情', '基本情', '案 情', '案 情'],
    'reasoning': ['裁判理由', '裁判 理由', '裁判理由', '本院认为', '本院认 为'],
    'ruling': ['裁判要旨', '裁判 要旨', '裁判要旨', '裁判要点', '裁判结果', '裁判 结果'],
    'related': ['关联索引', '关联 索引', '关联索引', '相关法条', '相关 法条', '法律依据'],
}

def make_header_pattern() -> re.Pattern:
    """编译段标题匹配正则"""
    all_headers = []
    for headers in SECTION_PATTERNS.values():
        all_headers.extend(headers)
    pattern = '|'.join(re.escape(h) for h in all_headers)
    return re.compile(pattern)

HEADER_PAT = make_header_pattern()

def find_header_in_line(line: str) -> tuple[str, str, str]:
    """
    在一行中查找段标题。
    返回: (section_type, header_text, rest_content)
    section_type: 'keywords'/'facts'/'reasoning'/'ruling'/'related'/None
    header_text: 匹配的标题文本
    rest_content: 标题后的剩余内容（可能为空）
    """
    match = HEADER_PAT.search(line)
    if not match:
        return None, "", ""

    matched_text = match.group()
    # 找到是哪个section_type
    sec_type = None
    for st, headers in SECTION_PATTERNS.items():
        if matched_text in headers:
            sec_type = st
            break
    if sec_type is None:
        return None, "", ""

    # 提取标题后的内容
    rest = line[match.end():].strip()
    # 去掉开头的冒号、空格
    rest = re.sub(r'^[\s:：\->\.。]+', '', rest)

    return sec_type, matched_text, rest

def build_section_type_lookup():
    """建立 截断变体 -> section_type 的映射"""
    lookup = {}
    for sec_type, headers in SECTION_PATTERNS.items():
        for h in headers:
            lookup[h] = sec_type
    return lookup

HEADER_LOOKUP = build_section_type_lookup()

def infer_section_type(line: str) -> Optional[str]:
    """从行内容推断可能的section_type（用于竖排截断场景）"""
    # 如果行很短且可能是截断的标题
    if len(line) <= 6:
        # 尝试还原：竖排截断通常是 2-4 个字符
        for sec_type, headers in SECTION_PATTERNS.items():
            for h in headers:
                if line in h or h in line:
                    return sec_type
    return None

def split_into_sections_v2(lines: list[str]) -> dict:
    """将行列表分割为sections，支持竖排截断和跨行合并"""
    sections = defaultdict(list)
    current_section = None
    pending_line = None  # 暂存当前行（用于跨行合并）

    i = 0
    while i < len(lines):
        line = lines[i]
        sec_type, header_text, rest_content = find_header_in_line(line)

        if sec_type:
            # 保存上一个section
            if current_section:
                sections[current_section] = current_content
            current_section = sec_type

            # 如果标题后有内容，直接用；否则暂存
            if rest_content and len(rest_content) > 2:
                current_content = [rest_content]
                pending_line = None
            else:
                # 标题单独一行，内容在下一行
                current_content = []
                pending_line = i + 1  # 标记下一行需要合并
            i += 1

        elif pending_line is not None and current_section:
            # 标题单独一行 → 下一行是实际内容
            if i == pending_line:
                # 检查下一行是否也是header（说明上一个标题是"空标题"）
                next_sec_type, _, _ = find_header_in_line(line)
                if next_sec_type:
                    # 标题是空的，跳过，使用当前行作为内容（它其实是metadata）
                    current_content.append(line)
                    pending_line = None
                else:
                    # 正常情况：标题后紧跟内容
                    current_content.append(line)
                    pending_line = None
            else:
                # 不是预期的pending行，正常处理
                if current_content or current_section not in ['keywords']:
                    # 如果有内容就归入当前section
                    if line and not is_noise_line(line):
                        current_content.append(line)
                else:
                    # 当前section无内容 → 把这行归metadata
                    sections.setdefault('metadata', []).append(line)
            i += 1

        else:
            # 普通行 → 加入当前section 或 metadata
            if current_section and current_content is not None:
                if line:
                    current_content.append(line)
            else:
                sections.setdefault('metadata', []).append(line)
            i += 1

    # 保存最后一个section
    if current_section and current_content is not None:
        sections[current_section] = current_content

    return sections


# ============================================================
# 字段提取
# ============================================================
CASE_NUM_PAT = re.compile(r'(\d{4}-\d{2,3}-\d{1,2}-\d{2,6}(?:-\d+)?)')

def extract_case_number(text: str) -> str:
    match = CASE_NUM_PAT.search(text)
    return match.group(1) if match else ""

def extract_date(text: str) -> str:
    patterns = [
        (r'(\d{4})年(\d{1,2})月(\d{1,2})日', '%s-%s-%s'),
        (r'(\d{4})-(\d{1,2})-(\d{1,2})', '%s-%s-%s'),
    ]
    for pat, fmt in patterns:
        m = re.search(pat, text)
        if m and len(m.groups()) == 3:
            return fmt % (m.group(1), m.group(2).zfill(2), m.group(3).zfill(2))
    return ""

def decompress_court(text: str) -> str:
    """还原竖排截断的法院名称"""
    if '人民法院' in text:
        return text
    result = text
    result = re.sub(r'([^院\s（　]{2,10}?)高级(?!人民法院)', r'\1高级人民法院', result)
    result = re.sub(r'([^院\s（　]{2,10}?)中级(?!人民法院)', r'\1中级人民法院', result)
    result = re.sub(r'([^院\s（　]{0,5})最高(?!人民法院)', r'\1最高人民法院', result)
    return result

def extract_court(text: str) -> str:
    # 1. 直接匹配完整法院名
    courts = re.findall(r'([^\s（（、，。：]{2,15}人民法院)', text)
    if courts:
        return courts[0]

    # 2. 截断场景：匹配"省/市+中级/高级"模式，然后还原
    truncated = re.findall(r'([^\s（（、，。：\n]{2,12}(?:中级|高级))(?![人民法院])', text)
    if truncated:
        best = max(truncated, key=len)
        return decompress_court(best)

    # 3. "最高"单独出现 → 还原
    if '最高' in text and '人民法院' not in text:
        m = re.search(r'([^，。：\n\s（（]{0,3}最高)', text)
        if m:
            return decompress_court(m.group(1))

    return ""

def infer_cause_of_action(title: str) -> str:
    title = title or ""
    pairs = [
        ('执行异议', '执行异议'),
        ('执行复议', '执行复议'),
        ('执行实施', '执行实施'),
        ('执行转重整', '执行转重整'),
        ('执行转破产', '执行转破产'),
        ('执行监督', '执行监督'),
        ('财产保全', '财产保全执行'),
        ('委托拍卖', '委托拍卖执行'),
        ('仲裁裁决', '仲裁裁决执行'),
        ('公证债权文书', '公证债权文书执行'),
        ('建设工程', '建设工程执行'),
        ('买卖合同', '买卖合同执行'),
        ('劳动争议', '劳动争议执行'),
        ('借款合同', '借款合同执行'),
        ('刑事', '刑事执行'),
        ('破产', '破产执行'),
        ('担保', '担保执行'),
        ('侵权', '侵权执行'),
        ('租赁', '租赁执行'),
    ]
    for kw, cause in pairs:
        if kw in title:
            return cause
    return "执行案件"

def infer_case_type(title: str) -> str:
    title = title or ""
    if '指导案例' in title or '指导性案例' in title:
        return "指导性案例"
    if '参考案例' in title:
        return "参考案例"
    if '典型案例' in title:
        return "典型案例"
    return "执行案件"

def extract_title_from_metadata(meta_lines: list[str], filename: str) -> str:
    for line in meta_lines:
        if '案' in line and 8 < len(line) < 80:
            return line
    return filename.replace('.pdf', '')

def parse_case_from_text(text: str, filename: str) -> Optional[ExtractedCase]:
    lines = text.split('\n')
    lines = [l.strip() for l in lines if l.strip()]
    if len(lines) < 3:
        return None

    sections = split_into_sections_v2(lines)

    case_number = extract_case_number(text)
    judgment_date = extract_date(text)
    court = extract_court(text)

    meta = sections.get('metadata', [])
    title = extract_title_from_metadata(meta, filename)
    if not title:
        title = filename.replace('.pdf', '')

    cause_of_action = infer_cause_of_action(title)
    case_type = infer_case_type(title)

    keywords = '\n'.join(sections.get('keywords', []))
    basic_facts = '\n'.join(sections.get('facts', []))
    reasoning = '\n'.join(sections.get('reasoning', []))
    ruling_points = '\n'.join(sections.get('ruling', []))
    related_laws = '\n'.join(sections.get('related', []))

    # 清理各section中的metadata残留
    # 如果keywords里混入了案情内容（超过一定长度），说明解析有问题
    if len(keywords) > 200:
        keywords = ""

    id_base = case_number.replace('-', '') if case_number else title
    case_id = f"case_{abs(hash(id_base)) % 100000:05d}"

    return ExtractedCase(
        id=case_id,
        case_number=case_number,
        title=title,
        court=court,
        judgment_date=judgment_date,
        case_type=case_type,
        cause_of_action=cause_of_action,
        keywords=keywords,
        basic_facts=basic_facts,
        reasoning=reasoning,
        ruling_points=ruling_points,
        related_laws=related_laws,
        full_text=text,
    )


# ============================================================
# 批量处理
# ============================================================
def process_all_pdfs(zip_path: str, max_workers: int = 4):
    cases, errors = [], []

    with zipfile.ZipFile(zip_path, 'r') as zf:
        pdfs = sorted([n for n in zf.namelist() if n.endswith('.pdf')])
        total = len(pdfs)
        print(f"待处理: {total} 个PDF")

        def process(pdf_name):
            try:
                pdf_bytes = zf.read(pdf_name)
                lines = extract_clean_pages(pdf_bytes)
                text = '\n'.join(lines)
                case = parse_case_from_text(text, os.path.basename(pdf_name))
                return case, None
            except Exception as e:
                return None, (pdf_name, str(e))

        done = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(process, p): p for p in pdfs}
            for f in as_completed(futures):
                case, err = f.result()
                if case:
                    cases.append(case)
                if err:
                    errors.append(err)
                done += 1
                if done % 50 == 0:
                    print(f"进度: {done}/{total}, 成功: {len(cases)}, 失败: {len(errors)}")

    print(f"\n完成: 成功 {len(cases)}, 失败 {len(errors)}")
    return cases, errors


def export(cases: list[ExtractedCase], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # JSON
    json_path = f"{output_dir}/cases.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump([c.to_rag_dict() for c in cases], f, ensure_ascii=False, indent=2)
    print(f"JSON: {json_path} ({len(cases)} 条)")

    # SQLite
    import sqlite3
    db_path = f"{output_dir}/cases.db"
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS cases (
        id, case_number, title, court, judgment_date, case_type,
        cause_of_action, keywords, basic_facts, reasoning,
        ruling_points, related_laws, full_text
    )""")
    for case in cases:
        c.execute("INSERT OR REPLACE INTO cases VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (case.id, case.case_number, case.title, case.court, case.judgment_date,
             case.case_type, case.cause_of_action, case.keywords, case.basic_facts,
             case.reasoning, case.ruling_points, case.related_laws, case.full_text))
    conn.commit()
    conn.close()
    print(f"SQLite: {db_path}")
    return json_path, db_path


def quick_test(zip_path: str, n: int = 5):
    """快速测试"""
    import zipfile

    with zipfile.ZipFile(zip_path, 'r') as zf:
        pdfs = sorted([n for n in zf.namelist() if n.endswith('.pdf')])[:n]

        for pdf_name in pdfs:
            print("=" * 70)
            print(f"文件: {pdf_name}")
            pdf_bytes = zf.read(pdf_name)
            lines = extract_clean_pages(pdf_bytes)
            text = '\n'.join(lines)

            case = parse_case_from_text(text, os.path.basename(pdf_name))
            if case:
                print(f"案号: {case.case_number}")
                print(f"标题: {case.title}")
                print(f"法院: {case.court}")
                print(f"日期: {case.judgment_date}")
                print(f"案由: {case.cause_of_action}")
                print(f"关键词: {case.keywords[:80] if case.keywords else 'N/A'}")
                print(f"裁判要旨: {case.ruling_points[:150] if case.ruling_points else 'N/A'}")
                print(f"关联法规: {case.related_laws[:100] if case.related_laws else 'N/A'}")
            else:
                print("解析失败")
            print()


if __name__ == "__main__":
    import time

    zip_path = "./执行案例356.zip"
    output_dir = "./extracted_cases"

    print("=" * 60)
    print("法眼AI - 执行案例提取 v3")
    print("=" * 60)

    print("\n[快速测试]\n")
    quick_test(zip_path, n=5)

    print("\n" + "=" * 60)
    print("[批量处理]\n")
    start = time.time()
    cases, errors = process_all_pdfs(zip_path, max_workers=4)
    print(f"耗时: {time.time() - start:.1f}秒")

    cases.sort(key=lambda x: x.judgment_date or "")

    # 字段非空率
    print("\n字段非空率:")
    fields = ['case_number', 'title', 'court', 'judgment_date',
               'cause_of_action', 'keywords', 'ruling_points', 'related_laws']
    for f in fields:
        n = sum(1 for c in cases if getattr(c, f))
        print(f"  {f}: {n}/{len(cases)} ({100*n/len(cases):.0f}%)")

    # 案由分布
    from collections import Counter
    causes = Counter(c.cause_of_action for c in cases)
    print("\n案由分布:")
    for k, v in causes.most_common():
        print(f"  {k}: {v}")

    # 法院分布
    courts = Counter(c.court for c in cases if c.court)
    print("\n法院分布(top10):")
    for k, v in courts.most_common(10):
        print(f"  {k}: {v}")

    print("\n[导出]")
    export(cases, output_dir)

    print("\n样本案例:")
    for c in cases[:3]:
        print(f"\n[{c.id}] {c.title}")
        print(f"  案号: {c.case_number} | 日期: {c.judgment_date} | 法院: {c.court}")
        print(f"  案由: {c.cause_of_action}")
        rp = c.ruling_points[:120] if c.ruling_points else 'N/A'
        print(f"  裁判要旨: {rp}...")
