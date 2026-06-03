"""
合并 extracted_cases 文件夹下所有 JSON 文件，统一去重
输出: /Users/tt/Desktop/hermes/项目开发/法眼ai/extracted_cases/merged_all_cases.json
"""
import json, os
from pathlib import Path

# 输入文件
INPUT_DIR = Path("/Users/tt/Desktop/hermes/项目开发/法眼ai/extracted_cases")
NEW_CASES_DIR = Path("/Users/tt/Desktop/hermes/项目开发/法眼ai代码/extracted_cases")

FILES = {
    "原all_cases": INPUT_DIR / "all_cases.json",
    "annual_2025": INPUT_DIR / "annual_2025.json",
    "cases": INPUT_DIR / "cases.json",
    "criminal_cases": INPUT_DIR / "criminal_cases.json",
    "guidance_cases": INPUT_DIR / "guidance_cases.json",
    "新all_cases": NEW_CASES_DIR / "all_cases.json",
}

OUT_PATH = INPUT_DIR / "merged_all_cases.json"

# 字段统一映射
FIELD_MAP = {
    # 目标字段: [可能的源字段列表，按优先级]
    "id": ["id", "case_number"],
    "case_number": ["case_number", "id"],
    "title": ["title"],
    "court": ["court"],
    "judgment_date": ["judgment_date"],
    "case_type": ["case_type", "type"],
    "cause_of_action": ["cause_of_action", "cause"],
    "trial_level": ["trial_level"],
    "province": ["province"],
    "content": ["content", "full_text", "facts"],
    "metadata": ["metadata"],
}

def normalize_case(case_dict, source_name):
    """将任意格式的案例标准化"""
    out = {}
    for target, sources in FIELD_MAP.items():
        for src in sources:
            if src in case_dict and case_dict[src]:
                out[target] = case_dict[src]
                break
    # 确保有id
    if "id" not in out or not out["id"]:
        out["id"] = f"unknown_{hash(str(case_dict))}"
    # 确保有content
    if "content" not in out or not out["content"]:
        for k in ["full_text", "facts", "ruling_points"]:
            if k in case_dict and case_dict[k]:
                out["content"] = case_dict[k]
                break
    out["_source"] = source_name
    return out

def make_key(case):
    """生成去重键"""
    k = case.get("id") or case.get("case_number") or ""
    return k.strip()

def main():
    all_cases = {}
    
    for name, path in FILES.items():
        if not path.exists():
            print(f"SKIP  {name}: 文件不存在")
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"ERR   {name}: {e}")
            continue
        
        if isinstance(data, dict) and "cases" in data:
            data = data["cases"]
        
        count = 0
        for item in data:
            if not isinstance(item, dict):
                continue
            norm = normalize_case(item, name)
            key = make_key(norm)
            if key and (key not in all_cases or all_cases[key].get("_source") != name):
                all_cases[key] = norm
                count += 1
        print(f"{name}: {count} 条入库")

    print(f"\n去重后合计: {len(all_cases)} 条")
    
    result = list(all_cases.values())
    
    # 统计来源
    from collections import Counter
    sources = Counter(c["_source"] for c in result)
    print("来源分布:")
    for s, n in sources.most_common():
        print(f"  {s}: {n}")
    
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n已保存到: {OUT_PATH}")

if __name__ == "__main__":
    main()