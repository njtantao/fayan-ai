#!/usr/bin/env python3
"""
处理流水线：读取 /tmp/parsed_clean_3181.csv
- 每 50 条为一批，调用 OpenRouter LLM 将"原始全文"智能分解为四段
- 每批写入 /tmp/batch_*.json
- 全部完成后合并为 /tmp/parsed_structured.csv

四段结构：
  1. 案件基本信息（案号、当事人、法院、案由、程序）
  2. 事实与理由（案件事实、诉辩主张）
  3. 法院观点（法院对事实与法律的认定）
  4. 裁判结果（判决/裁定主文）

用法：
  # 1. 先填入 OPENROUTER_API_KEY（见下方常量）
  # 2. 运行（默认干跑 DRY_RUN=True，验证流程）
  python3 pipeline_structured.py

  # 正式运行
  DRY_RUN=false python3 pipeline_structured.py
"""

import os
import csv
import json
import time
import math
import asyncio
from pathlib import Path
from typing import Optional

# ============ 配置 ============
INPUT_CSV  = "/tmp/parsed_clean_3181.csv"
OUTPUT_DIR = Path("/tmp")
BATCH_SIZE = 50

# ⚠️  请填入你的 OpenRouter API Key
OPENROUTER_API_KEY = os.environ.get(
    "OPENROUTER_API_KEY",
    "your-api-key-here"
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "anthropic/claude-3.5-sonnet"   # 可按需更换模型

DRY_RUN = os.environ.get("DRY_RUN", "true").lower() != "false"
REQUEST_TIMEOUT = 120   # 单次请求超时（秒）
MAX_RETRIES = 2         # 失败重试次数
RATE_LIMIT_DELAY = 1.0 # 每批之间间隔（秒），防止触发限流

# ============ CSV 字段名 ============
COL_CASE_NO     = "案号"
COL_CASE_NAME   = "案件名称"
COL_COURT       = "法院"
COL_REGION      = "所属地区"
COL_CASE_TYPE   = "案件类型"
COL_CAUSE        = "案由"
COL_PROCEDURE   = "审理程序"
COL_DATE        = "裁判日期"
COL_FACT_REASON = "案件事实与理由_原文"
COL_CLAIM       = "诉辩主张_原文"
COL_VIEW        = "法院观点_原文"
COL_RESULT      = "裁判结果_原文"
COL_FULL_TEXT   = "原始全文"

# ============ 核心 LLM 调用 ============

def build_prompt(full_text: str) -> str:
    return f"""你是一个法律文书结构化助手。请将以下法律裁判文书的【原始全文】智能分解为四个部分，输出标准 JSON（仅 JSON，无其他文字）。

输出格式：
{{
  "basic_info": "案件基本信息（案号、当事人、法院、案由、审理程序等）",
  "fact_claim": "案件事实与理由、诉辩主张",
  "court_view": "法院观点、法律认定与推理",
  "judgment": "裁判结果（判决/裁定主文）"
}}

要求：
- 仅从原文提取和概括，不要自行添加内容
- 如某部分内容为空，输出空字符串 ""
- 保持原文关键信息完整
- JSON 中不要包含转义换行符，用实际换行或空格代替

【原始全文】
{full_text}
"""


def call_openrouter_sync(full_text: str, retry: int = 0) -> dict:
    """同步调用 OpenRouter（使用 httpx）。"""
    import httpx

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://example.com",
        "X-Title": "LegalDocStructurer",
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": build_prompt(full_text)
            }
        ],
        "temperature": 0.1,
        "max_tokens": 4096,
    }

    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            resp = client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()
            # 尝试提取 JSON（可能包裹在 ```json ... ``` 中）
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:] if lines[0].startswith("```") else lines)
                if content.endswith("```"):
                    content = content[:-3]
            return json.loads(content)
    except Exception as e:
        if retry < MAX_RETRIES:
            time.sleep(2 ** retry)
            return call_openrouter_sync(full_text, retry + 1)
        raise RuntimeError(f"OpenRouter 调用失败（{retry+1}次）: {e}") from e


async def call_openrouter_async(full_text: str, retry: int = 0) -> dict:
    """异步调用 OpenRouter（使用 httpx.AsyncClient）。"""
    import httpx

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://example.com",
        "X-Title": "LegalDocStructurer",
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": build_prompt(full_text)
            }
        ],
        "temperature": 0.1,
        "max_tokens": 4096,
    }

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            resp = await client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:] if lines[0].startswith("```") else lines)
                if content.endswith("```"):
                    content = content[:-3]
            return json.loads(content)
    except Exception as e:
        if retry < MAX_RETRIES:
            await asyncio.sleep(2 ** retry)
            return await call_openrouter_async(full_text, retry + 1)
        raise RuntimeError(f"OpenRouter 调用失败（{retry+1}次）: {e}") from e


# ============ 流水线 ============

def load_records(csv_path: str) -> list[dict]:
    """读取 CSV，返回所有记录（含表头）。"""
    records = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records


async def process_batch_async(batch_records: list[dict], batch_idx: int) -> list[dict]:
    """异步并发处理一批（50条）记录。"""
    print(f"[Batch {batch_idx:03d}] 开始处理 {len(batch_records)} 条记录 ...")

    async def process_one(rec: dict) -> dict:
        full_text = rec.get(COL_FULL_TEXT, "") or ""
        if not full_text.strip():
            # 无原文时返回空结构
            return {
                "案号": rec.get(COL_CASE_NO, ""),
                "案件名称": rec.get(COL_CASE_NAME, ""),
                "法院": rec.get(COL_COURT, ""),
                "所属地区": rec.get(COL_REGION, ""),
                "案件类型": rec.get(COL_CASE_TYPE, ""),
                "案由": rec.get(COL_CAUSE, ""),
                "审理程序": rec.get(COL_PROCEDURE, ""),
                "裁判日期": rec.get(COL_DATE, ""),
                "案件事实与理由_原文": rec.get(COL_FACT_REASON, ""),
                "诉辩主张_原文": rec.get(COL_CLAIM, ""),
                "法院观点_原文": rec.get(COL_VIEW, ""),
                "裁判结果_原文": rec.get(COL_RESULT, ""),
                "原始全文": full_text,
                "basic_info": "",
                "fact_claim": "",
                "court_view": "",
                "judgment": "",
                "status": "empty_text",
            }

        try:
            result = await call_openrouter_async(full_text)
            return {
                "案号": rec.get(COL_CASE_NO, ""),
                "案件名称": rec.get(COL_CASE_NAME, ""),
                "法院": rec.get(COL_COURT, ""),
                "所属地区": rec.get(COL_REGION, ""),
                "案件类型": rec.get(COL_CASE_TYPE, ""),
                "案由": rec.get(COL_CAUSE, ""),
                "审理程序": rec.get(COL_PROCEDURE, ""),
                "裁判日期": rec.get(COL_DATE, ""),
                "案件事实与理由_原文": rec.get(COL_FACT_REASON, ""),
                "诉辩主张_原文": rec.get(COL_CLAIM, ""),
                "法院观点_原文": rec.get(COL_VIEW, ""),
                "裁判结果_原文": rec.get(COL_RESULT, ""),
                "原始全文": full_text,
                "basic_info": result.get("basic_info", ""),
                "fact_claim": result.get("fact_claim", ""),
                "court_view": result.get("court_view", ""),
                "judgment": result.get("judgment", ""),
                "status": "success",
            }
        except Exception as e:
            print(f"  [!] 案号 {rec.get(COL_CASE_NO, '?')} 处理失败: {e}")
            return {
                "案号": rec.get(COL_CASE_NO, ""),
                "案件名称": rec.get(COL_CASE_NAME, ""),
                "法院": rec.get(COL_COURT, ""),
                "所属地区": rec.get(COL_REGION, ""),
                "案件类型": rec.get(COL_CASE_TYPE, ""),
                "案由": rec.get(COL_CAUSE, ""),
                "审理程序": rec.get(COL_PROCEDURE, ""),
                "裁判日期": rec.get(COL_DATE, ""),
                "案件事实与理由_原文": rec.get(COL_FACT_REASON, ""),
                "诉辩主张_原文": rec.get(COL_CLAIM, ""),
                "法院观点_原文": rec.get(COL_VIEW, ""),
                "裁判结果_原文": rec.get(COL_RESULT, ""),
                "原始全文": full_text,
                "basic_info": "",
                "fact_claim": "",
                "court_view": "",
                "judgment": "",
                "status": f"error: {e}",
            }

    # 并发处理本批次（限制并发数为 10，避免瞬间请求过多）
    results = []
    semaphore = asyncio.Semaphore(10)

    async def sem_process(rec: dict) -> dict:
        async with semaphore:
            return await process_one(rec)

    tasks = [sem_process(rec) for rec in batch_records]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    return list(results)


def write_batch_json(batch_results: list[dict], batch_idx: int):
    """将一批结果写入 /tmp/batch_XXX.json。"""
    out_path = OUTPUT_DIR / f"batch_{batch_idx:03d}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(batch_results, f, ensure_ascii=False, indent=2)
    success = sum(1 for r in batch_results if r.get("status") == "success")
    empty   = sum(1 for r in batch_results if r.get("status") == "empty_text")
    err     = len(batch_results) - success - empty
    print(f"[Batch {batch_idx:03d}] 写入 {out_path}  成功={success} 空原文={empty} 失败={err}")
    return out_path


async def run_pipeline():
    """主流水线：读取CSV → 分批处理 → 写batch_*.json → 合并为CSV。"""
    print(f"=" * 60)
    print(f"流水线启动 | 输入={INPUT_CSV} | 每批={BATCH_SIZE} | DRY_RUN={DRY_RUN}")
    print(f"=" * 60)

    # 1. 读取
    records = load_records(INPUT_CSV)
    total = len(records)
    n_batch = math.ceil(total / BATCH_SIZE)
    print(f"共读取 {total} 条记录，分为 {n_batch} 批\n")

    # 2. 分批处理
    batch_files = []
    for i in range(n_batch):
        batch_idx = i + 1
        start = i * BATCH_SIZE
        end   = start + BATCH_SIZE
        batch = records[start:end]

        if DRY_RUN and batch_idx > 3:
            print(f"[DryRun] 跳过第 {batch_idx} 批及之后所有批次")
            break

        batch_file = OUTPUT_DIR / f"batch_{batch_idx:03d}.json"

        # 已有结果则跳过（断点续传）
        if batch_file.exists():
            print(f"[Batch {batch_idx:03d}] 已存在，跳过")
            batch_files.append(batch_file)
            continue

        results = await process_batch_async(batch, batch_idx)
        write_batch_json(results, batch_idx)
        batch_files.append(batch_file)

        if batch_idx < n_batch and not DRY_RUN:
            await asyncio.sleep(RATE_LIMIT_DELAY)

    # 3. 合并为 CSV
    print(f"\n合并 {len(batch_files)} 个批次为 CSV ...")
    output_csv = OUTPUT_DIR / "parsed_structured.csv"
    all_results: list[dict] = []

    for bf in sorted(batch_files):
        with open(bf, encoding="utf-8") as f:
            data = json.load(f)
            all_results.extend(data)

    if not all_results:
        print("没有可合并的数据，退出。")
        return

    # CSV 列顺序（保留原始列 + 新增4个结构化列 + status）
    csv_columns = [
        COL_CASE_NO, COL_CASE_NAME, COL_COURT, COL_REGION, COL_CASE_TYPE,
        COL_CAUSE, COL_PROCEDURE, COL_DATE,
        COL_FACT_REASON, COL_CLAIM, COL_VIEW, COL_RESULT, COL_FULL_TEXT,
        "basic_info", "fact_claim", "court_view", "judgment", "status"
    ]

    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)

    success_total = sum(1 for r in all_results if r.get("status") == "success")
    empty_total   = sum(1 for r in all_results if r.get("status") == "empty_text")
    err_total     = len(all_results) - success_total - empty_total

    print(f"\n完成！")
    print(f"  合并后CSV: {output_csv}")
    print(f"  总记录数: {len(all_results)}")
    print(f"  成功: {success_total} | 空原文: {empty_total} | 失败: {err_total}")
    print(f"  各批次JSON: {[str(bf) for bf in batch_files]}")


def main():
    # 参数检查
    if OPENROUTER_API_KEY == "your-api-key-here":
        print("=" * 60)
        print("【重要】请先在脚本中填入 OPENROUTER_API_KEY")
        print("=" * 60)
        print("方式1: 环境变量  export OPENROUTER_API_KEY='sk-...'")
        print("方式2: 直接编辑本脚本顶部的 OPENROUTER_API_KEY 常量")
        print()
        if DRY_RUN:
            print("当前 DRY_RUN=true，仅验证流程不调用 LLM")
        else:
            print("【错误】未配置 API Key 且 DRY_RUN=false，无法运行")
            return

    asyncio.run(run_pipeline())


if __name__ == "__main__":
    main()
