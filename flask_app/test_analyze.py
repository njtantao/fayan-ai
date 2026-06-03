#!/usr/bin/env python3
"""测试 analyze API 直接调用"""
import sys, os, warnings
warnings.filterwarnings('ignore')

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()

MINIMAX_API_KEY = os.environ.get('MINIMAX_API_KEY', '')
CSV_PATH = os.path.join(PROJECT_ROOT, 'data', 'all_cases_perfect.csv')

print(f"API_KEY set: {bool(MINIMAX_API_KEY)}")
print(f"CSV_PATH: {CSV_PATH}")
print(f"CSV exists: {os.path.exists(CSV_PATH)}")

from fayan_main import FaYanLegal, MINIMAX_BASE_URL, LLM_MODEL
print("FaYanLegal imported OK")

fayan = FaYanLegal(
    api_key=MINIMAX_API_KEY,
    base_url=MINIMAX_BASE_URL,
    model=LLM_MODEL,
    csv_path=CSV_PATH
)
print(f"FaYanLegal initialized, cases: {len(fayan.retriever.cases)}")

print("\n--- Running analyze ---")
result = fayan.analyze(
    case_text='甲向乙借款10万元，约定一年后还款，月利率2%，到期后甲无力还款，乙起诉甲',
    amount=100000,
    party_count=2
)
print(f"Analyze done, lawyer_referral={result.lawyer_referral}")

d = fayan.to_dict(result)
print(f"Conclusions: {len(d.get('conclusions', []))}")
if d.get('conclusions'):
    print(f"First conclusion: {d['conclusions'][0]['content'][:100]}")
print(f"Trace ID: {d.get('trace_id')}")
print("\nDONE")