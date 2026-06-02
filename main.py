#!/usr/bin/env python3
"""
Railway entry point
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")
CSV_PATH = "data/all_cases_perfect.csv"
PORT = int(os.environ.get("PORT", 5099))

import uvicorn
from fayan_main import create_app, FayanLegalRAG

print(f"API_KEY={'已设置' if MINIMAX_API_KEY else '未设置'}")
fayan = FayanLegalRAG(CSV_PATH)
app = create_app(fayan)
uvicorn.run(app, host="0.0.0.0", port=PORT)