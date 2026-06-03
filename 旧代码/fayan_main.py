"""
法眼AI - 打包入口
双击法眼AI.exe/mac法眼AI.app即可运行
"""
import sys
import os

# 获取可执行文件所在目录
if getattr(sys, 'frozen', False):
    APP_DIR = os.path.dirname(sys.executable)
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))

os.chdir(APP_DIR)
sys.path.insert(0, os.path.join(APP_DIR, 'web_app'))

import app as flask_app
flask_app.app.run(host='0.0.0.0', port=5099, debug=False)
