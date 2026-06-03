# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['fayan_main.py'],
    pathex=[],
    binaries=[],
    datas=[
        # 前端模板
        ('web_app/templates', 'templates'),
        # 案例库
        ('extracted_cases', 'extracted_cases'),
        # Web app 目录（包含 .env 和 fayan_api.py）
        ('web_app/.env', 'web_app'),
        ('web_app/fayan_api.py', 'web_app'),
    ],
    hiddenimports=[
        'flask', 'flask_cors', 'flask.env',
        'dotenv',
        'langchain_openai',
        'langchain_core',
        'openai',
        'jieba',
        'numpy',
        'sklearn', 'sklearn.feature_extraction', 'sklearn.metrics.pairwise',
        'rank_bm25',
        'scipy', 'scipy.sparse',
    ],
    hookspath=[],
    hooksconfig={},
    keys=[],
    exclude_binaries=False,
    name='法眼AI',
    debug=False,
    bootloader_ignore_signals=False,
    console=False,   # Windows GUI 模式，无黑窗口
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='法眼AI',
    debug=False,
    bootloader_ignore_signals=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx_exclude=[],
    name='法眼AI',
)
