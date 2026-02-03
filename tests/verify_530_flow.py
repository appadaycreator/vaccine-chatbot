#!/usr/bin/env python3
"""530 フローの契約チェック。app.js に必要な文字列が含まれることを確認する。"""
import sys
from pathlib import Path

APP_JS = Path(__file__).resolve().parent.parent / "docs" / "app.js"
content = APP_JS.read_text(encoding="utf-8")

checks = []
if "show530Banner" not in content:
    checks.append("FAIL: show530Banner が app.js に存在しません")
else:
    checks.append("OK: show530Banner が定義されている")

if "banner530" not in content:
    checks.append("FAIL: banner530 が app.js に存在しません")
else:
    checks.append("OK: banner530 バナーが実装されている")

if "isLikely530OrNetworkError" not in content:
    checks.append("FAIL: isLikely530OrNetworkError が app.js に存在しません")
else:
    checks.append("OK: isLikely530OrNetworkError が定義されている")

if "err.status = 0" not in content or "API に接続できません" not in content:
    checks.append("FAIL: fetch 失敗時に err.status = 0 とメッセージを設定しているか確認")
else:
    checks.append("OK: fetch 失敗時に err.status = 0 とメッセージを設定")

failed = any(c.startswith("FAIL") for c in checks)
for c in checks:
    print(c)
sys.exit(1 if failed else 0)
