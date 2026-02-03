#!/usr/bin/env python3
"""
SEARCH_ERROR 時のエラー表示の契約テスト。
API は detail に extra をマージするため、例外は detail.error に入る。
フロントは detail.error または detail.extra.error から「原因」を表示する。
"""
import sys


def stage_label(s):
    return str(s) if s else ""


def summarize_api_error(err):
    """app.js の summarizeApiError と同等のロジック（契約の検証用）"""
    parts = []
    if err and isinstance(err.get("status"), int):
        parts.append(f"状態コード: {err['status']}")
    msg = err.get("message", "") if err else ""
    if msg and msg not in parts:
        parts.append(msg)
    detail = err.get("detail") if err else None
    if detail and isinstance(detail, dict):
        m = detail.get("message") or detail.get("error")
        if m:
            parts.append(str(m))
        stage = detail.get("stage")
        code = detail.get("code")
        extra_parts = [
            f"段階: {stage_label(stage)}" if stage else "",
            f"コード: {code}" if code else "",
        ]
        extra = " / ".join(filter(None, extra_parts))
        if extra:
            parts.append(extra)
        raw_cause = (
            detail.get("extra", {}).get("error")
            if isinstance(detail.get("extra"), dict)
            else None
        ) or detail.get("error")
        cause = str(raw_cause).strip() if raw_cause is not None else ""
        if cause:
            parts.append(f"原因: {cause}")
    return " / ".join(parts) or "不明なエラー"


def main():
    # API が返す実際の形（extra が detail にマージ → detail.error）
    err = {
        "status": 500,
        "message": "Internal Server Error",
        "detail": {
            "message": "類似検索に失敗しました",
            "stage": "search",
            "code": "SEARCH_ERROR",
            "hints": [],
            "error": "readonly database",
        },
    }
    result = summarize_api_error(err)
    if "原因: readonly database" not in result:
        print(f"FAIL: expected 原因 in result. Got: {result}", file=sys.stderr)
        sys.exit(1)
    print("OK: detail.error is shown as 原因")
    return 0


if __name__ == "__main__":
    sys.exit(main())
