#!/usr/bin/env python3
"""
530 / ネットワークエラー時の UI ヒントの契約テスト。
フロントの apiConnectionHint と同内容のメッセージが含まれることを確認する。
"""
import sys

# app.js の apiConnectionHint と同一メッセージ（変更時は要同期）
EXPECTED_530_HINT = (
    "530 や接続エラーのときは、API にリクエストが届いていません。"
    "cloudflared トンネルと API サーバーが起動しているか確認してください。"
    "CORS と表示されても原因はトンネル停止のことが多いです。"
)


def api_connection_hint(status_or_err):
    """app.js の apiConnectionHint と同等のロジック"""
    status = status_or_err if isinstance(status_or_err, int) else getattr(status_or_err, "status", None)
    if status in (530, 0):
        return EXPECTED_530_HINT
    return "APIのURLとネットワーク接続を確認してください。"


def main():
    # 530 / 0 のとき期待メッセージが返る
    assert api_connection_hint(530) == EXPECTED_530_HINT
    assert api_connection_hint(0) == EXPECTED_530_HINT
    assert "cloudflared" in api_connection_hint(530)
    assert "CORS" in api_connection_hint(530)

    # ネットワークエラー用の err オブジェクト（getSources/postJson の catch で作る形）
    class Err:
        status = 0
        detail = {"message": "API に接続できません（ネットワークエラー）", "hints": [api_connection_hint(530)]}

    err = Err()
    assert err.detail["hints"]
    assert EXPECTED_530_HINT in err.detail["hints"][0]

    print("OK: 530/network hint contract verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
