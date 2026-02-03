# 530 を解消する手順（この PC で実施）

## 1. API が動いているか確認

```bash
curl -s http://127.0.0.1:8000/health
```

`{"status":"ok","ready":true}` が返れば OK。  
動いていなければ:

```bash
cd /Users/masayukitokunaga/workspace/vaccine-chatbot
source .venv/bin/activate
uvicorn api:app --host 127.0.0.1 --port 8000
```

（別ターミナルで実行するか、バックグラウンドで起動してください。）

## 2. cloudflared で新しいトンネル URL を取得

**新しいターミナル**で:

```bash
cloudflared tunnel --url http://localhost:8000
```

数秒すると、次のような行が表示されます:

```
https://xxxx-xx-xx-xx-xx.trycloudflare.com
```

この **https:// から .trycloudflare.com まで** をコピーしてください。

## 3. GitHub Pages の画面で設定

1. https://appadaycreator.github.io/vaccine-chatbot/ を開く
2. 「接続先（APIのURL）」を開く
3. 「APIのURL」に **コピーした URL を貼り付け**（末尾スラッシュは不要）
4. 「保存」をクリック

これで API に届く状態になり、コンソールの 530/CORS は出なくなります。

---

**注意**: Quick Tunnel（`cloudflared tunnel --url ...`）は **毎回 URL が変わります**。  
トンネルをやり直したら、必ず新しい URL をコピーして GitHub Pages 側で貼り直してください。
