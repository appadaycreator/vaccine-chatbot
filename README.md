# vaccine-chatbot

Ollama（ローカルLLM）とRAG（PDF検索）で、厚労省資料に基づいた質問応答を行うプロトタイプです。

## 前提

- macOS
- Python 3.13（推奨。最低でも Python 3.10 以上）
- Ollama がインストール済み・起動済み

## セットアップ

### 0) Python バージョン確認（重要）

`langchain-*` や `streamlit` は **Python 3.10 以上**が必要です。まずバージョンを確認してください。

```bash
python3 --version
```

もし `Python 3.9.x` など **3.10未満**の場合は、Python 3.13 をインストールして、以降の手順で `python3.13` を使って仮想環境を作成してください。

### 1) 仮想環境（初回のみ）

```bash
python3 -m venv .venv
```

（複数バージョンが入っていて `python3` が 3.10 未満を指す場合は、`python3.13 -m venv .venv` のように明示してください）

### 2) 依存関係のインストール

```bash
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3) Ollama モデル（初回のみ）

```bash
ollama pull gemma2
ollama pull llama3.1
ollama pull nomic-embed-text
```

## 使い方（CLI）

### シンプル版（固定の知識ベース）

```bash
. .venv/bin/activate
python app.py
```

### RAG版（PDF `vaccine_manual.pdf` を検索して回答）

```bash
. .venv/bin/activate
python app_rag.py
```

## 使い方（Web / Streamlit）

```bash
. .venv/bin/activate
streamlit run app_web.py
```

## 使い方（API / FastAPI）

### 起動

```bash
. .venv/bin/activate
export SSL_CERT_FILE="/opt/homebrew/etc/ca-certificates/cert.pem"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
uvicorn api:app --host 0.0.0.0 --port 8000
```

## GitHub Pages（フロントエンド）で叩く

`docs/` に GitHub Pages 用のシンプルなチャット画面（HTML/JS）を置いてあります。

- GitHub Pages を `docs/` から配信するように設定（Settings → Pages → **Deploy from a branch** → **/docs**）
- Pages を HTTPS で開き、API Base URL に `cloudflared` の公開URLを入力してテストします

### Mac mini に cloudflared をインストール

```bash
brew install cloudflared
```

### トンネルの作成（公開URLを取得）

まずAPIサーバーを起動:

```bash
. .venv/bin/activate
export SSL_CERT_FILE="/opt/homebrew/etc/ca-certificates/cert.pem"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
uvicorn api:app --host 127.0.0.1 --port 8000
```

次に別ターミナルで:

```bash
cloudflared tunnel --url http://localhost:8000
```

ログに表示される `https://xxxx.trycloudflare.com` のようなURLを、GitHub Pagesの画面で **API Base URL** に貼り付けて `/chat` が叩けるか確認してください。

### GitHub Pagesからソース（PDF）を追加する

GitHub Pagesの画面にある **「ソース追加（PDFアップロード）」** からPDFをアップロードできます。アップロードされたPDFは **Mac mini側に保存**され、RAGの検索対象に追加されます。

- API: `POST /sources/upload`（multipart）
- 一覧: `GET /sources`

`GET /sources` は **「元のファイル名（資料名）」** を優先して返します（保存時は重複排除のため `sha256_元ファイル名.pdf` のような保存名になります）。

保存先（既定）:

- `./uploads`（環境変数 `UPLOAD_DIR` で変更可能）

## 常時稼働（Mac miniを「止まらないサーバー」にする）

### 0) 先に Ollama を常駐（推奨）

```bash
brew services start ollama
```

### 1) cloudflared を常時起動（named tunnel 推奨）

Quick Tunnel（`cloudflared tunnel --url ...`）はURLが固定されないため、常時稼働には **named tunnel** を推奨します。

Cloudflare Zero Trust のダッシュボードでトンネルを作成して **Token** を取得したら、次を実行します。

```bash
sudo cloudflared service install [YOUR_TOKEN]
```

※ `[YOUR_TOKEN]` は **秘密情報** なのでGitに入れないでください。

### 2) API（uvicorn）を launchd で自動起動

`launchd` により、ログイン/再起動後も `uvicorn` が自動で立ち上がり、落ちても再起動します。

plistは既に作成済みです:

- 実体: `~/Library/LaunchAgents/com.vaccine.api.plist`
- テンプレ（リポジトリ内）: `launchd/com.vaccine.api.plist`

反映:

```bash
launchctl load -w ~/Library/LaunchAgents/com.vaccine.api.plist
```

状態確認:

```bash
launchctl list | grep vaccine
curl -sS http://127.0.0.1:8000/status
```

ログ確認:

- `api.log`
- `api.error.log`

### 3) macOS側の設定（推奨）

システム設定の「省エネルギー」で **「停電後に自動的に起動」** をオンにしておくと、不意の停電時も復帰しやすくなります。

### エンドポイント

- `GET /health`: ヘルスチェック
- `POST /chat`: RAGで回答

`POST /chat` の例:

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"接種後7日間に記録する項目は？","model":"gemma2:2b","k":2,"max_tokens":120,"timeout_s":120}'
```

### ベクトルDB（`chroma_db`）について

APIサーバーは起動時に `./chroma_db` を読み込みます。未作成の場合は、既定で `vaccine_manual.pdf`（または `PDF_PATH` 環境変数で指定したPDF）から自動構築します。

環境変数:

- `PDF_PATH`: 取り込むPDFパス（既定: `vaccine_manual.pdf`）
- `CHROMA_PERSIST_DIR`: Chroma永続化ディレクトリ（既定: `./chroma_db`）

### できること

- **ソース（根拠）の表示**: 回答の下に「参照ページ」を表示し、抜粋を展開できます
- **サイドバー設定**:
  - 回答モデル切替（`gemma2` ↔ `llama3.1`）
  - 検索の強さ（k値）をスライダーで調整
- **PDFアップロード**: 既定の `vaccine_manual.pdf` ではなく、ブラウザからPDFをアップロードして検索対象にできます

※初回はPDF解析・ベクトル化が走るため数分かかる場合があります。

## 注意（PDFについて）

`.gitignore` で `*.pdf` を除外しているため、**PDFはGitにコミットされません**。  
手元に `vaccine_manual.pdf` を置くか、Web版ではサイドバーからPDFをアップロードして利用してください。

## トラブルシューティング

### `No matching distribution found for langchain-community==...` が出る

多くの場合、**仮想環境を作ったPythonが3.10未満**です（例: システムの `python3` が 3.9）。

- `python3 --version` を確認する
- Python 3.10以上（例: 3.13）で作り直す:

```bash
rm -rf .venv
python3.13 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### `SSLError(SSLCertVerificationError('OSStatus -26276'))` が出て `pip install` に失敗する

macOS環境で証明書検証に失敗して PyPI に接続できないケースがあります。Homebrew の CA バンドルを明示すると改善することがあります。

```bash
. .venv/bin/activate
export SSL_CERT_FILE="/opt/homebrew/etc/ca-certificates/cert.pem"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
pip install -r requirements.txt
```

