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

### 開発（コード変更を即時反映したい場合）

開発時は **launchd は使わず**、別手順で `uvicorn --reload` を使って起動します。

```bash
. .venv/bin/activate
export RUN_MODE=dev
uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

### 状態確認（「今の実行コード」を断定）

`GET /status` で、**実行中APIがどのコード（gitコミット）で、いつ起動したか**が分かります。

- `version`: APIのバージョン（FastAPIの `app.version`）
- `git_sha`: 実行コードの git commit hash（取得できない場合は `unknown`）
- `started_at`: APIの起動時刻（UTC）
- `run_mode`: `RUN_MODE`（`prod` / `dev` 等）

例:

```bash
curl -sS http://127.0.0.1:8000/status | python -m json.tool
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

### 404（Not Found）が出る場合

フロント（`docs/app.js`）は **基本的に `POST /chat` のみ**を叩きます。  
それでも 404 が出る場合は、API Base URL が **FastAPI（uvicorn）ではなく別サービス（例: Streamlit 等）**を指している可能性が高いです。

- まず確認: `GET /health` と `GET /status` が返るか
- 404 になる場合: `cloudflared tunnel --url http://localhost:8000` で発行されたURLを貼り直してください

### PDFを追加する（アップロード機能は無し）

ブラウザからのPDFアップロードは使わず、**Mac mini 側でPDFを配置して参照**します。

- 既定の配置先: `./pdfs/`（環境変数 `PDF_DIR` で変更可能）
- 既定の単体PDF: `vaccine_manual.pdf`（環境変数 `PDF_PATH` で変更可能）
  - 互換: `./uploads/` に置かれたPDFも参照対象に含めます（旧構成の名残で、アップロード機能ではありません）

配置後の反映:

- 自動: 次回の `/chat` 実行時に、PDFの増減/更新を検知して自動で再インデックスします
- 手動: `POST /reload` で強制的に再インデックスできます

一覧: `GET /sources`

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

起動直後に `api.log` へ、起動設定（`PDF_DIR` / `CHROMA_PERSIST_DIR` 等）と `git_sha` / `started_at` が1行JSONで出ます。  
**「修正したのに挙動が変わらない」** と感じたら、まず `/status` の `git_sha` が想定どおりか確認してください。

### 3) macOS側の設定（推奨）

システム設定の「省エネルギー」で **「停電後に自動的に起動」** をオンにしておくと、不意の停電時も復帰しやすくなります。

### エンドポイント

- `GET /health`: ヘルスチェック
- `GET /status`: 状態確認（PDF数、初期化エラー、timings、embeddingウォームアップ/キャッシュなど）
- `GET /sources`: 参照するPDF一覧
- `POST /reload`: PDFを再インデックス（強制）
- `POST /search`: 検索（embedding → 類似検索）
- `POST /generate`: 生成（LLM応答）
- `POST /chat`: 検索＋生成（フロントの既定ルート。内部的には `/search`→`/generate` 相当）

`POST /chat` の例:

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"接種後7日間に記録する項目は？","model":"gemma2:2b","k":2,"max_tokens":120,"timeout_s":240,"embedding_timeout_s":240,"search_timeout_s":120,"generate_timeout_s":240}'
```

`POST /search`（検索のみ）:

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"接種後7日間に記録する項目は？","k":2,"embedding_timeout_s":240,"search_timeout_s":120}'
```

`POST /generate`（生成のみ。`/search` の `context` を渡す）:

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"接種後7日間に記録する項目は？","model":"gemma2:2b","max_tokens":120,"generate_timeout_s":240,"context":"..."}'
```

### ベクトルDB（`chroma_db`）について

APIサーバーは起動時に `./chroma_db` を読み込みます。未作成の場合は、既定で `vaccine_manual.pdf`（または `PDF_PATH` 環境変数で指定したPDF）から自動構築します。

環境変数:

- `PDF_PATH`: 取り込むPDFパス（既定: `vaccine_manual.pdf`）
- `PDF_DIR`: 取り込むPDFディレクトリ（既定: `./pdfs`）
- `CHROMA_PERSIST_DIR`: Chroma永続化ディレクトリ（既定: `./chroma_db`）

### できること

- **ソース（根拠）の表示**: 回答の下に「参照ページ」を表示し、抜粋を展開できます
- **サイドバー設定**:
  - 回答モデル切替（`gemma2` ↔ `llama3.1`）
  - 検索の強さ（k値）をスライダーで調整
- **PDF配置で知識追加**: `./pdfs/` にPDFを置くと、次回の実行時に自動で再インデックスして検索対象に反映します

※初回はPDF解析・ベクトル化が走るため数分かかる場合があります。

## 注意（PDFについて）

`.gitignore` で `*.pdf` を除外しているため、**PDFはGitにコミットされません**。  
手元に `vaccine_manual.pdf` を置くか、`./pdfs/` にPDFを配置して利用してください。

## トラブルシューティング
### 検索や生成が遅い / タイムアウトする

遅い環境（Mac mini 等）でも「待てば返る」「どこが遅いか分かる」ように、処理を分割しています。

- まず `GET /status` を確認（`ready` / `init_error` / `pdf_count` / `embed_warmup` / `last_request_timings`）
- **embedding が遅い/失敗**:
  - Ollama が起動しているか確認（例: `brew services start ollama`）
  - `nomic-embed-text` が入っているか確認（`ollama pull nomic-embed-text`）
  - 初回はモデル起動で遅くなります。しばらく待つか `embedding_timeout_s` を延長
- **類似検索が遅い/失敗**:
  - PDF数が多い場合は時間がかかります（`pdf_count` を確認）
  - `k` を小さくして試す
  - PDFを追加/更新したら `POST /reload` で再インデックス
- **生成が遅い/失敗**:
  - 軽量モデル（例: `gemma2:2b`）を試す
  - 初回はモデル起動で遅くなります。しばらく待つか `generate_timeout_s` を延長


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

