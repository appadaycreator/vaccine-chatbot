# vaccine-chatbot

Ollama（ローカルLLM）とRAG（PDF検索）で、厚労省資料に基づいた質問応答を行うプロトタイプです。

## 回答品質の下限（医療系としての“言い方”と根拠の強制）

本プロジェクトは、**断定・誤誘導・根拠不明回答を減らす**ため、回答の構造を固定しています（横展開: **API / Streamlit**）。

- **回答フォーマット固定**: つねに `結論 / 根拠 / 相談先` の3セクションで返します
- **根拠が取れない場合は必ず「資料にない」**: `sources` が空（=参照PDFから該当箇所が取れない）場合は、LLM生成を行わず固定文を返します
- **次の行動（相談先）を必ず提示**: 相談先（医療機関・自治体窓口・緊急時の119）を必ず含めます
- **免責と相談導線の常設（UI）**: GitHub Pages（`docs/`）と Streamlit で「診断ではない」旨を常設表示します

## 利用技術（システム構成）

- **GitHub（リポジトリ上）**: `docs/tech.html`
- **GitHub Pages（配信）**: Pages のURL配下の `tech.html`（例: `.../tech.html`）

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

### Streamlit UI の操作（最低限のUX）

- **例ボタン**: よくある質問をワンクリック送信します
- **再送**: 最後の質問をもう一度送ります（不安定なときのリトライ用）
- **エラー表示**: 画面上に要点を表示し、ログ全文は展開して確認できます

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
- `recent_errors`: 直近エラー（最大20件、簡易）
- `error_counts`: エラーコード別の累計カウント（簡易）

例:

```bash
curl -sS http://127.0.0.1:8000/status | python -m json.tool
```

## GitHub Pages（フロントエンド）で叩く

`docs/` に GitHub Pages 用のシンプルなチャット画面（HTML/JS）を置いてあります。
画面上部の **「環境チェック」** で、Ollama未起動・モデル未DLなどの“動かない原因”を確認できます。

- GitHub Pages を `docs/` から配信するように設定（Settings → Pages → **Deploy from a branch** → **/docs**）
- Pages を HTTPS で開き、API Base URL に `cloudflared` の公開URLを入力してテストします（初期値は空です）

### 静的チャットUI（docs/）の操作（最低限のUX）

- **送信中は多重実行しない**: 送信ボタン/入力欄を無効化（キャンセル可能）
- **Enterで送信 / Shift+Enterで改行**
- **入力履歴**: 入力欄の先頭/末尾で **↑↓** を押すと履歴を呼び出せます
- **再送**: 最後の質問を「再送」でリトライできます
- **エラー表示**: エラーはメッセージカードで **要点＋対処** を提示し、ログ全文は折りたたみ表示します
- **Markdown表示**: AIの回答はMarkdownとして表示し、箇条書き等が崩れないようにしています（marked が利用できない場合はプレーン表示にフォールバック）

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

フロント（`docs/app.js`）は主に `POST /chat` を叩きますが、起動時に **環境チェック（`GET /diagnostics`）** と
**参照ソース確認（`GET /sources`）** も行います。  
これらが 404 の場合は、API Base URL が **FastAPI（uvicorn）ではなく別サービス（例: Streamlit 等）**を指している可能性が高いです。

- まず確認: `GET /health` と `GET /status` が返るか
- 404 になる場合: `cloudflared tunnel --url http://localhost:8000` で発行されたURLを貼り直してください

### PDFを追加する（アップロード機能は無し）

ブラウザからのPDFアップロードは使わず、**Mac mini 側でPDFを配置して参照**します。

- **正（唯一の置き場所）**: `PDF_DIR`（既定: `./pdfs/`）
- **任意（補助）**: `PDF_PATH`（既定: `vaccine_manual.pdf`）を指定すると、その単体PDFも参照対象に含めます
- **旧互換（非推奨 / 既定で無効）**: どうしても旧構成を参照したい場合のみ、`ENABLE_LEGACY_UPLOADS=1` と `LEGACY_PDF_DIR` を設定してください  
  - 例: `ENABLE_LEGACY_UPLOADS=1 LEGACY_PDF_DIR=./uploads`
  - これは**アップロード機能ではありません**（管理UI/保存期限/サイズ制限/認証は提供しません）

配置後の反映:

- 自動: 次回の `/chat` 実行時に、PDFの増減/更新を検知して自動で再インデックスします
- 手動: `POST /reload` で再インデックスを開始できます（**非同期**で受け付けます）
  - すでに実行中の場合は **409**（多重実行防止）
  - 実行状況は `GET /sources` の `indexing.running` で確認できます

一覧/状態: `GET /sources`
  - `indexed`: インデックス完了（UIは未完了だと送信不可）
  - `indexing.running`: 実行中フラグ（少なくとも「実行中か」をユーザーに見える形で提示）
  - `error` / `next_actions`: 失敗理由（OCR不足など）と「次にやること」
  - `items[].ingest`: PDFごとの取り込み状況（`ok` / `ocr_needed` / `error`）

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

`api.log` は **JSONの構造化ログ** で出ます（stdout）。`api.error.log` は **1行で見やすいエラーログ**（stderr）です。  
起動直後に `api.log` へ、起動設定（`PDF_DIR` / `CHROMA_PERSIST_DIR` 等）と `git_sha` / `started_at` が1行JSONで出ます。  
**「修正したのに挙動が変わらない」** と感じたら、まず `/status` の `git_sha` が想定どおりか確認してください。

#### 実働コード（ブランチ/コミット）を固定する運用

- 本番の実働は **launchd が `uvicorn api:app` をこのリポジトリの作業ディレクトリで起動**します（テンプレ: `launchd/com.vaccine.api.plist`）。
- コード更新（`git pull` 等）をしたら、**プロセスを再起動しない限り動作は変わりません**。
  - 目視の唯一の正: `GET /status` の `git_sha` / `started_at`
  - 反映: `launchctl kickstart -k gui/$UID/com.vaccine.api`（または unload/load）

#### request_id と工程時間（障害切り分け）

“遅い/落ちた/答えない” を短時間で原因特定できるよう、APIは次をログ化します（横展開: **API**）。

- **request_id**: すべてのリクエスト/レスポンスに `X-Request-ID` を付与
  - クライアントが `X-Request-ID` を送ればそれを採用、無ければサーバーが生成します
  - 失敗時は `api.error.log` の `request_id=...` で1回の失敗を追跡できます
- **工程時間**（例: `/chat`）:
  - `index_check_ms`: インデックス確認（PDF差分チェック/必要なら再インデックス）
  - `embedding_ms`: クエリのベクトル化
  - `search_ms`: 類似検索（ベクトルDB）
  - `generate_ms`: 生成（LLM）
  - `total_ms`: 合計
- **失敗理由**: `stage` / `code` / `message`（例: `EMBEDDING_TIMEOUT`, `SEARCH_TIMEOUT`, `GENERATE_TIMEOUT`, `INDEX_CHECK_TIMEOUT` など）

補足:

- `INDEX_CHECK_TIMEOUT_S`（環境変数）で **index確認のタイムアウト**を調整できます（既定: 120秒）

### 3) macOS側の設定（推奨）

システム設定の「省エネルギー」で **「停電後に自動的に起動」** をオンにしておくと、不意の停電時も復帰しやすくなります。

### エンドポイント

- `GET /health`: ヘルスチェック
- `GET /status`: 状態確認（PDF数、初期化エラー、timings、embeddingウォームアップ/キャッシュなど）
- `GET /diagnostics`: 環境チェック（Ollama疎通/応答時間、モデル有無、embedding/生成のスモークテスト。UIの「環境チェック」で表示）
- `GET /sources`: 参照PDF一覧＋インデックス状態（実行中/最終成功/エラー/次にやること）
- `POST /reload`: 再インデックス開始（非同期。実行中は409で多重実行防止）
- `POST /search`: 検索（embedding → 類似検索）
- `POST /generate`: 生成（LLM応答）
- `POST /chat`: 検索＋生成（フロントの既定ルート。内部的には `/search`→`/generate` 相当）

補足:

- 再インデックス実行中（`GET /sources` の `indexing.running=true`）は、`/chat` は **503** を返します（待たされずに「いま答えられない理由」が分かるようにするため）

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
### まず「なぜ動かないか」を確認する（推奨）

GitHub Pages の画面にある **「環境チェック」**（APIの `GET /diagnostics`）で、次を診断できます。

- **Ollama疎通**: Ollama が起動していない/到達できない
- **生成モデルの有無**: 選択中モデルが未インストール（例: `ollama pull gemma2:2b`）
- **Embeddingモデルの有無/動作**: `nomic-embed-text` が未インストール、または embedding が失敗している（RAGが動かない）
- **生成の動作**: 生成がタイムアウト/失敗している（初回起動・高負荷・モデル不備など）

コマンド例:

```bash
curl -sS http://127.0.0.1:8000/diagnostics | python -m json.tool
```

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

