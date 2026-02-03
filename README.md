# vaccine-chatbot

Ollama（ローカルLLM）とRAG（PDF検索）で、参照PDF（厚労省等の資料）に基づいた質問応答を行うプロトタイプです。

## 回答の方針（資料にある範囲のみ）

本プロジェクトは、**資料に書かれている内容に限定して回答**します（横展開: **API / Streamlit / docs UI / CLI**）。

- **回答本文**: 普通のチャット会話として返します（`結論/根拠/相談先` などの固定見出しは出しません）
- **資料にない場合**: `sources` が空（=参照PDFから該当箇所を特定できない）場合は、LLM生成を行わず固定文（「資料にない」）を返します
- **一般論は返さない**: 「一般的には…」のような一般論・推測による補完は行いません
- **根拠の提示**: 回答本文とは別に、UIが `sources`（資料名/ページ/抜粋）を表示します

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
# 生成（回答）モデル: どれか1つ以上
# 例: 迷ったらまずはこれ（タグは gemma2:latest 等になることがあります）
ollama pull gemma2
# 例: 軽量モデルを明示する場合（環境により gemma2:2b を選びたい場合）
# ollama pull gemma2:2b
# 例: 追加で別モデルも使う場合
ollama pull llama3.1

# embedding（RAG / PDF検索に必須）
ollama pull nomic-embed-text
```

補足:

- **RAG（PDF検索）には embedding モデル（`nomic-embed-text`）が必須**です。未導入だと環境チェックで **「赤: 要対応」** になり、検索できません。
- 対処:
  - `ollama pull nomic-embed-text`
  - 生成モデルは、`gemma2` のような「タグ無し名」で pull/run しても、Ollama 側で `:latest` 等のタグが付くことがあります（UIはインストール済み名を候補表示します）。

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

- **再送**: 最後の質問をもう一度送ります（不安定なときのリトライ用）
- **会話ログをコピー**: 画面の会話ログをまとめてクリップボードにコピーできます
- **エラー表示**: 画面上に要点を表示し、ログ全文は展開して確認できます

## 使い方（API / FastAPI）

### 推奨: 1つのURLで完結（API配下の <code>/ui</code>）

FastAPI が静的UI（`docs/`）を配信するため、**`http://127.0.0.1:8000/ui/` を開くだけで**チャットまで到達できます。  
この導線では **同一オリジン**で `POST /chat` 等を呼ぶため、Mixed Content や「APIのURL貼り替え」が不要です。

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
画面上部の **「環境チェック」** で、Ollamaが未起動・モデル未ダウンロードなどの“動かない原因”を確認できます。

- GitHub Pages を `docs/` から配信するように設定（Settings → Pages → **Deploy from a branch** → **/docs**）
- Pages を HTTPS で開き、APIのURL に `cloudflared` の公開URLを入力してテストします（初期値は空です）

### 静的チャットUI（docs/）の操作（最低限のUX）

- **ファーストビュー最適化**: チャットを最上部に配置し、接続先/設定/環境チェック/参照ソース/免責はアコーディオンで折りたたみます（API未設定時は「接続先」を自動で開きます）
- **主要操作はキーボードだけで完結**: スキップリンク（「チャット入力へ移動」）で入力欄へスクロールし、**入力欄へフォーカス**も当てます（モバイルでも入力開始しやすい）。フォーカス可視化（Tab移動）も実装
- **送信中は多重実行しない**: 送信ボタンを無効化（入力欄は送信中のみ読み取り専用、キャンセル可能）
- **処理段階の表示**: `/chat` の処理を **埋め込み → 検索 → 生成** の段階に分けて「いま何をしているか」を表示します（目安。直近の処理時間（内訳）をもとに推定）
- **タイムアウト後の自己解決**: 失敗時は `detail.stage` を見て **次に試すこと**（例: `k` を下げる／軽量モデルへ切替）をボタンで提示します
- **Enterキーで送信 / Shift+Enterで改行**
- **入力履歴**: 入力欄の先頭/末尾で **↑↓** を押すと履歴を呼び出せます
- **再送**: 最後の質問を「再送」でリトライできます
- **会話ログをコピー**: 「会話ログをコピー」ボタンで、画面の会話ログをまとめてクリップボードにコピーできます（根拠のページラベルも同梱）
- **エラー表示**: エラーはメッセージカードで **要点＋対処** を提示し、ログ全文は折りたたみ表示します
- **Markdown表示**: AIの回答はMarkdownとして表示し、箇条書き等が崩れないようにしています（marked が利用できない場合はプレーン表示にフォールバック）
  - ※回答は資料にある範囲に限定します。該当箇所が見つからない場合は「資料にない」と返します。

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

ログに表示される `https://xxxx.trycloudflare.com` のようなURLを、GitHub Pagesの画面で **APIのURL** に貼り付けて `/chat` が叩けるか確認してください。

### 404（Not Found）が出る場合

フロント（`docs/app.js`）は主に `POST /chat` を叩きますが、起動時に **環境チェック（`GET /diagnostics`）** と
**参照ソース確認（`GET /sources`）** も行います。  
これらが 404 の場合は、APIのURL が **FastAPI（uvicorn）ではなく別サービス（例: Streamlit 等）**を指している可能性が高いです。

- まず確認: `GET /health` と `GET /status` が返るか
- 404 になる場合: `cloudflared tunnel --url http://localhost:8000` で発行されたURLを貼り直してください
- 補足: **環境チェック（`/diagnostics`）が404だったURLは、次回のページ再読み込みでは自動的にスキップ**します（コンソールが汚れないようにするため）。URLを直した/サーバーを更新した場合は、APIのURLを入力し直して「保存」を押すと再判定します。

### CORSエラーに見えるが、実はAPIの500（Internal Server Error）の場合

ブラウザのコンソールで次のように見えることがあります:

- `blocked by CORS policy: No 'Access-Control-Allow-Origin' header ...`
- `GET https://xxxx.trycloudflare.com/sources net::ERR_FAILED 500`

このケースは「CORS設定が無い」よりも、**APIが500を返していて（またはトンネル先が落ちていて）結果的にCORSに見えている**ことが多いです。
まずAPI自体が生きているか、次の順で確認してください:

- `GET /health`（200が返るか）
- `GET /status`（200が返るか）
- `GET /sources`（200が返るか。UIは起動時に `/sources` を呼びます）

補足:

- GitHub Pages から使う場合でも、API側は CORS を許可している必要があります（本リポジトリの FastAPI は CORS を有効化しています）。
- **推奨導線**（CORS回避）: API配下の `http://127.0.0.1:8000/ui/`（または公開URLの `/ui/`）で開くと、同一オリジンになり CORS の影響を受けません。

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
- **失敗理由**: `stage` / `code` / `message`（例: `EMBEDDING_TIMEOUT`, `SEARCH_TIMEOUT`, `SEARCH_ERROR`, `GENERATE_TIMEOUT`, `INDEX_CHECK_TIMEOUT` など）
  - `hints` は常に配列で返します（UIが一貫して「対処」を表示できるようにするため）
  - **検索で 500（SEARCH_ERROR）** のとき: 下記「検索で 500 のときの対処手順」を参照してください

**なぜ 500（SEARCH_ERROR）になるか**

- `/chat` や `/search` では、**embedding（質問のベクトル化）** のあと、**Chroma（ベクトルDB）で類似検索**を実行します。
- この「類似検索」の処理中に、**タイムアウト以外の例外**が発生すると、API はそれを **500 SEARCH_ERROR** として返します（HTTP の 500 = サーバー側の内部エラー）。
- 想定される原因の例:
  - **Chroma（SQLite）の状態**: 永続化先が読み取り専用・ロック・破損（`readonly database` / `database is locked` など）
  - **次元不一致**: インデックス構築時と現在で embedding モデルが違う（例: 別モデルで再インデックスしたが古い DB を参照している）
  - **リソース・Chroma の不具合**: メモリ不足、Chroma の内部エラー、ディスク障害
- **実際の例外メッセージ**は、エラーレスポンスの `detail.extra.error` や `GET /status` の `recent_errors` に含まれるので、そこを見ると「なぜ」が分かります。

**検索で 500（SEARCH_ERROR）のときの対処手順**

1. **API サーバー側で原因確認**
   - `GET /status` を開く（例: ブラウザで `https://<APIのURL>/status`、または `curl https://<APIのURL>/status`）。
   - レスポンスの **`init_error`** に値があれば、インデックス構築失敗などの原因です。
   - **`recent_errors`** に直近のエラー詳細（メッセージ）が出ます。
2. **再インデックス**
   - `POST /reload` で再インデックスを実行。完了するまで待ってから、再度 `/chat` を試す。
3. **k を小さくする**
   - UI の検索の強さ（k）を 2 や 1 に下げて再送する（エラー時の「k=2 にして再送」「k=1 にして再送」ボタンでも可）。
4. **サーバー（API を動かしている端末）側の確認**
   - Ollama が起動しているか（`brew services list` や `ollama list`）。
   - `nomic-embed-text` が入っているか（`ollama pull nomic-embed-text`）。
   - Chroma の永続化先（既定は `./chroma_db`）の権限・読み書き可能か（「readonly database」が出ていないか）。
   - ターミナルや launchd のログに、検索時に出ている Python の例外メッセージがないか確認。

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
  -d '{"prompt":"（質問をここに入力。資料にある範囲で）","model":"gemma2","k":2,"max_tokens":120,"timeout_s":240,"embedding_timeout_s":240,"search_timeout_s":120,"generate_timeout_s":240}'
```

`POST /search`（検索のみ）:

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"（質問をここに入力。資料にある範囲で）","k":2,"embedding_timeout_s":240,"search_timeout_s":120}'
```

`POST /generate`（生成のみ。`/search` の `context` を渡す）:

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"（質問をここに入力。資料にある範囲で）","model":"gemma2","max_tokens":120,"generate_timeout_s":240,"context":"..."}'
```

### ベクトルDB（`chroma_db`）について

APIサーバーは起動時に `./chroma_db` を読み込みます。未作成の場合は、既定で `vaccine_manual.pdf`（または `PDF_PATH` 環境変数で指定したPDF）から自動構築します。

注意:

- **永続化ディレクトリは書き込み可能である必要があります**（`attempt to write a readonly database` が出る場合は、権限/所有者/マウント設定の問題です）
- **安全のため、APIが自動で削除/置換する永続化ディレクトリ名は `chroma_db*` のみに制限**しています（例: `chroma_db_8001` のようなポート別分割は可）
- `CHROMA_PERSIST_DIR` が書き込み不可の場合、APIは **ユーザー書き込み領域へフォールバック**します（macOS: `~/Library/Application Support/vaccine-chatbot/chroma_db`）

環境変数:

- `PDF_PATH`: 取り込むPDFパス（既定: `vaccine_manual.pdf`）
- `PDF_DIR`: 取り込むPDFディレクトリ（既定: `./pdfs`）
- `CHROMA_PERSIST_DIR`: Chroma永続化ディレクトリ（既定: `./chroma_db`）
- `PDF_LOADER`: PDF抽出ローダの選択（既定: `auto`）
  - `auto`: 可能なら `PyMuPDFLoader` を優先し、無ければ `PyPDFLoader` にフォールバック
  - `pymupdf`: `PyMuPDFLoader` を強制（未導入の場合はエラー）
  - `pypdf`: `PyPDFLoader` を強制
- `CHUNK_SIZE`: PDFチャンクサイズ（既定: `900`）
- `CHUNK_OVERLAP`: PDFチャンクの重なり（既定: `120`）
- `DIAG_PING_TIMEOUT_S`: 環境チェック（`GET /diagnostics`）の **Ollama疎通（/api/version）** のタイムアウト秒（既定: `2.5`）
- `DIAG_LIST_TIMEOUT_S`: 環境チェック（`GET /diagnostics`）の **モデル一覧取得（ollama.list）** のタイムアウト秒（既定: `8.0`）
- `DIAG_LIST_RETRIES`: モデル一覧取得のリトライ回数（既定: `1`）
- `DIAG_EMBED_TIMEOUT_S`: 環境チェック（`GET /diagnostics`）の **embeddingスモークテスト** のタイムアウト秒（既定: `8.0`）

### できること

- **ソース（根拠）の表示**: 回答の下に **資料名 + ページラベル（例: `[P3]`）** と **抜粋（前後数行）** を表示します（クリックで展開 / コピー可）
- **サイドバー設定**:
  - 回答モデル切替（`gemma2` ↔ `llama3.1`）
  - 検索の強さ（k値）をスライダーで調整
- **PDF配置で知識追加**: `./pdfs/` にPDFを置くと、次回の実行時に自動で再インデックスして検索対象に反映します

根拠が取れない場合（`sources` が0件）の挙動:

- **生成しない**: “それっぽい要約”を避けるため、LLM生成を行わず固定文（「資料にない」）を返します
- **次のアクション提示（UI）**: GitHub Pages（`docs/`）側で、質問の言い換え候補をボタンで提示します

※初回はPDF解析・ベクトル化が走るため数分かかる場合があります。

## 注意（PDFについて）

`.gitignore` で `*.pdf` を除外しているため、**PDFはGitにコミットされません**。  
手元に `vaccine_manual.pdf` を置くか、`./pdfs/` にPDFを配置して利用してください。

## OCR（スキャンPDF対応）

スキャンPDFなどで **テキスト抽出がほぼ0** の場合、本プロジェクトは `ocrmypdf`（内部で tesseract / ghostscript 等を利用）で **自動OCRを試します**（横展開: **API / Streamlit / CLI**）。

### 事前準備（macOS例）

```bash
brew install ocrmypdf tesseract
```

※OCRはPython依存ではなく **外部コマンド依存**です。未導入の場合、APIの `GET /sources` の `items[].ingest.ocr.error` が `ocrmypdf_not_found` になります。

### OCRの設定（環境変数）

- **`OCR_MODE`**: `off` / `auto` / `force`（既定: `auto`）
  - `auto`: テキスト抽出できないPDFのみOCR（推奨）
  - `force`: 既にテキストがあるPDFでも再OCR（精度確認や再生成向け）
- **`OCR_LANG`**: 既定 `jpn+eng`
- **`OCR_CACHE_DIR`**: 既定 `./ocr_cache`（OCR済みPDFのキャッシュ）
- **`OCRMYPDF_ARGS`**: `ocrmypdf` への追加引数（必要な場合のみ）

反映手順:

- PDF追加/更新後、またはOCR設定を変えた後は **`POST /reload`** で再インデックスしてください（UIの「再インデックス」）。

### OCR精度の判定（CER/WER）

OCR結果を **定量評価**したい場合は `ocr_eval.py` を使います（正解テキストとの差で CER/WER を出します）。

```bash
. .venv/bin/activate

# ページ別の正解テキスト（例: gt/1.txt, gt/2.txt ...）で評価
python3 ocr_eval.py --pdf ./pdfs/your.pdf --gt-dir ./gt --out ocr_eval_report.json

# 全文の正解テキスト（1ファイル）で評価
python3 ocr_eval.py --pdf ./pdfs/your.pdf --gt ./gt.txt --out ocr_eval_report.json
```

補足:

- スキャン品質の影響が大きいので、OCRがうまくいかない場合は `OCR_MODE=force`、`OCR_LANG`、`OCRMYPDF_ARGS` の調整を試してください。

### 「どんな質問でも“資料にない”になる」時の原因チェック

「資料に書かれているはずの質問」でも、つねに次のような固定文が返る場合:

> 資料に記載がないため、この資料に基づく回答はできません。（質問: ...）

これは多くの場合、**RAGの検索結果が0件**になっている状態です（LLMが忘れたのではなく、検索が空）。

- **PDFが参照できていない**
  - `PDF_PATH`（既定: `vaccine_manual.pdf`）/ `PDF_DIR`（既定: `./pdfs`）の配置を確認
  - 実行時の作業ディレクトリが想定と違うと、相対パスの `vaccine_manual.pdf` / `./pdfs` が見つかりません
- **スキャンPDFでOCR不足（テキスト抽出できていない）**
  - PDFは見つかっても **抽出文字数がほぼ0** だと、検索が常に0件になり全質問で「資料にない」になります
  - 対処: OCRして **テキスト入りPDF** にしてから配置してください（または `ocrmypdf` を導入し `OCR_MODE=auto` で自動OCRを有効化）
  - 追加対策（抽出精度）: `pip install pymupdf` → `export PDF_LOADER=pymupdf`
- **（APIの場合）インデックス未完了/失敗**
  - `GET /sources` の `error` / `next_actions` を確認し、必要なら `POST /reload` を実行
  - `GET /status` の `pdf_count` / `init_error` も参考になります

### PDF解析の精度を上げる（段組/ヘッダ/改行ノイズ対策）

本プロジェクトは、PDF取り込み時に **抽出テキストのクリーニング**（過剰改行・空白・英単語ハイフン改行）と、**繰り返しヘッダ/フッタの軽い除去**を行います（横展開: **API / Streamlit**）。

さらにPDFによっては、`PyPDFLoader` より `PyMuPDFLoader`（PyMuPDF）の方が抽出精度が上がることがあります:

```bash
. .venv/bin/activate
pip install pymupdf
```

切替:

```bash
export PDF_LOADER=pymupdf
```

分割（検索の当たり方）が合わない場合は、チャンク設定を調整してください:

```bash
export CHUNK_SIZE=900
export CHUNK_OVERLAP=120
```

## トラブルシューティング
### まず「なぜ動かないか」を確認する（推奨）

GitHub Pages の画面にある **「環境チェック」**（APIの `GET /diagnostics`）で、次を診断できます。

- **Ollama疎通**: Ollama が起動していない/到達できない
- **生成モデルの有無**: 選択中モデルが未インストール（例: `ollama pull gemma2`）
- **Embeddingモデルの有無/動作**: `nomic-embed-text` が未インストール、または embedding が失敗している（RAGが動かない）
- **生成の動作**: 生成がタイムアウト/失敗している（初回起動・高負荷・モデル不備など）

コマンド例:

```bash
curl -sS http://127.0.0.1:8000/diagnostics | python -m json.tool
```

### 全てのPDFファイルから回答されない

複数PDFを配置しているのに、**一部のPDFの内容だけが根拠に出る**、または**特定のPDFが一切参照されない**場合の主な原因と対処です。

- **一部PDFが取り込みに失敗している（テキスト抽出不可・OCR未実施）**
  - スキャンPDFでテキストがほぼ0かつOCRが off / 失敗していると、そのPDFはチャンクが0件になり検索に一切出てきません
  - 対処: **GET /sources** や **POST /reload** のレポートで、各PDFの `status` を確認する（`ocr_needed` / `error` のPDFは要対応）
  - 該当PDFをOCRしてテキスト入りにしてから、**POST /reload** で再インデックスする
- **検索結果が top-k のみ（デフォルト k=3）**
  - 全PDFはインデックスされていても、1回の質問で返すのは**類似度上位 k 件**だけです。複数PDFがあると、ある質問ではAのPDFばかりが top-k に入り、BのPDFは入らないことがあります（仕様）
  - 対処: UIやAPIで **k を増やす**（例: 5〜10）と、複数PDFから選ばれやすくなります
- **キーワードフィルタで全件落ちている**
  - 質問のキーワードが資料の文言と一致しない（言い換え・同義語など）と、`min_overlap` を満たさず検索結果がすべて捨てられ、「資料にない」と返ることがあります
  - 対処: 環境変数 **`SEARCH_KEYWORD_OVERLAP_MIN=0`** でフィルタを無効化して試す（API起動時または `.env`）
- **インデックスに一部PDFしか含まれていない**
  - 再インデックスが途中で失敗した、または古いインデックスのままになっていると、新しく置いたPDFが入っていません
  - 対処: **POST /reload** で再インデックスし、レポートで全PDFが `status: ok` になっているか確認する

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

