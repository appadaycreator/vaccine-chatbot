# vaccine-chatbot

Ollama（ローカルLLM）とRAG（PDF検索）で、厚労省資料に基づいた質問応答を行うプロトタイプです。

## 前提

- macOS
- Python 3.13（`.venv` は Python 3.13 で作成されています）
- Ollama がインストール済み・起動済み

## セットアップ

### 1) 仮想環境（初回のみ）

```bash
python3 -m venv .venv
```

### 2) 依存関係のインストール

```bash
. .venv/bin/activate
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

