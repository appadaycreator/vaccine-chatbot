import ollama
import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings


def _normalize_newlines(text: str) -> str:
    return (text or "").replace("\r\n", "\n").replace("\r", "\n")


def _model_names_from_ollama_list(payload) -> list[str]:
    if not isinstance(payload, dict):
        return []
    models = payload.get("models")
    if not isinstance(models, list):
        return []
    out: list[str] = []
    for m in models:
        if isinstance(m, dict) and isinstance(m.get("name"), str):
            out.append(m["name"])
    return out


def _has_model(model_names: list[str], wanted: str) -> bool:
    w = (wanted or "").strip()
    if not w:
        return False
    return any(n == w or n.startswith(w + ":") for n in model_names)


def _ensure_embedding_model(model: str = "nomic-embed-text") -> None:
    try:
        info = ollama.list()
    except Exception as e:
        raise RuntimeError(
            "Ollama に接続できません（未起動の可能性）。\n"
            "対処:\n"
            "- Ollama が起動しているか確認してください（例: brew services start ollama）"
        ) from e
    names = _model_names_from_ollama_list(info)
    if not _has_model(names, model):
        raise RuntimeError(
            f"Embeddingモデル（{model}）が見つかりません。\n"
            "対処:\n"
            f"- ollama pull {model}\n"
            "- RAG（PDF検索）は embedding モデルが無いと動きません"
        )


def _clean_pdf_text(text: str) -> str:
    t = _normalize_newlines(text)
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"(?<=\w)-\n(?=\w)", "", t)
    return t.strip()


def _get_splitter() -> RecursiveCharacterTextSplitter:
    try:
        chunk_size = int(os.environ.get("CHUNK_SIZE", "900"))
    except Exception:
        chunk_size = 900
    try:
        chunk_overlap = int(os.environ.get("CHUNK_OVERLAP", "120"))
    except Exception:
        chunk_overlap = 120
    chunk_size = max(200, min(chunk_size, 5000))
    chunk_overlap = max(0, min(chunk_overlap, max(0, chunk_size - 1)))
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def _load_pdf_docs_best_effort(pdf_path: str):
    prefer = (os.environ.get("PDF_LOADER", "auto") or "auto").strip().lower()

    def _try_pymupdf():
        try:
            from langchain_community.document_loaders import PyMuPDFLoader  # type: ignore

            return PyMuPDFLoader(pdf_path).load()
        except Exception:
            return None

    if prefer in ("pymupdf", "fitz"):
        docs = _try_pymupdf()
        if docs is None:
            raise RuntimeError("PDF_LOADER=pymupdf が指定されていますが、PyMuPDFLoader（pymupdf）が利用できません。")
        return docs
    if prefer in ("pypdf", "pdf"):
        return PyPDFLoader(pdf_path).load()

    docs = _try_pymupdf()
    if docs is not None:
        return docs
    return PyPDFLoader(pdf_path).load()


# 1. PDFの読み込みと分割
print("PDFを解析中...")
data = _load_pdf_docs_best_effort("vaccine_manual.pdf")  # PDFファイルを指定
for d in data:
    d.page_content = _clean_pdf_text(getattr(d, "page_content", "") or "")

text_splitter = _get_splitter()
chunks = [c for c in text_splitter.split_documents(data) if (c.page_content or "").strip()]

# 2. ベクトルデータベースの作成 (OllamaのEmbeddingモデルを使用)
# 注意: embedding モデルは自動ダウンロードされません。事前に `ollama pull nomic-embed-text` が必要です。
print("知識ベースを構築中（これには数分かかる場合があります）...")
_ensure_embedding_model("nomic-embed-text")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)

def _no_sources_answer(question: str) -> str:
    q = (question or "").strip()
    qline = f"（質問: {q}）" if q else ""
    return (
        "結論:\n"
        "資料に記載がないため、この資料に基づく回答はできません。"
        f"{qline}\n\n"
        "根拠:\n"
        "- 資料にない（参照PDFから該当箇所を特定できませんでした）\n\n"
        "相談先:\n"
        "- 接種を受けた医療機関\n"
        "- お住まいの自治体の予防接種相談窓口\n"
        "- 症状が強い／急に悪化した／緊急性が疑われる場合: 119（救急）\n"
    )


def _build_answer_prompt(*, question: str, context: str) -> str:
    return f"""
あなたは医療情報の文脈で、厚労省等の配布資料（下の【資料】）に基づいて回答するアシスタントです。
推測や一般論で補完してはいけません。【資料】に書かれていないことは「資料にない」と明確に述べてください。

必ず次の3セクションだけで出力してください（見出し名は固定）:
結論:
根拠:
相談先:

ルール:
- 【資料】に書かれていない内容を断定しない（曖昧にそれっぽく言わない）
- 「根拠」には【資料】から該当箇所を引用/要約して箇条書きで示す。該当がなければ「資料にない」と書く
- 「相談先」は必ず1つ以上。緊急性が疑われる場合は救急（119）も含める

【資料】:
{context}

質問: {question}
""".strip()

def rag_chatbot(user_query):
    # 3. 関連情報の検索
    docs = vectorstore.similarity_search(user_query, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    # 4. LLMへの問い合わせ
    if not docs or not context.strip():
        return _no_sources_answer(user_query)

    prompt = _build_answer_prompt(question=user_query, context=context)
    
    response = ollama.generate(model='gemma2', prompt=prompt)
    return response['response']

if __name__ == "__main__":
    while True:
        query = input("\n資料について質問してください: ")
        if query.lower() == 'exit': break
        print("\n資料を検索して回答を生成中...")
        print(f"\n【回答】:\n{rag_chatbot(query)}")