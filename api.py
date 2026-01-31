import os
from typing import Any

import ollama
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field


DEFAULT_PDF_PATH = "vaccine_manual.pdf"
DEFAULT_PERSIST_DIR = "./chroma_db"
DEFAULT_EMBED_MODEL = "nomic-embed-text"


app = FastAPI(title="vaccine-chatbot API", version="0.1.0")

# GitHub Pages（外部）からのアクセスを許可する設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 公開時はGitHub PagesのURLに制限すると安全です
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    model: str = Field(default="gemma2", min_length=1, max_length=100)
    k: int = Field(default=3, ge=1, le=20)


def _format_context(docs) -> str:
    lines: list[str] = []
    for doc in docs:
        meta = doc.metadata or {}
        # PyPDFLoader は page=0始まりで入ることが多いので+1して表示
        page = meta.get("page")
        if isinstance(page, int):
            page_label = f"P{page + 1}"
        else:
            page_label = "P?"
        text = (doc.page_content or "").strip()
        lines.append(f"[{page_label}] {text}")
    return "\n".join(lines)


def _extract_sources(docs) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    seen: set[tuple[str | None, int | None]] = set()
    for doc in docs:
        meta = doc.metadata or {}
        src = meta.get("source")
        page = meta.get("page")
        page_num = page + 1 if isinstance(page, int) else None
        key = (src, page_num)
        if key in seen:
            continue
        seen.add(key)
        excerpt = (doc.page_content or "").strip().replace("\n", " ")
        if len(excerpt) > 300:
            excerpt = excerpt[:300] + "…"
        sources.append({"source": src, "page": page_num, "excerpt": excerpt})
    return sources


def _build_vectorstore_from_pdf(pdf_path: str, persist_dir: str) -> Chroma:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model=DEFAULT_EMBED_MODEL)
    vs = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_dir)

    # langchainの版によっては明示persistが必要なことがあるため、あれば呼ぶ
    persist = getattr(vs, "persist", None)
    if callable(persist):
        persist()
    return vs


def _load_vectorstore(persist_dir: str) -> Chroma:
    embeddings = OllamaEmbeddings(model=DEFAULT_EMBED_MODEL)
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)


def _is_vectorstore_ready(vs: Chroma) -> bool:
    try:
        # Chromaのラッパーが提供するcountがあれば使う
        if hasattr(vs, "_collection") and getattr(vs, "_collection") is not None:
            return int(vs._collection.count()) > 0  # type: ignore[attr-defined]
    except Exception:
        pass
    return False


@app.on_event("startup")
def _startup() -> None:
    """
    起動時に一度だけRAGエンジンを初期化します。

    - 既に `./chroma_db` がある: そこから読み込み
    - 無い/空: 手元のPDF（既定: vaccine_manual.pdf）から構築して永続化
    """
    persist_dir = os.environ.get("CHROMA_PERSIST_DIR", DEFAULT_PERSIST_DIR)
    pdf_path = os.environ.get("PDF_PATH", DEFAULT_PDF_PATH)

    # 起動失敗で落とすのではなく、/status と /chat で理由を返せるようにする
    app.state.vectorstore = None
    app.state.init_error = None
    app.state.persist_dir = persist_dir
    app.state.pdf_path = pdf_path

    try:
        vs = _load_vectorstore(persist_dir)
        if _is_vectorstore_ready(vs):
            app.state.vectorstore = vs
            return

        if not os.path.exists(pdf_path):
            app.state.init_error = (
                f"ベクトルDBが未作成で、PDFも見つかりませんでした: {pdf_path}\n"
                f"PDFを配置するか、環境変数 PDF_PATH を指定してください。"
            )
            return

        app.state.vectorstore = _build_vectorstore_from_pdf(pdf_path, persist_dir)
    except Exception as e:
        app.state.init_error = str(e)


@app.get("/health")
def health() -> dict[str, str | bool]:
    ready = app.state.vectorstore is not None
    return {"status": "ok", "ready": ready}


@app.get("/status")
def status() -> dict[str, Any]:
    return {
        "ready": app.state.vectorstore is not None,
        "persist_dir": getattr(app.state, "persist_dir", DEFAULT_PERSIST_DIR),
        "pdf_path": getattr(app.state, "pdf_path", DEFAULT_PDF_PATH),
        "init_error": getattr(app.state, "init_error", None),
    }


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        vs = app.state.vectorstore
        if vs is None:
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "RAGエンジンが未初期化です（PDF/ベクトルDBが不足している可能性があります）",
                    "persist_dir": getattr(app.state, "persist_dir", DEFAULT_PERSIST_DIR),
                    "pdf_path": getattr(app.state, "pdf_path", DEFAULT_PDF_PATH),
                    "init_error": getattr(app.state, "init_error", None),
                },
            )

        docs = vs.similarity_search(request.prompt, k=request.k)

        context = _format_context(docs)
        full_prompt = f"""
あなたは厚労省の資料に基づいて回答する専門アシスタントです。
以下の【資料】の内容に基づいて、日本語で簡潔に回答してください。
資料に記載がない場合は「資料内には該当する情報が見当たりません」と答え、自治体の相談窓口または接種を受けた医療機関への相談を促してください。

【資料】:
{context}

質問: {request.prompt}
回答:
""".strip()

        response = ollama.generate(model=request.model, prompt=full_prompt)

        return {
            "answer": response.get("response", ""),
            "sources": _extract_sources(docs),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

