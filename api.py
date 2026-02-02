import asyncio
import json
import os
import shutil
from datetime import datetime, timezone
from typing import Any

import ollama
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field


DEFAULT_PDF_PATH = "vaccine_manual.pdf"  # 単体PDF（任意）
DEFAULT_PDF_DIR = "./pdfs"  # ここにPDFを置くと自動で参照（推奨）
DEFAULT_PERSIST_DIR = "./chroma_db"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
MAX_CONTEXT_CHARS = 5000
MAX_DOC_CHARS = 1400
INDEX_MANIFEST_NAME = "source_index.json"


app = FastAPI(title="vaccine-chatbot API", version="0.2.0")

# GitHub Pages（外部）からのアクセスを許可する設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 公開時はGitHub PagesのURLに制限すると安全です
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    model: str = Field(default="gemma2:2b", min_length=1, max_length=100)
    k: int = Field(default=3, ge=1, le=20)
    max_tokens: int = Field(default=256, ge=32, le=1024)
    timeout_s: int = Field(default=180, ge=10, le=600)


def _format_context(docs) -> str:
    lines: list[str] = []
    total = 0
    for doc in docs:
        meta = doc.metadata or {}
        page = meta.get("page")
        page_label = f"P{page + 1}" if isinstance(page, int) else "P?"
        text = (doc.page_content or "").strip()
        if len(text) > MAX_DOC_CHARS:
            text = text[:MAX_DOC_CHARS] + "…"
        line = f"[{page_label}] {text}"
        if total + len(line) > MAX_CONTEXT_CHARS:
            break
        lines.append(line)
        total += len(line)
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


def _persist_if_supported(vs: Chroma) -> None:
    persist = getattr(vs, "persist", None)
    if callable(persist):
        persist()


def _list_pdf_paths(pdf_path: str, pdf_dir: str) -> list[str]:
    paths: list[str] = []
    if pdf_path and os.path.exists(pdf_path) and pdf_path.lower().endswith(".pdf"):
        paths.append(pdf_path)
    try:
        if pdf_dir and os.path.isdir(pdf_dir):
            for f in sorted(os.listdir(pdf_dir)):
                if f.lower().endswith(".pdf"):
                    paths.append(os.path.join(pdf_dir, f))
    except Exception:
        pass
    # 重複除去（順序保持）
    uniq: list[str] = []
    seen: set[str] = set()
    for p in paths:
        ap = os.path.abspath(p)
        if ap in seen:
            continue
        seen.add(ap)
        uniq.append(p)
    return uniq


def _source_signature(paths: list[str]) -> list[dict[str, Any]]:
    sig: list[dict[str, Any]] = []
    for p in paths:
        try:
            st = os.stat(p)
            sig.append(
                {
                    "path": os.path.abspath(p),
                    "filename": os.path.basename(p),
                    "size_bytes": int(st.st_size),
                    "mtime": float(st.st_mtime),
                }
            )
        except Exception:
            sig.append({"path": os.path.abspath(p), "filename": os.path.basename(p), "size_bytes": None, "mtime": None})
    sig.sort(key=lambda x: x.get("path") or "")
    return sig


def _manifest_path(persist_dir: str) -> str:
    return os.path.join(persist_dir, INDEX_MANIFEST_NAME)


def _load_index_manifest(persist_dir: str) -> dict[str, Any]:
    try:
        with open(_manifest_path(persist_dir), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def _save_index_manifest(persist_dir: str, data: dict[str, Any]) -> None:
    os.makedirs(persist_dir, exist_ok=True)
    path = _manifest_path(persist_dir)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _safe_reset_persist_dir(persist_dir: str) -> None:
    # 誤爆防止：基本は "chroma_db" 直下のみ削除を許可
    base = os.path.basename(os.path.abspath(persist_dir))
    if base != "chroma_db":
        raise RuntimeError(f"安全のため persist_dir の自動削除は chroma_db のみに限定しています: {persist_dir}")
    if os.path.isdir(persist_dir):
        shutil.rmtree(persist_dir)
    os.makedirs(persist_dir, exist_ok=True)


def _split_pdf_docs(pdf_path: str, source_label: str):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    for d in docs:
        d.metadata = dict(d.metadata or {})
        d.metadata["source"] = source_label
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    return [c for c in chunks if (c.page_content or "").strip()]


def _rebuild_vectorstore(paths: list[str], persist_dir: str) -> Chroma:
    # 既存DBがある場合も、ソースが変わったら作り直す（重複/ゴミ混入を避ける）
    _safe_reset_persist_dir(persist_dir)

    chunks = []
    for p in paths:
        if not os.path.exists(p):
            continue
        chunks.extend(_split_pdf_docs(p, os.path.basename(p)))

    if not chunks:
        raise RuntimeError(
            "PDFが見つからない、またはPDFからテキストを抽出できませんでした。"
            "（スキャンPDFの場合はOCRしてから配置してください）"
        )

    embeddings = OllamaEmbeddings(model=DEFAULT_EMBED_MODEL)
    vs = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_dir)
    _persist_if_supported(vs)
    return vs


def _load_vectorstore(persist_dir: str) -> Chroma:
    embeddings = OllamaEmbeddings(model=DEFAULT_EMBED_MODEL)
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)


def _is_vectorstore_ready(vs: Chroma) -> bool:
    try:
        if hasattr(vs, "_collection") and getattr(vs, "_collection") is not None:
            return int(vs._collection.count()) > 0  # type: ignore[attr-defined]
    except Exception:
        pass
    return False


def _ensure_index_uptodate(force: bool = False) -> None:
    persist_dir = app.state.persist_dir
    pdf_path = app.state.pdf_path
    pdf_dir = app.state.pdf_dir

    paths = _list_pdf_paths(pdf_path, pdf_dir)
    sig = _source_signature(paths)

    last_sig = getattr(app.state, "source_signature", None)
    if (not force) and app.state.vectorstore is not None and last_sig == sig:
        return

    # 既存DBがあり、ソースが変わっていないならロードで済ませる
    if (not force) and os.path.isdir(persist_dir):
        try:
            vs = _load_vectorstore(persist_dir)
            if _is_vectorstore_ready(vs) and last_sig == sig:
                app.state.vectorstore = vs
                return
        except Exception:
            pass

    # 作り直し
    app.state.vectorstore = _rebuild_vectorstore(paths, persist_dir)
    app.state.source_signature = sig
    app.state.last_indexed_at = datetime.now(tz=timezone.utc).isoformat()
    _save_index_manifest(
        persist_dir,
        {"source_signature": sig, "last_indexed_at": app.state.last_indexed_at, "pdf_path": pdf_path, "pdf_dir": pdf_dir},
    )


@app.on_event("startup")
def _startup() -> None:
    persist_dir = os.environ.get("CHROMA_PERSIST_DIR", DEFAULT_PERSIST_DIR)
    pdf_path = os.environ.get("PDF_PATH", DEFAULT_PDF_PATH)
    pdf_dir = os.environ.get("PDF_DIR", DEFAULT_PDF_DIR)

    app.state.vectorstore = None
    app.state.init_error = None
    app.state.persist_dir = persist_dir
    app.state.pdf_path = pdf_path
    app.state.pdf_dir = pdf_dir
    app.state.ingest_lock = asyncio.Lock()
    app.state.source_signature = None
    app.state.last_indexed_at = None

    # 起動時にmanifestがあれば拾う（参考情報）
    try:
        m = _load_index_manifest(persist_dir)
        if isinstance(m, dict) and "source_signature" in m:
            app.state.source_signature = m.get("source_signature")
            app.state.last_indexed_at = m.get("last_indexed_at")
    except Exception:
        pass

    # 起動時にベクトルDBを準備（失敗しても落とさず /status に理由を出す）
    try:
        _ensure_index_uptodate(force=False)
    except Exception as e:
        app.state.init_error = str(e)


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {"status": "ok", "ready": app.state.vectorstore is not None}


@app.get("/status")
def status() -> dict[str, Any]:
    pdfs = _list_pdf_paths(getattr(app.state, "pdf_path", DEFAULT_PDF_PATH), getattr(app.state, "pdf_dir", DEFAULT_PDF_DIR))
    return {
        "ready": app.state.vectorstore is not None,
        "persist_dir": getattr(app.state, "persist_dir", DEFAULT_PERSIST_DIR),
        "pdf_path": getattr(app.state, "pdf_path", DEFAULT_PDF_PATH),
        "pdf_dir": getattr(app.state, "pdf_dir", DEFAULT_PDF_DIR),
        "pdf_count": len(pdfs),
        "last_indexed_at": getattr(app.state, "last_indexed_at", None),
        "init_error": getattr(app.state, "init_error", None),
    }


@app.get("/sources")
def list_sources() -> dict[str, Any]:
    pdf_path = getattr(app.state, "pdf_path", DEFAULT_PDF_PATH)
    pdf_dir = getattr(app.state, "pdf_dir", DEFAULT_PDF_DIR)
    persist_dir = getattr(app.state, "persist_dir", DEFAULT_PERSIST_DIR)

    paths = _list_pdf_paths(pdf_path, pdf_dir)
    sig = _source_signature(paths)

    items: list[dict[str, Any]] = []
    for s in sig:
        items.append(
            {
                "type": "pdf",
                "filename": s.get("filename"),
                "original_filename": s.get("filename"),
                "path": s.get("path"),
                "size_bytes": s.get("size_bytes"),
                "mtime": s.get("mtime"),
            }
        )

    indexed_sig = getattr(app.state, "source_signature", None)
    indexed = bool(app.state.vectorstore is not None and indexed_sig == sig)
    return {
        "ok": True,
        "indexed": indexed,
        "last_indexed_at": getattr(app.state, "last_indexed_at", None),
        "persist_dir": persist_dir,
        "items": items,
    }


@app.post("/reload")
async def reload_sources() -> dict[str, Any]:
    lock: asyncio.Lock = app.state.ingest_lock
    async with lock:
        try:
            await asyncio.to_thread(_ensure_index_uptodate, True)
            return {"ok": True, "reindexed": True, "last_indexed_at": getattr(app.state, "last_indexed_at", None)}
        except Exception as e:
            app.state.init_error = str(e)
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # PDF配置の変更があれば自動追従（重い処理なので排他）
        lock: asyncio.Lock = app.state.ingest_lock
        async with lock:
            try:
                await asyncio.to_thread(_ensure_index_uptodate, False)
                app.state.init_error = None
            except Exception as e:
                app.state.vectorstore = None
                app.state.init_error = str(e)

        vs = app.state.vectorstore
        if vs is None:
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "RAGエンジンが未初期化です（PDFが不足、またはテキスト抽出できない可能性）",
                    "persist_dir": getattr(app.state, "persist_dir", DEFAULT_PERSIST_DIR),
                    "pdf_path": getattr(app.state, "pdf_path", DEFAULT_PDF_PATH),
                    "pdf_dir": getattr(app.state, "pdf_dir", DEFAULT_PDF_DIR),
                    "init_error": getattr(app.state, "init_error", None),
                },
            )

        try:
            docs = await asyncio.wait_for(
                asyncio.to_thread(vs.similarity_search, request.prompt, k=request.k),
                timeout=request.timeout_s,
            )
        except TimeoutError:
            raise HTTPException(
                status_code=504,
                detail={"message": "検索（embedding/類似検索）がタイムアウトしました", "timeout_s": request.timeout_s},
            )

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

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    ollama.generate,
                    model=request.model,
                    prompt=full_prompt,
                    options={"num_predict": request.max_tokens},
                ),
                timeout=request.timeout_s,
            )
        except TimeoutError:
            raise HTTPException(
                status_code=504,
                detail={"message": "生成（LLM応答）がタイムアウトしました", "timeout_s": request.timeout_s},
            )

        return {"answer": response.get("response", ""), "sources": _extract_sources(docs)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

