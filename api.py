import asyncio
import os
import re
import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import ollama
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field


DEFAULT_PDF_PATH = "vaccine_manual.pdf"
DEFAULT_PERSIST_DIR = "./chroma_db"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_UPLOAD_DIR = "./uploads"
MAX_CONTEXT_CHARS = 5000
MAX_DOC_CHARS = 1400
MAX_UPLOAD_BYTES = 30 * 1024 * 1024  # 30MB
UPLOAD_MANIFEST_NAME = "_manifest.json"


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
    max_tokens: int = Field(default=256, ge=32, le=1024)
    timeout_s: int = Field(default=180, ge=10, le=600)


def _format_context(docs) -> str:
    lines: list[str] = []
    total = 0
    for doc in docs:
        meta = doc.metadata or {}
        # PyPDFLoader は page=0始まりで入ることが多いので+1して表示
        page = meta.get("page")
        if isinstance(page, int):
            page_label = f"P{page + 1}"
        else:
            page_label = "P?"
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


def _sanitize_filename(name: str) -> str:
    name = (name or "").strip()
    if not name:
        return "uploaded.pdf"
    name = name.replace("\\", "/").split("/")[-1]
    name = re.sub(r"[^0-9A-Za-z._ -]", "_", name)
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return name


def _manifest_path(upload_dir: str) -> str:
    return os.path.join(upload_dir, UPLOAD_MANIFEST_NAME)


def _load_manifest(upload_dir: str) -> dict[str, Any]:
    path = _manifest_path(upload_dir)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:
        # 破損などがあっても、API全体は落とさない
        return {}


def _save_manifest(upload_dir: str, data: dict[str, Any]) -> None:
    path = _manifest_path(upload_dir)
    tmp = path + ".tmp"
    os.makedirs(upload_dir, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _infer_original_from_saved_name(filename: str) -> str:
    # 例: "0123abcd..._original-name.pdf" -> "original-name.pdf"
    if not filename:
        return filename
    parts = filename.split("_", 1)
    if len(parts) == 2 and len(parts[0]) >= 8:
        return parts[1]
    return filename


def _persist_if_supported(vs: Chroma) -> None:
    persist = getattr(vs, "persist", None)
    if callable(persist):
        persist()


def _split_pdf_docs(pdf_path: str, source_label: str):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    for d in docs:
        d.metadata = dict(d.metadata or {})
        d.metadata["source"] = source_label
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)


def _create_vectorstore_from_chunks(chunks, persist_dir: str) -> Chroma:
    embeddings = OllamaEmbeddings(model=DEFAULT_EMBED_MODEL)
    vs = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_dir)
    _persist_if_supported(vs)
    return vs


def _add_chunks_to_vectorstore(vs: Chroma, chunks) -> None:
    # Chromaは add_documents を持つ（langchain版により引数は変わらない想定）
    vs.add_documents(chunks)
    _persist_if_supported(vs)


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
    upload_dir = os.environ.get("UPLOAD_DIR", DEFAULT_UPLOAD_DIR)

    # 起動失敗で落とすのではなく、/status と /chat で理由を返せるようにする
    app.state.vectorstore = None
    app.state.init_error = None
    app.state.persist_dir = persist_dir
    app.state.pdf_path = pdf_path
    app.state.upload_dir = upload_dir
    app.state.ingest_lock = asyncio.Lock()

    try:
        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(persist_dir, exist_ok=True)
    except Exception:
        # ここは致命的ではない（/statusに出す）
        pass

    try:
        vs = _load_vectorstore(persist_dir)
        if _is_vectorstore_ready(vs):
            app.state.vectorstore = vs
            return

        # DB未作成の場合は、既定PDF or uploads内PDFから構築
        upload_pdfs = []
        try:
            upload_pdfs = [
                os.path.join(upload_dir, f)
                for f in os.listdir(upload_dir)
                if f.lower().endswith(".pdf")
            ]
        except Exception:
            upload_pdfs = []

        chunks = []
        if os.path.exists(pdf_path):
            chunks.extend(_split_pdf_docs(pdf_path, os.path.basename(pdf_path)))
        for p in sorted(upload_pdfs):
            chunks.extend(_split_pdf_docs(p, os.path.basename(p)))

        if not chunks:
            app.state.init_error = (
                f"ベクトルDBが未作成で、PDFも見つかりませんでした。\n"
                f"- 既定PDF: {pdf_path}\n"
                f"- 追加PDF: {upload_dir}/*.pdf\n"
                f"PDFを配置するか、環境変数 PDF_PATH / UPLOAD_DIR を指定してください。"
            )
            return

        app.state.vectorstore = _create_vectorstore_from_chunks(chunks, persist_dir)
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
        "upload_dir": getattr(app.state, "upload_dir", DEFAULT_UPLOAD_DIR),
        "init_error": getattr(app.state, "init_error", None),
    }


@app.get("/sources")
def list_sources() -> dict[str, Any]:
    pdf_path = getattr(app.state, "pdf_path", DEFAULT_PDF_PATH)
    upload_dir = getattr(app.state, "upload_dir", DEFAULT_UPLOAD_DIR)

    manifest = _load_manifest(upload_dir)
    items: list[dict[str, Any]] = []
    if os.path.exists(pdf_path):
        base_name = os.path.basename(pdf_path)
        try:
            st = os.stat(pdf_path)
            size_bytes = int(st.st_size)
            mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
        except Exception:
            size_bytes = None
            mtime = None
        items.append(
            {
                "type": "base",
                "filename": base_name,
                "original_filename": base_name,
                "path": pdf_path,
                "size_bytes": size_bytes,
                "mtime": mtime,
            }
        )
    try:
        for f in sorted(os.listdir(upload_dir)):
            if f == UPLOAD_MANIFEST_NAME:
                continue
            if not f.lower().endswith(".pdf"):
                continue
            path = os.path.join(upload_dir, f)
            entry = manifest.get(f, {}) if isinstance(manifest, dict) else {}
            original = None
            if isinstance(entry, dict):
                original = entry.get("original_filename") or entry.get("original") or None
            if not original:
                original = _infer_original_from_saved_name(f)
            try:
                st = os.stat(path)
                size_bytes = int(st.st_size)
                mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
            except Exception:
                size_bytes = None
                mtime = None
            items.append(
                {
                    "type": "upload",
                    "filename": f,  # 保存名（ハッシュ付き）
                    "original_filename": original,  # 元ファイル名（表示用）
                    "path": path,
                    "size_bytes": size_bytes,
                    "mtime": mtime,
                }
            )
    except Exception as e:
        return {"ok": False, "error": str(e), "items": items}
    return {"ok": True, "items": items}


@app.post("/sources/upload")
async def upload_source(file: UploadFile = File(...)):
    upload_dir = getattr(app.state, "upload_dir", DEFAULT_UPLOAD_DIR)
    persist_dir = getattr(app.state, "persist_dir", DEFAULT_PERSIST_DIR)

    if file.content_type not in (None, "", "application/pdf"):
        raise HTTPException(status_code=400, detail="PDFのみ対応です（content-type: application/pdf）")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="ファイルが空です")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"ファイルが大きすぎます（最大 {MAX_UPLOAD_BYTES} bytes）")

    # 表示用の元ファイル名（ユーザーが選んだ名前）
    original_name = (file.filename or "").strip()
    safe_name = _sanitize_filename(original_name or "uploaded.pdf")
    digest = hashlib.sha256(raw).hexdigest()[:16]
    save_name = f"{digest}_{safe_name}"
    os.makedirs(upload_dir, exist_ok=True)
    save_path = os.path.join(upload_dir, save_name)
    if not os.path.exists(save_path):
        with open(save_path, "wb") as f:
            f.write(raw)

    # 取り込み（排他）
    lock: asyncio.Lock = app.state.ingest_lock
    async with lock:
        # 元ファイル名をmanifestに保存（フロント表示用）
        try:
            manifest = _load_manifest(upload_dir)
            if not isinstance(manifest, dict):
                manifest = {}
            manifest[save_name] = {
                "original_filename": original_name or safe_name,
                "saved_filename": save_name,
                "sha256_16": digest,
                "size_bytes": len(raw),
                "uploaded_at": datetime.now(tz=timezone.utc).isoformat(),
            }
            _save_manifest(upload_dir, manifest)
        except Exception:
            # ここで失敗してもRAG取り込み自体は継続
            pass

        vs = app.state.vectorstore
        try:
            chunks = await asyncio.to_thread(_split_pdf_docs, save_path, save_name)
            if vs is None:
                app.state.vectorstore = await asyncio.to_thread(_create_vectorstore_from_chunks, chunks, persist_dir)
            else:
                await asyncio.to_thread(_add_chunks_to_vectorstore, vs, chunks)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return {"ok": True, "filename": save_name, "original_filename": original_name or safe_name}


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

        return {
            "answer": response.get("response", ""),
            "sources": _extract_sources(docs),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

