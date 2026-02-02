import asyncio
import json
import os
import platform
import shutil
import subprocess
import time
from datetime import datetime, timezone
from typing import Any, Optional

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
LEGACY_PDF_DIR = "./uploads"  # 互換: 旧アップロード先を参照対象としても見る（アップロード機能ではない）
DEFAULT_PERSIST_DIR = "./chroma_db"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
MAX_CONTEXT_CHARS = 5000
MAX_DOC_CHARS = 1400
INDEX_MANIFEST_NAME = "source_index.json"

DEFAULT_EMBEDDING_TIMEOUT_S = 180
DEFAULT_SEARCH_TIMEOUT_S = 60
DEFAULT_GENERATE_TIMEOUT_S = 180
DEFAULT_CACHE_MAX_ENTRIES = 512


APP_STARTED_AT = datetime.now(tz=timezone.utc).isoformat()


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
    embedding_timeout_s: Optional[int] = Field(default=None, ge=1, le=900)
    search_timeout_s: Optional[int] = Field(default=None, ge=1, le=900)
    generate_timeout_s: Optional[int] = Field(default=None, ge=1, le=900)


class SearchRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    k: int = Field(default=3, ge=1, le=20)
    timeout_s: int = Field(default=120, ge=5, le=900)  # 互換（まとめて指定）
    embedding_timeout_s: Optional[int] = Field(default=None, ge=1, le=900)
    search_timeout_s: Optional[int] = Field(default=None, ge=1, le=900)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    model: str = Field(default="gemma2:2b", min_length=1, max_length=100)
    max_tokens: int = Field(default=256, ge=32, le=2048)
    timeout_s: int = Field(default=180, ge=5, le=900)  # 互換（まとめて指定）
    generate_timeout_s: Optional[int] = Field(default=None, ge=1, le=900)
    context: str = Field(default="", max_length=MAX_CONTEXT_CHARS + 500)


class _LRUCache:
    def __init__(self, max_entries: int):
        self.max_entries = max(1, int(max_entries))
        self._data: dict[str, Any] = {}
        self._order: list[str] = []
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any | None:
        if key in self._data:
            self.hits += 1
            try:
                self._order.remove(key)
            except ValueError:
                pass
            self._order.append(key)
            return self._data[key]
        self.misses += 1
        return None

    def set(self, key: str, value: Any) -> None:
        if key in self._data:
            self._data[key] = value
            try:
                self._order.remove(key)
            except ValueError:
                pass
            self._order.append(key)
            return
        self._data[key] = value
        self._order.append(key)
        while len(self._order) > self.max_entries:
            oldest = self._order.pop(0)
            self._data.pop(oldest, None)

    def stats(self) -> dict[str, Any]:
        return {
            "max_entries": self.max_entries,
            "entries": len(self._data),
            "hits": self.hits,
            "misses": self.misses,
        }


def _repo_root() -> str:
    # launchd の WorkingDirectory を信頼しつつ、スクリプト配置も基準にできるようにする
    return os.path.dirname(os.path.abspath(__file__))


def _read_text(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None


def _try_git_rev_parse(repo_root: str) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            timeout=1.0,
        )
        sha = out.decode("utf-8", errors="ignore").strip()
        return sha or None
    except Exception:
        return None


def _try_read_git_dir(repo_root: str) -> str | None:
    git_dir = os.path.join(repo_root, ".git")
    head = _read_text(os.path.join(git_dir, "HEAD"))
    if not head:
        return None
    head = head.strip()

    # detached HEAD
    if not head.startswith("ref:"):
        return head if len(head) >= 7 else None

    # ref: refs/heads/main
    ref = head.split(":", 1)[1].strip()
    ref_path = os.path.join(git_dir, ref)
    sha = _read_text(ref_path)
    if sha:
        sha = sha.strip()
        return sha or None

    # packed-refs fallback
    packed = _read_text(os.path.join(git_dir, "packed-refs"))
    if not packed:
        return None
    for line in packed.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("^"):
            continue
        parts = line.split(" ")
        if len(parts) == 2 and parts[1] == ref:
            return parts[0]
    return None


def _get_git_sha(repo_root: str) -> str:
    # デプロイ時に .git を含めない場合は、環境変数で注入できるようにする
    for k in ("GIT_SHA", "GIT_COMMIT", "SOURCE_VERSION"):
        v = os.environ.get(k)
        if v and v.strip():
            return v.strip()
    return _try_git_rev_parse(repo_root) or _try_read_git_dir(repo_root) or "unknown"


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
    # 互換: 旧構成で uploads/ に置かれているPDFも参照対象に含める
    try:
        if os.path.isdir(LEGACY_PDF_DIR):
            for f in sorted(os.listdir(LEGACY_PDF_DIR)):
                if f.lower().endswith(".pdf"):
                    paths.append(os.path.join(LEGACY_PDF_DIR, f))
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
async def _startup() -> None:
    persist_dir = os.environ.get("CHROMA_PERSIST_DIR", DEFAULT_PERSIST_DIR)
    pdf_path = os.environ.get("PDF_PATH", DEFAULT_PDF_PATH)
    pdf_dir = os.environ.get("PDF_DIR", DEFAULT_PDF_DIR)
    run_mode = os.environ.get("RUN_MODE", "prod").strip() or "prod"
    git_sha = _get_git_sha(_repo_root())

    app.state.vectorstore = None
    app.state.init_error = None
    app.state.persist_dir = persist_dir
    app.state.pdf_path = pdf_path
    app.state.pdf_dir = pdf_dir
    app.state.started_at = APP_STARTED_AT
    app.state.version = getattr(app, "version", None)
    app.state.git_sha = git_sha
    app.state.run_mode = run_mode
    app.state.ingest_lock = asyncio.Lock()
    app.state.source_signature = None
    app.state.last_indexed_at = None
    app.state.last_request_timings = None
    app.state.embeddings = OllamaEmbeddings(model=DEFAULT_EMBED_MODEL)
    cache_max = int(os.environ.get("EMBED_CACHE_MAX", str(DEFAULT_CACHE_MAX_ENTRIES)))
    app.state.embedding_cache = _LRUCache(max_entries=cache_max)
    app.state.embed_warmup = {"ok": False, "started_at": None, "finished_at": None, "error": None, "ms": None}

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

    # "初回だけ遅い" 対策: embeddingモデルを軽くウォームアップ（失敗しても落とさない）
    async def _warmup_embed():
        app.state.embed_warmup["started_at"] = datetime.now(tz=timezone.utc).isoformat()
        t0 = time.perf_counter()
        try:
            await asyncio.wait_for(
                asyncio.to_thread(app.state.embeddings.embed_query, "warmup"),
                timeout=float(os.environ.get("EMBED_WARMUP_TIMEOUT_S", str(DEFAULT_EMBEDDING_TIMEOUT_S))),
            )
            ms = int((time.perf_counter() - t0) * 1000)
            app.state.embed_warmup.update(
                {"ok": True, "error": None, "finished_at": datetime.now(tz=timezone.utc).isoformat(), "ms": ms}
            )
        except Exception as e:
            ms = int((time.perf_counter() - t0) * 1000)
            app.state.embed_warmup.update(
                {"ok": False, "error": str(e), "finished_at": datetime.now(tz=timezone.utc).isoformat(), "ms": ms}
            )

    asyncio.create_task(_warmup_embed())

    # launchd の StandardOutPath に落ちる想定（起動時の設定・バージョンを必ず残す）
    try:
        print(
            json.dumps(
                {
                    "ts": datetime.now(tz=timezone.utc).isoformat(),
                    "event": "startup",
                    "version": getattr(app.state, "version", None),
                    "git_sha": getattr(app.state, "git_sha", None),
                    "started_at": getattr(app.state, "started_at", None),
                    "run_mode": getattr(app.state, "run_mode", None),
                    "pdf_path": pdf_path,
                    "pdf_dir": pdf_dir,
                    "legacy_pdf_dir": LEGACY_PDF_DIR,
                    "persist_dir": persist_dir,
                    "embed_model": DEFAULT_EMBED_MODEL,
                    "embed_cache_max": int(os.environ.get("EMBED_CACHE_MAX", str(DEFAULT_CACHE_MAX_ENTRIES))),
                    "python": platform.python_version(),
                    "cwd": os.getcwd(),
                },
                ensure_ascii=False,
            )
        )
    except Exception:
        pass


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {"status": "ok", "ready": app.state.vectorstore is not None}


@app.get("/status")
def status() -> dict[str, Any]:
    pdfs = _list_pdf_paths(getattr(app.state, "pdf_path", DEFAULT_PDF_PATH), getattr(app.state, "pdf_dir", DEFAULT_PDF_DIR))
    return {
        "version": getattr(app.state, "version", None),
        "git_sha": getattr(app.state, "git_sha", None),
        "started_at": getattr(app.state, "started_at", None),
        "run_mode": getattr(app.state, "run_mode", None),
        "ready": app.state.vectorstore is not None,
        "persist_dir": getattr(app.state, "persist_dir", DEFAULT_PERSIST_DIR),
        "pdf_path": getattr(app.state, "pdf_path", DEFAULT_PDF_PATH),
        "pdf_dir": getattr(app.state, "pdf_dir", DEFAULT_PDF_DIR),
        "legacy_pdf_dir": LEGACY_PDF_DIR,
        "pdf_count": len(pdfs),
        "last_indexed_at": getattr(app.state, "last_indexed_at", None),
        "init_error": getattr(app.state, "init_error", None),
        "embed_model": DEFAULT_EMBED_MODEL,
        "embed_warmup": getattr(app.state, "embed_warmup", None),
        "embedding_cache": getattr(app.state, "embedding_cache", _LRUCache(DEFAULT_CACHE_MAX_ENTRIES)).stats(),
        "last_request_timings": getattr(app.state, "last_request_timings", None),
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


def _resolve_timeouts(
    timeout_s: int,
    embedding_timeout_s: Optional[int],
    search_timeout_s: Optional[int],
    generate_timeout_s: Optional[int],
) -> tuple[int, int, int]:
    # 新フィールドが未指定なら、互換の timeout_s を使用
    et = int(embedding_timeout_s if embedding_timeout_s is not None else timeout_s)
    st = int(search_timeout_s if search_timeout_s is not None else timeout_s)
    gt = int(generate_timeout_s if generate_timeout_s is not None else timeout_s)
    return max(1, et), max(1, st), max(1, gt)


def _http_error(
    *,
    stage: str,
    code: str,
    message: str,
    timeout_s: int | None = None,
    hints: Optional[list[str]] = None,
    timings: Optional[dict[str, Any]] = None,
    extra: Optional[dict[str, Any]] = None,
    status_code: int = 500,
) -> None:
    detail: dict[str, Any] = {"stage": stage, "code": code, "message": message}
    if timeout_s is not None:
        detail["timeout_s"] = int(timeout_s)
    if hints:
        detail["hints"] = hints
    if timings:
        detail["timings"] = timings
    if extra:
        detail.update(extra)
    raise HTTPException(status_code=status_code, detail=detail)


def _is_ollama_down_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(
        s in msg
        for s in [
            "connection refused",
            "failed to connect",
            "connecterror",
            "connectionerror",
            "cannot connect",
            "no such host",
        ]
    )


async def _embed_query_with_cache(prompt: str, timeout_s: int) -> tuple[list[float], dict[str, Any]]:
    cache: _LRUCache = app.state.embedding_cache
    key = prompt.strip()
    cached = cache.get(key)
    if cached is not None:
        return cached, {"cached_embedding": True}

    retries = int(os.environ.get("EMBED_RETRIES", "2"))
    base_delay = float(os.environ.get("EMBED_RETRY_BASE_DELAY_S", "0.25"))
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            vec = await asyncio.wait_for(
                asyncio.to_thread(app.state.embeddings.embed_query, prompt),
                timeout=timeout_s,
            )
            cache.set(key, vec)
            return vec, {"cached_embedding": False, "attempt": attempt + 1}
        except TimeoutError as e:
            last_err = e
        except Exception as e:
            last_err = e
        if attempt < retries:
            await asyncio.sleep(base_delay * (2**attempt))
    assert last_err is not None
    raise last_err


async def _run_search(prompt: str, k: int, embedding_timeout_s: int, search_timeout_s: int) -> dict[str, Any]:
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
        _http_error(
            stage="search",
            code="RAG_NOT_READY",
            message="RAGエンジンが未初期化です（PDFが不足、またはテキスト抽出できない可能性）",
            hints=[
                "PDFを ./pdfs/（環境変数 PDF_DIR）に配置してください",
                "スキャンPDFの場合はOCRしてから配置してください",
                "PDFを追加・更新したら POST /reload で再インデックスしてください",
                "GET /status で init_error / pdf_count を確認してください",
            ],
            extra={
                "persist_dir": getattr(app.state, "persist_dir", DEFAULT_PERSIST_DIR),
                "pdf_path": getattr(app.state, "pdf_path", DEFAULT_PDF_PATH),
                "pdf_dir": getattr(app.state, "pdf_dir", DEFAULT_PDF_DIR),
                "init_error": getattr(app.state, "init_error", None),
            },
            status_code=503,
        )

    timings: dict[str, Any] = {}
    t_total0 = time.perf_counter()

    # embedding
    t_embed0 = time.perf_counter()
    try:
        vec, embed_meta = await _embed_query_with_cache(prompt, timeout_s=embedding_timeout_s)
        timings["embedding_ms"] = int((time.perf_counter() - t_embed0) * 1000)
        timings.update(embed_meta)
    except TimeoutError:
        timings["embedding_ms"] = int((time.perf_counter() - t_embed0) * 1000)
        _http_error(
            stage="embedding",
            code="EMBEDDING_TIMEOUT",
            message="embedding（クエリのベクトル化）がタイムアウトしました",
            timeout_s=embedding_timeout_s,
            hints=[
                "Ollama が起動しているか確認してください（Mac mini なら brew services start ollama など）",
                f"Ollama に {DEFAULT_EMBED_MODEL} が入っているか確認してください（ollama pull {DEFAULT_EMBED_MODEL}）",
                "初回はモデル起動で時間がかかります。しばらく待つか、タイムアウトを延長してください",
            ],
            timings=timings,
            status_code=504,
        )
    except Exception as e:
        timings["embedding_ms"] = int((time.perf_counter() - t_embed0) * 1000)
        if _is_ollama_down_error(e):
            _http_error(
                stage="embedding",
                code="OLLAMA_UNAVAILABLE",
                message="embedding に失敗しました（Ollama に接続できない可能性）",
                hints=[
                    "Ollama が起動しているか確認してください",
                    f"モデルが存在するか確認してください（ollama pull {DEFAULT_EMBED_MODEL}）",
                ],
                timings=timings,
                extra={"error": str(e)},
                status_code=502,
            )
        _http_error(
            stage="embedding",
            code="EMBEDDING_ERROR",
            message="embedding に失敗しました",
            hints=["GET /status で ready / init_error を確認してください"],
            timings=timings,
            extra={"error": str(e)},
            status_code=500,
        )

    # similarity search (vector search)
    t_search0 = time.perf_counter()
    try:
        fn = getattr(vs, "similarity_search_by_vector", None)
        if callable(fn):
            docs = await asyncio.wait_for(asyncio.to_thread(fn, vec, k=k), timeout=search_timeout_s)
        else:
            docs = await asyncio.wait_for(asyncio.to_thread(vs.similarity_search, prompt, k=k), timeout=search_timeout_s)
        timings["search_ms"] = int((time.perf_counter() - t_search0) * 1000)
    except TimeoutError:
        timings["search_ms"] = int((time.perf_counter() - t_search0) * 1000)
        _http_error(
            stage="search",
            code="SEARCH_TIMEOUT",
            message="類似検索（ベクトルDBクエリ）がタイムアウトしました",
            timeout_s=search_timeout_s,
            hints=[
                "PDF数が多い/初回インデックス直後は遅くなることがあります",
                "k を小さくして試してください（例: 3→2）",
                "PDF追加・更新後は POST /reload で再インデックスしてください",
            ],
            timings=timings,
            extra={"pdf_count": len(_list_pdf_paths(app.state.pdf_path, app.state.pdf_dir))},
            status_code=504,
        )
    except Exception as e:
        timings["search_ms"] = int((time.perf_counter() - t_search0) * 1000)
        _http_error(
            stage="search",
            code="SEARCH_ERROR",
            message="類似検索に失敗しました",
            hints=["POST /reload で再インデックスを試してください", "GET /status で init_error を確認してください"],
            timings=timings,
            extra={"error": str(e)},
            status_code=500,
        )

    timings["total_ms"] = int((time.perf_counter() - t_total0) * 1000)
    context = _format_context(docs)
    return {"docs": docs, "context": context, "timings": timings}


async def _run_generate(prompt: str, context: str, model: str, max_tokens: int, generate_timeout_s: int) -> dict[str, Any]:
    t0 = time.perf_counter()
    full_prompt = f"""
あなたは厚労省の資料に基づいて回答する専門アシスタントです。
以下の【資料】の内容に基づいて、日本語で簡潔に回答してください。
資料に記載がない場合は「資料内には該当する情報が見当たりません」と答え、自治体の相談窓口または接種を受けた医療機関への相談を促してください。

【資料】:
{context}

質問: {prompt}
回答:
""".strip()
    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(
                ollama.generate,
                model=model,
                prompt=full_prompt,
                options={"num_predict": max_tokens},
            ),
            timeout=generate_timeout_s,
        )
    except TimeoutError:
        _http_error(
            stage="generate",
            code="GENERATE_TIMEOUT",
            message="生成（LLM応答）がタイムアウトしました",
            timeout_s=generate_timeout_s,
            hints=[
                "初回はモデル起動で時間がかかります。しばらく待つか、タイムアウトを延長してください",
                "軽量モデル（例: gemma2:2b）を選んでください",
            ],
            timings={"generate_ms": int((time.perf_counter() - t0) * 1000)},
            status_code=504,
        )
    except Exception as e:
        if _is_ollama_down_error(e):
            _http_error(
                stage="generate",
                code="OLLAMA_UNAVAILABLE",
                message="生成に失敗しました（Ollama に接続できない可能性）",
                hints=["Ollama が起動しているか確認してください", f"モデルが存在するか確認してください（ollama pull {model}）"],
                extra={"error": str(e)},
                status_code=502,
            )
        _http_error(
            stage="generate",
            code="GENERATE_ERROR",
            message="生成に失敗しました",
            hints=["モデル名を確認してください", "Ollama のログを確認してください"],
            extra={"error": str(e)},
            status_code=500,
        )

    return {"answer": response.get("response", ""), "timings": {"generate_ms": int((time.perf_counter() - t0) * 1000)}}


@app.post("/search")
async def search_endpoint(request: SearchRequest):
    et, st, _ = _resolve_timeouts(request.timeout_s, request.embedding_timeout_s, request.search_timeout_s, None)
    result = await _run_search(request.prompt, request.k, embedding_timeout_s=et, search_timeout_s=st)
    docs = result["docs"]
    timings = result["timings"]
    app.state.last_request_timings = {"stage": "search", **timings}
    try:
        print(json.dumps({"ts": datetime.now(tz=timezone.utc).isoformat(), "event": "search", "k": request.k, "timings": timings}, ensure_ascii=False))
    except Exception:
        pass
    return {"context": result["context"], "sources": _extract_sources(docs), "timings": timings}


@app.post("/generate")
async def generate_endpoint(request: GenerateRequest):
    _, _, gt = _resolve_timeouts(request.timeout_s, None, None, request.generate_timeout_s)
    result = await _run_generate(
        prompt=request.prompt,
        context=(request.context or "")[: MAX_CONTEXT_CHARS + 500],
        model=request.model,
        max_tokens=request.max_tokens,
        generate_timeout_s=gt,
    )
    timings = result["timings"]
    app.state.last_request_timings = {"stage": "generate", **timings}
    try:
        print(json.dumps({"ts": datetime.now(tz=timezone.utc).isoformat(), "event": "generate", "model": request.model, "timings": timings}, ensure_ascii=False))
    except Exception:
        pass
    return {"answer": result["answer"], "timings": timings}


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        et, st, gt = _resolve_timeouts(
            request.timeout_s,
            request.embedding_timeout_s,
            request.search_timeout_s,
            request.generate_timeout_s,
        )

        search_res = await _run_search(request.prompt, request.k, embedding_timeout_s=et, search_timeout_s=st)
        docs = search_res["docs"]
        context = search_res["context"]
        timings = dict(search_res["timings"])

        gen_res = await _run_generate(
            prompt=request.prompt,
            context=context,
            model=request.model,
            max_tokens=request.max_tokens,
            generate_timeout_s=gt,
        )
        timings["generate_ms"] = gen_res["timings"]["generate_ms"]
        timings["total_ms"] = int((timings.get("embedding_ms", 0) + timings.get("search_ms", 0) + timings.get("generate_ms", 0)))

        app.state.last_request_timings = {"stage": "chat", **timings}
        try:
            print(
                json.dumps(
                    {
                        "ts": datetime.now(tz=timezone.utc).isoformat(),
                        "event": "chat",
                        "k": request.k,
                        "model": request.model,
                        "timings": timings,
                    },
                    ensure_ascii=False,
                )
            )
        except Exception:
            pass
        return {"answer": gen_res["answer"], "sources": _extract_sources(docs), "timings": timings}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

