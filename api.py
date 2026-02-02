import asyncio
import contextvars
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter, deque
from datetime import datetime, timezone
from typing import Any, Optional

import ollama
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import RedirectResponse
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
# 生成モデル（回答モデル）の既定。環境差が大きいので、必要なら環境変数で上書きする。
DEFAULT_LLM_MODEL = os.environ.get("DEFAULT_LLM_MODEL", "gemma2")
MAX_CONTEXT_CHARS = 5000
MAX_DOC_CHARS = 1400
INDEX_MANIFEST_NAME = "source_index.json"

DEFAULT_EMBEDDING_TIMEOUT_S = 180
DEFAULT_SEARCH_TIMEOUT_S = 60
DEFAULT_GENERATE_TIMEOUT_S = 180
DEFAULT_CACHE_MAX_ENTRIES = 512
DEFAULT_INDEX_CHECK_TIMEOUT_S = 120

REQUEST_ID_HEADER = "X-Request-ID"
_request_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        rid = getattr(record, "request_id", None) or _request_id_ctx.get()
        record.request_id = rid or "-"
        # stderr formatter が KeyError にならないようデフォルトを入れる
        if not hasattr(record, "event"):
            record.event = "-"
        if not hasattr(record, "stage"):
            record.stage = "-"
        if not hasattr(record, "code"):
            record.code = "-"
        return True


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", _request_id_ctx.get()) or "-",
        }
        for k in (
            "event",
            "method",
            "path",
            "status_code",
            "stage",
            "code",
            "timeout_s",
            "timings",
            "extra",
        ):
            v = getattr(record, k, None)
            if v is not None:
                payload[k] = v
        if record.exc_info:
            try:
                etype = record.exc_info[0].__name__ if record.exc_info[0] else None
                payload["error_type"] = etype
                payload["error"] = str(record.exc_info[1]) if record.exc_info[1] else None
            except Exception:
                pass
        return json.dumps(payload, ensure_ascii=False)


def _get_logger() -> logging.Logger:
    logger = logging.getLogger("vaccine_api")
    if getattr(logger, "_configured", False):
        return logger
    logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
    logger.propagate = False

    h_out = logging.StreamHandler(sys.stdout)
    h_out.setLevel(logger.level)
    h_out.addFilter(_RequestIdFilter())
    h_out.setFormatter(_JsonFormatter())

    h_err = logging.StreamHandler(sys.stderr)
    h_err.setLevel(logging.WARNING)
    h_err.addFilter(_RequestIdFilter())
    h_err.setFormatter(
        logging.Formatter(
            "%(asctime)sZ %(levelname)s request_id=%(request_id)s event=%(event)s stage=%(stage)s code=%(code)s msg=%(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )

    logger.addHandler(h_out)
    logger.addHandler(h_err)
    logger._configured = True  # type: ignore[attr-defined]
    return logger


logger = _get_logger()


_recent_errors: deque[dict[str, Any]] = deque(maxlen=50)
_error_counts: Counter[str] = Counter()
_error_lock = threading.Lock()


def _record_error(item: dict[str, Any]) -> None:
    code = str(item.get("code") or "UNKNOWN")
    with _error_lock:
        _recent_errors.append(item)
        _error_counts[code] += 1


def _new_request_id() -> str:
    return uuid.uuid4().hex

# 回答品質（医療系の言い方）を最低ラインで保証するための固定フォーマット
# - 断定・誤誘導を避けるため、根拠（sources）が取れない場合は必ず「資料にない」を返す
# - ユーザーが次に取る行動（相談先）を必ず出す
ANSWER_SECTIONS = ("結論", "根拠", "相談先")


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
    """
    LLMに「結論/根拠/相談先」の構造を強制し、資料外の推測を抑止するテンプレ。
    注意: sources が空のときは呼び出し側で生成せず _no_sources_answer を返す（DoD対策）。
    """
    return f"""
あなたは医療情報の文脈で、厚労省等の配布資料（下の【資料】）に基づいて回答するアシスタントです。
推測や一般論で補完してはいけません。【資料】に書かれていないことは「資料にない」と明確に述べてください。

必ず次の3セクションだけで出力してください（見出し名は固定）:
結論:
根拠:
相談先:

ルール:
- 【資料】に書かれていない内容を断定しない（曖昧にそれっぽく言わない）
- 「根拠」には、【資料】から該当箇所をページラベル（例: [P3]）つきで引用/要約して箇条書きで示す
- 「相談先」は必ず1つ以上。緊急性が疑われる場合は救急（119）も含める
- 余計な免責文や追加セクション（注意/補足など）は出さない（UI側で常設するため）

【資料】:
{context}

質問: {question}
""".strip()


# 根拠（PDFの該当箇所）が取れない場合のフォールバック用「参考情報」。
# - 注意: 参照PDFに基づく“根拠”ではない。UIで根拠（引用）カードは出ない想定。
# - それでも「まったく回答が返らない/常に資料にない」状態を避けるための“安全側の一般説明”。
FALLBACK_KNOWLEDGE_BASE = """
【プロトタイプ知識ベース（資料外・一般情報） 抜粋】
- 観察期間：接種当日（0日目）から7日間
- 記録項目：体温、接種部位の反応（腫れ・痛み）、全身反応（発熱、頭痛、倦怠感）
- 報告が必要な症状：37.5度以上の発熱、日常生活に支障が出るほどの痛みや腫れ
- 連絡先：各自治体の相談窓口、または接種を受けた医療機関
""".strip()


def _build_general_fallback_prompt(*, question: str, reference: str) -> str:
    """
    参照PDFから該当箇所が取れなかったときの一般説明用プロンプト。
    - 参照PDFに基づくと“断定しない”
    - ただしユーザーが次に取れる行動（相談先）を必ず出す
    """
    return f"""
あなたは医療情報の文脈で回答するアシスタントです。
現在、参照PDFから質問の該当箇所を特定できていません。
そのため、以下の【参考情報】と一般的な注意として、断定を避けつつ回答してください。

必ず次の3セクションだけで出力してください（見出し名は固定）:
結論:
根拠:
相談先:

ルール:
- 参照PDFに基づくと断定しない（ページラベルの引用もしない）
- 「根拠」には、【参考情報】からの引用/要約と、「一般的には…」のような前置きを使って不確実性を明示する
- 医療判断（診断/治療の指示）をしない。迷う場合は相談先へ誘導する
- 「相談先」は必ず1つ以上。緊急性が疑われる場合は救急（119）も含める
- 余計な追加セクション（注意/補足など）は出さない

【参考情報】:
{reference}

質問: {question}
""".strip()


APP_STARTED_AT = datetime.now(tz=timezone.utc).isoformat()


app = FastAPI(title="vaccine-chatbot API", version="0.2.0")

# UI（docs/）をAPI配下で配信（推奨導線: /ui）
# 注: ここは import 時に評価されるので、未定義の関数（_repo_root 等）を参照しない
_ui_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
if os.path.isdir(_ui_dir):

    @app.get("/ui", include_in_schema=False)
    def _ui_redirect():
        # 相対パス（./app.js 等）が壊れないよう末尾スラッシュへ寄せる
        return RedirectResponse(url="/ui/", status_code=307)

    app.mount("/ui", StaticFiles(directory=_ui_dir, html=True), name="ui")
else:
    logger.warning(
        "ui_dir_missing",
        extra={"event": "ui_dir_missing", "stage": "startup", "code": "UI_DIR_MISSING", "extra": {"path": _ui_dir}},
    )

# GitHub Pages（外部）からのアクセスを許可する設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 公開時はGitHub PagesのURLに制限すると安全です
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _request_observability_middleware(request: Request, call_next):
    rid = request.headers.get(REQUEST_ID_HEADER) or _new_request_id()
    token = _request_id_ctx.set(rid)
    request.state.request_id = rid
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        _request_id_ctx.reset(token)
    try:
        response.headers[REQUEST_ID_HEADER] = rid
    except Exception:
        pass
    try:
        logger.info(
            "request_finished",
            extra={
                "event": "request_finished",
                "request_id": rid,
                "method": request.method,
                "path": request.url.path,
                "status_code": getattr(response, "status_code", None),
                "timings": {"request_ms": int((time.perf_counter() - t0) * 1000)},
            },
        )
    except Exception:
        pass
    return response


@app.exception_handler(HTTPException)
async def _observability_http_exception_handler(request: Request, exc: HTTPException):
    rid = getattr(request.state, "request_id", None) or _new_request_id()
    detail = _normalize_error_detail(exc.detail)
    stage = detail.get("stage") if isinstance(detail, dict) else None
    code = detail.get("code") if isinstance(detail, dict) else None
    timings = detail.get("timings") if isinstance(detail, dict) else None
    timeout_s = detail.get("timeout_s") if isinstance(detail, dict) else None

    item = {
        "ts": datetime.now(tz=timezone.utc).isoformat(),
        "request_id": rid,
        "method": request.method,
        "path": request.url.path,
        "status_code": exc.status_code,
        "stage": stage,
        "code": code,
        "message": detail.get("message") if isinstance(detail, dict) else str(exc.detail),
        "timings": timings,
    }
    # 422（バリデーション）はノイズになりがちなので除外
    if exc.status_code != 422:
        _record_error(item)

    level = logging.ERROR if exc.status_code >= 500 else logging.WARNING
    try:
        logger.log(
            level,
            "http_error",
            extra={
                "event": "http_error",
                "request_id": rid,
                "method": request.method,
                "path": request.url.path,
                "status_code": exc.status_code,
                "stage": stage,
                "code": code,
                "timeout_s": timeout_s,
                "timings": timings,
                "extra": {"detail": detail},
            },
        )
    except Exception:
        pass

    resp = JSONResponse(status_code=exc.status_code, content={"detail": detail})
    resp.headers[REQUEST_ID_HEADER] = rid
    return resp


@app.exception_handler(Exception)
async def _observability_unhandled_exception_handler(request: Request, exc: Exception):
    rid = getattr(request.state, "request_id", None) or _new_request_id()
    item = {
        "ts": datetime.now(tz=timezone.utc).isoformat(),
        "request_id": rid,
        "method": request.method,
        "path": request.url.path,
        "status_code": 500,
        "stage": "unhandled",
        "code": "UNHANDLED_EXCEPTION",
        "message": str(exc),
    }
    _record_error(item)
    logger.exception(
        "unhandled_exception",
        extra={
            "event": "unhandled_exception",
            "request_id": rid,
            "method": request.method,
            "path": request.url.path,
            "status_code": 500,
            "stage": "unhandled",
            "code": "UNHANDLED_EXCEPTION",
        },
    )
    resp = JSONResponse(
        status_code=500,
        content={"detail": {"stage": "unhandled", "code": "UNHANDLED_EXCEPTION", "message": "Internal Server Error"}},
    )
    resp.headers[REQUEST_ID_HEADER] = rid
    return resp


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    model: str = Field(default=DEFAULT_LLM_MODEL, min_length=1, max_length=100)
    k: int = Field(default=3, ge=1, le=20)
    max_tokens: int = Field(default=256, ge=32, le=1024)
    timeout_s: int = Field(default=180, ge=10, le=600)
    embedding_timeout_s: Optional[int] = Field(default=None, ge=1, le=900)
    search_timeout_s: Optional[int] = Field(default=None, ge=1, le=900)
    generate_timeout_s: Optional[int] = Field(default=None, ge=1, le=900)
    # 根拠（PDF該当箇所）が取れない場合でも一般説明を返す（横展開: docs UI / Streamlit）
    # - 未指定（None）の場合は環境変数 ALLOW_GENERAL_FALLBACK_DEFAULT を参照
    allow_general_fallback: Optional[bool] = Field(default=None)


class SearchRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    k: int = Field(default=3, ge=1, le=20)
    timeout_s: int = Field(default=120, ge=5, le=900)  # 互換（まとめて指定）
    embedding_timeout_s: Optional[int] = Field(default=None, ge=1, le=900)
    search_timeout_s: Optional[int] = Field(default=None, ge=1, le=900)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    model: str = Field(default=DEFAULT_LLM_MODEL, min_length=1, max_length=100)
    max_tokens: int = Field(default=256, ge=32, le=2048)
    timeout_s: int = Field(default=180, ge=5, le=900)  # 互換（まとめて指定）
    generate_timeout_s: Optional[int] = Field(default=None, ge=1, le=900)
    context: str = Field(default="", max_length=MAX_CONTEXT_CHARS + 500)
    allow_general_fallback: Optional[bool] = Field(default=None)


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


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _init_index_status() -> dict[str, Any]:
    return {
        "running": False,
        "run_id": None,
        "trigger": None,  # startup / reload / auto
        "started_at": None,
        "finished_at": None,
        "last_success_at": None,
        "last_error": None,  # {"message": str, "code": str, "hints": [..], "at": iso, "items": [...]}
        "last_report": None,  # {"items": [...], "summary": {...}}
    }


class IndexBuildError(RuntimeError):
    def __init__(self, message: str, report: dict[str, Any]):
        super().__init__(message)
        self.report = report


def _get_index_status() -> dict[str, Any]:
    lock: threading.Lock | None = getattr(app.state, "index_status_lock", None)
    if lock is None:
        return dict(getattr(app.state, "index_status", _init_index_status()))
    with lock:
        return dict(getattr(app.state, "index_status", _init_index_status()))


def _update_index_status(patch: dict[str, Any]) -> dict[str, Any]:
    lock: threading.Lock | None = getattr(app.state, "index_status_lock", None)
    if lock is None:
        cur = getattr(app.state, "index_status", _init_index_status())
        cur.update(patch)
        app.state.index_status = cur
        return dict(cur)
    with lock:
        cur = getattr(app.state, "index_status", _init_index_status())
        cur.update(patch)
        app.state.index_status = cur
        return dict(cur)


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


def _normalize_newlines(text: str) -> str:
    return (text or "").replace("\r\n", "\n").replace("\r", "\n")


def _make_excerpt(text: str, *, max_lines: int = 10, max_chars: int = 900) -> str:
    """
    根拠表示用の抜粋を作る（UIでそのまま表示できるよう改行を残す）。
    - 行が多い場合は先頭/末尾を残して間を "…" にする（前後数行の雰囲気）。
    """
    raw = _normalize_newlines(text).strip()
    if not raw:
        return ""
    lines = [ln.strip() for ln in raw.split("\n")]
    lines = [ln for ln in lines if ln]
    if not lines:
        return ""

    if len(lines) <= max_lines:
        picked = lines
    else:
        head_n = max(1, max_lines // 2)
        tail_n = max(1, max_lines - head_n - 1)
        picked = lines[:head_n] + ["…"] + lines[-tail_n:]

    out: list[str] = []
    total = 0
    for ln in picked:
        # "…" は常に残す（長文切り捨ての合図）
        if ln != "…" and total + len(ln) + 1 > max_chars:
            break
        out.append(ln)
        total += len(ln) + 1
        if total >= max_chars:
            break
    return "\n".join(out).strip()


def _extract_sources(docs) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    seen: set[tuple[str | None, int | None]] = set()
    for doc in docs:
        meta = doc.metadata or {}
        src = meta.get("source") or "資料"
        page = meta.get("page")
        page_num = page + 1 if isinstance(page, int) else None
        key = (src, page_num)
        if key in seen:
            continue
        seen.add(key)
        page_label = f"[P{page_num}]" if isinstance(page_num, int) else "[P?]"
        excerpt = _make_excerpt(doc.page_content or "")
        sources.append(
            {
                "source": str(src),
                "page": page_num,
                "page_label": page_label,
                "excerpt": excerpt,
                "location": f"{src} {page_label}",
            }
        )
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


def _is_safe_persist_dir(persist_dir: str) -> bool:
    """
    誤爆防止のため、APIが自動削除/置換してよい永続化ディレクトリ名を制限する。
    - 既定: chroma_db
    - 互換: chroma_db_8001 等（ポート別に分けたい用途）
    - 内部用: chroma_db.__building__/chroma_db.__backup__ など（アトミック更新）
    """
    base = os.path.basename(os.path.abspath(persist_dir))
    return bool(base) and base.startswith("chroma_db")


def _is_readonly_db_error(e: Exception) -> bool:
    """
    Chroma（SQLite）永続化先が read-only のときに出る典型的な文言を検出する。
    例: "attempt to write a readonly database"
    """
    msg = str(e or "")
    low = msg.lower()
    return any(
        s in low
        for s in [
            "attempt to write a readonly database",
            "readonly database",
            "read-only database",
            "sqlite_readonly",
            "permission denied",
        ]
    )


def _is_dir_writable(path: str) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".write_test")
        with open(probe, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(probe)
        return True
    except Exception:
        return False


def _default_persist_base_dir() -> str:
    home = os.path.expanduser("~")
    if sys.platform == "darwin":
        return os.path.join(home, "Library", "Application Support", "vaccine-chatbot")
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return os.path.join(xdg, "vaccine-chatbot")
    return os.path.join(home, ".local", "share", "vaccine-chatbot")


def _resolve_persist_dir(requested: str) -> tuple[str, list[str]]:
    """
    永続化先が書き込み不可の場合にフォールバックする。
    - requested が書き込み可: そのまま
    - 書き込み不可: OS標準のユーザー書き込み領域へ退避
    """
    req = (requested or "").strip() or DEFAULT_PERSIST_DIR
    warnings: list[str] = []

    # 相対パスは作業ディレクトリ基準のまま（launchd で WorkingDirectory が設定される想定）
    if _is_dir_writable(req):
        return req, warnings

    base = _default_persist_base_dir()
    fallback = os.path.join(base, "chroma_db")
    warnings.append(f"CHROMA_PERSIST_DIR（{req}）が書き込み不可のため、{fallback} を使用します。")
    return fallback, warnings


def _safe_reset_persist_dir(persist_dir: str) -> None:
    # 誤爆防止：chroma_db 系のみ削除/再作成を許可
    if not _is_safe_persist_dir(persist_dir):
        raise RuntimeError(f"安全のため persist_dir の自動削除は chroma_db* のみに限定しています: {persist_dir}")
    if os.path.isdir(persist_dir):
        shutil.rmtree(persist_dir)
    os.makedirs(persist_dir, exist_ok=True)


def _get_splitter() -> RecursiveCharacterTextSplitter:
    """
    PDF取り込み用の分割設定。
    PDFは改行ノイズが多く短文が増えやすいので、デフォルトはやや大きめに取る。
    """
    try:
        chunk_size = int(os.environ.get("CHUNK_SIZE", "900"))
    except Exception:
        chunk_size = 900
    try:
        chunk_overlap = int(os.environ.get("CHUNK_OVERLAP", "120"))
    except Exception:
        chunk_overlap = 120
    # 安全域
    chunk_size = max(200, min(chunk_size, 5000))
    chunk_overlap = max(0, min(chunk_overlap, max(0, chunk_size - 1)))
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def _normalize_newlines(text: str) -> str:
    return (text or "").replace("\r\n", "\n").replace("\r", "\n")


def _clean_pdf_text(text: str) -> str:
    """
    PDF抽出テキストの最低限のクリーニング。
    - 過剰な空白/空行を整理
    - 英単語のハイフネーション改行を結合（日本語本文への影響を最小化）
    """
    t = _normalize_newlines(text)
    # 行末空白の除去
    t = re.sub(r"[ \t]+\n", "\n", t)
    # 連続空行の圧縮
    t = re.sub(r"\n{3,}", "\n\n", t)
    # 英単語ハイフネーション: "exam-\nple" -> "example"
    t = re.sub(r"(?<=\w)-\n(?=\w)", "", t)
    return t.strip()


def _strip_repeated_header_footer(docs: list[Any], *, top_n: int = 2, bottom_n: int = 2, min_ratio: float = 0.6) -> dict[str, Any]:
    """
    ページ毎に繰り返されるヘッダ/フッタ（タイトル行やページ番号等）を軽く除去する。
    完全自動は危険なので、各ページの先頭/末尾行だけを対象に、出現頻度が高い短い行のみ除去する。
    """
    pages = len(docs)
    if pages <= 1:
        return {"removed_top": 0, "removed_bottom": 0}

    def _edge_lines(text: str) -> tuple[list[str], list[str]]:
        lines = [ln.strip() for ln in _normalize_newlines(text).split("\n")]
        lines = [ln for ln in lines if ln]
        return lines[:top_n], (lines[-bottom_n:] if lines else [])

    top_counter: Counter[str] = Counter()
    bottom_counter: Counter[str] = Counter()
    for d in docs:
        top, bottom = _edge_lines(getattr(d, "page_content", "") or "")
        top_counter.update(top)
        bottom_counter.update(bottom)

    threshold = max(2, int(pages * min_ratio))
    # 短すぎる/長すぎる行は除外（誤除去を減らす）
    def _pick(counter: Counter[str]) -> set[str]:
        out: set[str] = set()
        for ln, c in counter.items():
            if c >= threshold and 2 <= len(ln) <= 80:
                out.add(ln)
        return out

    top_rm = _pick(top_counter)
    bottom_rm = _pick(bottom_counter)

    removed_top = 0
    removed_bottom = 0
    for d in docs:
        raw = _normalize_newlines(getattr(d, "page_content", "") or "")
        lines = [ln.rstrip() for ln in raw.split("\n")]
        # 先頭側
        i = 0
        while i < len(lines) and lines[i].strip() == "":
            i += 1
        for _ in range(top_n):
            if i < len(lines) and lines[i].strip() in top_rm:
                lines[i] = ""
                removed_top += 1
                i += 1
            else:
                break
        # 末尾側
        j = len(lines) - 1
        while j >= 0 and lines[j].strip() == "":
            j -= 1
        for _ in range(bottom_n):
            if j >= 0 and lines[j].strip() in bottom_rm:
                lines[j] = ""
                removed_bottom += 1
                j -= 1
            else:
                break
        d.page_content = _clean_pdf_text("\n".join(lines))

    return {"removed_top": removed_top, "removed_bottom": removed_bottom}


def _load_pdf_docs_best_effort(pdf_path: str) -> tuple[list[Any], str]:
    """
    可能ならPyMuPDF系ローダを優先（レイアウト/段組に強いことが多い）。
    依存が無い場合は PyPDFLoader にフォールバック。
    `PDF_LOADER` で明示指定も可能（auto/pymupdf/pypdf）。
    """
    prefer = (os.environ.get("PDF_LOADER", "auto") or "auto").strip().lower()

    def _try_pymupdf() -> Optional[list[Any]]:
        try:
            # langchain-communityに存在し、かつ pymupdf が入っていると動く
            from langchain_community.document_loaders import PyMuPDFLoader  # type: ignore

            return PyMuPDFLoader(pdf_path).load()
        except Exception:
            return None

    if prefer in ("pymupdf", "fitz"):
        docs = _try_pymupdf()
        if docs is None:
            raise RuntimeError("PDF_LOADER=pymupdf が指定されていますが、PyMuPDFLoader（pymupdf）が利用できません。")
        return docs, "pymupdf"
    if prefer in ("pypdf", "pdf"):
        return PyPDFLoader(pdf_path).load(), "pypdf"

    # auto
    docs = _try_pymupdf()
    if docs is not None:
        return docs, "pymupdf"
    return PyPDFLoader(pdf_path).load(), "pypdf"


def _split_pdf_docs(pdf_path: str, source_label: str):
    docs, loader_used = _load_pdf_docs_best_effort(pdf_path)
    for d in docs:
        d.metadata = dict(d.metadata or {})
        d.metadata["source"] = source_label
        d.metadata["loader"] = loader_used
        d.page_content = _clean_pdf_text(getattr(d, "page_content", "") or "")
    _strip_repeated_header_footer(docs)
    splitter = _get_splitter()
    chunks = splitter.split_documents(docs)
    return [c for c in chunks if (c.page_content or "").strip()]


def _analyze_and_split_pdf(pdf_path: str) -> tuple[list[Any], dict[str, Any]]:
    """
    PDFを読み込み、チャンク化して返す。
    併せて「テキスト抽出できたか（OCR不足か）」等のユーザー向け情報を返す。
    """
    docs, loader_used = _load_pdf_docs_best_effort(pdf_path)
    pages = len(docs)
    for d in docs:
        d.page_content = _clean_pdf_text(getattr(d, "page_content", "") or "")

    strip_rep = _strip_repeated_header_footer(docs)

    extracted_chars = 0
    for d in docs:
        extracted_chars += len((getattr(d, "page_content", "") or "").strip())

    # source は表示用に basename を固定
    source_label = os.path.basename(pdf_path)
    for d in docs:
        d.metadata = dict(d.metadata or {})
        d.metadata["source"] = source_label
        d.metadata["loader"] = loader_used

    splitter = _get_splitter()
    chunks = splitter.split_documents(docs)
    chunks = [c for c in chunks if (c.page_content or "").strip()]
    chunk_count = len(chunks)

    ocr_needed = (pages > 0 and extracted_chars < 20) or chunk_count == 0
    status = "ok"
    hints: list[str] = []
    if ocr_needed:
        status = "ocr_needed"
        hints = [
            "スキャンPDFの場合はOCRして、テキスト入りのPDFにしてから配置してください",
            "OCR後に POST /reload（UIの「再インデックス」）で再インデックスしてください",
        ]

    report = {
        "status": status,
        "pages": pages,
        "extracted_chars": extracted_chars,
        "chunk_count": chunk_count,
        "loader": loader_used,
        "cleanup": strip_rep,
        "hints": hints,
    }
    return chunks, report


def _build_chunks_with_report(paths: list[str]) -> tuple[list[Any], list[dict[str, Any]], dict[str, Any]]:
    chunks: list[Any] = []
    items: list[dict[str, Any]] = []
    summary = {
        "pdf_total": len(paths),
        "pdf_ok": 0,
        "pdf_ocr_needed": 0,
        "pdf_error": 0,
        "chunks_total": 0,
        "pages_total": 0,
        "extracted_chars_total": 0,
    }
    for p in paths:
        base = {"path": os.path.abspath(p), "filename": os.path.basename(p)}
        if not os.path.exists(p):
            items.append({**base, "status": "missing", "error": "ファイルが存在しません", "hints": ["PDFの配置先（PDF_DIR）を確認してください"]})
            summary["pdf_error"] += 1
            continue
        try:
            ch, rep = _analyze_and_split_pdf(p)
            items.append({**base, **rep})
            summary["pages_total"] += int(rep.get("pages") or 0)
            summary["extracted_chars_total"] += int(rep.get("extracted_chars") or 0)
            summary["chunks_total"] += int(rep.get("chunk_count") or 0)
            if rep.get("status") == "ok":
                summary["pdf_ok"] += 1
            elif rep.get("status") == "ocr_needed":
                summary["pdf_ocr_needed"] += 1
            else:
                summary["pdf_error"] += 1
            chunks.extend(ch)
        except Exception as e:
            items.append(
                {
                    **base,
                    "status": "error",
                    "error": str(e),
                    "hints": ["PDFが壊れていないか確認してください", "スキャンPDFの場合はOCRしてから配置してください"],
                }
            )
            summary["pdf_error"] += 1

    return chunks, items, summary


def _rebuild_vectorstore(paths: list[str], persist_dir: str) -> tuple[Chroma, dict[str, Any]]:
    # 先にPDFを解析して「空/失敗」を判断（失敗した場合に既存DBを消さない）
    chunks, report_items, summary = _build_chunks_with_report(paths)
    report = {"items": report_items, "summary": summary}

    if not chunks:
        if not paths:
            raise IndexBuildError(
                "PDFが未配置です。./pdfs/（環境変数 PDF_DIR）にPDFを配置して、再インデックスしてください。",
                report,
            )
        if summary.get("pdf_ocr_needed", 0) >= 1 and summary.get("extracted_chars_total", 0) < 20:
            raise IndexBuildError(
                "PDFは見つかりましたが、テキストを抽出できませんでした。スキャンPDFの可能性があります（OCRしてから配置してください）。",
                report,
            )
        raise IndexBuildError(
            "PDFは見つかりましたが、取り込みに失敗しました。PDFが壊れていないか、OCR済みかを確認してください。",
            report,
        )

    # embedding モデルが無いと RAG は構築できない（ここで明示的にエラーにする）
    # 失敗しても既存DBを消さないよう、persist_dir を触る前にチェックする。
    try:
        timeout_s = float(os.environ.get("INDEX_MODEL_CHECK_TIMEOUT_S", "3.0"))
        model_names = _ollama_cli_list_models(timeout_s=timeout_s)
        if not _has_model(model_names, DEFAULT_EMBED_MODEL):
            raise IndexBuildError(f"Embeddingモデル（{DEFAULT_EMBED_MODEL}）が見つかりません。", report)
    except IndexBuildError:
        raise
    except FileNotFoundError:
        raise IndexBuildError("Ollama（ollama コマンド）が見つかりません。", report)
    except subprocess.TimeoutExpired:
        raise IndexBuildError("Ollama のモデル一覧取得（ollama list）がタイムアウトしました。", report)
    except Exception as e:
        raise IndexBuildError(f"Embeddingモデルの準備確認に失敗しました: {e}", report)

    # 誤爆防止（危険なパスは触らない）
    if not _is_safe_persist_dir(persist_dir):
        raise IndexBuildError(
            f"安全のため persist_dir の自動削除は chroma_db* のみに限定しています: {persist_dir}",
            report,
        )

    # 既存DBを壊さないため、まず別ディレクトリにビルドしてからアトミックに置換する
    stamp = int(time.time() * 1000)
    build_dir = f"{persist_dir}.__building__{stamp}"
    backup_dir = f"{persist_dir}.__backup__{stamp}"

    # 古い途中生成物があれば掃除（安全域のみ）
    try:
        if os.path.isdir(build_dir):
            shutil.rmtree(build_dir)
        if os.path.isdir(backup_dir):
            shutil.rmtree(backup_dir)
    except Exception:
        pass

    os.makedirs(build_dir, exist_ok=True)
    embeddings = OllamaEmbeddings(model=DEFAULT_EMBED_MODEL)
    vs_build = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=build_dir)
    _persist_if_supported(vs_build)

    # 置換（失敗時はできるだけロールバック）
    try:
        if os.path.isdir(persist_dir):
            os.replace(persist_dir, backup_dir)
        elif os.path.exists(persist_dir):
            # 予期せぬファイル形状は危険なので中断
            raise RuntimeError(f"persist_dir がディレクトリではありません: {persist_dir}")
        os.replace(build_dir, persist_dir)
        if os.path.isdir(backup_dir):
            shutil.rmtree(backup_dir)
    except Exception:
        # ロールバック（best effort）
        try:
            if (not os.path.isdir(persist_dir)) and os.path.isdir(backup_dir):
                os.replace(backup_dir, persist_dir)
        except Exception:
            pass
        try:
            if os.path.isdir(build_dir):
                shutil.rmtree(build_dir)
        except Exception:
            pass
        raise

    # 最終パスでロードし直す（内部参照のズレを避ける）
    vs = _load_vectorstore(persist_dir)
    return vs, report


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


def _ensure_index_uptodate(force: bool = False, trigger: str = "auto") -> None:
    persist_dir = app.state.persist_dir
    pdf_path = app.state.pdf_path
    pdf_dir = app.state.pdf_dir

    paths = _list_pdf_paths(pdf_path, pdf_dir)
    sig = _source_signature(paths)

    last_sig = getattr(app.state, "source_signature", None)
    if (not force) and app.state.vectorstore is not None and last_sig == sig:
        return

    # 既存DBがあり、ソースが変わっていないならロードで済ませる
    #
    # 重要:
    # - last_sig が一致しない（=未知/変更あり）状態で Chroma を先に初期化すると、
    #   その後の再構築で persist_dir を削除した際に内部状態が不整合になり、
    #   SQLite が「readonly database」になるケースがあるため、先に一致確認を行う。
    if (not force) and last_sig == sig and os.path.isdir(persist_dir):
        try:
            vs = _load_vectorstore(persist_dir)
            if _is_vectorstore_ready(vs):
                app.state.vectorstore = vs
                return
        except Exception:
            pass

    run_id = int(time.time() * 1000)
    t0 = time.perf_counter()
    _update_index_status(
        {
            "running": True,
            "run_id": run_id,
            "trigger": trigger,
            "started_at": _utc_now_iso(),
            "finished_at": None,
            "last_error": None,
        }
    )
    try:
        logger.info(
            "index_start",
            extra={
                "event": "index_start",
                "stage": "index",
                "extra": {"run_id": run_id, "trigger": trigger, "force": bool(force), "pdf_count": len(paths), "persist_dir": persist_dir},
            },
        )
    except Exception:
        pass

    try:
        prev_sig = getattr(app.state, "source_signature", None)

        # 作り直し（失敗時に既存DBを維持できるよう、_rebuild_vectorstore 側で先に解析する）
        try:
            vs, report = _rebuild_vectorstore(paths, persist_dir)
        except Exception as e:
            # 永続化先が read-only/権限不足なら、ユーザー書き込み領域へ退避して1回だけ再試行
            # - 既存の DB が root 所有/権限不整合になっているケースで自己回復しやすくする
            if _is_readonly_db_error(e):
                req = getattr(app.state, "persist_dir_requested", persist_dir) or persist_dir
                fallback, warns = _resolve_persist_dir(str(req))
                if fallback and fallback != persist_dir:
                    app.state.persist_dir = fallback
                    app.state.persist_dir_warnings = list(warns or []) + [
                        f"永続化先の書き込みエラーのため、{fallback} へ切り替えて再試行します。"
                    ]
                    vs, report = _rebuild_vectorstore(paths, fallback)
                else:
                    raise
            else:
                raise
        app.state.vectorstore = vs
        app.state.source_signature = sig
        app.state.last_indexed_at = _utc_now_iso()
        app.state.init_error = None
        _update_index_status(
            {
                "running": False,
                "finished_at": _utc_now_iso(),
                "last_success_at": app.state.last_indexed_at,
                "last_report": report,
                "last_error": None,
            }
        )
        _save_index_manifest(
            persist_dir,
            {
                "source_signature": sig,
                "last_indexed_at": app.state.last_indexed_at,
                "pdf_path": pdf_path,
                "pdf_dir": pdf_dir,
                "index_status": _get_index_status(),
            },
        )
        try:
            logger.info(
                "index_success",
                extra={
                    "event": "index_success",
                    "stage": "index",
                    "timings": {"index_ms": int((time.perf_counter() - t0) * 1000)},
                    "extra": {"run_id": run_id, "trigger": trigger, "summary": (report or {}).get("summary")},
                },
            )
        except Exception:
            pass
    except Exception as e:
        # 失敗時は、既存の vectorstore / signature をなるべく維持する（途中で消していない想定）
        app.state.init_error = str(e)
        report = e.report if isinstance(e, IndexBuildError) else None
        # 失敗理由に応じて「次にやること」を出し分け（PDF問題と Ollama/モデル不足を分離）
        err_code = "INDEX_BUILD_FAILED"
        err_hints: list[str] = [
            "PDFを ./pdfs/（環境変数 PDF_DIR）に配置してください",
            "スキャンPDFの場合はOCRしてから配置してください",
            "PDFを追加・更新したら POST /reload で再インデックスしてください",
        ]
        if _is_readonly_db_error(e):
            err_code = "PERSIST_DIR_READONLY"
            err_hints = [
                "CHROMA_PERSIST_DIR（既定: ./chroma_db）が書き込み可能か確認してください（権限/所有者/マウント設定）",
                "必要なら CHROMA_PERSIST_DIR をユーザー書き込み可能な場所へ変更してください（例: macOS は ~/Library/Application Support/vaccine-chatbot/chroma_db）",
                "書き込み先を直した後に POST /reload を実行してください",
            ]
        elif _is_ollama_down_error(e):
            err_code = "OLLAMA_UNAVAILABLE"
            err_hints = [
                "Ollama が起動しているか確認してください",
                f"Embeddingモデルが存在するか確認してください（ollama pull {DEFAULT_EMBED_MODEL}）",
            ]
        elif _is_model_not_found_error(e, DEFAULT_EMBED_MODEL):
            err_code = "EMBEDDING_MODEL_NOT_FOUND"
            err_hints = [
                f"ollama pull {DEFAULT_EMBED_MODEL}",
                "RAG（PDF検索）は embedding モデルが無いと動きません",
            ]
        _update_index_status(
            {
                "running": False,
                "finished_at": _utc_now_iso(),
                "last_report": report,
                "last_error": {
                    "message": str(e),
                    "code": err_code,
                    "hints": err_hints,
                    "at": _utc_now_iso(),
                    "report": report,
                },
            }
        )
        _save_index_manifest(
            persist_dir,
            {
                "source_signature": getattr(app.state, "source_signature", prev_sig),
                "last_indexed_at": getattr(app.state, "last_indexed_at", None),
                "pdf_path": pdf_path,
                "pdf_dir": pdf_dir,
                "index_status": _get_index_status(),
            },
        )
        try:
            logger.error(
                "index_failed",
                extra={
                    "event": "index_failed",
                    "stage": "index",
                    "code": err_code,
                    "timings": {"index_ms": int((time.perf_counter() - t0) * 1000)},
                    "extra": {"run_id": run_id, "trigger": trigger, "error": str(e), "report": report},
                },
            )
        except Exception:
            pass
        # 例外は上位へ（/reload などでエラー表示できるように）
        raise


@app.on_event("startup")
async def _startup() -> None:
    persist_dir_req = os.environ.get("CHROMA_PERSIST_DIR", DEFAULT_PERSIST_DIR)
    persist_dir, persist_warnings = _resolve_persist_dir(persist_dir_req)
    pdf_path = os.environ.get("PDF_PATH", DEFAULT_PDF_PATH)
    pdf_dir = os.environ.get("PDF_DIR", DEFAULT_PDF_DIR)
    run_mode = os.environ.get("RUN_MODE", "prod").strip() or "prod"
    git_sha = _get_git_sha(_repo_root())

    app.state.vectorstore = None
    app.state.init_error = None
    app.state.persist_dir = persist_dir
    app.state.persist_dir_requested = persist_dir_req
    app.state.persist_dir_warnings = persist_warnings
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
    app.state.index_status_lock = threading.Lock()
    app.state.index_status = _init_index_status()
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
        if isinstance(m, dict) and isinstance(m.get("index_status"), dict):
            # 起動後に /sources で理由を出せるよう、前回の情報を引き継ぐ
            _update_index_status(m.get("index_status"))
    except Exception:
        pass

    # 起動時にベクトルDBを準備（失敗しても落とさず /status に理由を出す）
    try:
        _ensure_index_uptodate(force=False, trigger="startup")
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
        logger.info(
            "startup",
            extra={
                "event": "startup",
                "stage": "startup",
                "extra": {
                    "version": getattr(app.state, "version", None),
                    "git_sha": getattr(app.state, "git_sha", None),
                    "started_at": getattr(app.state, "started_at", None),
                    "run_mode": getattr(app.state, "run_mode", None),
                    "pdf_path": pdf_path,
                    "pdf_dir": pdf_dir,
                    "legacy_pdf_dir": LEGACY_PDF_DIR,
                    "persist_dir": persist_dir,
                    "persist_dir_requested": persist_dir_req,
                    "persist_dir_warnings": persist_warnings,
                    "embed_model": DEFAULT_EMBED_MODEL,
                    "embed_cache_max": int(os.environ.get("EMBED_CACHE_MAX", str(DEFAULT_CACHE_MAX_ENTRIES))),
                    "python": platform.python_version(),
                    "cwd": os.getcwd(),
                },
            },
        )
    except Exception:
        pass


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {"status": "ok", "ready": app.state.vectorstore is not None}


@app.get("/status")
def status() -> dict[str, Any]:
    pdfs = _list_pdf_paths(getattr(app.state, "pdf_path", DEFAULT_PDF_PATH), getattr(app.state, "pdf_dir", DEFAULT_PDF_DIR))
    with _error_lock:
        recent = list(_recent_errors)
        counts = dict(_error_counts)
    return {
        "version": getattr(app.state, "version", None),
        "git_sha": getattr(app.state, "git_sha", None),
        "started_at": getattr(app.state, "started_at", None),
        "run_mode": getattr(app.state, "run_mode", None),
        "ready": app.state.vectorstore is not None,
        "persist_dir": getattr(app.state, "persist_dir", DEFAULT_PERSIST_DIR),
        "persist_dir_requested": getattr(app.state, "persist_dir_requested", None),
        "persist_dir_warnings": getattr(app.state, "persist_dir_warnings", None),
        "pdf_path": getattr(app.state, "pdf_path", DEFAULT_PDF_PATH),
        "pdf_dir": getattr(app.state, "pdf_dir", DEFAULT_PDF_DIR),
        "legacy_pdf_dir": LEGACY_PDF_DIR,
        "pdf_count": len(pdfs),
        "last_indexed_at": getattr(app.state, "last_indexed_at", None),
        "init_error": getattr(app.state, "init_error", None),
        "indexing": _get_index_status(),
        "embed_model": DEFAULT_EMBED_MODEL,
        "embed_warmup": getattr(app.state, "embed_warmup", None),
        "embedding_cache": getattr(app.state, "embedding_cache", _LRUCache(DEFAULT_CACHE_MAX_ENTRIES)).stats(),
        "last_request_timings": getattr(app.state, "last_request_timings", None),
        "recent_errors": recent[-20:],
        "error_counts": counts,
    }


@app.get("/sources")
def list_sources() -> dict[str, Any]:
    pdf_path = getattr(app.state, "pdf_path", DEFAULT_PDF_PATH)
    pdf_dir = getattr(app.state, "pdf_dir", DEFAULT_PDF_DIR)
    persist_dir = getattr(app.state, "persist_dir", DEFAULT_PERSIST_DIR)

    paths = _list_pdf_paths(pdf_path, pdf_dir)
    sig = _source_signature(paths)

    index_status = _get_index_status()
    report = index_status.get("last_report") or {}
    report_items = report.get("items") if isinstance(report, dict) else None
    report_map: dict[str, Any] = {}
    if isinstance(report_items, list):
        for it in report_items:
            if isinstance(it, dict) and isinstance(it.get("path"), str):
                report_map[it["path"]] = it

    items: list[dict[str, Any]] = []
    for s in sig:
        p = s.get("path")
        rep = report_map.get(p) if isinstance(p, str) else None
        items.append(
            {
                "type": "pdf",
                "filename": s.get("filename"),
                "original_filename": s.get("filename"),
                "path": s.get("path"),
                "size_bytes": s.get("size_bytes"),
                "mtime": s.get("mtime"),
                "ingest": rep,
            }
        )

    indexed_sig = getattr(app.state, "source_signature", None)
    running = bool(index_status.get("running") is True)
    indexed = bool(app.state.vectorstore is not None and indexed_sig == sig and (not running))

    next_actions: list[str] = []
    if not items:
        next_actions = [
            f"PDFを {pdf_dir}/（環境変数 PDF_DIR）に配置してください（*.pdf）",
            "配置後に「再インデックス」（POST /reload）を実行してください",
            "スキャンPDFの場合はOCRしてテキスト入りPDFにしてから配置してください",
        ]
    else:
        if running:
            next_actions = ["インデックス実行中です。完了までお待ちください（実行中は送信できません）。"]
        else:
            any_ocr = any(
                isinstance(x, dict)
                and isinstance(x.get("ingest"), dict)
                and x["ingest"].get("status") == "ocr_needed"
                for x in items
            )
            if any_ocr:
                next_actions.append("一部PDFでテキスト抽出できていません（スキャンPDFの可能性）。OCRしてから再インデックスしてください。")
            if not indexed:
                next_actions.append("PDFを追加・更新した場合は「再インデックス」（POST /reload）を実行してください。")

    return {
        "ok": True,
        "indexed": indexed,
        "indexing": index_status,
        "last_indexed_at": getattr(app.state, "last_indexed_at", None),
        "persist_dir": persist_dir,
        "persist_dir_requested": getattr(app.state, "persist_dir_requested", None),
        "persist_dir_warnings": getattr(app.state, "persist_dir_warnings", None),
        "pdf_path": pdf_path,
        "pdf_dir": pdf_dir,
        "legacy_pdf_dir": LEGACY_PDF_DIR,
        "init_error": getattr(app.state, "init_error", None),
        "error": index_status.get("last_error"),
        "next_actions": next_actions,
        "items": items,
    }


@app.post("/reload")
async def reload_sources(request: Request) -> dict[str, Any]:
    lock: asyncio.Lock = app.state.ingest_lock

    # 多重実行は待たずに拒否（UIからも分かるように running を返す）
    if lock.locked():
        logger.warning(
            "reload_rejected",
            extra={
                "event": "reload_rejected",
                "method": request.method,
                "path": request.url.path,
                "stage": "index",
                "code": "INDEXING_IN_PROGRESS",
                "extra": {"indexing": _get_index_status()},
            },
        )
        return JSONResponse(
            status_code=409,
            content={
                "ok": False,
                "accepted": False,
                "running": True,
                "indexing": _get_index_status(),
                "message": "すでに再インデックスが実行中です。",
            },
        )

    # 非同期で受け付け（UIは /sources をポーリングして状態を見る）
    run_id = int(time.time() * 1000)
    _update_index_status(
        {
            "running": True,
            "run_id": run_id,
            "trigger": "reload",
            "started_at": _utc_now_iso(),
            "finished_at": None,
            "last_error": None,
        }
    )

    async def _task():
        async with lock:
            try:
                await asyncio.to_thread(_ensure_index_uptodate, True, "reload")
            except Exception:
                # 失敗内容は _ensure_index_uptodate 側で index_status / init_error に記録される
                pass

    app.state.index_task = asyncio.create_task(_task())
    logger.info(
        "reload_accepted",
        extra={
            "event": "reload_accepted",
            "method": request.method,
            "path": request.url.path,
            "stage": "index",
            "extra": {"run_id": run_id},
        },
    )
    return JSONResponse(
        status_code=202,
        content={
            "ok": True,
            "accepted": True,
            "running": True,
            "indexing": _get_index_status(),
            "message": "再インデックスを開始しました。完了までしばらくお待ちください。",
        },
    )


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
    detail: dict[str, Any] = {
        "stage": str(stage or "unknown"),
        "code": str(code or "UNKNOWN"),
        "message": str(message or ""),
        # UIが常に配列として扱えるよう、未指定でも空配列を入れる
        "hints": [str(x) for x in (hints or [])],
    }
    if timeout_s is not None:
        detail["timeout_s"] = int(timeout_s)
    if timings:
        detail["timings"] = timings
    if extra:
        detail.update(extra)
    raise HTTPException(status_code=status_code, detail=detail)


def _normalize_error_detail(detail: Any) -> dict[str, Any]:
    """
    UIが stage/code/hints を前提にできるよう、HTTPException.detail を正規化する。
    - stage/code/message: 常に文字列
    - hints: 常に文字列配列
    """
    if isinstance(detail, dict):
        d = dict(detail)
    else:
        d = {"message": str(detail)}

    d.setdefault("stage", "unknown")
    d.setdefault("code", "HTTP_ERROR")
    d.setdefault("message", "")
    d["stage"] = str(d.get("stage") or "unknown")
    d["code"] = str(d.get("code") or "HTTP_ERROR")
    d["message"] = str(d.get("message") or "")

    hints = d.get("hints")
    if isinstance(hints, list):
        d["hints"] = [str(x) for x in hints]
    elif hints is None:
        d["hints"] = []
    else:
        d["hints"] = [str(hints)]

    return d


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


def _is_model_not_found_error(e: Exception, model: str) -> bool:
    """
    Ollama 側にモデルが存在しない（pull が必要）ケースを雑に検出する。
    例: "model 'nomic-embed-text' not found" 等
    """
    wanted = (model or "").strip()
    if not wanted:
        return False
    msg = str(e)
    low = msg.lower()
    if wanted.lower() not in low and f"（{wanted}）" not in msg:
        return False
    return any(s in low for s in ["not found", "model not found", "no such file", "missing"]) or "見つかりません" in msg


def _ollama_cli_list_models(timeout_s: float = 3.0) -> list[str]:
    """
    `ollama list` の出力をモデル名配列にする（タイムアウト制御のため CLI を使用）。
    出力形式の揺れに備えて、先頭カラム（NAME）だけを拾う。
    """
    if shutil.which("ollama") is None:
        raise FileNotFoundError("ollama command not found")
    proc = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True,
        timeout=float(timeout_s),
        check=False,
    )
    out = (proc.stdout or "").splitlines()
    names: list[str] = []
    for ln in out:
        t = (ln or "").strip()
        if not t:
            continue
        if t.upper().startswith("NAME"):
            continue
        name = t.split()[0].strip()
        if name:
            names.append(name)
    return names


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
    timings: dict[str, Any] = {}
    t_total0 = time.perf_counter()

    # PDF配置の変更があれば自動追従（重い処理なので排他）
    lock: asyncio.Lock = app.state.ingest_lock
    if lock.locked() and bool(_get_index_status().get("running") is True):
        _http_error(
            stage="search",
            code="INDEXING_IN_PROGRESS",
            message="インデックス処理が実行中のため、いまは回答できません",
            hints=["インデックス完了を待ってから再送してください", "UIの「参照ソース（PDF一覧）」で実行状況とエラー内容を確認できます"],
            extra={"indexing": _get_index_status()},
            status_code=503,
        )

    index_check_timeout_s = int(os.environ.get("INDEX_CHECK_TIMEOUT_S", str(DEFAULT_INDEX_CHECK_TIMEOUT_S)))
    t_index0 = time.perf_counter()
    async with lock:
        try:
            await asyncio.wait_for(asyncio.to_thread(_ensure_index_uptodate, False, "auto"), timeout=index_check_timeout_s)
            app.state.init_error = None
            timings["index_check_ms"] = int((time.perf_counter() - t_index0) * 1000)
        except TimeoutError:
            timings["index_check_ms"] = int((time.perf_counter() - t_index0) * 1000)
            _http_error(
                stage="index_check",
                code="INDEX_CHECK_TIMEOUT",
                message="index確認（PDF差分チェック/必要なら再インデックス）がタイムアウトしました",
                timeout_s=index_check_timeout_s,
                hints=[
                    "PDFの追加直後はインデックスが重い場合があります。POST /reload を実行し、完了後に再送してください",
                    "PDF数が多い場合は分割・整理を検討してください",
                    "サーバー負荷が高い場合は時間をおいて再送してください",
                ],
                timings={**timings, "total_ms": int((time.perf_counter() - t_total0) * 1000)},
                extra={"indexing": _get_index_status()},
                status_code=504,
            )
        except Exception as e:
            timings["index_check_ms"] = int((time.perf_counter() - t_index0) * 1000)
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
            timings={**timings, "total_ms": int((time.perf_counter() - t_total0) * 1000)},
            extra={
                "persist_dir": getattr(app.state, "persist_dir", DEFAULT_PERSIST_DIR),
                "pdf_path": getattr(app.state, "pdf_path", DEFAULT_PDF_PATH),
                "pdf_dir": getattr(app.state, "pdf_dir", DEFAULT_PDF_DIR),
                "init_error": getattr(app.state, "init_error", None),
            },
            status_code=503,
        )

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
        if _is_model_not_found_error(e, DEFAULT_EMBED_MODEL):
            _http_error(
                stage="embedding",
                code="EMBEDDING_MODEL_NOT_FOUND",
                message=f"Embeddingモデル（{DEFAULT_EMBED_MODEL}）が見つかりません。",
                hints=[f"ollama pull {DEFAULT_EMBED_MODEL}", "RAG（PDF検索）は embedding モデルが無いと動きません"],
                timings=timings,
                extra={"error": str(e)},
                status_code=503,
            )
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

    # まれに検索結果が0件になるケースを救う（kを増やして1回だけ再試行）
    # - “資料にあるはずなのに常に資料にない” を軽減する目的
    # - 追加のembeddingは不要（同じ vec を再利用）
    if (not docs) and k < 20:
        retry_k = min(20, max(8, k * 4))
        t_retry0 = time.perf_counter()
        try:
            fn = getattr(vs, "similarity_search_by_vector", None)
            if callable(fn):
                docs = await asyncio.wait_for(asyncio.to_thread(fn, vec, k=retry_k), timeout=search_timeout_s)
            else:
                docs = await asyncio.wait_for(asyncio.to_thread(vs.similarity_search, prompt, k=retry_k), timeout=search_timeout_s)
            timings["search_retry_k"] = retry_k
            timings["search_retry_ms"] = int((time.perf_counter() - t_retry0) * 1000)
        except Exception:
            # 再試行が失敗しても、元の検索結果（空）のまま返す
            pass

    timings["total_ms"] = int((time.perf_counter() - t_total0) * 1000)
    context = _format_context(docs)
    return {"docs": docs, "context": context, "timings": timings}


async def _run_generate(prompt: str, context: str, model: str, max_tokens: int, generate_timeout_s: int) -> dict[str, Any]:
    t0 = time.perf_counter()
    # context が空のときは、生成せずに固定フォーマットで返す（断定/hallucination防止）
    if not (context or "").strip():
        return {"answer": _no_sources_answer(prompt), "timings": {"generate_ms": int((time.perf_counter() - t0) * 1000)}}

    full_prompt = _build_answer_prompt(question=prompt, context=context)
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


def _env_allow_general_fallback_default() -> bool:
    v = (os.environ.get("ALLOW_GENERAL_FALLBACK_DEFAULT", "0") or "0").strip().lower()
    return v in ("1", "true", "yes", "y", "on")


async def _run_generate_general_fallback(prompt: str, model: str, max_tokens: int, generate_timeout_s: int) -> dict[str, Any]:
    t0 = time.perf_counter()
    full_prompt = _build_general_fallback_prompt(question=prompt, reference=FALLBACK_KNOWLEDGE_BASE)
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
async def search_endpoint(payload: SearchRequest, request: Request):
    et, st, _ = _resolve_timeouts(payload.timeout_s, payload.embedding_timeout_s, payload.search_timeout_s, None)
    result = await _run_search(payload.prompt, payload.k, embedding_timeout_s=et, search_timeout_s=st)
    docs = result["docs"]
    timings = result["timings"]
    app.state.last_request_timings = {"stage": "search", **timings}
    logger.info(
        "search",
        extra={
            "event": "search",
            "method": request.method,
            "path": request.url.path,
            "stage": "search",
            "extra": {"k": payload.k},
            "timings": timings,
        },
    )
    return {"context": result["context"], "sources": _extract_sources(docs), "timings": timings}


@app.post("/generate")
async def generate_endpoint(payload: GenerateRequest, request: Request):
    _, _, gt = _resolve_timeouts(payload.timeout_s, None, None, payload.generate_timeout_s)
    allow_fallback = payload.allow_general_fallback if payload.allow_general_fallback is not None else _env_allow_general_fallback_default()
    context = (payload.context or "")[: MAX_CONTEXT_CHARS + 500]
    if allow_fallback and not context.strip():
        result = await _run_generate_general_fallback(
            prompt=payload.prompt,
            model=payload.model,
            max_tokens=payload.max_tokens,
            generate_timeout_s=gt,
        )
    else:
        result = await _run_generate(
            prompt=payload.prompt,
            context=context,
            model=payload.model,
            max_tokens=payload.max_tokens,
            generate_timeout_s=gt,
        )
    timings = result["timings"]
    app.state.last_request_timings = {"stage": "generate", **timings}
    logger.info(
        "generate",
        extra={
            "event": "generate",
            "method": request.method,
            "path": request.url.path,
            "stage": "generate",
            "extra": {"model": payload.model},
            "timings": timings,
        },
    )
    return {"answer": result["answer"], "timings": timings}


@app.post("/chat")
async def chat_endpoint(payload: ChatRequest, request: Request):
    et, st, gt = _resolve_timeouts(
        payload.timeout_s,
        payload.embedding_timeout_s,
        payload.search_timeout_s,
        payload.generate_timeout_s,
    )

    search_res = await _run_search(payload.prompt, payload.k, embedding_timeout_s=et, search_timeout_s=st)
    docs = search_res["docs"]
    context = search_res["context"]
    timings = dict(search_res["timings"])
    sources = _extract_sources(docs)

    allow_fallback = payload.allow_general_fallback if payload.allow_general_fallback is not None else _env_allow_general_fallback_default()

    # sources が取れない（=根拠0件）の場合:
    # - 既定: 生成せず「資料にない」（従来どおり）
    # - allow_fallback: “参照PDFの根拠なし”を明示した上で、一般説明として回答を返す
    if not sources:
        if allow_fallback:
            gen_res = await _run_generate_general_fallback(
                prompt=payload.prompt,
                model=payload.model,
                max_tokens=payload.max_tokens,
                generate_timeout_s=gt,
            )
            answer = gen_res["answer"]
            timings["generate_ms"] = gen_res["timings"]["generate_ms"]
            timings["total_ms"] = int(
                (timings.get("index_check_ms", 0) + timings.get("embedding_ms", 0) + timings.get("search_ms", 0) + timings.get("generate_ms", 0))
            )
        else:
            answer = _no_sources_answer(payload.prompt)
            timings["generate_ms"] = 0
            timings["total_ms"] = int((timings.get("index_check_ms", 0) + timings.get("embedding_ms", 0) + timings.get("search_ms", 0)))
        app.state.last_request_timings = {"stage": "chat", **timings}
        logger.info(
            "chat",
            extra={
                "event": "chat",
                "method": request.method,
                "path": request.url.path,
                "stage": "chat",
                "extra": {"k": payload.k, "model": payload.model, "sources": 0, "fallback": bool(allow_fallback)},
                "timings": timings,
            },
        )
        return {"answer": answer, "sources": [], "timings": timings}

    gen_res = await _run_generate(
        prompt=payload.prompt,
        context=context,
        model=payload.model,
        max_tokens=payload.max_tokens,
        generate_timeout_s=gt,
    )
    timings["generate_ms"] = gen_res["timings"]["generate_ms"]
    timings["total_ms"] = int(
        (timings.get("index_check_ms", 0) + timings.get("embedding_ms", 0) + timings.get("search_ms", 0) + timings.get("generate_ms", 0))
    )

    app.state.last_request_timings = {"stage": "chat", **timings}
    logger.info(
        "chat",
        extra={
            "event": "chat",
            "method": request.method,
            "path": request.url.path,
            "stage": "chat",
            "extra": {"k": payload.k, "model": payload.model, "sources": len(sources)},
            "timings": timings,
        },
    )
    return {"answer": gen_res["answer"], "sources": sources, "timings": timings}


def _level_rank(level: str) -> int:
    lv = (level or "").strip().lower()
    if lv == "red":
        return 2
    if lv == "yellow":
        return 1
    return 0


def _max_level(levels: list[str]) -> str:
    best = "green"
    for lv in levels:
        if _level_rank(lv) > _level_rank(best):
            best = lv
    return best


def _model_names_from_ollama_list(payload: Any) -> list[str]:
    """
    ollama.list() の戻りはバージョン差があり得るため、防御的にモデル名配列へ正規化する。
    期待値: {"models":[{"name":"gemma2:2b", ...}, ...]}
    """
    # 新しめの ollama Python SDK では ListResponse(models=[Model(model="gemma2:2b", ...), ...]) のような
    # “dict ではない”戻りになることがあるため、両方を吸収する。
    try:
        # pydantic v2 系: model_dump
        if not isinstance(payload, dict) and hasattr(payload, "model_dump"):
            dumped = payload.model_dump()  # type: ignore[attr-defined]
            if isinstance(dumped, dict):
                payload = dumped
    except Exception:
        pass

    models = payload.get("models") if isinstance(payload, dict) else getattr(payload, "models", None)
    if models is None:
        return []
    if not isinstance(models, list):
        try:
            models = list(models)  # type: ignore[arg-type]
        except Exception:
            return []

    names: list[str] = []
    for m in models:
        name: str | None = None
        if isinstance(m, dict):
            v = m.get("name") or m.get("model")
            if isinstance(v, str) and v.strip():
                name = v.strip()
        else:
            for attr in ("name", "model"):
                v = getattr(m, attr, None)
                if isinstance(v, str) and v.strip():
                    name = v.strip()
                    break
            if name is None:
                try:
                    md = m.model_dump() if hasattr(m, "model_dump") else (m.dict() if hasattr(m, "dict") else None)
                    if isinstance(md, dict):
                        v = md.get("name") or md.get("model")
                        if isinstance(v, str) and v.strip():
                            name = v.strip()
                except Exception:
                    pass
        if name:
            names.append(name)
    return names


def _has_model(model_names: list[str], wanted: str) -> bool:
    w = (wanted or "").strip()
    if not w:
        return False
    # ollama list では :latest が付くことがあるので、prefix も許容
    return any(n == w or n.startswith(w + ":") for n in model_names)


@app.get("/diagnostics")
async def diagnostics(model: str | None = None) -> dict[str, Any]:
    """
    GitHub Pages（docs/）の「環境チェック」から呼ばれる診断API。
    - Ollama 疎通
    - モデルの有無（生成/embedding）
    - embedding の簡易スモークテスト
    """
    requested_model = (model or DEFAULT_LLM_MODEL).strip() or DEFAULT_LLM_MODEL
    embed_model = DEFAULT_EMBED_MODEL

    checks: list[dict[str, Any]] = []

    def add_check(level: str, label: str, message: str, hints: Optional[list[str]] = None) -> None:
        item: dict[str, Any] = {"level": level, "label": label, "message": message}
        if hints:
            item["hints"] = [str(x) for x in hints]
        checks.append(item)

    # タイムアウトは短め（UIが固まらないように）
    list_timeout_s = float(os.environ.get("DIAG_LIST_TIMEOUT_S", "3.0"))
    embed_timeout_s = float(os.environ.get("DIAG_EMBED_TIMEOUT_S", "6.0"))

    model_names: list[str] = []
    ollama_ok = False
    try:
        info = await asyncio.wait_for(asyncio.to_thread(ollama.list), timeout=list_timeout_s)
        model_names = _model_names_from_ollama_list(info)
        ollama_ok = True
        add_check("green", "Ollama疎通", "Ollama に接続できました。")
    except TimeoutError:
        add_check(
            "red",
            "Ollama疎通",
            "Ollama への接続がタイムアウトしました。",
            hints=[
                "Ollama が起動しているか確認してください（例: brew services start ollama）",
                "サーバー負荷が高い場合は時間をおいて再試行してください",
            ],
        )
    except Exception as e:
        add_check(
            "red",
            "Ollama疎通",
            f"Ollama に接続できませんでした: {e}",
            hints=[
                "Ollama が起動しているか確認してください",
                "環境変数 OLLAMA_HOST を変更している場合は値を確認してください",
            ],
        )

    if ollama_ok:
        if _has_model(model_names, requested_model):
            add_check("green", "回答モデル", f"回答モデル（{requested_model}）が見つかりました。")
        else:
            add_check(
                "red",
                "回答モデル",
                f"回答モデル（{requested_model}）が見つかりません。",
                hints=[f"ollama pull {requested_model}", "UIのモデル選択を、インストール済みモデルに変更してください"],
            )

        if _has_model(model_names, embed_model):
            add_check("green", "Embeddingモデル", f"Embeddingモデル（{embed_model}）が見つかりました。")
        else:
            add_check(
                "red",
                "Embeddingモデル",
                f"Embeddingモデル（{embed_model}）が見つかりません。",
                hints=[f"ollama pull {embed_model}", "RAG（PDF検索）は embedding モデルが無いと動きません"],
            )

        # embedding の簡易チェック（存在しないと確実に失敗するので、存在確認が通った場合だけ実施）
        if _has_model(model_names, embed_model):
            try:
                embeddings = getattr(app.state, "embeddings", None) or OllamaEmbeddings(model=embed_model)
                vec = await asyncio.wait_for(asyncio.to_thread(embeddings.embed_query, "diag"), timeout=embed_timeout_s)
                ok = isinstance(vec, list) and len(vec) > 0
                add_check("green" if ok else "yellow", "Embedding動作", "Embedding のスモークテストが完了しました。" if ok else "Embedding の結果が不正です。")
            except TimeoutError:
                add_check(
                    "yellow",
                    "Embedding動作",
                    "Embedding のスモークテストがタイムアウトしました（初回起動や高負荷の可能性）。",
                    hints=[
                        "初回はモデル起動で時間がかかる場合があります。少し待って再試行してください",
                        "必要なら DIAG_EMBED_TIMEOUT_S を延長してください",
                    ],
                )
            except Exception as e:
                add_check(
                    "red",
                    "Embedding動作",
                    f"Embedding のスモークテストに失敗しました: {e}",
                    hints=[
                        "Ollama が起動しているか確認してください",
                        f"ollama pull {embed_model} を実行してください",
                    ],
                )

    overall_level = _max_level([str(c.get("level") or "green") for c in checks] or ["yellow"])
    summary = {"green": "OK", "yellow": "一部注意", "red": "要対応"}.get(overall_level, "未確認")

    return {
        "ok": True,
        "overall": {"level": overall_level, "summary": summary},
        "checks": checks,
        "models": {"count": len(model_names), "names": model_names[:50]},
        "meta": {
            "requested_model": requested_model,
            "embed_model": embed_model,
            "run_mode": getattr(app.state, "run_mode", None),
            "version": getattr(app.state, "version", None),
            "git_sha": getattr(app.state, "git_sha", None),
            "started_at": getattr(app.state, "started_at", None),
        },
    }

