from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
from collections import Counter
from typing import Any, Optional


def _normalize_newlines(text: str) -> str:
    return (text or "").replace("\r\n", "\n").replace("\r", "\n")


def clean_pdf_text(text: str) -> str:
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


def strip_repeated_header_footer(
    docs: list[Any], *, top_n: int = 2, bottom_n: int = 2, min_ratio: float = 0.6
) -> dict[str, Any]:
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
        d.page_content = clean_pdf_text("\n".join(lines))

    return {"removed_top": removed_top, "removed_bottom": removed_bottom}


def _default_ocr_cache_dir() -> str:
    # 既定は作業ディレクトリ直下（launchd の WorkingDirectory を信頼）
    return os.environ.get("OCR_CACHE_DIR", "./ocr_cache").strip() or "./ocr_cache"


def _ocr_settings_from_env() -> dict[str, Any]:
    """
    OCR設定（環境変数）を取得する。

    - OCR_MODE:
      - off: OCRしない（従来どおり「OCRしてから置いてね」）
      - auto: テキスト抽出できないPDFのみ OCR を試す（推奨）
      - force: 常にOCR（既にテキストがあるPDFでも再OCR）
    """
    mode = (os.environ.get("OCR_MODE", "auto") or "auto").strip().lower()
    if mode not in ("off", "auto", "force"):
        mode = "auto"
    lang = (os.environ.get("OCR_LANG", "jpn+eng") or "jpn+eng").strip()
    # ocrmypdf の追加引数（必要ならユーザー側で微調整できるように）
    extra = (os.environ.get("OCRMYPDF_ARGS", "") or "").strip()
    return {"mode": mode, "lang": lang, "extra_args": extra}


def ocr_settings_signature() -> dict[str, Any]:
    s = _ocr_settings_from_env()
    # 署名は “変化検知” のためなので、必要最小限だけ
    return {
        "type": "ocr_settings",
        "ocr_mode": s.get("mode"),
        "ocr_lang": s.get("lang"),
        "ocr_cache_dir": _default_ocr_cache_dir(),
        "ocrmypdf_args": s.get("extra_args"),
    }


def _ocr_key_for_file(path: str) -> str:
    ap = os.path.abspath(path)
    try:
        st = os.stat(ap)
        basis = f"{ap}|{int(st.st_size)}|{float(st.st_mtime)}"
    except Exception:
        basis = f"{ap}|NA|NA"
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()  # nosec - 非暗号用途（キャッシュキー）


def _ocrmypdf_available() -> bool:
    return shutil.which("ocrmypdf") is not None


def _run_ocrmypdf(src_pdf: str, dst_pdf: str, *, mode: str, lang: str, extra_args: str) -> None:
    """
    ocrmypdf でテキストレイヤー付きPDFを生成する。
    - 実体は外部コマンド（ocrmypdf / tesseract / ghostscript）に依存する
    """
    cmd: list[str] = ["ocrmypdf"]

    # auto: 既にテキストがあるページはスキップ（通常はこれが最も安全）
    # force: 常にOCR（既存テキストがあっても再OCR）
    if mode == "force":
        cmd += ["--force-ocr"]
    else:
        cmd += ["--skip-text"]

    # 精度に効くことが多い前処理（ocrmypdf側）
    cmd += ["--rotate-pages", "--deskew", "--clean"]

    # 出力最適化は精度に寄与しないので抑制（互換性・速度優先）
    cmd += ["--output-type", "pdf", "--optimize", "0"]

    if lang:
        # `-l` は ocrmypdf/tesseract の一般的な指定
        cmd += ["-l", lang]

    if extra_args:
        # ユーザー指定をそのまま分割して追記（シェル展開しない）
        cmd += extra_args.split()

    cmd += [src_pdf, dst_pdf]

    os.makedirs(os.path.dirname(dst_pdf) or ".", exist_ok=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            "ocrmypdf に失敗しました。\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout: {(proc.stdout or '').strip()}\n"
            f"stderr: {(proc.stderr or '').strip()}"
        )


def maybe_ocr_pdf(pdf_path: str) -> tuple[str, dict[str, Any]]:
    """
    必要なら OCR を実行し、OCR済みPDFのパスを返す。
    - OCR_MODE=off: 何もしない
    - ocrmypdf が無い: 何もしない（レポートに理由を入れる）
    """
    s = _ocr_settings_from_env()
    mode = str(s.get("mode") or "auto")
    lang = str(s.get("lang") or "jpn+eng")
    extra = str(s.get("extra_args") or "")

    report: dict[str, Any] = {"enabled": mode != "off", "mode": mode, "lang": lang, "used": False, "tool": None, "path": None, "error": None}
    if mode == "off":
        return pdf_path, report

    if not _ocrmypdf_available():
        report["error"] = "ocrmypdf_not_found"
        return pdf_path, report

    key = _ocr_key_for_file(pdf_path)
    cache_dir = _default_ocr_cache_dir()
    out = os.path.join(cache_dir, f"ocr_{key}.pdf")
    report["tool"] = "ocrmypdf"
    report["path"] = out

    if os.path.exists(out):
        report["used"] = True
        return out, report

    _run_ocrmypdf(pdf_path, out, mode=mode, lang=lang, extra_args=extra)
    report["used"] = True
    return out, report


def load_pdf_docs_best_effort(pdf_path: str) -> tuple[list[Any], str]:
    """
    可能ならPyMuPDF系ローダを優先（レイアウト/段組に強いことが多い）。
    依存が無い場合は PyPDFLoader にフォールバック。
    `PDF_LOADER` で明示指定も可能（auto/pymupdf/pypdf）。
    """
    from langchain_community.document_loaders import PyPDFLoader

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


def load_pdf_docs_with_ocr_best_effort(pdf_path: str, *, min_chars_for_text: int = 20) -> tuple[list[Any], dict[str, Any]]:
    """
    PDFを読み込み、テキスト抽出がほぼ0なら OCR を試してから読み直す。
    """
    docs, loader_used = load_pdf_docs_best_effort(pdf_path)
    extracted_chars = 0
    for d in docs:
        d.page_content = clean_pdf_text(getattr(d, "page_content", "") or "")
        extracted_chars += len((getattr(d, "page_content", "") or "").strip())

    ocr_report: dict[str, Any] = {"attempted": False}
    if extracted_chars >= int(min_chars_for_text):
        strip_rep = strip_repeated_header_footer(docs)
        return docs, {"loader": loader_used, "ocr": {**ocr_report, "attempted": False}, "cleanup": strip_rep}

    # OCRを試す（可能なら）
    ocr_path, rep = maybe_ocr_pdf(pdf_path)
    ocr_report = {**rep, "attempted": True}
    if ocr_path == pdf_path:
        # OCRできない/しない場合は、元のdocsを返す
        strip_rep = strip_repeated_header_footer(docs)
        return docs, {"loader": loader_used, "ocr": ocr_report, "cleanup": strip_rep}

    # OCR済みPDFを読み直す
    docs2, loader_used2 = load_pdf_docs_best_effort(ocr_path)
    for d in docs2:
        d.page_content = clean_pdf_text(getattr(d, "page_content", "") or "")
    strip_rep2 = strip_repeated_header_footer(docs2)
    return docs2, {"loader": loader_used2, "ocr": ocr_report, "cleanup": strip_rep2}

