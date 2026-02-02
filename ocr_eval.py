from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from dataclasses import dataclass
from typing import Any

from pdf_ingest import load_pdf_docs_with_ocr_best_effort, ocr_settings_signature


def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _safe_json_load(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_newlines(text: str) -> str:
    return (text or "").replace("\r\n", "\n").replace("\r", "\n")


def _normalize_text(text: str, *, ignore_whitespace: bool) -> str:
    t = _normalize_newlines(text)
    # NFKC: 全角/半角等の正規化（OCR比較ではこの方が実用的）
    t = unicodedata.normalize("NFKC", t)
    # NBSP等のゆれを吸収
    t = t.replace("\u00a0", " ")
    if ignore_whitespace:
        # 改行/スペース/タブ差はOCRで揺れやすいので既定で無視
        t = re.sub(r"\s+", "", t)
    else:
        t = re.sub(r"[ \t]+", " ", t)
        t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t


def _tokenize_for_wer(text: str) -> list[str]:
    """
    WERのための簡易トークナイズ。
    - スペースが少ない日本語本文は WER が意味を持ちにくいので、基本は whitespace split。
    """
    t = unicodedata.normalize("NFKC", _normalize_newlines(text)).strip()
    t = t.replace("\u00a0", " ")
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return []
    return t.split(" ")


def _levenshtein_distance(a: str, b: str) -> int:
    """
    1次元DPの編集距離（Levenshtein）。
    - O(len(a)*len(b)) のため、1ページ程度のテキスト比較を想定
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    # メモリ節約: 常に b を短くする
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


def _levenshtein_distance_tokens(a: list[str], b: list[str]) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    # メモリ節約: 常に b を短くする
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ta in enumerate(a, start=1):
        cur = [i]
        for j, tb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ta == tb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


def _cer(gt: str, hyp: str, *, ignore_whitespace: bool) -> dict[str, Any]:
    ng = _normalize_text(gt, ignore_whitespace=ignore_whitespace)
    nh = _normalize_text(hyp, ignore_whitespace=ignore_whitespace)
    if not ng:
        return {"cer": None, "distance": None, "gt_len": 0, "hyp_len": len(nh)}
    dist = _levenshtein_distance(ng, nh)
    return {"cer": float(dist) / float(len(ng)), "distance": dist, "gt_len": len(ng), "hyp_len": len(nh)}


def _wer(gt: str, hyp: str) -> dict[str, Any]:
    tg = _tokenize_for_wer(gt)
    th = _tokenize_for_wer(hyp)
    if len(tg) < 5:
        return {"wer": None, "distance": None, "gt_tokens": len(tg), "hyp_tokens": len(th)}
    dist = _levenshtein_distance_tokens(tg, th)
    return {"wer": float(dist) / float(len(tg)), "distance": dist, "gt_tokens": len(tg), "hyp_tokens": len(th)}


def _quality_heuristics(text: str) -> dict[str, Any]:
    raw = unicodedata.normalize("NFKC", _normalize_newlines(text)).strip()
    n = len(raw)
    if n == 0:
        return {"chars": 0, "replacement_ratio": None, "suspicious_ratio": None}

    repl = raw.count("\ufffd")  # replacement character
    # 日本語OCRでありがちなノイズ（記号の連発等）を雑に拾う
    suspicious = 0
    for ch in raw:
        o = ord(ch)
        if ch.isspace():
            continue
        # ひらがな/カタカナ/漢字/英数/一般記号以外を「怪しい」寄りにカウント
        is_jp = (0x3040 <= o <= 0x30ff) or (0x3400 <= o <= 0x9fff)
        is_basic = ch.isalnum() or is_jp or (0x2000 <= o <= 0x206f) or (0x3000 <= o <= 0x303f)
        if not is_basic:
            suspicious += 1

    return {
        "chars": n,
        "replacement_ratio": float(repl) / float(n),
        "suspicious_ratio": float(suspicious) / float(n),
    }


@dataclass
class _EnvPatch:
    key: str
    old: str | None


def _apply_env(overrides: dict[str, str | None]) -> list[_EnvPatch]:
    patches: list[_EnvPatch] = []
    for k, v in overrides.items():
        patches.append(_EnvPatch(k, os.environ.get(k)))
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)
    return patches


def _restore_env(patches: list[_EnvPatch]) -> None:
    for p in patches:
        if p.old is None:
            os.environ.pop(p.key, None)
        else:
            os.environ[p.key] = p.old


def _load_gt_pages_from_dir(dir_path: str) -> dict[int, str]:
    out: dict[int, str] = {}
    for name in os.listdir(dir_path):
        if not name.lower().endswith(".txt"):
            continue
        m = re.search(r"(\d+)", name)
        if not m:
            continue
        page = int(m.group(1))
        out[page] = _read_text_file(os.path.join(dir_path, name))
    return out


def _load_gt_pages_from_json(path: str) -> dict[int, str]:
    data = _safe_json_load(path)
    out: dict[int, str] = {}

    # 1) {"pages":[{"page":1,"text":"..."}, ...]}
    if isinstance(data, dict) and isinstance(data.get("pages"), list):
        for it in data["pages"]:
            if isinstance(it, dict) and isinstance(it.get("page"), int) and isinstance(it.get("text"), str):
                out[int(it["page"])] = it["text"]
        if out:
            return out

    # 2) {"1":"...", "2":"..."} のようなマップ
    if isinstance(data, dict):
        for k, v in data.items():
            try:
                page = int(str(k))
            except Exception:
                continue
            if isinstance(v, str):
                out[page] = v
        if out:
            return out

    # 3) ["page1 text", "page2 text", ...] （1始まり）
    if isinstance(data, list) and all(isinstance(x, str) for x in data):
        for i, t in enumerate(data, start=1):
            out[i] = t
        return out

    raise ValueError("GT JSON形式を解釈できませんでした（pages配列 / ページマップ / 文字列配列のいずれかを想定）。")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="OCR結果の精度（CER/WER）を判定します。")
    ap.add_argument("--pdf", required=True, help="評価対象のPDFパス")
    ap.add_argument("--gt", default="", help="正解テキスト（.txt または .json）。未指定ならヒューリスティックのみ")
    ap.add_argument("--gt-dir", default="", help="ページ別正解テキストのディレクトリ（*.txt）")
    ap.add_argument("--ignore-whitespace", action="store_true", help="CER計算で空白差を無視（既定ON）")
    ap.add_argument("--keep-whitespace", action="store_true", help="CER計算で空白差も評価する")
    ap.add_argument("--ocr-mode", choices=["off", "auto", "force"], default="", help="OCR_MODE を上書き")
    ap.add_argument("--ocr-lang", default="", help="OCR_LANG を上書き（例: jpn+eng）")
    ap.add_argument("--ocr-cache-dir", default="", help="OCR_CACHE_DIR を上書き")
    ap.add_argument("--ocrmypdf-args", default="", help="OCRMYPDF_ARGS を上書き（例: \"--tesseract-timeout 120\"）")
    ap.add_argument("--json", action="store_true", help="JSONのみを標準出力に出す（機械処理向け）")
    ap.add_argument("--out", default="", help="結果JSONの保存先（例: ocr_eval_report.json）")
    args = ap.parse_args(argv)

    ignore_ws = True
    if args.keep_whitespace:
        ignore_ws = False
    if args.ignore_whitespace:
        ignore_ws = True

    env_overrides: dict[str, str | None] = {}
    if args.ocr_mode:
        env_overrides["OCR_MODE"] = args.ocr_mode
    if args.ocr_lang:
        env_overrides["OCR_LANG"] = args.ocr_lang
    if args.ocr_cache_dir:
        env_overrides["OCR_CACHE_DIR"] = args.ocr_cache_dir
    if args.ocrmypdf_args:
        env_overrides["OCRMYPDF_ARGS"] = args.ocrmypdf_args

    patches = _apply_env(env_overrides)
    try:
        docs, ingest_meta = load_pdf_docs_with_ocr_best_effort(args.pdf)
        ocr_sig = ocr_settings_signature()
    finally:
        _restore_env(patches)

    pages: list[dict[str, Any]] = []
    for d in docs:
        meta = getattr(d, "metadata", {}) or {}
        page0 = meta.get("page")
        page_num = (int(page0) + 1) if isinstance(page0, int) else None
        pages.append({"page": page_num, "text": str(getattr(d, "page_content", "") or "")})

    full_text = "\n\n".join([p["text"] for p in pages if (p.get("text") or "").strip()])
    heur = _quality_heuristics(full_text)

    gt_pages: dict[int, str] = {}
    gt_full: str | None = None
    gt_path = (args.gt or "").strip()
    gt_dir = (args.gt_dir or "").strip()
    if gt_dir:
        gt_pages = _load_gt_pages_from_dir(gt_dir)
    if gt_path:
        if gt_path.lower().endswith(".json"):
            gt_pages = {**gt_pages, **_load_gt_pages_from_json(gt_path)}
        else:
            gt_full = _read_text_file(gt_path)

    report: dict[str, Any] = {
        "ok": True,
        "pdf": {"path": os.path.abspath(args.pdf), "pages": len(pages)},
        "ingest": ingest_meta,
        "ocr_settings": ocr_sig,
        "heuristics": heur,
        "metrics": None,
        "per_page": None,
    }

    # ページ別GTがある場合は、それを優先して評価する
    if gt_pages:
        per_page: list[dict[str, Any]] = []
        total_dist = 0
        total_len = 0
        for p in pages:
            page_num = p.get("page")
            if not isinstance(page_num, int):
                continue
            gt = gt_pages.get(page_num)
            if gt is None:
                continue
            hyp = str(p.get("text") or "")
            cer = _cer(gt, hyp, ignore_whitespace=ignore_ws)
            wer = _wer(gt, hyp)
            if isinstance(cer.get("distance"), int) and isinstance(cer.get("gt_len"), int):
                total_dist += int(cer["distance"])
                total_len += int(cer["gt_len"])
            per_page.append(
                {
                    "page": page_num,
                    "cer": cer.get("cer"),
                    "wer": wer.get("wer"),
                    "gt_len": cer.get("gt_len"),
                    "hyp_len": cer.get("hyp_len"),
                    "gt_tokens": wer.get("gt_tokens"),
                    "hyp_tokens": wer.get("hyp_tokens"),
                }
            )
        per_page.sort(key=lambda x: int(x.get("page") or 0))
        report["per_page"] = per_page
        report["metrics"] = {
            "mode": "per_page",
            "ignore_whitespace": bool(ignore_ws),
            "cer": (float(total_dist) / float(total_len)) if total_len > 0 else None,
            "evaluated_pages": len(per_page),
        }

    # 全文GTがある場合は全文で評価（ページ別があっても併記）
    if gt_full is not None:
        cer = _cer(gt_full, full_text, ignore_whitespace=ignore_ws)
        wer = _wer(gt_full, full_text)
        report.setdefault("metrics", {})
        if isinstance(report["metrics"], dict):
            report["metrics"] = {
                **report["metrics"],
                "mode": (report["metrics"].get("mode") if report["metrics"] else "full_text"),
                "ignore_whitespace": bool(ignore_ws),
                "cer_full": cer.get("cer"),
                "wer_full": wer.get("wer"),
            }

    out_json = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out_json)

    if args.json:
        sys.stdout.write(out_json + "\n")
        return 0

    # 人間向けの最小サマリ
    metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
    cer_v = metrics.get("cer_full") or metrics.get("cer")
    wer_v = metrics.get("wer_full")
    print(f"PDF: {report['pdf']['path']}")
    print(f"ページ数: {report['pdf']['pages']} / 抽出文字数: {heur.get('chars')}")
    print(f"OCR設定: {json.dumps(report.get('ocr_settings') or {}, ensure_ascii=False)}")
    if isinstance(report.get("ingest"), dict) and isinstance(report["ingest"].get("ocr"), dict):
        print(f"OCR実行: {json.dumps(report['ingest']['ocr'], ensure_ascii=False)}")
    if cer_v is not None:
        print(f"CER: {cer_v:.4f}（小さいほど良い）")
    if wer_v is not None:
        print(f"WER: {wer_v:.4f}（小さいほど良い）")
    if cer_v is None and gt_full is None and not gt_pages:
        print("GT（正解テキスト）が無いため、ヒューリスティックのみです。--gt または --gt-dir を指定してください。")
    if args.out:
        print(f"JSON出力: {os.path.abspath(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

