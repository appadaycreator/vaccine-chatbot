"""
RAG検索結果のキーワードフィルタとフォールバック（API / Streamlit / CLI で共通利用）。

- 質問キーワードが資料に含まれない doc を落とし、ズレた根拠を抑える
- キーワードで全落ちしたが検索結果はある場合は、検索結果をフォールバックでコンテキストに使う
"""

import os
import re
from typing import Any

_JP_TERM_RE = re.compile(r"[一-龥ぁ-んァ-ン]{2,}|[A-Za-z0-9]{2,}")
_DEFAULT_TERM_STOPWORDS = {
    "です", "ます", "する", "した", "して", "いる", "ある", "こと", "これ", "それ",
    "ため", "場合", "どの", "どれ", "どこ", "いつ", "なに", "何", "方法", "目安",
    "目的", "要約", "教えて", "ください", "について", "ですか", "どうすれば", "どのくらい",
    "範囲", "資料", "記載", "アンケート",
}


def _keyword_terms(text: str) -> set[str]:
    s = (text or "").strip()
    if not s:
        return set()
    terms = {m.group(0) for m in _JP_TERM_RE.finditer(s)}
    out: set[str] = set()
    for t in terms:
        tt = t.strip()
        if not tt or len(tt) < 2:
            continue
        if tt.isascii():
            tt = tt.lower()
        if tt in _DEFAULT_TERM_STOPWORDS:
            continue
        out.add(tt)
    return out


def _keyword_overlap_score(terms: set[str], text: str) -> int:
    if not terms:
        return 0
    body = (text or "").strip()
    if not body:
        return 0
    low = body.lower()
    score = 0
    for t in terms:
        needle = t.lower() if t.isascii() else t
        if needle and needle in low:
            score += 1
    return score


def filter_docs_by_keyword_overlap(
    prompt: str,
    docs: list[Any],
    *,
    min_overlap: int | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    """
    質問のキーワードが含まれない doc を落とす。戻り値は (kept_docs, meta)。
    """
    if min_overlap is None:
        min_overlap = int(os.environ.get("SEARCH_KEYWORD_OVERLAP_MIN", "1") or "1")
    min_overlap = max(0, min(min_overlap, 10))
    terms = _keyword_terms(prompt)
    if re.search(r"\d+\s*(日|日間|週間|週|か月|ヶ月|ヵ月)", prompt or ""):
        min_overlap = max(min_overlap, 2)
    if min_overlap <= 0 or not terms or not docs:
        return docs, {"keyword_filter": {"enabled": bool(min_overlap > 0), "terms": len(terms), "kept": len(docs), "dropped": 0}}

    kept: list[Any] = []
    dropped = 0
    best = 0
    for d in docs:
        txt = getattr(d, "page_content", "") or ""
        sc = _keyword_overlap_score(terms, txt)
        best = max(best, sc)
        if sc >= min_overlap:
            kept.append(d)
        else:
            dropped += 1
    meta = {
        "keyword_filter": {
            "enabled": True,
            "terms": len(terms),
            "min_overlap": min_overlap,
            "best": best,
            "kept": len(kept),
            "dropped": dropped,
            "all_dropped": bool(len(kept) == 0 and len(docs) > 0),
        }
    }
    return kept, meta


def apply_search_filter(
    prompt: str,
    docs: list[Any],
    *,
    min_overlap: int | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    """
    キーワードフィルタをかけ、全落ち時は検索結果をフォールバックでそのまま返す。
    戻り値: (使用する doc リスト, meta)。
    """
    raw_docs = list(docs or [])
    filtered_docs, meta = filter_docs_by_keyword_overlap(prompt, raw_docs, min_overlap=min_overlap)
    kf = (meta or {}).get("keyword_filter") if isinstance(meta, dict) else None
    all_dropped = bool(isinstance(kf, dict) and kf.get("all_dropped") is True)
    if all_dropped and raw_docs:
        return raw_docs, {**(meta or {}), "keyword_fallback_used": True}
    return filtered_docs, meta
