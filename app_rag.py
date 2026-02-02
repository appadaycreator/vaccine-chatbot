import ollama
import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

from pdf_ingest import load_pdf_docs_with_ocr_best_effort


def _normalize_newlines(text: str) -> str:
    return (text or "").replace("\r\n", "\n").replace("\r", "\n")


def _model_names_from_ollama_list(payload) -> list[str]:
    # ollama.list() は SDK バージョンで戻り型が変わる（dict / ListResponse）ため両対応する
    try:
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
            models = list(models)
        except Exception:
            return []

    out: list[str] = []
    for m in models:
        name = None
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
        if name:
            out.append(name)
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


def _strip_repeated_header_footer(docs: list, *, top_n: int = 2, bottom_n: int = 2, min_ratio: float = 0.6) -> None:
    """
    ページ毎に繰り返されるヘッダ/フッタ（タイトル行やページ番号等）を軽く除去する。
    - 先頭/末尾の少数行のみを対象
    - 出現頻度が高い短い行のみ除去（誤除去を減らす）
    """
    pages = len(docs)
    if pages <= 1:
        return

    def _edge_lines(text: str):
        lines = [ln.strip() for ln in _normalize_newlines(text).split("\n")]
        lines = [ln for ln in lines if ln]
        return lines[:top_n], (lines[-bottom_n:] if lines else [])

    from collections import Counter

    top_counter = Counter()
    bottom_counter = Counter()
    for d in docs:
        top, bottom = _edge_lines(getattr(d, "page_content", "") or "")
        top_counter.update(top)
        bottom_counter.update(bottom)

    threshold = max(2, int(pages * min_ratio))

    def _pick(counter):
        out = set()
        for ln, c in counter.items():
            if c >= threshold and 2 <= len(ln) <= 80:
                out.add(ln)
        return out

    top_rm = _pick(top_counter)
    bottom_rm = _pick(bottom_counter)

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
                j -= 1
            else:
                break
        d.page_content = _clean_pdf_text("\n".join(lines))


def _list_pdf_paths() -> list[str]:
    """
    参照対象PDFを列挙する（API/Streamlit と同じ導線）。
    - PDF_PATH: 単体PDF（任意）
    - PDF_DIR: ディレクトリ配下の *.pdf（推奨）
    """
    pdf_dir = os.environ.get("PDF_DIR", "./pdfs")
    pdf_path = os.environ.get("PDF_PATH", "vaccine_manual.pdf")
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


def _analyze_and_split_pdf(pdf_path: str) -> tuple[list, dict]:
    docs, ingest_meta = load_pdf_docs_with_ocr_best_effort(pdf_path)
    pages = len(docs)
    for d in docs:
        d.metadata = dict(getattr(d, "metadata", {}) or {})
        d.metadata["source"] = os.path.basename(pdf_path)
        d.page_content = _clean_pdf_text(getattr(d, "page_content", "") or "")

    extracted_chars = sum(len((getattr(d, "page_content", "") or "").strip()) for d in docs)
    splitter = _get_splitter()
    chunks = [c for c in splitter.split_documents(docs) if (c.page_content or "").strip()]
    report = {
        "path": os.path.abspath(pdf_path),
        "filename": os.path.basename(pdf_path),
        "pages": pages,
        "extracted_chars": extracted_chars,
        "chunk_count": len(chunks),
        "loader": (ingest_meta or {}).get("loader") if isinstance(ingest_meta, dict) else None,
        "ocr": (ingest_meta or {}).get("ocr") if isinstance(ingest_meta, dict) else None,
        "cleanup": (ingest_meta or {}).get("cleanup") if isinstance(ingest_meta, dict) else None,
    }
    return chunks, report


def build_vectorstore() -> Chroma:
    paths = _list_pdf_paths()
    if not paths:
        raise RuntimeError(
            "参照するPDFが見つかりませんでした。\n"
            "対処:\n"
            "- 単体PDF: `vaccine_manual.pdf` を作業ディレクトリに置く（または環境変数 PDF_PATH で指定）\n"
            "- 複数PDF: `./pdfs/`（環境変数 PDF_DIR）に *.pdf を置く\n"
            "補足:\n"
            "- このリポジトリは .gitignore で *.pdf を除外しているため、PDFはGitに含まれません"
        )

    print("PDFを解析中...")
    chunks_all: list = []
    items: list[dict] = []
    for p in paths:
        ch, rep = _analyze_and_split_pdf(p)
        items.append(rep)
        chunks_all.extend(ch)

    extracted_total = sum(int(it.get("extracted_chars") or 0) for it in items)
    chunk_total = len(chunks_all)
    print(f"PDF件数: {len(paths)} / 総抽出文字数: {extracted_total} / 総チャンク数: {chunk_total}")
    for it in items:
        print(f"- {it.get('filename')} pages={it.get('pages')} extracted_chars={it.get('extracted_chars')} chunks={it.get('chunk_count')}")

    # スキャンPDF（OCR不足）等でテキスト抽出できないと、検索結果が常に0件になり全質問で「資料にない」になる。
    if chunk_total == 0 or extracted_total < 20:
        raise RuntimeError(
            "PDFは見つかりましたが、テキストを抽出できませんでした（スキャンPDFでOCR不足の可能性）。\n"
            "この状態だと検索結果が常に0件になり、すべての質問で「資料にない」と返ります。\n"
            "対処:\n"
            "- OCRしてテキスト入りPDFにしてから配置してください\n"
            "- 可能なら抽出精度が上がる場合があります: `pip install pymupdf` → `export PDF_LOADER=pymupdf`"
        )

    print("知識ベースを構築中（これには数分かかる場合があります）...")
    _ensure_embedding_model("nomic-embed-text")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    persist_dir = (os.environ.get("CHROMA_PERSIST_DIR") or "").strip()
    if persist_dir:
        return Chroma.from_documents(documents=chunks_all, embedding=embeddings, persist_directory=persist_dir)
    return Chroma.from_documents(documents=chunks_all, embedding=embeddings)


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


vectorstore = build_vectorstore()

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

FALLBACK_KNOWLEDGE_BASE = """
【プロトタイプ知識ベース（資料外・一般情報）】
- 接種後は体調変化（発熱、痛み、倦怠感など）が起こり得ます
- 症状が強い／長引く／急に悪化する場合は、接種を受けた医療機関またはお住まいの自治体の相談窓口に相談してください
- 呼吸が苦しい、意識がもうろう、急激な悪化など緊急性が疑われる場合は 119（救急）を利用してください
""".strip()


def _env_allow_general_fallback_default() -> bool:
    # 既定はON（「何も返らない/資料にないだけ」状態を避ける）。必要なら環境変数でOFFにできる。
    v = (os.environ.get("ALLOW_GENERAL_FALLBACK_DEFAULT", "1") or "1").strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def _build_general_fallback_prompt(*, question: str, reference: str) -> str:
    return f"""
あなたは医療情報の文脈で回答するアシスタントです。
現在、参照PDFから質問の該当箇所を特定できていません。
そのため、以下の【参考情報】と一般的な注意として、断定を避けつつ回答してください。

必ず次の3セクションだけで出力してください（見出し名は固定、Markdownの # や ## は使わない）:
結論:
根拠:
相談先:

ルール:
- 参照PDFに基づくと断定しない（ページラベルの引用もしない）
- ページ番号や `[P12]` のような表記は一切出力しない
- 見出しは必ず `結論:` / `根拠:` / `相談先:` のプレーンテキストのみ
- 「根拠」には、【参考情報】からの引用/要約と、「一般的には…」のような前置きを使って不確実性を明示する
- 医療判断（診断/治療の指示）をしない。迷う場合は相談先へ誘導する
- 「相談先」は必ず1つ以上。緊急性が疑われる場合は救急（119）も含める
- 余計な追加セクション（注意/補足など）は出さない

【参考情報】:
{reference}

質問: {question}
""".strip()


def _build_answer_prompt(*, question: str, context: str) -> str:
    return f"""
あなたは医療情報の文脈で、厚労省等の配布資料（下の【資料】）に基づいて回答するアシスタントです。
推測や一般論で補完してはいけません。【資料】に書かれていないことは「資料にない」と明確に述べてください。

必ず次の3セクションだけで出力してください（見出し名は固定、Markdownの # や ## は使わない）:
結論:
根拠:
相談先:

ルール:
- 【資料】に書かれていない内容を断定しない（曖昧にそれっぽく言わない）
- 「根拠」には【資料】から該当箇所を引用/要約して箇条書きで示す。該当がなければ「資料にない」と書く
- 「相談先」は必ず1つ以上。緊急性が疑われる場合は救急（119）も含める
- 見出しは必ず `結論:` / `根拠:` / `相談先:` のプレーンテキストのみ（Markdown見出しにしない）

【資料】:
{context}

質問: {question}
""".strip()

def rag_chatbot(user_query, *, allow_general_fallback: bool | None = None):
    # 3. 関連情報の検索
    docs = vectorstore.similarity_search(user_query, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    # 4. LLMへの問い合わせ
    if not docs or not context.strip():
        allow = allow_general_fallback if allow_general_fallback is not None else _env_allow_general_fallback_default()
        if not allow:
            return _no_sources_answer(user_query)
        prompt = _build_general_fallback_prompt(question=user_query, reference=FALLBACK_KNOWLEDGE_BASE)
        response = ollama.generate(model="gemma2", prompt=prompt)
        return response["response"]

    prompt = _build_answer_prompt(question=user_query, context=context)
    
    response = ollama.generate(model='gemma2', prompt=prompt)
    return response['response']

if __name__ == "__main__":
    while True:
        query = input("\n資料について質問してください: ")
        if query.lower() == 'exit': break
        print("\n資料を検索して回答を生成中...")
        print(f"\n【回答】:\n{rag_chatbot(query)}")