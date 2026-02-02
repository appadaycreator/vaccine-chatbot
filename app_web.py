import os

import ollama
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ページの設定
st.set_page_config(page_title="ワクチン接種後健康観察アシスタント", page_icon="🏥")
st.title("🏥 健康観察アシスタント")
st.caption("厚労省の実施要領に基づいたプロトタイプ")

# 免責（UIに常設）＋相談導線
st.info(
    "このツールは資料に基づく情報提供を目的としており、診断や治療の代替ではありません。\n"
    "体調が悪い・不安が強い場合は、接種を受けた医療機関や自治体の予防接種相談窓口に相談してください。\n"
    "緊急性が疑われる場合（呼吸が苦しい、意識がもうろう等）は 119（救急）を利用してください。"
)

# サイドバー（設定）
with st.sidebar:
    st.header("設定")
    llm_model = st.selectbox("回答モデル", ["gemma2", "llama3.1"], index=0)
    k = st.slider("検索の強さ（k値）", min_value=1, max_value=10, value=3, step=1)
    st.caption("埋め込みモデル: `nomic-embed-text`（固定）")
    st.caption("PDFはサーバー側で `./pdfs/`（環境変数 `PDF_DIR`）に配置して利用します。")
    st.divider()
    st.caption("操作ヒント: 「例」ボタンで質問を自動送信 / 「再送」で最後の質問をもう一度送れます。")


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
- 「根拠」には、【資料】から該当箇所を引用/要約して箇条書きで示す
- 「相談先」は必ず1つ以上。緊急性が疑われる場合は救急（119）も含める
- 余計な免責文や追加セクション（注意/補足など）は出さない（UI側で常設するため）

【資料】:
{context}

質問: {question}
""".strip()


def _normalize_docs_source(docs, source_label: str):
    for d in docs:
        d.metadata = dict(d.metadata or {})
        d.metadata["source"] = source_label
    return docs


def _normalize_newlines(text: str) -> str:
    return (text or "").replace("\r\n", "\n").replace("\r", "\n")


def _make_excerpt(text: str, max_lines: int = 10, max_chars: int = 900) -> str:
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
        if ln != "…" and total + len(ln) + 1 > max_chars:
            break
        out.append(ln)
        total += len(ln) + 1
        if total >= max_chars:
            break
    return "\n".join(out).strip()


@st.cache_resource(show_spinner=False)
def _build_vectorstore_from_paths(paths: list[str], signature: str):
    # signature はキャッシュキー安定化のため
    chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    for p in paths:
        loader = PyPDFLoader(p)
        docs = loader.load()
        docs = _normalize_docs_source(docs, os.path.basename(p))
        chunks.extend([c for c in splitter.split_documents(docs) if (c.page_content or "").strip()])
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma.from_documents(documents=chunks, embedding=embeddings)


def _list_pdf_paths() -> list[str]:
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
    # 重複除去
    uniq: list[str] = []
    seen: set[str] = set()
    for p in paths:
        ap = os.path.abspath(p)
        if ap in seen:
            continue
        seen.add(ap)
        uniq.append(p)
    return uniq


def _signature(paths: list[str]) -> str:
    parts: list[str] = []
    for p in paths:
        try:
            st_ = os.stat(p)
            parts.append(f"{os.path.abspath(p)}|{int(st_.st_size)}|{float(st_.st_mtime)}")
        except Exception:
            parts.append(f"{os.path.abspath(p)}|NA|NA")
    return "\n".join(sorted(parts))

paths = _list_pdf_paths()
try:
    if not paths:
        st.error("参照するPDFが見つかりませんでした。`vaccine_manual.pdf` または `./pdfs/` にPDFを配置してください。")
        st.stop()
    sig = _signature(paths)
    with st.spinner("PDFを解析して知識ベースを構築中...（初回は時間がかかります）"):
        vectorstore = _build_vectorstore_from_paths(paths, sig)
    st.success(f"資料の読み込みが完了しました（{len(paths)}件）。")
except Exception as e:
    st.error(f"PDFを読み込めませんでした: {e}")
    st.stop()

# リセットボタン
col1, col2 = st.columns([1, 3])
with col1:
    if st.button("履歴をリセット"):
        st.session_state.messages = []
        st.rerun()

# 横展開（UX最低限）: 例ボタン / 再送
if "queued_prompt" not in st.session_state:
    st.session_state.queued_prompt = ""
if "last_user_prompt" not in st.session_state:
    st.session_state.last_user_prompt = ""

quick_items = [
    "接種後7日間に記録する項目は？",
    "37.5度以上の発熱が出たらどうすればいい？",
    "接種部位の腫れ・痛みはどのくらい続く？（資料にある範囲で）",
    "相談先（医療機関/自治体/119）の判断の目安は？",
]

qcols = st.columns([1, 1, 1, 1])
for i, text in enumerate(quick_items):
    with qcols[i]:
        if st.button(f"例: {text}", use_container_width=True):
            st.session_state.queued_prompt = text
            st.rerun()

rs_col1, rs_col2 = st.columns([1, 3])
with rs_col1:
    if st.button("再送", disabled=not bool(st.session_state.last_user_prompt)):
        st.session_state.queued_prompt = st.session_state.last_user_prompt
        st.rerun()
with rs_col2:
    if st.session_state.last_user_prompt:
        st.caption(f"最後の質問: {st.session_state.last_user_prompt}")

# チャット履歴の初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

# 履歴の表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            sources = message["sources"]
            locs = [str(s.get("location") or f"{s.get('source','資料')} {s.get('page_label','[P?]')}") for s in sources]
            st.markdown("**根拠（引用）**: " + " / ".join(locs))
            with st.expander("根拠の抜粋を表示"):
                for s in sources:
                    title = str(s.get("location") or f"{s.get('source','資料')} {s.get('page_label','[P?]')}")
                    st.markdown(f"- {title}")
                    if s.get("excerpt"):
                        st.code(str(s["excerpt"]))


def _extract_sources(docs):
    sources = []
    seen = set()
    for d in docs:
        meta = d.metadata or {}
        src = meta.get("source") or "資料"
        page = meta.get("page")
        page_num = page + 1 if isinstance(page, int) else None
        key = (src, page_num)
        if key in seen:
            continue
        seen.add(key)
        page_label = f"[P{page_num}]" if isinstance(page_num, int) else "[P?]"
        excerpt = _make_excerpt(d.page_content or "")
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

# ユーザー入力
prompt = st.chat_input("質問をどうぞ")
if not prompt and st.session_state.queued_prompt:
    prompt = st.session_state.queued_prompt
    st.session_state.queued_prompt = ""

if prompt:
    st.session_state.last_user_prompt = prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # RAGロジック
    with st.chat_message("assistant"):
        try:
            with st.spinner("資料を確認中..."):
                # 検索
                docs = vectorstore.similarity_search(prompt, k=k)
                context = "\n".join([doc.page_content for doc in docs])
                sources = _extract_sources(docs)

                # 根拠が取れない場合は、生成せずに固定フォーマットで返す（断定/hallucination防止）
                if not sources:
                    answer = _no_sources_answer(prompt)
                else:
                    full_prompt = _build_answer_prompt(question=prompt, context=context)
                    response = ollama.generate(model=llm_model, prompt=full_prompt)
                    answer = response["response"]

            st.markdown(answer)
            if sources:
                locs = [str(s.get("location") or f"{s.get('source','資料')} {s.get('page_label','[P?]')}") for s in sources]
                st.markdown("**根拠（引用）**: " + " / ".join(locs))
                with st.expander("根拠の抜粋を表示"):
                    for s in sources:
                        title = str(s.get("location") or f"{s.get('source','資料')} {s.get('page_label','[P?]')}")
                        st.markdown(f"- {title}")
                        if s.get("excerpt"):
                            st.code(str(s["excerpt"]))

            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
        except Exception as e:
            st.error("エラーが発生しました。まずは Ollama / PDF / 設定（k値・モデル）を確認してください。")
            with st.expander("ログ全文（展開）"):
                st.code(str(e))
            st.session_state.messages.append({"role": "assistant", "content": f"エラー: {e}"})