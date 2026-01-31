import streamlit as st
import ollama
import hashlib
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# ページの設定
st.set_page_config(page_title="ワクチン接種後健康観察アシスタント", page_icon="🏥")
st.title("🏥 健康観察アシスタント")
st.caption("厚労省の実施要領に基づいたプロトタイプ")

# サイドバー（設定）
with st.sidebar:
    st.header("設定")
    llm_model = st.selectbox("回答モデル", ["gemma2", "llama3.1"], index=0)
    k = st.slider("検索の強さ（k値）", min_value=1, max_value=10, value=3, step=1)
    st.caption("埋め込みモデル: `nomic-embed-text`（固定）")

    uploaded_pdf = st.file_uploader("PDFアップロード（任意）", type=["pdf"])
    st.caption("未アップロード時は `vaccine_manual.pdf` を使用します。")


def _normalize_docs_source(docs, source_label: str):
    for d in docs:
        d.metadata = dict(d.metadata or {})
        d.metadata["source"] = source_label
    return docs


@st.cache_resource(show_spinner=False)
def _build_vectorstore_from_path(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    docs = _normalize_docs_source(docs, os.path.basename(pdf_path))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma.from_documents(documents=chunks, embedding=embeddings)


@st.cache_resource(show_spinner=False)
def _build_vectorstore_from_bytes(pdf_bytes: bytes, filename: str, file_hash_hex: str):
    # file_hash_hex はキャッシュキー安定化のため（bytesだけでも良いが、明示しておく）
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(pdf_bytes)
        tmp_path = f.name
    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        docs = _normalize_docs_source(docs, filename or "uploaded.pdf")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(docs)
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        return Chroma.from_documents(documents=chunks, embedding=embeddings)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

try:
    if uploaded_pdf is not None:
        pdf_bytes = uploaded_pdf.getvalue()
        pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()
        with st.spinner("アップロードPDFを解析して知識ベースを構築中...（初回は時間がかかります）"):
            vectorstore = _build_vectorstore_from_bytes(pdf_bytes, uploaded_pdf.name, pdf_hash)
        st.success(f"資料の読み込みが完了しました（{uploaded_pdf.name}）。")
    else:
        with st.spinner("既定PDFを解析して知識ベースを構築中...（初回は時間がかかります）"):
            vectorstore = _build_vectorstore_from_path("vaccine_manual.pdf")
        st.success("資料の読み込みが完了しました（vaccine_manual.pdf）。")
except Exception as e:
    st.error(f"PDFを読み込めませんでした: {e}")
    st.stop()

# リセットボタン
col1, col2 = st.columns([1, 3])
with col1:
    if st.button("履歴をリセット"):
        st.session_state.messages = []
        st.rerun()

# チャット履歴の初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

# 履歴の表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            sources = message["sources"]
            pages = []
            for s in sources:
                if s.get("page") is not None:
                    pages.append(f"{s.get('source','資料')} p.{s['page']}")
                else:
                    pages.append(f"{s.get('source','資料')}")
            st.markdown("**根拠（参照ページ）**: " + " / ".join(pages))
            with st.expander("根拠の抜粋を表示"):
                for s in sources:
                    title = f"{s.get('source','資料')}"
                    if s.get("page") is not None:
                        title += f" p.{s['page']}"
                    st.markdown(f"- {title}")
                    if s.get("excerpt"):
                        st.caption(s["excerpt"])


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
        excerpt = (d.page_content or "").strip().replace("\n", " ")
        if len(excerpt) > 400:
            excerpt = excerpt[:400] + "…"
        sources.append({"source": src, "page": page_num, "excerpt": excerpt})
    return sources

# ユーザー入力
if prompt := st.chat_input("質問をどうぞ"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # RAGロジック
    with st.chat_message("assistant"):
        with st.spinner("資料を確認中..."):
            # 検索
            docs = vectorstore.similarity_search(prompt, k=k)
            context = "\n".join([doc.page_content for doc in docs])
            sources = _extract_sources(docs)
            
            # 生成
            full_prompt = f"""
あなたは厚労省の資料に基づいて回答する専門アシスタントです。
以下の【資料抜粋】の内容に基づいて、日本語で簡潔に回答してください。
資料に記載がない場合は「資料内には該当する情報が見当たりません」と答え、自治体の相談窓口または接種を受けた医療機関への相談を促してください。

【資料抜粋】
{context}

質問: {prompt}
回答:
""".strip()
            response = ollama.generate(model=llm_model, prompt=full_prompt)
            answer = response['response']
            
            st.markdown(answer)
            if sources:
                pages = []
                for s in sources:
                    if s.get("page") is not None:
                        pages.append(f"{s.get('source','資料')} p.{s['page']}")
                    else:
                        pages.append(f"{s.get('source','資料')}")
                st.markdown("**根拠（参照ページ）**: " + " / ".join(pages))
                with st.expander("根拠の抜粋を表示"):
                    for s in sources:
                        title = f"{s.get('source','資料')}"
                        if s.get("page") is not None:
                            title += f" p.{s['page']}"
                        st.markdown(f"- {title}")
                        if s.get("excerpt"):
                            st.caption(s["excerpt"])

            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})