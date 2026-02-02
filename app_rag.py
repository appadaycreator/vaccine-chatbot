import ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# 1. PDFの読み込みと分割
print("PDFを解析中...")
loader = PyPDFLoader("vaccine_manual.pdf") # PDFファイルを指定
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(data)

# 2. ベクトルデータベースの作成 (OllamaのEmbeddingモデルを使用)
# 初回実行時に 'nomic-embed-text' モデルがダウンロードされます
print("知識ベースを構築中（これには数分かかる場合があります）...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)

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
- 「根拠」には【資料】から該当箇所を引用/要約して箇条書きで示す。該当がなければ「資料にない」と書く
- 「相談先」は必ず1つ以上。緊急性が疑われる場合は救急（119）も含める

【資料】:
{context}

質問: {question}
""".strip()

def rag_chatbot(user_query):
    # 3. 関連情報の検索
    docs = vectorstore.similarity_search(user_query, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    # 4. LLMへの問い合わせ
    if not docs or not context.strip():
        return _no_sources_answer(user_query)

    prompt = _build_answer_prompt(question=user_query, context=context)
    
    response = ollama.generate(model='gemma2', prompt=prompt)
    return response['response']

if __name__ == "__main__":
    while True:
        query = input("\n資料について質問してください: ")
        if query.lower() == 'exit': break
        print("\n資料を検索して回答を生成中...")
        print(f"\n【回答】:\n{rag_chatbot(query)}")