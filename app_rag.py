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

def rag_chatbot(user_query):
    # 3. 関連情報の検索
    docs = vectorstore.similarity_search(user_query, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    # 4. LLMへの問い合わせ
    prompt = f"""
    あなたは厚労省の資料に基づき回答する専門アシスタントです。
    以下の【参考資料】の内容に基づいて、質問に答えてください。
    資料に記載がない場合は「資料内には該当する情報が見当たりません」と答え、窓口への相談を促してください。

    【参考資料】:
    {context}

    質問: {user_query}
    回答:
    """
    
    response = ollama.generate(model='gemma2', prompt=prompt)
    return response['response']

if __name__ == "__main__":
    while True:
        query = input("\n資料について質問してください: ")
        if query.lower() == 'exit': break
        print("\n資料を検索して回答を生成中...")
        print(f"\n【回答】:\n{rag_chatbot(query)}")