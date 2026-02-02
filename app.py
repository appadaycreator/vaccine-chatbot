import ollama

def health_chatbot(user_query: str) -> str:
    """
    互換: 旧 `app.py` は「固定の知識ベース（資料外）」で一般論を返すデモでしたが、
    現在は「資料（PDF）に書かれている内容のみで回答する」方針に統一します。
    そのため、ここでは RAG版（PDF検索）の実装を呼び出します。
    """
    try:
        from app_rag import rag_chatbot  # import時にPDF解析/知識ベース構築が走る
    except Exception as e:
        return f"起動に失敗しました（RAGの初期化エラー）: {e}"
    # 互換引数は残しているが、一般論フォールバックは無効化済み
    return rag_chatbot(user_query, allow_general_fallback=False)

if __name__ == "__main__":
    print("-" * 50)
    print("ワクチン接種後健康観察アシスタント（プロトタイプ）起動中...")
    print("※終了するには 'exit' と入力してください。")
    print("-" * 50)
    
    while True:
        user_input = input("\n質問を入力してください: ")
        if user_input.lower() == 'exit':
            break
        
        print("\nAIが回答を生成中...")
        answer = health_chatbot(user_input)
        print(f"\n【回答】:\n{answer}")