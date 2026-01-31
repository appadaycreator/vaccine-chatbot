import ollama

# 1. 知識ベースの定義（プロトタイプ用の抜粋）
KNOWLEDGE_BASE = """
【厚労省 ワクチン接種後健康状況調査 抜粋】
- 観察期間：接種当日（0日目）から7日間
- 記録項目：体温、接種部位の反応（腫れ・痛み）、全身反応（発熱、頭痛、倦怠感）
- 報告が必要な症状：37.5度以上の発熱、日常生活に支障が出るほどの痛みや腫れ
- 連絡先：各自治体の相談窓口、または接種を受けた医療機関
"""

def health_chatbot(user_query):
    # 2. Gemma 2 を使用して回答を生成
    # 知識ベースに基づいた回答を促すシステムプロンプトを設定
    prompt = f"""
    あなたは厚労省の健康状況調査をサポートする専門のアシスタントです。
    以下の【知識ベース】の内容のみに基づいて、ユーザーの質問に正確に答えてください。
    知識ベースにない情報は「分かりかねます」と答え、相談窓口を案内してください。

    【知識ベース】:
    {KNOWLEDGE_BASE}

    質問: {user_query}
    回答:
    """
    
    # stream=True にすると、ChatGPTのように逐次回答が表示されます
    response = ollama.generate(model='gemma2', prompt=prompt)
    return response['response']

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