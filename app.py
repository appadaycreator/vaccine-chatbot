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
あなたは医療情報の文脈で、下の【知識ベース】に基づいて回答するアシスタントです。
推測や一般論で補完してはいけません。【知識ベース】に書かれていないことは「資料にない」と明確に述べてください。

必ず次の3セクションだけで出力してください（見出し名は固定）:
結論:
根拠:
相談先:

ルール:
- 【知識ベース】に書かれていない内容を断定しない（曖昧にそれっぽく言わない）
- 「根拠」には【知識ベース】の該当箇所を引用/要約して箇条書きで示す。該当がなければ「資料にない」と書く
- 「相談先」は必ず1つ以上。緊急性が疑われる場合は救急（119）も含める

【知識ベース】:
{KNOWLEDGE_BASE}

質問: {user_query}
""".strip()
    
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