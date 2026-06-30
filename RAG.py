from lmstudio_base import generate_response
from embedding_base import retrieve

docs = [
    {"text": "名前はアレックス", "source": "internal_profile"},
    {"text": "年齢は30歳", "source": "internal_profile"},
    {"text": "趣味は読書と旅行", "source": "internal_profile"},
    {"text": "好きな食べ物は寿司", "source": "internal_profile"},
    {"text": "好きなアニメは弱虫ペダル", "source": "internal_profile"},
]

# 質問に対して関連するドキュメントを検索し、応答を生成する(RAG)関数
# 引数 question: ユーザの質問
# 戻り値: LLMの応答（簡潔化済み）
# 仮実装のサンプルっす
def rag_sample(question):
    retrieved = retrieve(question, docs, topk=2)
    context_text = "\n".join(f"- {d['text']}（出典: {d['source']}）" for d in retrieved)

    # システム指示（簡潔化ルール）
    system_inst = (
        "あなたは親しみやすい日常会話用AIです。応答は原則1〜2文、50トークン以下で世間話にふさわしく簡潔に答えてください。"
        "あなたに関するプロフィール情報を参考にし，なりきって答えてください。"
        "情報源を使用した場合は末尾に出典を1つだけ示してください。"
    )

    few_shot = (
        "例:\nユーザ: 今日の天気は？\nあなた: 天気予報だと晴れって言ってたよ．\n\n"
        "ユーザ: おすすめの映画は？\nあなた: コメディなら『映画A』がおすすめかな。"
    )

    prompt = (
        f"{system_inst}\n\n{few_shot}\n\n参考情報:\n{context_text}\n\n質問: {question}\n回答:"
    )

    return generate_response(prompt)

if __name__ == "__main__":
    question = input("質問を入力してください: ").strip()
    if not question:
        print("質問を入力してください")
        exit()

    print(rag_sample(question))