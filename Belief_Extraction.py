import lmstudio as lms
import math
import re

docs = [
    {"text": "名前はアレックス", "source": "internal_profile"},
    {"text": "年齢は30歳", "source": "internal_profile"},
    {"text": "趣味は読書と旅行", "source": "internal_profile"},
    {"text": "好きな食べ物は寿司", "source": "internal_profile"},
    {"text": "好きなアニメは弱虫ペダル", "source": "internal_profile"},
]

embed_model = lms.embedding_model("nomic-embed-text-v1.5")
doc_embeddings = [embed_model.embed(d["text"]) for d in docs]

# 類似度計算関数
# 引数 a, b: 埋め込みベクトル
def similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)) + 1e-8)

# 質問に対して関連するドキュメントを検索する関数
# 引数 question: ユーザの質問、topk: 返すドキュメントの数
def retrieve(question, topk=2):
    q_emb = embed_model.embed(question) # クエリの埋め込み(ベクトル化)
    ranked = sorted(zip(docs, doc_embeddings), key=lambda item: similarity(q_emb, item[1]), reverse=True)
    return [doc for doc, _ in ranked[:topk]]

# LLM インスタンス
llm = lms.llm("openai/gpt-oss-20b")

# システム指示（簡潔化ルール）
system_inst = (
    "これは文章から信念を抽出するタスクです．"
    "与えられた相手の発言から，それを受けた人物が考えることを抽出してください．"
    "事実ではなく，その発言を受けた人物が考えることを抽出してください．"
)

few_shot = (
    "正例①:\nユーザ: この前の旅行すっごく良かったんだよね！\n抽出すべき信念: 旅行は良いものだ\n\n"
    "正例②:\nユーザ: 勉強さぼりすぎて補習になっちゃったよ...\n抽出すべき信念: 勉強は大切だ\n\n"

    "負例①:\nユーザ: この前の旅行すっごく良かったんだよね！\n抽出すべきでない事実: ユーザは旅行に行った\n\n"
    "負例②:\nユーザ: 勉強さぼりすぎて補習になっちゃったよ...\n抽出すべきでない事実: ユーザは勉強をしなかった ユーザは補習になった 等\n\n"
)

question = input("質問を入力してください: ").strip()
if not question:
    print("質問を入力してください")
    exit()
retrieved = retrieve(question)
context_text = "\n".join(f"- {d['text']}（出典: {d['source']}）" for d in retrieved)

prompt = (
    f"{system_inst}\n\n{few_shot}\n\n参考情報:\n{context_text}\n\n質問: {question}\n回答:"
)

resp = llm.respond(prompt)
content = resp.content

# 対話応答部分だけを抽出して返すための関数
# 引数 text: LLMの応答、max_sentences: 最大文数
# 戻り値: プレフィックスを削除した簡潔な応答
def extract_response(text, max_sentences=2):
    text = text.strip()
    # LMStudio のチャネル付加テキストを削除
    if '<|start|>assistant<|channel|>final<|message|>' in text:
        text = text.split('<|start|>assistant<|channel|>final<|message|>', 1)[1]
    text = re.sub(r'<\|/?(?:assistant|channel|message|start|end|analysis)\|>', '', text)
    text = text.strip()
    text = re.sub(r'^(?:アシスタント|回答|Assistant)(?:：|:)?\s*', '', text)
    parts = re.split('(?<=[。！？\n])', text)
    out = ''
    cnt = 0
    for p in parts:
        if p.strip() == '':
            continue
        out += p
        cnt += 1
        if cnt >= max_sentences:
            break
    out = out.strip()
    if not out:
        out = text[:200].strip()
    return out

print(extract_response(content, 2))