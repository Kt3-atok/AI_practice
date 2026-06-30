import lmstudio as lms
import math
from lmstudio_base import generate_response

embed_model = lms.embedding_model("nomic-embed-text-v1.5")

# 類似度計算関数
# 引数 a, b: 埋め込みベクトル
def similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)) + 1e-8)

# 質問に対して関連するドキュメントを検索する関数
# 引数 question: ユーザの質問、docs: ドキュメントのリスト、topk: 返すドキュメントの数
def retrieve(question, docs, topk=2):
    doc_embeddings = [embed_model.embed(d["text"]) for d in docs]
    q_emb = embed_model.embed(question) # クエリの埋め込み(ベクトル化)
    ranked = sorted(zip(docs, doc_embeddings), key=lambda item: similarity(q_emb, item[1]), reverse=True)
    return [doc for doc, _ in ranked[:topk]]