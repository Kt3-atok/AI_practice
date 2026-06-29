import lmstudio as lms
import re

# LLM インスタンス
llm = lms.llm("openai/gpt-oss-20b")


# プロンプトから応答を生成する関数
# 引数 prompt: LLMに渡すプロンプト
# 戻り値: LLMの応答（簡潔化済み）
def generate_response(prompt):
    resp = llm.respond(prompt)
    content = resp.content
    return extract_response(content, 2)


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


if __name__ == "__main__":
    question = input("質問を入力してください: ").strip()
    if not question:
        print("質問を入力してください")
        exit()

    print(generate_response(question))