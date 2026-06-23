import lmstudio as lms
import math
import re



# LLM インスタンス
llm = lms.llm("openai/gpt-oss-20b")

# システム指示（簡潔化ルール）
system_inst = (
    "これは文章から信念を抽出するタスクです．"
    "与えられた相手の発言から，それを受けた人物が考えることを抽出してください．"
    "事実ではなく，その発言を受けた人物が考えることを抽出してください．"

    "#ルール"
    "- 信念は「～は～」の命題の形で出力する"
    "- 意味と方向（肯定/否定・好き/嫌い）をそのまま正確に保つ。ぼかさない"
    "- 「〜が変化した」のような曖昧な要約は禁止。変化したなら「変化後の内容」を具体的に書く"
    "- あいさつ・相槌・一時的な雑談は抽出しない"
    "- 重要な事実がなければ、空の文字列を返す"
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


prompt = (
    f"{system_inst}\n\n{few_shot}\n\n質問: {question}\n回答:"
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