# AI_practice

LM Studioを使用したRAG（Retrieval-Augmented Generation）の実装練習プロジェクトです。

## 概要

`practice5_rag4.py`は、埋め込みモデルを用いた質問応答システムです。
プロフィール情報をベクトル化して検索し、LLMで自然な応答を生成します。

## 必要な環境

- Python 3.8以上
- LM Studio（ローカルLLMサーバー）

## セットアップ手順

### 1. リポジトリをクローン

```bash
git clone <repository-url>
cd AI_practice
```

### 2. 仮想環境を作成・有効化

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. 依存パッケージをインストール

```bash
pip install -r requirements.txt
```

## 実行方法

```bash
python practice5_rag4.py
```

実行後、質問を入力してください。デフォルト質問は「好きなアニメは何ですか？」です。

## 依存パッケージ

- `lmstudio`: LM Studioのクライアントライブラリ
- `numpy`: 数値計算
- `httpx`, `httpx-ws`: HTTP通信
- その他： 詳細は [requirements.txt](requirements.txt) を参照

## ライセンス

MIT
