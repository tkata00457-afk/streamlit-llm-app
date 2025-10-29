# app.py
# Python 3.11 / Streamlit Community Cloud対応
# ローカルでは .env から OPENAI_API_KEY を読み込む
# 本番では Streamlit Secrets の OPENAI_API_KEY を読む

import os
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


def get_api_key() -> str:
    """
    1. ローカル:
       .env の OPENAI_API_KEY を load_dotenv() で読み込み -> os.getenv()
    2. デプロイ:
       Streamlit Community Cloud の Secrets の OPENAI_API_KEY を参照
    """
    # ローカル用に .env を読む
    load_dotenv()

    # まず環境変数から試す
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key

    # Streamlit Cloud の Secrets から試す
    if "OPENAI_API_KEY" in st.secrets:
        return st.secrets["OPENAI_API_KEY"]

    # どこにも無い場合
    raise ValueError(
        "OPENAI_API_KEY が見つかりません。\n"
        ".env に OPENAI_API_KEY=... を書くか、"
        "デプロイ時はStreamlitのSecretsにOPENAI_API_KEYを設定してください。"
    )


@st.cache_resource(show_spinner=False)
def get_llm():
    """
    LangChain の ChatOpenAI を初期化して返す。
    キャッシュして複数回使い回す。
    """
    api_key = get_api_key()

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.4,
        api_key=api_key,
    )
    return llm


def build_system_prompt(expert_choice: str) -> str:
    """
    ラジオボタンの選択肢に応じた「専門家キャラ」のシステムメッセージを返す。
    """
    if expert_choice == "医療コンサルタント":
        return (
            "あなたは病院経営と臨床現場の両方に詳しい医療コンサルタントです。"
            "医療安全、現場オペレーション、人員配置、患者説明のリスク管理について、"
            "現実的で実行可能なアドバイスを提示してください。"
            "専門用語は使ってもよいですが、必要に応じて平易な説明も加えてください。"
        )

    elif expert_choice == "スタートアップ経営アドバイザー":
        return (
            "あなたはシード〜シリーズA段階のスタートアップ経営アドバイザーです。"
            "採用、資金繰り、事業の優先順位付け、プロダクトの絞り込み、"
            "チームマネジメントなどについて、明日から動ける具体的な助言をしてください。"
            "机上の空論ではなく、実務に落とし込んでください。"
        )

    # 念のためのデフォルト
    return (
        "あなたは丁寧で誠実な専門家アシスタントです。"
        "ユーザーの質問に対し、わかりやすく・実用的に回答してください。"
    )


def ask_llm(user_input: str, expert_choice: str) -> str:
    """
    【課題要件の関数】
    - 引数:
        user_input: 画面のテキスト入力
        expert_choice: ラジオボタンで選んだ専門家ロール
    - 戻り値:
        LLMからの回答テキスト
    - LangChain経由でLLMを呼び出す
    """
    llm = get_llm()
    system_prompt = build_system_prompt(expert_choice)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input),
    ]

    # LangChainのChatOpenAIは .invoke(messages) で実行できる
    response = llm.invoke(messages)

    # 応答は ChatMessage型で返ってくるので .content を読む
    return response.content


# =======================
# Streamlit UI
# =======================

st.set_page_config(
    page_title="LLM相談アプリ",
    page_icon="💬",
    layout="centered",
)

st.title("💬 LLM相談アプリ")
st.caption("あなたの相談をLLMに送り、選んだ専門家の視点で回答します。")

st.markdown(
    """
### 使い方
1. 回答スタイル（専門家イメージ）を選びます  
2. 質問・相談内容を入力します  
3. 「送信」を押すと、LLMからの回答が表示されます  

### 専門家ロール
- 医療コンサルタント  
  - 医療安全や現場オペレーション、人員配置、説明義務リスクなどのアドバイス  
- スタートアップ経営アドバイザー  
  - 採用・事業の優先順位・資金の考え方・チーム運営などのアドバイス  

### セキュリティ/キー管理
- ローカル実行時：`.env` に `OPENAI_API_KEY=sk-...` を保存し、GitHubには上げない  
- デプロイ時：Streamlit Community Cloud の Secrets に `OPENAI_API_KEY` を設定  
  （.envはアップしない）  

### バージョン
- Streamlit Community Cloud 側の Python は 3.11 を指定してください。
"""
)

st.divider()

expert_choice = st.radio(
    "回答スタイル（専門家の立場）を選んでください：",
    options=[
        "医療コンサルタント",
        "スタートアップ経営アドバイザー",
    ],
)

user_input = st.text_area(
    "相談内容・質問を入力してください：",
    placeholder="例）複数の医師で同じ患者を診ると責任が曖昧になります。安全に回す運用の考え方は？",
    height=150,
)

if st.button("送信"):
    if not user_input.strip():
        st.warning("質問が空です。入力してください。")
    else:
        with st.spinner("LLMに問い合わせ中..."):
            try:
                answer = ask_llm(user_input, expert_choice)
                st.markdown("### 回答")
                st.write(answer)
            except Exception as e:
                st.error(
                    "エラーが発生しました。APIキーの設定などを確認してください。\n\n"
                    f"詳細: {e}"
                )

st.divider()
st.caption("Powered by LangChain + OpenAI API")
