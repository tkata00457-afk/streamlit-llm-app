import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

@st.cache_resource(show_spinner=False)
def get_llm(model: str = "gpt-4o-mini", temperature: float = 0.4) -> ChatOpenAI:
    api_key = (st.secrets.get("OPENAI_API_KEY") or "").strip()
    if not api_key or not api_key.startswith("sk-"):
        st.error("サーバ側の設定が不足しています。管理者に連絡してください。")
        st.stop()

    os.environ["OPENAI_API_KEY"] = api_key

    try:
        return ChatOpenAI(model=model, temperature=temperature, api_key=api_key)
    except TypeError:
        return ChatOpenAI(model_name=model, temperature=temperature, openai_api_key=api_key)



def build_system_prompt(expert_choice: str) -> str:
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
    return (
        "あなたは丁寧で誠実な専門家アシスタントです。"
        "ユーザーの質問に対し、わかりやすく・実用的に回答してください。"
    )

def ask_llm(user_input: str, expert_choice: str) -> str:
    llm = get_llm()
    messages = [
        SystemMessage(content=build_system_prompt(expert_choice)),
        HumanMessage(content=user_input),
    ]
    resp = llm.invoke(messages)
    return resp.content

st.set_page_config(page_title="LLM相談アプリ", page_icon="💬", layout="centered")

st.title("💬 LLM相談アプリ")

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
"""
)

st.divider()

expert_choice = st.radio(
    "回答スタイル（専門家の立場）を選んでください：",
    options=["医療コンサルタント", "スタートアップ経営アドバイザー"],
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
                st.error("エラーが発生しました。設定を確認してください。\n\n" + str(e))

st.divider()
st.caption("Powered by LangChain + OpenAI API")