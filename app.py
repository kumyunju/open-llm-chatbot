from uuid import UUID

import requests
import streamlit as st
from groq import APIError, Groq
from bike_programs import answer_chat

st.title("오픈소스 LLM 기반 분석 프로그램 운용 챗봇")

# 같은 브라우저에서 앱을 다시 열면 이전 대화를 찾을 수 있게 해요.
browser_memory = st.components.v2.component(
    "browser_chat_memory",
    html="<span hidden></span>",
    js="""
    export default function ({ data, setStateValue }) {
        if (data) return;
        let id;
        try {
            const key = "open-llm-chatbot.conversation.v1";
            id = window.localStorage.getItem(key);
            if (!id) {
                id = window.crypto.randomUUID();
                window.localStorage.setItem(key, id);
            }
        } catch {
            id = "unavailable";
        }
        setStateValue("browser_id", id);
    }
    """,
)

browser = browser_memory(
    key="browser_memory",
    data=st.session_state.get("browser_id"),
    on_browser_id_change=lambda: None,
)

if browser.browser_id == "unavailable":
    st.error("브라우저에서 사이트 데이터 저장을 허용해 주세요.")
    st.stop()

if not browser.browser_id:
    st.caption("대화 기록을 불러오는 중이에요…")
    st.stop()

try:
    conversation = UUID(browser.browser_id)
    if conversation.version != 4:
        raise ValueError
except (ValueError, TypeError, AttributeError):
    st.error("브라우저에 저장된 대화 정보가 올바르지 않아요.")
    st.stop()

conversation_id = str(conversation)
st.session_state["browser_id"] = conversation_id

# Streamlit Secrets에서 연결 정보를 읽어요.
try:
    table_url = st.secrets["SUPABASE_URL"].rstrip("/") + "/rest/v1/chat_messages"
    headers = {"apikey": st.secrets["SUPABASE_SECRET_KEY"]}
    groq_key = st.secrets["GROQ_API_KEY"]
except (FileNotFoundError, KeyError):
    st.error("Streamlit의 Secrets에서 연결 정보를 확인해 주세요.")
    st.stop()

# Supabase에서 최근 메시지를 읽고, 오래된 것부터 화면에 표시해요.
try:
    response = requests.get(
        table_url,
        headers=headers,
        params={
            "conversation_id": f"eq.{conversation_id}",
            "select": "role,content",
            "order": "id.desc",
            "limit": 100,
        },
        timeout=10,
    )
    response.raise_for_status()
    history = list(reversed(response.json()))
except requests.RequestException:
    st.error("대화 기록을 불러오지 못했어요. Supabase 연결을 확인해 주세요.")
    st.stop()

for entry in history:
    st.chat_message(entry["role"]).write(entry["content"])

message = st.chat_input("메시지를 입력하세요", max_chars=4000)

if message and message.strip():
    message = message.strip()
    st.chat_message("user").write(message)
    user_entry = {"role": "user", "content": message}

        try:
        with st.spinner("답변을 작성하고 있어요…"):
            client = Groq(api_key=groq_key)
            answer = answer_chat(client, history[-20:], user_entry)
    except APIError:
        st.error("모델에서 답변을 가져오지 못했어요. 잠시 후 다시 시도해 주세요.")
        st.stop()
    if not answer:
        st.warning("답변이 비어 있어요. 다시 물어봐 주세요.")
        st.stop()

    st.chat_message("assistant").write(answer)
    assistant_entry = {"role": "assistant", "content": answer}

    # 질문과 답변을 한 번에 저장해요.
    try:
        saved = requests.post(
            table_url,
            headers={**headers, "Prefer": "return=minimal"},
            json=[
                {"conversation_id": conversation_id, **entry}
                for entry in (user_entry, assistant_entry)
            ],
            timeout=10,
        )
        saved.raise_for_status()
    except requests.RequestException:
        st.warning("답변은 받았지만 이번 대화의 저장 여부를 확인하지 못했어요.")
