import streamlit as st

st.title("오픈소스 LLM 기반 분석 프로그램 운용 챗봇")

history = st.session_state.setdefault("chat_history", [])

for entry in history:
    st.chat_message(entry["role"]).write(entry["text"])

message = st.chat_input("메시지를 입력하세요")

if message:
    history.append({"role": "user", "text": message})
    st.chat_message("user").write(message)
