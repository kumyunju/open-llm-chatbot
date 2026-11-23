import streamlit as st
from groq import APIError, Groq

st.title("오픈소스 LLM 기반 분석 프로그램 운용 챗봇")

history = st.session_state.setdefault("chat_history", [])

for entry in history:
    st.chat_message(entry["role"]).write(entry["content"])

message = st.chat_input("메시지를 입력하세요")

if message:
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except (FileNotFoundError, KeyError):
        st.error("Streamlit의 Secrets에 GROQ_API_KEY를 설정해 주세요.")
        st.stop()

    user_entry = {"role": "user", "content": message}
    st.chat_message("user").write(message)

    try:
        with st.spinner("답변을 작성하고 있어요…"):
            client = Groq(api_key=api_key)
            result = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {"role": "system", "content": "사용자의 질문에 한국어로 답해 줘."},
                    *history[-20:],
                    user_entry,
                ],
            )
    except APIError:
        st.error("모델에서 답변을 받지 못했어요. 잠시 후 다시 시도해 주세요.")
        st.stop()

    answer = result.choices[0].message.content or "답변이 비어 있어요. 다시 물어봐 주세요."
    st.chat_message("assistant").write(answer)
    history.extend([user_entry, {"role": "assistant", "content": answer}])
