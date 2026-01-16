import streamlit as st
from chatbot_logic import chatbot_response

st.set_page_config(page_title="FAQ Chatbot", page_icon="🤖")
st.title("🤖 FAQ Chatbot")
st.caption("ChatGPT-style FAQ chatbot using NLP & cosine similarity")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask your question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    response = chatbot_response(user_input)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
