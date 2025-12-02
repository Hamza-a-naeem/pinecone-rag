import streamlit as st

st.title("My Chatbot")

# store messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# user input
user_input = st.chat_input("Ask something...")

if user_input:
    # add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # bot response (replace with real RAG or LLM)
    response = f"You said: {user_input}"

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
