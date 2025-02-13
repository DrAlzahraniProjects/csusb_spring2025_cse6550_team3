import streamlit as st
from langchain.chat_models import init_chat_model
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import os

# Set up the chatbot
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("Error: Please set your OPENAI_API_KEY environment variable.")
    st.stop()

chat = init_chat_model("llama3-8b-8192", model_provider="groq")
messages = [SystemMessage(content="You are a helpful AI assistant.")]

st.title("Chatbot Web App")
st.write("Hello, this is my chatbot!")

# Input field for user message
user_input = st.text_input("Type your message:")

if st.button("Send"):
    if user_input:
        messages.append(HumanMessage(content=user_input))
        response = chat.invoke(messages)
        ai_message = AIMessage(content=response.content)
        messages.append(ai_message)
        st.write(f"Chatbot response: {response.content}")
    else:
        st.warning("Please enter a message before sending.")
