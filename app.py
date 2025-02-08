import streamlit as st

st.title("Chatbot Web App")
st.write("Hello, this is my chatbot!")

# Input field for user message
user_input = st.text_input("Type your message:")

if st.button("Send"):
    st.write(f"Chatbot response: {user_input}")  # Replace with actual chatbot logic

