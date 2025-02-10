import streamlit as st
import os
import requests
from llama_cpp import Llama

st.title("Chatbot Web App")
st.write("Hello, this is my chatbot!")

MODEL_URL = f"https://drive.google.com/uc?id=132wcY2EADKUW2XDkKNKXnNGf5V9CgPwV"
MODEL_PATH = "llama-2-7b.Q4_K_M.gguf"

# Function to download the model if not present
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading Llama model... (this may take a while)")
        response = requests.get(MODEL_URL, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        with open(MODEL_PATH, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
        st.success("Download complete!")

# Ensure model is downloaded
download_model()

# Load the model
llm = Llama(model_path=MODEL_PATH)

# Input field for user message
user_input = st.text_input("Type your message:")

if st.button("Send") and user_input:
    response = llm(user_input, max_tokens=200)
    chatbot_reply = response["choices"][0]["text"].strip()
    st.write(f"Chatbot response: {chatbot_reply}")
