# # Jupyter Notebook Documentation for `app.py`
# 
# ## **Overview**
# This script sets up a **Streamlit chatbot web app** that uses **LangChain** and the **Llama 3 (8B) model from Groq** 
# to provide AI-powered interactions.

# ## **Dependencies**
# Ensure you have the required Python libraries installed before running the script:
# ```
# pip install streamlit langchain groq
# ```

# ## **Environment Variable**
# Before running the chatbot, set your `GROQ_API_KEY` environment variable:
# ```bash
# export GROQ_API_KEY="your_api_key_here"  # Linux/macOS
# set GROQ_API_KEY="your_api_key_here"  # Windows (CMD)
# $env:GROQ_API_KEY="your_api_key_here"  # Windows (PowerShell)
# ```

# ## **Code Breakdown**

# ### **1. Import Required Modules**
import streamlit as st
from langchain.chat_models import init_chat_model
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import os

# Explanation:
# - `streamlit` → Used for creating the chatbot UI
# - `init_chat_model` → Initializes the Llama 3 chatbot model
# - `SystemMessage`, `HumanMessage`, `AIMessage` → Defines conversation roles
# - `os` → Accesses environment variables

# ### **2. Set Up API Key & Chat Model**
api_key = os.getenv("GROQ_API_KEY")

# Check if API key is set
if not api_key:
    st.error("Error: Please set your GROQ_API_KEY environment variable.")
    st.stop()

# Initialize the chat model (Llama 3 - 8B)
chat = init_chat_model("llama3-8b-8192", model_provider="groq")

# Set initial system message for AI behavior
messages = [SystemMessage(content="You are a helpful AI assistant.")]

# ### **3. Streamlit Web App UI**
st.title("Chatbot Web App")
st.write("Hello, this is my chatbot!")

# Create input field for user messages
user_input = st.text_input("Type your message:")

# ### **4. Handling User Input & Response**
if st.button("Send"):
    if user_input:
        messages.append(HumanMessage(content=user_input))  # Add user input
        response = chat.invoke(messages)  # Get AI response
        ai_message = AIMessage(content=response.content)
        messages.append(ai_message)  # Store AI response
        st.write(f"Chatbot response: {response.content}")  # Display AI response
    else:
        st.warning("Please enter a message before sending.")  # Warning if input is empty

# ## **Running the Chatbot**
# Run the chatbot using Streamlit:
# ```bash
# streamlit run app.py
# ```

# The chatbot will now be accessible in a web browser.
