# # Jupyter Notebook Documentation for app.py
#
# ## Overview
# This script sets up a Streamlit chatbot web app that uses LangChain and 
# the Llama 3 (8B) model from Groq to provide AI-powered interactions.
#
# ## Dependencies
# Ensure you have the required Python libraries installed before running the script:
#
#   pip install streamlit langchain groq
#
# ## Environment Variable
# Before running the chatbot, set your GROQ_API_KEY environment variable:
#
#   export GROQ_API_KEY="your_api_key_here"  # Linux/macOS
#   set GROQ_API_KEY="your_api_key_here"     # Windows (CMD)
#   $env:GROQ_API_KEY="your_api_key_here"    # Windows (PowerShell)
#
# ## Code Breakdown

import streamlit as st
import numpy as np
import pandas as pd
import os
from langchain.chat_models import init_chat_model
from langchain.schema import SystemMessage, HumanMessage, AIMessage

# 1. Check for API Key
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("Please set your GROQ_API_KEY environment variable.")
    st.stop()

# 2. Initialize Chat Model
chat = init_chat_model("llama3-8b-8192", model_provider="groq")

# 3. Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="Hey, I'm an AI research helper. Feel free to ask me anything about AI research.")
    ]
if "conf_matrix" not in st.session_state:
    st.session_state.conf_matrix = np.zeros((2, 2), dtype=int)
if "last_ai_response" not in st.session_state:
    st.session_state.last_ai_response = None

# 4. Apply Custom CSS for Better UI
st.markdown(
    """
    <style>
    body, .main {
        background-color: #F5F8FA;
        color: #333;
    }
    .chat-container {
        max-width: 700px;
        margin: 2rem auto;
        padding: 1rem;
    }
    .title {
        font-family: "Helvetica Neue", Arial, sans-serif;
        text-align: center;
        color: #333333;
        font-size: 1.8rem;
    }
    .subtitle {
        font-family: "Helvetica", sans-serif;
        text-align: center;
        color: #555555;
        margin-bottom: 1.5rem;
        font-size: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 5. Page Layout
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
st.markdown("<h2 class='title'>TEAM3 Chatbot - AI Research Helper</h2>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Welcome! Ask me about AI research, and I'll do my best to assist you.</p>", unsafe_allow_html=True)

# 6. Display Chat Messages
for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(message.content)

# 7. Create Placeholder for Rating Buttons Below Chat Input
rating_placeholder = st.empty()

# 8. Chat Input at the Bottom
user_input = st.chat_input("Type your message here...")

if user_input:
    st.session_state.messages.append(HumanMessage(content=user_input))

    # Send messages to AI
    messages_to_send = st.session_state.messages
    with st.spinner("Thinking..."):
        response = chat.invoke(messages_to_send)
        ai_message = AIMessage(content=response.content)
        st.session_state.messages.append(ai_message)
        st.session_state.last_ai_response = ai_message.content  # Store last AI response for rating
    
    st.rerun()

# 9. Display Rating Buttons Below Chat Input (Dynamically Appearing)
if st.session_state.last_ai_response:
    with rating_placeholder:
        st.markdown("### Rate the AI's Latest Response:")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Correct", key="correct"):
                st.session_state.conf_matrix[0, 0] += 1
                st.session_state.last_ai_response = None  # Hide buttons after rating
                st.success("Thank you for your feedback!")
                st.rerun()
        with col2:
            if st.button("❌ Incorrect", key="incorrect"):
                st.session_state.conf_matrix[0, 1] += 1
                st.session_state.last_ai_response = None  # Hide buttons after rating
                st.warning("Thank you for your feedback! We will improve.")
                st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

# 10. Confusion Matrix & Metrics (in the sidebar)
st.sidebar.write("### Confusion Matrix")
cm_df = pd.DataFrame(
    st.session_state.conf_matrix,
    index=["True answerable (Yes/No)", "True unanswerable"],
    columns=["LLM: Answerable (Yes/No)", "LLM: Unanswerable"]
)
st.sidebar.table(cm_df)

TP = st.session_state.conf_matrix[0, 0]
FN = st.session_state.conf_matrix[0, 1]
FP = st.session_state.conf_matrix[1, 0]
TN = st.session_state.conf_matrix[1, 1]

# Calculate Metrics
sensitivity = TP / (TP + FN) if (TP + FN) else 0
specificity = TN / (TN + FP) if (TN + FP) else 0
accuracy = (TP + TN) / np.sum(st.session_state.conf_matrix) if np.sum(st.session_state.conf_matrix) else 0
precision = TP / (TP + FP) if (TP + FP) else 0
f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) else 0

st.sidebar.write("### Metrics")
st.sidebar.write(f"Sensitivity: {sensitivity:.2f}")
st.sidebar.write(f"Specificity: {specificity:.2f}")
st.sidebar.write(f"Accuracy: {accuracy:.2f}")
st.sidebar.write(f"Precision: {precision:.2f}")
st.sidebar.write(f"F1 Score: {f1_score:.2f}")

# 11. Reset Confusion Matrix Button
if st.sidebar.button("Reset Confusion Matrix"):
    st.session_state.conf_matrix = np.zeros((2, 2), dtype=int)
    st.rerun()

