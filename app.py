import streamlit as st
from langchain.chat_models import init_chat_model
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import os

# 🚨 Check for GROQ API Key
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("🚨 Error: Please set your GROQ_API_KEY environment variable.")
    st.stop()

# 🔮 Initialize Chat Model (Groq + llama3-8b-8192)
chat = init_chat_model("llama3-8b-8192", model_provider="groq")

# 💬 Store conversation in session state
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content="Hey I'm csusb chatbot, You can ask questions about csusb.")]

# 🎨 Custom CSS for a light, clean UI in blue/white/gray
st.markdown(
    """
    <style>
    /* Overall page background: light grayish-blue */
    body, .main {
        background-color: #F5F8FA;
        color: #333333;
    }
    .main > div {
        padding: 0 !important;
    }

    /* Remove default Streamlit chat message styling (outer boxes) */
    .stChatMessage {
        background: transparent !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    /* Container for centered content (optional) */
    .chat-container {
        max-width: 700px;
        margin: 2rem auto;
        padding: 1rem;
    }

    /* Title & Subtitle */
    .title {
        font-family: "Helvetica Neue", Arial, sans-serif;
        text-align: center;
        color: #333333;
        margin-top: 0;
        margin-bottom: 0.3rem;
        font-size: 1.8rem;
    }
    .subtitle {
        font-family: "Helvetica", sans-serif;
        text-align: center;
        color: #555555;
        margin-bottom: 1.5rem;
        font-size: 1rem;
    }

    /* User & Assistant Bubbles */
    .stChatMessageUser .stMarkdown {
        background-color: #0078D4;  /* Blue bubble for user */
        color: #FFFFFF;
        padding: 8px 12px;
        border-radius: 8px;
        display: inline-block;
        max-width: 80%;
        margin: 6px 0px;
        line-height: 1.4;
    }
    .stChatMessageAssistant .stMarkdown {
        background-color: #E9ECEF;  /* Light gray bubble for assistant */
        color: #333333;
        padding: 8px 12px;
        border-radius: 8px;
        display: inline-block;
        max-width: 80%;
        margin: 6px 0px;
        line-height: 1.4;
    }

    /* Chat Input */
    .stTextInput > div > div > input {
        background-color: #FFFFFF;
        color: #333333;
        border: 1px solid #CCCCCC;
        border-radius: 8px;
        padding: 10px;
        font-size: 16px;
    }
    .stTextInput > div > div > input::placeholder {
        color: #999999;
    }

    /* Spinner (Loading) */
    .stSpinner > div > div {
        color: #0078D4;
    }

    /* Button (if needed) */
    .stButton>button {
        background-color: #0078D4;
        color: #FFFFFF;
        font-size: 16px;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        transition: 0.3s;
        border: none;
    }
    .stButton>button:hover {
        background-color: #005999;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 🏷️ Container Start
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# 🏷️ Title & Subtitle
st.markdown('<h2 class="title">TEAM3 Chatbot - CSUSB</h2>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Welcome! Ask me anything, and I\'ll do my best to assist you.</p>', unsafe_allow_html=True)

# 💬 Display chat messages
for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        # Let st.chat_message auto-render the content
        pass

        # If you want to manually control the text:
        st.write(message.content)

# 📝 Chat input at bottom
user_input = st.chat_input("Type your message here...")

if user_input:
    # Add user message
    st.session_state.messages.append(HumanMessage(content=user_input))

    # Generate AI response
    with st.spinner("Thinking..."):
        response = chat.invoke(st.session_state.messages)
        ai_message = AIMessage(content=response.content)
        st.session_state.messages.append(ai_message)

    # Refresh UI
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)  # Close container


