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
import numpy as np
import pandas as pd
import os
from langchain.chat_models import init_chat_model
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import faiss
from sentence_transformers import SentenceTransformer

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

# 4. Apply Custom CSS
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
    .stChatMessage {
        background: transparent !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    .stChatMessageUser .stMarkdown {
        background-color: #0078D4;
        color: #FFFFFF;
        padding: 8px 12px;
        border-radius: 8px;
        display: inline-block;
        max-width: 80%;
        margin: 6px 0;
    }
    .stChatMessageAssistant .stMarkdown {
        background-color: #E9ECEF;
        color: #333333;
        padding: 8px 12px;
        border-radius: 8px;
        display: inline-block;
        max-width: 80%;
        margin: 6px 0;
    }
    .stTextInput>div>div>input {
        background-color: #FFFFFF;
        color: #333333;
        border: 1px solid #CCCCCC;
        border-radius: 8px;
        padding: 10px;
        font-size: 16px;
    }
    /* Make rating buttons look smaller and aligned */
    .rating-container {
        display: flex;
        flex-direction: row;
        gap: 0.5rem;
        margin: 1rem 0;
    }
    .rating-button {
        background-color: #0078D4;
        color: #FFFFFF;
        font-size: 12px;
        padding: 0.3rem 1rem;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .rating-button:hover {
        background-color: #005999;
    }
    /* Sidebar Button */
    .stButton>button {
        background-color: #0078D4;
        color: #FFFFFF;
        font-size: 14px;
        border-radius: 6px;
        padding: 0.4rem 1rem;
        border: none;
        margin-top: 10px;
    }
    .stButton>button:hover {
        background-color: #005999;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 5. Page Layout
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
st.markdown("<h2 class='title'>TEAM3 Chatbot - AI Research Helper</h2>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Welcome! Ask me about AI research, and I'll do my best to assist you.</p>", unsafe_allow_html=True)

# Path to the output file in the mounted volume
output_file_path = "/data/output.csv" 

df = pd.read_csv(output_file_path)

# Ensure the CSV file has a column named 'text'
if 'text' not in df.columns:
    raise ValueError("CSV file must have a 'text' column")

# Extract sentences from the CSV file
sentences = df['text'].tolist()

# Load a pre-trained sentence embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')  # You can choose any model you prefer

# Generate embeddings for the sentences
embeddings = model.encode(sentences).astype('float32')

# Create a FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance
index.add(embeddings)  # Add embeddings to the index

def retrieve_similar_sentences(query_sentence, k=1):
    # Generate embedding for the query sentence
    query_embedding = model.encode(query_sentence).astype('float32').reshape(1, -1)  # Reshape to 2D array

    # Search the index
    distances, indices = index.search(query_embedding, k)

    # Retrieve and return the most similar sentences
    similar_sentences = [sentences[indices[0][i]] for i in range(k)]
    return similar_sentences


# 6. Display Chat Messages
for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(message.content)

# 7. Rating Container (below messages)
st.write("#### Rate the Latest Chatbot Response")
rating_area = st.container()
with rating_area:
    st.markdown("<div class='rating-container'>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Correct ✅", key="correct_btn", help="Mark the last response as Correct"):
            st.session_state.conf_matrix[0, 0] += 1
            st.rerun()
    with col2:
        if st.button("Incorrect ❌", key="incorrect_btn", help="Mark the last response as Incorrect"):
            st.session_state.conf_matrix[0, 1] += 1
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# AI-to-AI Conversation (Alpha asks, Beta answers)
def ai_to_ai_conversation():
    alpha_prompt = "You are Alpha, an AI researcher. Ask a question about AI research."
    beta_prompt = "You are Beta, an AI assistant. Answer the question using available knowledge. If the question is unanswerable, say you don't know."

    messages = [SystemMessage(content="This is an AI-to-AI conversation. Alpha will ask, and Beta will respond.")]

    for i in range(10):  # 10 turns: 5 answerable + 5 unanswerable
        # Alpha asks a question
        with st.spinner(f"Alpha is thinking... ({i+1}/10)"):
            response_alpha = chat.invoke(messages + [HumanMessage(content=alpha_prompt)])
            alpha_question = response_alpha.content
            messages.append(HumanMessage(content=alpha_question))

        # Beta responds
        with st.spinner(f"Beta is thinking... ({i+1}/10)"):
            response_beta = chat.invoke(messages + [SystemMessage(content=beta_prompt)])
            beta_answer = response_beta.content
            messages.append(AIMessage(content=beta_answer))

        # Display the dialogue
        with st.chat_message("user"):
            st.write(f"**Alpha:** {alpha_question}")
        with st.chat_message("assistant"):
            st.write(f"**Beta:** {beta_answer}")

    st.success("AI-to-AI conversation completed!")

# Run AI-to-AI dialogue when triggered
if st.button("Run AI-to-AI Conversation"):
    ai_to_ai_conversation()

# 8. Chat Input at the Bottom
user_input = st.chat_input("Type your message here...")
if user_input:
    st.session_state.messages.append(HumanMessage(content=user_input))

    # Retrieve similar sentences based on user input
    similar_sentences = retrieve_similar_sentences(user_input)
    context = " ".join(similar_sentences)  # Combine similar sentences for context

    with st.spinner("Thinking..."):
        # Create a new list of messages to send to the model, including context
        messages_to_send = st.session_state.messages + [SystemMessage(content=f"Context: {context}")]
        response = chat.invoke(messages_to_send)

        ai_message = AIMessage(content=response.content)
        st.session_state.messages.append(ai_message)
    st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

# 9. Confusion Matrix & Metrics
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

# True Positive (TP) → The LLM correctly identifies that the question is answerable and provides an answer ("yes" or "no")
# False Positive (FP) → The LLM incorrectly tries to answer a question that is actually unanswerable
# True Negative (TN) → The LLM correctly identifies that the question is unanswerable and responds accordingly
# False Negative (FN) → The LLM fails to answer a question that is actually answerable

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

# 10. Reset Confusion Matrix Button
if st.sidebar.button("Reset Confusion Matrix"):
    st.session_state.conf_matrix = np.zeros((2, 2), dtype=int)
    st.rerun()

# 5 answerable questions by our chatbot - 

# 1. What is the github link of the dataset used in the research paper Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms? 
# 2. Which dataset was used by 'Brain Tumor Segmentation with Deep Neural Networks' paper?
# 3. Who are the authors for the paper 'Deep Residual Learning for Image Recognition'?
# 4. When was the paper 'U-Net: Convolutional Networks for Biomedical Image Segmentation' published?
# 5. What are the five histologic patterns of non-mucinous lung adenocarcinoma?


# 5 unanswerable questions by our chatbot - 

# 1. What is the complete list of references for the paper YOLOv3: An Incremental Improvement?
# 2. Who are the authors of the paper - Knowledge Transfer for Melanoma Screening with Deep Learning ?
# 3. When was the paper 'Skin Lesion Synthesis with Generative Adversarial Networks' published?
# 4. In the paper 'MED3D: TRANSFER LEARNING FOR 3D MEDICAL IMAGE ANALYSIS', what dice coefficient was achieved?
# 5. Which dataset was used in the paper 'V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation' ?
