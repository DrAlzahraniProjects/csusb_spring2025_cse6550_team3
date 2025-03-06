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
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time
import re


# Predefined questions
original_questions = [
    # Answerable Questions
    "What is the main advantage of using Curvature-based Feature Selection (CFS) over PCA for dimensionality reduction?",
    "How does the Inception-ResNet-v2 model contribute to feature extraction in breast tumor classification?",
    "What are the key classifiers used in the ensemble method for breast tumor classification, and why were they chosen?",
    "How does Menger Curvature help in ranking features in Electronic Health Records (EHR) data classification?",
    "What are the main challenges in handling missing data in medical datasets, and how does the first paper address this issue?",
    # Unanswerable Questions
    "How does the performance of CFS compare to other feature selection methods on completely different datasets outside the ones mentioned?",
    "What specific preprocessing steps were used for data normalization in each classification experiment?",
    "How would the proposed breast cancer classification model perform on a realtime clinical setup?",
    "Can the Curvature-based Feature Selection method be adapted to non-medical domains like finance or cybersecurity?",
    "How does the ensemble of CatBoost, XGBoost, and LightGBM compare to deep learning models trained end-to-end on histopathology images?"
]

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
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

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

if 'text' not in df.columns:
    raise ValueError("CSV file must have a 'text' column")

sentences = df['text'].tolist()

# Combine all sentences into a single text string (if needed)
csv_text = " ".join(sentences)

# Split text into smaller chunks for better embedding and retrieval
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
chunks = text_splitter.split_text(csv_text)

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

# Updated ai_to_ai_conversation function
def ai_to_ai_conversation():
    alpha_prompt = "You are Alpha, an AI researcher. Provide only the rephrased version of this question, keeping its meaning the same, without any additional text: '{}'"
    beta_prompt = "You are Beta, an AI assistant. Answer the following question based on the available corpus: '{}'"

    messages = [SystemMessage(content="This is an AI-to-AI conversation. Alpha will ask a question, and Beta will respond using the corpus.")]

    # Clear previous conversation history but keep messages for model context
    st.session_state.conversation_history = []

    for i, original_question in enumerate(original_questions):
        is_answerable = i < 5

        with st.spinner(f"Alpha is asking... ({i+1}/10)"):
            alpha_input = alpha_prompt.format(original_question)
            response_alpha = chat.invoke(messages + [HumanMessage(content=alpha_input)])
            rephrased_question = response_alpha.content.strip()
            rephrased_question = rephrased_question.split('\n')[0].strip()
            messages.append(HumanMessage(content=rephrased_question))

        with st.chat_message("user"):
            st.write(f"**Alpha:** {rephrased_question}")
        st.session_state.conversation_history.append(("user", f"**Alpha:** {rephrased_question}"))

        beta_placeholder = st.empty()
        with beta_placeholder.chat_message("assistant"):
            st.write("**Beta:** Thinking...")

        time.sleep(3)

        with st.spinner(f"Beta is responding... ({i+1}/10)"):
            similar_sentences = retrieve_similar_sentences(rephrased_question)
            context = " ".join(similar_sentences)

            # Create a new list of messages to send to the model, including context
            messages_to_send = st.session_state.messages + [SystemMessage(content=f"Context: {context}")]
            response = chat.invoke(messages_to_send)

            # beta_answer = AIMessage(content=response.content)
            beta_answer_str = ""

            # Get the full response string
            # Ensure response is a string; if not, extract its content.
            if hasattr(response, 'content'):
                beta_answer_str = response.content
            else:
                beta_answer_str = response

            # Define the prefix that precedes the actual content
            prefix = 'content="'
            suffix = '" additional_kwargs='

            # Check that the string starts with the expected prefix
            if beta_answer_str.startswith(prefix):
                # Remove the prefix
                temp = beta_answer_str[len(prefix):]
                # Find the end of the content using the suffix
                end_index = temp.find(suffix)
                if end_index != -1:
                    clean_text = temp[:end_index]
                else:
                    # Fallback: if the suffix isn't found, take the rest of the string
                    clean_text = temp
            else:
                # If the structure isn't as expected, fallback to using the whole string
                clean_text = beta_answer_str

            # Remove newline characters and extra spaces
            clean_text = clean_text.replace("\n", " ").strip()

            beta_answer = clean_text
            messages.append(AIMessage(content=beta_answer))

        with beta_placeholder.chat_message("assistant"):
            st.write(f"**Beta:** {beta_answer}")
        st.session_state.conversation_history.append(("assistant", f"**Beta:** {beta_answer}"))

        # Update confusion matrix
        if is_answerable and "No context available" not in beta_answer:
            st.session_state.conf_matrix[0, 0] += 1  # TP
        elif is_answerable and "No context available" in beta_answer:
            st.session_state.conf_matrix[0, 1] += 1  # FN
        elif not is_answerable and "No context available" in beta_answer:
            st.session_state.conf_matrix[1, 1] += 1  # TN
        else:
            st.session_state.conf_matrix[1, 0] += 1  # FP

        if i < len(original_questions) - 1:
            time.sleep(2)

    st.success("AI-to-AI conversation completed!")

# Display conversation history
for role, content in st.session_state.conversation_history:
    with st.chat_message(role):
        st.write(content)

# Run AI-to-AI dialogue when triggered
if st.button("Run AI-to-AI Conversation", key="run_conversation"):
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

# Sidebar with reset button
st.sidebar.write("### Confusion Matrix")
cm_df = pd.DataFrame(
    st.session_state.conf_matrix,
    index=["True Answerable", "True Unanswerable"],
    columns=["Predicted Answerable", "Predicted Unanswerable"]
)
st.sidebar.table(cm_df)

TP = st.session_state.conf_matrix[0, 0]
FN = st.session_state.conf_matrix[0, 1]
FP = st.session_state.conf_matrix[1, 0]
TN = st.session_state.conf_matrix[1, 1]

sensitivity = TP / (TP + FN) if (TP + FN) else 0
specificity = TN / (TN + FP) if (TN + FP) else 0
accuracy = (TP + TN) / np.sum(st.session_state.conf_matrix) if np.sum(st.session_state.conf_matrix) else 0
precision = TP / (TP + FP) if (TP + FP) else 0
f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) else 0

st.sidebar.write("### Metrics")
st.sidebar.write(f"Sensitivity (True Answerable Detection): {sensitivity:.2f}")
st.sidebar.write(f"Specificity (True Unanswerable Detection): {specificity:.2f}")
st.sidebar.write(f"Accuracy: {accuracy:.2f}")
st.sidebar.write(f"Precision: {precision:.2f}")
st.sidebar.write(f"F1 Score: {f1_score:.2f}")

# 10. Reset Confusion Matrix Button
if st.sidebar.button("Reset Confusion Matrix", key="reset_matrix"):
    st.session_state.conf_matrix = np.zeros((2, 2), dtype=int)
    st.session_state.conversation_history = []  # Clear conversation history
    st.rerun()  # Rerun to refresh UI


