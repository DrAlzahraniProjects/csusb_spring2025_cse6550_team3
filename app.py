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

# ------------------- Custom CSS for Light/Dark Mode and Unified Button Styling -------------------
st.markdown(
    """
    <style>
    /* Light mode variables */
    :root {
        --background-color: #F5F8FA;
        --text-color: #333333;
        --subtitle-color: #555555;
        --button-bg: transparent;
        --button-text: red;      /* Red text in light mode */
        --button-border: red;    /* Red border in light mode */
        --hover-bg: rgba(255, 0, 0, 0.1);
    }
    /* Dark mode variables */
    @media (prefers-color-scheme: dark) {
        :root {
            --background-color: #121212;
            --text-color: #E0E0E0;
            --subtitle-color: #B0B0B0;
            --button-bg: transparent;
            --button-text: red;    /* Red text in dark mode */
            --button-border: red;  /* Red border in dark mode */
            --hover-bg: rgba(255, 0, 0, 0.1);
        }
    }
    
    body, .main {
        background-color: var(--background-color);
        color: var(--text-color);
    }
    .chat-container {
        max-width: 700px;
        margin: 2rem auto;
        padding: 1rem;
    }
    .title {
        font-family: "Helvetica Neue", Arial, sans-serif;
        text-align: center;
        color: var(--text-color);
        font-size: 1.8rem;
    }
    .subtitle {
        font-family: "Helvetica", sans-serif;
        text-align: center;
        color: var(--subtitle-color);
        margin-bottom: 1.5rem;
        font-size: 1rem;
    }
    .stChatMessage {
        background: transparent !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    /* Chat bubbles for both user and assistant use a transparent background with a border */
    .stChatMessageUser .stMarkdown, 
    .stChatMessageAssistant .stMarkdown {
        background-color: transparent;
        color: var(--text-color);
        padding: 8px 12px;
        border: 1px solid var(--text-color);
        border-radius: 8px;
        display: inline-block;
        max-width: 80%;
        margin: 6px 0;
    }
    .stTextInput>div>div>input {
        background-color: var(--background-color);
        color: var(--text-color);
        border: 1px solid #CCCCCC;
        border-radius: 8px;
        padding: 10px;
        font-size: 16px;
    }
    /* Unified button styling for all buttons */
    .stButton>button {
        background-color: var(--button-bg);
        color: var(--button-text);
        border: 2px solid var(--button-border);
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
        transition: background-color 0.3s ease, color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: var(--hover-bg);
    }
    /* Sidebar button styling to match the main buttons */
    .stSidebar .stButton>button {
        background-color: var(--button-bg);
        color: var(--button-text);
        border: 2px solid var(--button-border);
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
        transition: background-color 0.3s ease, color 0.3s ease;
    }
    .stSidebar .stButton>button:hover {
        background-color: var(--hover-bg);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------- Main App Code -------------------

# 1. Check for API Key
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("Please set your GROQ_API_KEY environment variable.")
    st.stop()

# 2. Initialize the Chat Model
chat = init_chat_model("llama3-8b-8192", model_provider="groq")

# 3. Initialize Session State Variables
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="Hey, I'm an AI research helper. Feel free to ask me anything about AI research.")
    ]
if "conf_matrix" not in st.session_state:
    st.session_state.conf_matrix = np.zeros((2, 2), dtype=int)
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "last_ai_response" not in st.session_state:
    st.session_state.last_ai_response = None

# 4. Layout the Main Chat Container
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
st.markdown("<h2 class='title'>TEAM3 Chatbot - AI Research Helper</h2>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Welcome! Ask me about AI research, and I'll do my best to assist you.</p>", unsafe_allow_html=True)

# 5. Load Text Data from CSV File
output_file_path = "/data/output.csv" 
df = pd.read_csv(output_file_path)
if 'text' not in df.columns:
    raise ValueError("CSV file must have a 'text' column")
sentences = df['text'].tolist()
csv_text = " ".join(sentences)

# 6. Split Text and Build FAISS Index
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
chunks = text_splitter.split_text(csv_text)
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(sentences).astype('float32')
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

def retrieve_similar_sentences(query_sentence, k=1):
    query_embedding = model.encode(query_sentence).astype('float32').reshape(1, -1)
    distances, indices = index.search(query_embedding, k)
    similar_sentences = [sentences[indices[0][i]] for i in range(k)]
    return similar_sentences

# 7. Display Chat Messages
for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(message.content)

# 8. Display AI-to-AI Conversation History
for role, content in st.session_state.conversation_history:
    with st.chat_message(role):
        st.write(content)

# 9. "Start AI-to-AI Conversation" Button (Unified Styling)
if st.button("Start AI-to-AI Conversation", key="run_conversation", help="Click to start AI-to-AI conversation"):
    def ai_to_ai_conversation():
        original_questions = [
            "What is the main advantage of using Curvature-based Feature Selection (CFS) over PCA for dimensionality reduction?",
            "How does the Inception-ResNet-v2 model contribute to feature extraction in breast tumor classification?",
            "What are the key classifiers used in the ensemble method for breast tumor classification, and why were they chosen?",
            "How does Menger Curvature help in ranking features in Electronic Health Records (EHR) data classification?",
            "What are the main challenges in handling missing data in medical datasets, and how does the first paper address this issue?",
            "How does the performance of CFS compare to other feature selection methods on completely different datasets outside the ones mentioned?",
            "What specific preprocessing steps were used for data normalization in each classification experiment?",
            "How would the proposed breast cancer classification model perform on a realtime clinical setup?",
            "Can the Curvature-based Feature Selection method be adapted to non-medical domains like finance or cybersecurity?",
            "How does the ensemble of CatBoost, XGBoost, and LightGBM compare to deep learning models trained end-to-end on histopathology images?"
        ]
        alpha_prompt = "You are Alpha, an AI researcher. Provide only the rephrased version of this question, keeping its meaning the same, without any additional text: '{}'"
        beta_prompt = "You are Beta, an AI assistant. Using the provided context, generate a concise answer to the following question: '{}'"
    
        messages = [SystemMessage(content="This is an AI-to-AI conversation. Alpha will ask a question, and Beta will respond using the corpus as context.")]
        st.session_state.conversation_history = []
    
        for i, original_question in enumerate(original_questions):
            is_answerable = i < 5
            with st.spinner(f"Alpha is asking... ({i+1}/10)"):
                alpha_input = alpha_prompt.format(original_question)
                response_alpha = chat.invoke([HumanMessage(content=alpha_input)])
                rephrased_question = response_alpha.content.strip().split('\n')[0].strip()
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
                beta_input = f"You are Beta, an AI assistant. Answer the following question using the given context:\n\nQuestion: {rephrased_question}\nContext: {context}\n\nProvide a clear, well-structured response based on the information available."
                response_beta = chat.invoke([HumanMessage(content=beta_input)])
                beta_answer = response_beta.content.strip()
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
    
    ai_to_ai_conversation()

# 10. Chat Input at the Bottom
user_input = st.chat_input("Type your message here...")
if user_input:
    st.session_state.messages.append(HumanMessage(content=user_input))
    similar_sentences = retrieve_similar_sentences(user_input)
    context = " ".join(similar_sentences)
    with st.spinner("Thinking..."):
        messages_to_send = st.session_state.messages + [SystemMessage(content=f"Context: {context}")]
        response = chat.invoke(messages_to_send)
        ai_message = AIMessage(content=response.content)
        st.session_state.messages.append(ai_message)
        st.session_state.last_ai_response = ai_message.content  # Save latest AI response for rating
    st.rerun()

# 11. Display Rating Buttons Below Chat Input (Unified Styling)
if st.session_state.last_ai_response:
    rating_placeholder = st.empty()
    with rating_placeholder:
        st.markdown("### Rate the AI's Latest Response:")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Correct", key="correct"):
                st.session_state.conf_matrix[0, 0] += 1
                st.session_state.last_ai_response = None  # Hide rating buttons after rating
                st.success("Thank you for your feedback!")
                st.rerun()
        with col2:
            if st.button("❌ Incorrect", key="incorrect"):
                st.session_state.conf_matrix[0, 1] += 1
                st.session_state.last_ai_response = None  # Hide rating buttons after rating
                st.warning("Thank you for your feedback! We will improve.")
                st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

# 12. Sidebar: Display Confusion Matrix and Metrics
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

if st.sidebar.button("Reset Confusion Matrix", key="reset_matrix"):
    st.session_state.conf_matrix = np.zeros((2, 2), dtype=int)
    st.session_state.conversation_history = []
    st.rerun()

