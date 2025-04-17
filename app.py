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
import requests

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
    /* IP verification status styles */
    .ip-status-allowed {
        background-color: #d4edda;
        color: #155724;
        padding: 10px 15px;
        border-radius: 5px;
        margin-bottom: 20px;
        border: 1px solid #c3e6cb;
        text-align: center;
    }
    .ip-status-denied {
        background-color: #f8d7da;
        color: #721c24;
        padding: 10px 15px;
        border-radius: 5px;
        margin-bottom: 20px;
        border: 1px solid #f5c6cb;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------- IP Verification Functions -------------------

def get_user_ip() -> str:
    try:
        # When Streamlit is running inside a container, the Request object might not be accessible
        response = requests.get('https://api.ipify.org?format=json', timeout=5)
        return response.json().get("ip", "")
    except Exception:
        return ""

def is_csusb_ip(ip: str) -> bool:
    return any([
        ip.startswith("138.23."),
        ip.startswith("139.182."),
        ip.startswith("152.79.") 
    ])

# ------------------- Main App Code -------------------

# Create page title
# Verify IP address with a subtle but informative indicator at the top
user_ip = get_user_ip()
if not is_csusb_ip(user_ip):
    # Show denied access message with custom HTML
    st.markdown(
        f"""
        <div class="ip-status-denied">
            <h3>🚫 Access Denied</h3>
            <p>Your IP address ({user_ip}) is not from the CSUSB campus network.</p>
            <p>Only users within the CSUSB campus network can access this application.</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    st.stop()
else:
    # Add a subtle text indicator at the very top with more descriptive text
    st.markdown(
        f"""
        <div style="text-align: right; font-size: 11px; color: #779977; padding: 2px; margin-top: -15px;">
        CSUSB IP verification successful: {user_ip} ✓
        </div>
        """, 
        unsafe_allow_html=True
    )

# Create page title and welcome message - keeping original UI intact
st.markdown("<h2 class='title'>TEAM3 Chatbot - AI Research Helper</h2>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Welcome! Ask me about AI research, and I'll do my best to assist you.</p>", unsafe_allow_html=True)

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
        SystemMessage(content="You are an AI research helper. Respond with concise answers limited to one paragraph. Keep responses focused and brief while maintaining helpfulness.")
    ]
if "conf_matrix" not in st.session_state:
    st.session_state.conf_matrix = np.zeros((2, 2), dtype=int)
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "last_ai_response" not in st.session_state:
    st.session_state.last_ai_response = None
if "question_times" not in st.session_state:
    st.session_state.question_times = []

# 4. Layout the Main Chat Container
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

is_new__papers_path = "/data/is_new_pdfs.txt"
faiss_index_file_path = "/app/data/faiss_index.index"
chunks_file_path = "/app/data/chunks.txt"
is_new_vector_database = True
model = None
index = None
chunks = []

model = SentenceTransformer('all-MiniLM-L6-v2')

#Load chunks
# Open the file in read mode
with open(chunks_file_path, 'r') as f:
    # Read each line from the file
    for line in f:
        # Strip the newline character and add the line to the chunks list
        chunks.append(line.strip())

# Step 2: Read the index using FAISS
index = faiss.read_index(faiss_index_file_path)

# def create_vector_database():
#     global model, chunks, index

#     # Path to your JSON file
#     input_file_path = "/data/papers_output.json"

#     # Download the FAISS index file from Google Drive
#     json_file_id = '1Rtcdans4EUxuZK16kub0em2L4sONK5BY'
#     gdown.download(f'https://drive.google.com/uc?id={json_file_id}', input_file_path, quiet=False)

#     # === Load JSON ===
#     with open(input_file_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)

#     if not isinstance(data, list):
#         data = [data]  # Ensure list of entries

#     # === Normalize JSON to flat DataFrame ===
#     df = pd.json_normalize(data)
#     df.fillna("", inplace=True)

#     # === Combine all fields into a single text column ===
#     df["text"] = df.astype(str).agg(" ".join, axis=1)

#     # === Prepare sentences ===
#     sentences = df["text"].tolist()
#     json_text = " ".join(sentences)


#     # Split text into smaller chunks for better embedding and retrieval
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
#     chunks = text_splitter.split_text(json_text)

#     # Load a pre-trained sentence embedding model
#     model = SentenceTransformer('all-MiniLM-L6-v2')

#     # Open the file in write mode
#     with open(chunks_file_path, 'w') as f:
#         for chunk in chunks:
#             f.write(chunk + '\n')  # Write each chunk on a new line

#     # Generate embeddings for the sentences
#     embeddings = model.encode(chunks).astype('float32')

#     # Create a FAISS index
#     index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance
#     index.add(embeddings)  # Add embeddings to the index
#     faiss.write_index(index, faiss_index_file_path)

# #check if vector database already exists
# # Load the persisted set of processed URLs when the spider starts
# if os.path.exists(is_new__papers_path):
#     try:
#         # Open the text file
#         with open(is_new__papers_path, mode='r') as txtfile:
#             first_line = txtfile.readline().strip()  # Read the first line and strip whitespace
#             if first_line:  # Check if the line is not empty
#                 first_char = first_line[0]

#                 if first_char == '0':
#                     #Load the vector database
#                     is_new_vector_database = False
#                 elif first_char == '1':
#                     #Make a new vector database
#                     is_new_vector_database = True
#     except Exception as e:
#         print(f"Error saving processed files: {e}")

    
# if is_new_vector_database:
#     create_vector_database()
# else:
#     if os.path.exists(faiss_index_file_path):
#         # Load a pre-trained sentence embedding model
#         model = SentenceTransformer('all-MiniLM-L6-v2')

#         #Load chunks
#         # Open the file in read mode
#         with open(chunks_file_path, 'r') as f:
#             # Read each line from the file
#             for line in f:
#                 # Strip the newline character and add the line to the chunks list
#                 chunks.append(line.strip())

#         #Load a saved FAISS index
#         index = faiss.read_index(faiss_index_file_path)
#     else:
#         create_vector_database()


def check_rate_limit():
    """Check if the user has exceeded 10 questions in the last 60 seconds."""
    current_time = time.time()
    # Remove timestamps older than 60 seconds
    st.session_state.question_times = [t for t in st.session_state.question_times if current_time - t < 60]
    if len(st.session_state.question_times) >= 10:
        return False, "You’ve reached the limit of 10 questions per minute because the server has limited resources. Please try again in 3 minutes."
    return True, None

def retrieve_similar_sentences(query_sentence, k=3):
    """Retrieve top-k similar sentences from the corpus."""
    query_embedding = model.encode(query_sentence).astype('float32').reshape(1, -1)
    distances, indices = index.search(query_embedding, k)
    similar_sentences = [chunks[indices[0][i]] for i in range(min(k, len(indices[0])))]
    return similar_sentences, distances[0].tolist()

def rerank_sentences(query, sentences):
    """Use LLM to re-rank retrieved sentences based on relevance to the query with retry on rate limit."""
    rerank_prompt = (
        "You are an AI tasked with ranking sentences based on their relevance to a query. "
        "For each sentence, provide a relevance score between 0 and 1 (where 1 is highly relevant) "
        "and a brief explanation. Return the results in this format exactly, do not bold the words:\n"
        "Sentence: <sentence>\nScore: <score>\nExplanation: <explanation>\n\n"
        "Query: '{}'\n\n"
        "Sentences to rank:\n{}"
    ).format(query, "\n".join([f"{i+1}. {s}" for i, s in enumerate(sentences)]))

    max_retries = 3
    retry_delay = 60  # Wait 60 seconds to ensure TPM limit resets

    for attempt in range(max_retries):
        try:
            response = chat.invoke([HumanMessage(content=rerank_prompt)])
            reranked = []
            
            # Parse LLM response
            lines = response.content.strip().split("\n")
            for i in range(0, len(lines), 3):
                try:
                    sentence = lines[i].replace("Sentence: ", "").strip()
                    score = float(lines[i+1].replace("Score: ", "").strip())
                    reranked.append((sentence, score))
                except (IndexError, ValueError):
                    continue
            
            # Sort by score in descending order
            reranked.sort(key=lambda x: x[1], reverse=True)
            return reranked

        except Exception as e:
            if "Error code: 413" in str(e) and "rate_limit_exceeded" in str(e):
                if attempt < max_retries - 1:
                    st.warning(f"Rate limit exceeded. Waiting {retry_delay} seconds before retrying... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
            # If not a rate limit error or max retries reached, raise the exception
            raise e
    
    # If all retries fail, return an empty list to avoid breaking downstream logic
    return []

# 7. Display Chat Messages
for message in st.session_state.messages:
    if isinstance(message, SystemMessage):
        continue  # Skip displaying system messages
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(message.content)

# 8. Display AI-to-AI Conversation History
for role, content in st.session_state.conversation_history:
    with st.chat_message(role):
        st.write(content)

# 9. "Start AI-to-AI Conversation" Button (Unified Styling)
# if st.button("Start AI-to-AI Conversation", key="run_conversation", help="Click to start AI-to-AI conversation"):
#     def ai_to_ai_conversation():
#         original_questions = [
#             "What is the main advantage of using Curvature-based Feature Selection (CFS) over PCA for dimensionality reduction?",
#             "How does the Inception-ResNet-v2 model contribute to feature extraction in breast tumor classification?",
#             "What are the key classifiers used in the ensemble method for breast tumor classification, and why were they chosen?",
#             "How does Menger Curvature help in ranking features in Electronic Health Records (EHR) data classification?",
#             "What are the main challenges in handling missing data in medical datasets, and how does the first paper address this issue?",
#             "How does the performance of CFS compare to other feature selection methods on completely different datasets outside the ones mentioned?",
#             "What specific preprocessing steps were used for data normalization in each classification experiment?",
#             "How would the proposed breast cancer classification model perform on a realtime clinical setup?",
#             "Can the Curvature-based Feature Selection method be adapted to non-medical domains like finance or cybersecurity?",
#             "How does the ensemble of CatBoost, XGBoost, and LightGBM compare to deep learning models trained end-to-end on histopathology images?"
#         ]
#         alpha_prompt = "You are Alpha, an AI researcher. Provide only the rephrased version of this question, keeping its meaning the same, without any additional text: '{}'"
#         beta_prompt = "You are Beta, an AI assistant. Generate a concise answer limited to one paragraph to this question using only the provided context: '{}'"
    
#         messages = [SystemMessage(content="This is an AI-to-AI conversation. Alpha will ask a question, and Beta will respond using the re-ranked corpus context.")]
#         st.session_state.conversation_history = []
    
#         for i, original_question in enumerate(original_questions):
#             is_answerable = i < 5
#             with st.spinner(f"Alpha is asking... ({i+1}/10)"):
#                 alpha_input = alpha_prompt.format(original_question)
#                 response_alpha = chat.invoke([HumanMessage(content=alpha_input)])
#                 rephrased_question = response_alpha.content.strip().split('\n')[0].strip()
#                 messages.append(HumanMessage(content=rephrased_question))
    
#             with st.chat_message("user"):
#                 st.write(f"**Alpha:** {rephrased_question}")
#             st.session_state.conversation_history.append(("user", f"**Alpha:** {rephrased_question}"))
    
#             beta_placeholder = st.empty()
#             with beta_placeholder.chat_message("assistant"):
#                 st.write("**Beta:** Thinking...")
    
#             time.sleep(1.5)
    
#             with st.spinner(f"Beta is responding... ({i+1}/10)"):
#               # Step 1: Retrieve top-k similar sentences
#               similar_sentences, distances = retrieve_similar_sentences(rephrased_question, k=3)
            
#               if not similar_sentences:
#                   beta_answer = "No context available."
#               else:
#                   # Step 2: Re-rank using LLM (restored as original)
#                   reranked = rerank_sentences(rephrased_question, similar_sentences)

#                   # Step 3: Use top-ranked sentence(s) as context
#                   if reranked:
#                       context = reranked[0][0]  # Use the top-ranked sentence
#                       beta_input = beta_prompt.format(rephrased_question)
#                       messages_to_send = [
#                           SystemMessage(content=f"Context: {context}\n\nYou MUST respond with a concise answer limited to one paragraph."), 
#                           HumanMessage(content=beta_input)
#                       ]
#                       response_beta = chat.invoke(messages_to_send)
#                       beta_answer = response_beta.content.strip()
#                   else:
#                       beta_input = beta_prompt.format(rephrased_question)
#                       messages_to_send = [SystemMessage(content=f"Context: {similar_sentences}"), HumanMessage(content=beta_input)]
#                       response_beta = chat.invoke(messages_to_send)
#                       beta_answer = response_beta.content.strip()

#             messages.append(AIMessage(content=beta_answer))
    
#             with beta_placeholder.chat_message("assistant"):
#                 st.write(f"**Beta:** {beta_answer}")
#             st.session_state.conversation_history.append(("assistant", f"**Beta:** {beta_answer}"))
    
#             # Update confusion matrix
#             if is_answerable and "No context available" not in beta_answer:
#                 st.session_state.conf_matrix[0, 0] += 1  # TP
#             elif is_answerable and "No context available" in beta_answer:
#                 st.session_state.conf_matrix[0, 1] += 1  # FN
#             elif not is_answerable and "No context available" in beta_answer:
#                 st.session_state.conf_matrix[1, 1] += 1  # TN
#             else:
#                 st.session_state.conf_matrix[1, 0] += 1  # FP
    
#             if i < len(original_questions) - 1:
#                 time.sleep(0.5)
    
#         st.success("AI-to-AI conversation completed!")
    
#     ai_to_ai_conversation()

# 10. Chat Input at the Bottom
user_input = st.chat_input("Type your message here...")
if user_input:
    can_ask, error_message = check_rate_limit()
    if not can_ask:
        st.error(error_message)
    else:
        st.session_state.question_times.append(time.time())
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("Thinking..."):
            # Retrieve and re-rank for user input
            similar_sentences, _ = retrieve_similar_sentences(user_input, k=3)
            if not similar_sentences:
                context = "No context available."
            else:
                reranked = rerank_sentences(user_input, similar_sentences)
                context = reranked[0][0] if reranked else "No context available."
            
            messages_to_send = st.session_state.messages + [
                SystemMessage(content=f"Context: {context}\n\nYou MUST respond with a concise answer limited to one paragraph.")
            ]
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
