import hashlib
import json
import subprocess
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
import threading

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
    """
    Get the user's IP address by making a request to an external service.
    
    Returns:
        str: The user's IP address, or an empty string if it cannot be determined.
    """
    try:
        response = requests.get('https://api.ipify.org?format=json', timeout=None)
        return response.json().get("ip", "")
    except Exception:
        return ""

def is_US_ip(ip: str) -> bool:
    """
    Check if an IP address is from the United States using ip-api.com.
    
    Args:
        ip (str): The IP address to check.
        
    Returns:
        bool: True if the IP is from the US, False otherwise.
    """
    try:
        response = requests.get(f"http://ip-api.com/json/{ip}?fields=country", timeout=None)
        return response.json().get("country", "").lower() == "united states"
    except Exception:
        return False

# ------------------- Main App Code -------------------

# Create page title
# Verify IP address with a subtle but informative indicator at the top
user_ip = get_user_ip()
if not is_US_ip(user_ip):
    # Show denied access message with custom HTML
    st.markdown(
        f"""
        <div class="ip-status-denied">
            <h3>🚫 Access Denied</h3>
            <p>Access to this webpage is prohibited.</p>
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
        US IP verification successful: {user_ip} ✓
        </div>
        """, 
        unsafe_allow_html=True
    )

# Create page title and welcome message - keeping original UI intact
st.markdown("<h2 class='title'>TEAM3 Chatbot - AI Research Helper</h2>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Welcome! I'm here to assist you with your research. Ask me research-related questions, and I'll provide answers based on datasets, models and research papers from paperswithcode.com/sota.</p>", unsafe_allow_html=True)

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
            SystemMessage(content="""
    You are an expert research assistant.
    You are designed to help users with questions related to research papers, methodologies, benchmarks, datasets, models, and advancements, based on information from Papers with Code.

    Rules & Restrictions:
    - **Stay on Topic:** Only respond to questions related to research, including models, datasets, benchmarks, research papers, and machine learning techniques.
    - **No Negative Responses:** Maintain a factual, supportive, and encouraging tone at all times.
    - **Support and Guide:** Provide clear, concise, and precise responses focused on research.
    - **No Controversial Discussions:** Avoid unrelated topics such as politics, ethics debates, or opinions outside of research.
    - **Keep Responses Concise:** Limit answers to 2-3 sentences to ensure clarity, focus, and academic professionalism.

    Provide a concise and accurate answer based solely on the context below.
    If the conte    xt does not contain enough information to answer the question, respond with "I don't have enough information to answer this question." Do not generate, assume, or fabricate any details beyond the given context.
    """)
    ]

# Commented out confusion matrix
# if "conf_matrix" not in st.session_state:
#     st.session_state.conf_matrix = np.zeros((2, 2), dtype=int)

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Commented out last_ai_response for the rating buttons
# if "last_ai_response" not in st.session_state:
#     st.session_state.last_ai_response = None

if "question_times" not in st.session_state:
    st.session_state.question_times = []

# 4. Layout the Main Chat Container
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

PAPER_HASHES_PATH = "/data/paper_hashes.json"
PAPER_HASHES = {}

if os.path.exists(PAPER_HASHES_PATH):
    with open(PAPER_HASHES_PATH, "r") as f:
        try:
            PAPER_HASHES = json.load(f)
        except json.JSONDecodeError:
            PAPER_HASHES = {}

def compute_md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

faiss_index_file_path = "/app/data/faiss_index.index"
chunks_file_path = "/app/data/chunks.txt"
model = None
index = None
chunks = []
new_chunks = []

model = SentenceTransformer('all-MiniLM-L6-v2')

#Load chunks
# Open the file in read mode
with open(chunks_file_path, 'r') as f:
    # Read each line from the file
    for line in f:
        # Strip the newline character and add the line to the chunks list
        chunks.append(line.strip())

index = faiss.read_index(faiss_index_file_path)


def create_chunks(output_file_path="/data/papers_output.json"):
    global chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)

    df = pd.read_json(output_file_path)

    for _, row in df.iterrows():
        uid = row.get('url')
        full_text = json.dumps(row.to_dict(), sort_keys=True, default=str)
        current_hash = compute_md5(full_text)

        if PAPER_HASHES.get(uid) == current_hash:
            continue  # Skip unchanged papers

        PAPER_HASHES[uid] = current_hash  # Track updated paper

        # Dataset–Model Pairs
        datasets = row.get('dataset_names', [])
        models = row.get('best_model_names', [])
        dataset_links = row.get('dataset_links', [])
        for i in range(min(len(datasets), len(models))):
            dataset_text = f"Dataset: {datasets[i]}, Best Model: {models[i]}"
            if i < len(dataset_links):
                dataset_text += f", Dataset Link: {dataset_links[i]}"
            new_chunks.append(dataset_text)

        # Main Metadata
        title = row.get('title') or ""
        main_text = " ".join(filter(None, [
            f"Title: {title.strip()}",
            f"Abstract: {row.get('abstract', '')}",
            f"Description: {row.get('description', '')}",
            f"URL: {row.get('url', '')}",
            f"Date: {row.get('date', '')}",
            f"Authors: {', '.join(row.get('authors', []))}",
            f"Artefacts: {', '.join(row.get('artefact-information', []))}"
        ]))
        new_chunks.extend(text_splitter.split_text(main_text))

        # Paper List Entries
        paper_titles = row.get('paper_list_titles', [])
        paper_abstracts = row.get('paper_list_abstracts', [])
        paper_authors = row.get('paper_list_authors', [])
        paper_dates = row.get('paper_list_dates', [])
        paper_links = row.get('paper_list_title_links', [])
        author_links = row.get('paper_list_author_links', [])
        for i in range(len(paper_titles)):
            paper_text = " ".join(filter(None, [
                f"Paper Title: {paper_titles[i]}",
                f"Link: {paper_links[i]}" if i < len(paper_links) else "",
                f"Abstract: {paper_abstracts[i]}" if i < len(paper_abstracts) else "",
                f"Authors: {paper_authors[i]}" if i < len(paper_authors) else "",
                f"Author Links: {author_links[i]}" if i < len(author_links) else "",
                f"Date: {paper_dates[i]}" if i < len(paper_dates) else ""
            ]))
            new_chunks.extend(text_splitter.split_text(paper_text))

    # Load existing chunks
    existing_chunks = []
    if os.path.exists("/data/chunks.txt"):
        with open("/data/chunks.txt", 'r') as f:
            existing_chunks = [line.strip() for line in f if line.strip()]

    if new_chunks != []:
        all_chunks = existing_chunks + new_chunks

        with open("/data/chunks.txt", 'w') as f:
            for chunk in all_chunks:
                f.write(chunk + '\n')

        chunks = all_chunks  # update global chunks
    else:
        print("No new chunks to add.")

    if new_chunks != []:
        # Update paper hashes
        with open(PAPER_HASHES_PATH, "w") as f:
            json.dump(PAPER_HASHES, f, indent=2)

def create_vector_database():
    """
    Create a vector database using FAISS by encoding chunks of text.
    This function loads text chunks, generates embeddings, and creates a searchable index.
    The index is then saved to disk for future use.
    """
    global model, index, chunks

    if new_chunks == []:
        print("No new content. FAISS index unchanged.")
        return

    # Create embeddings and FAISS index
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss_index_file_path = "/data/faiss_index.index"

    # Save index to file
    faiss.write_index(index, faiss_index_file_path)


# Step 2: Create the index using FAISS
def load_existing_index():
    """
    Load an existing FAISS index from disk if available.
    If not available, start a background thread to create a new index.
    """
    global index
    
    saved_faiss_index_file_path = "/data/faiss_index.index"

    if os.path.exists(saved_faiss_index_file_path):
        index = faiss.read_index(saved_faiss_index_file_path)
        print("Loaded existing FAISS index.")

load_existing_index()

def run_scraper():
    """Run the Scrapy spider via subprocess."""
    result = subprocess.run(["python3", "go-paper-spider.py"], capture_output=True)


def full_refresh_pipeline():
    run_scraper()
    create_chunks()
    create_vector_database()

def schedule_next_run(interval_hours: int = 24):
    """Run `full_refresh_pipeline` every `interval_hours` after an initial delay."""
    def run_and_reschedule():
        full_refresh_pipeline()
        schedule_next_run(interval_hours)  # Reschedule after each run

    delay = interval_hours * 3600
    timer = threading.Timer(delay, run_and_reschedule)
    timer.daemon = True  # Allows the program (e.g., Streamlit) to exit cleanly
    timer.start()

# Start the first scheduled run in a background thread
thread_scrape_increment = threading.Thread(target=schedule_next_run)
thread_scrape_increment.daemon = True
thread_scrape_increment.start()

def check_rate_limit():
    """
    Check if the user has exceeded 10 questions in the last 60 seconds.
    
    Returns:
        tuple: (bool, str or None) A tuple containing:
            - A boolean indicating if the user can ask another question
            - An error message if the rate limit is exceeded, None otherwise
    """
    current_time = time.time()
    # Remove timestamps older than 60 seconds
    st.session_state.question_times = [t for t in st.session_state.question_times if current_time - t < 60]
    if len(st.session_state.question_times) >= 10:
        return False, "You've reached the limit of 10 questions per minute because the server has limited resources. Please try again in 3 minutes."
    return True, None

def retrieve_similar_sentences(query_sentence, k):
    """Retrieve top-k similar sentences from the corpus."""
    query_embedding = model.encode(query_sentence).astype('float32').reshape(1, -1)
    distances, indices = index.search(query_embedding, k)
    similar_sentences = [chunks[indices[0][i]] for i in range(min(k, len(indices[0])))]
    return similar_sentences, distances[0].tolist()

def rerank_sentences(query, sentences):
    """
    Use LLM to re-rank retrieved sentences based on relevance to the query with retry on rate limit.
    
    Args:
        query (str): The original user query
        sentences (list): List of sentences to rank
        
    Returns:
        list: List of tuples (sentence, relevance_score) sorted by relevance
    """
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

# The AI-to-AI conversation button section has been left commented out in the original code

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
            similar_sentences, _ = retrieve_similar_sentences(user_input, k=10)
            if not similar_sentences:
                context = "No context available."
            else:
                reranked = rerank_sentences(user_input, similar_sentences)
                threshold = 0.3

                # Check if reranked is not empty and if the score is low
                if reranked and reranked[-1][1] < threshold:
                    context = "Not enough context available."
                else:
                    context = reranked[-1][0] if reranked else "No context available."
            
            messages_to_send = st.session_state.messages + [
                SystemMessage(content=f"Context: {context}\n\nYou MUST respond with a concise answer limited to one paragraph.")
            ]
            response = chat.invoke(messages_to_send)
            ai_message = AIMessage(content=response.content)
            st.session_state.messages.append(ai_message)
            # Commented out to remove rating buttons
            # st.session_state.last_ai_response = ai_message.content  # Save latest AI response for rating
        st.rerun()

# 11. Rating Buttons section - commented out
# if st.session_state.last_ai_response:
#     rating_placeholder = st.empty()
#     with rating_placeholder:
#         st.markdown("### Rate the AI's Latest Response:")
#         col1, col2 = st.columns(2)
#         with col1:
#             if st.button("✅ Correct", key="correct"):
#                 st.session_state.conf_matrix[0, 0] += 1
#                 st.session_state.last_ai_response = None  # Hide rating buttons after rating
#                 st.success("Thank you for your feedback!")
#                 st.rerun()
#         with col2:
#             if st.button("❌ Incorrect", key="incorrect"):
#                 st.session_state.conf_matrix[0, 1] += 1
#                 st.session_state.last_ai_response = None  # Hide rating buttons after rating
#                 st.warning("Thank you for your feedback! We will improve.")
#                 st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

# 12. Sidebar: Confusion Matrix section - commented out
# st.sidebar.write("### Confusion Matrix")
# cm_df = pd.DataFrame(
#     st.session_state.conf_matrix,
#     index=["True Answerable", "True Unanswerable"],
#     columns=["Predicted Answerable", "Predicted Unanswerable"]
# )
# st.sidebar.table(cm_df)

# TP = st.session_state.conf_matrix[0, 0]
# FN = st.session_state.conf_matrix[0, 1]
# FP = st.session_state.conf_matrix[1, 0]
# TN = st.session_state.conf_matrix[1, 1]
# sensitivity = TP / (TP + FN) if (TP + FN) else 0
# specificity = TN / (TN + FP) if (TN + FP) else 0
# accuracy = (TP + TN) / np.sum(st.session_state.conf_matrix) if np.sum(st.session_state.conf_matrix) else 0
# precision = TP / (TP + FP) if (TP + FP) else 0
# f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) else 0

# st.sidebar.write("### Metrics")
# st.sidebar.write(f"Sensitivity (True Answerable Detection): {sensitivity:.2f}")
# st.sidebar.write(f"Specificity (True Unanswerable Detection): {specificity:.2f}")
# st.sidebar.write(f"Accuracy: {accuracy:.2f}")
# st.sidebar.write(f"Precision: {precision:.2f}")
# st.sidebar.write(f"F1 Score: {f1_score:.2f}")

# if st.sidebar.button("Reset Confusion Matrix", key="reset_matrix"):
#     st.session_state.conf_matrix = np.zeros((2, 2), dtype=int)
#     st.session_state.conversation_history = []
#     st.rerun()
