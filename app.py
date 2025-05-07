import hashlib
import json
import subprocess
import tempfile
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
import subprocess


# Start Apache in background
def start_apache():
    subprocess.run(["apache2ctl", "start"])

# Start these services in threads
threading.Thread(target=start_apache).start()

# Then run Streamlit app
os.system("streamlit run app.py --server.port=2503 --server.baseUrlPath=/team3s25")

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
# Removed the IP verification success message as requested

# Create page title and welcome message - keeping original UI intact
st.markdown("<h2 class='title'>TEAM3 Chatbot - AI Research Helper</h2>", unsafe_allow_html=True)
st.markdown(
    "<p class='subtitle'>Welcome! I'm here to assist you with your research. Ask me research-related questions, and I'll provide answers based on datasets, models and research papers from "
    "<a href='https://paperswithcode.com/sota' target='_blank'>paperswithcode.com/sota</a>"
    ".</p>",
    unsafe_allow_html=True
)

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
    If the context does not contain enough information to answer the question, respond with "I don't have enough information to answer this question." Do not generate, assume, or fabricate any details beyond the given context.
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

def sanitize(text):
    """Preserve data integrity during line-based storage"""
    return text.replace('\n', ' [NL] ').replace('\r', ' [CR] ')

def desanitize(text):
    return text.replace(' [NL] ', '\n').replace(' [CR] ', '\r')

# ------------------- Add Version File Handling -------------------
VERSION_FILE = "/data/version.txt"

def update_version_file():
    """Update the version file with current timestamp to invalidate caches."""
    with open(VERSION_FILE, "w") as f:
        f.write(str(time.time()))

# Initialize version file if it doesn't exist
if not os.path.exists(VERSION_FILE):
    update_version_file()

def check_and_reload_data():
    """Check if data version has changed and reload if needed"""
    current_version = open(VERSION_FILE).read().strip()
    
    if current_version != st.session_state.current_data_version:
        # Clear existing cache entries
        load_faiss_wrapper.clear()
        load_chunks_wrapper.clear()
        load_sources_wrapper.clear()
        
        # Reload data
        st.session_state.faiss_index = load_faiss_wrapper(current_version)
        st.session_state.chunks = load_chunks_wrapper(current_version)
        st.session_state.chunks_sources = load_sources_wrapper(current_version)
        st.session_state.current_data_version = current_version

@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def load_chunks(chunks_file_path):
    with open(chunks_file_path, 'r', encoding='latin-1') as f:
        return [desanitize(line.strip()) for line in f]

def load_faiss_index(index_file_path, updated_at=None):
    return faiss.read_index(index_file_path)

# Load saved data from Docker Volume
@st.cache_resource(show_spinner=False)
def load_faiss_wrapper(_version: str):
    """Wrapper with version-based cache invalidation"""
    return faiss.read_index(saved_faiss_index_file_path)

@st.cache_data(show_spinner=False)
def load_chunks_wrapper(_version: str):
    """Wrapper with version-based cache invalidation"""
    with open(saved_chunks_file_path, 'r', encoding='latin-1') as f:
        return [desanitize(line.strip()) for line in f]

@st.cache_data(show_spinner=False)
def load_sources_wrapper(_version: str):
    """Wrapper with version-based cache invalidation"""
    with open(saved_chunk_sources_file_path, 'r', encoding='latin-1') as f:
        return [desanitize(line.strip()) for line in f]



faiss_index_file_path = "/app/data/faiss_pq.index"
chunks_file_path = "/app/data/saved_chunks.txt"
chunk_sources_file_path = "/app/data/saved_chunks_sources.txt"
model = None
index = None
chunks = []
chunks_sources = []
new_chunks = []
# Docker Volume Paths
saved_faiss_index_file_path = "/data/faiss_index_newest.index"
saved_chunks_file_path = "/data/chunks_newest.txt"
saved_chunk_sources_file_path = "/data/chunks_sources_newest.txt"

# Then assign them to variables
model = load_model()

# Initialize version file if it doesn't exist
if not os.path.exists(VERSION_FILE):
    update_version_file()
    # Load current version
    current_version = open(VERSION_FILE).read().strip()

if os.path.exists(saved_faiss_index_file_path) and os.path.exists(saved_chunks_file_path):
    # Check if we need to load initial data
    if "data_initialized" not in st.session_state:
        # Load current version
        current_version = open(VERSION_FILE).read().strip()


        # Load all components using versioned wrappers
        st.session_state.faiss_index = load_faiss_wrapper(current_version)
        st.session_state.chunks = load_chunks_wrapper(current_version)
        st.session_state.chunks_sources = load_sources_wrapper(current_version)
        st.session_state.current_data_version = current_version
        st.session_state.data_initialized = True
else:
    st.session_state.faiss_index = load_faiss_index(faiss_index_file_path)
    st.session_state.chunks = load_chunks(chunks_file_path)
    st.session_state.chunks_sources = load_chunks(chunk_sources_file_path)


def create_chunks(output_file_path="/data/papers_output.json"):
    global new_chunks, PAPER_HASHES
    create_chunks = []
    new_chunks_sources = []
    create_chunks_sources = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)

    df = pd.read_json(output_file_path)

    for _, row in df.iterrows():
        uid = row.get('url')
        full_text = json.dumps(row.to_dict(), sort_keys=True, default=str)
        current_hash = compute_md5(full_text)

        if PAPER_HASHES.get(uid) == current_hash:
            continue  # Skip unchanged papers

        PAPER_HASHES[uid] = current_hash  # Mark this paper as updated

        datasets = row.get('dataset_names', [])
        models = row.get('best_model_names', [])
        dataset_links = row.get('dataset_links', [])

        # Dataset–Model Pairs (tagged with SOURCE)
        for i in range(min(len(datasets), len(models))):
            dataset_text = f"Dataset: {datasets[i]}, Best Model: {models[i]}"
            if i < len(dataset_links):
                dataset_text += f", Dataset Link: {dataset_links[i]}"
            new_chunks.append(dataset_text)
            new_chunks_sources.append(uid)

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
        main_text_split = text_splitter.split_text(main_text)
        new_chunks.extend(main_text_split)
        new_chunks_sources.extend([uid] * len(main_text_split))

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
            paper_text_split = text_splitter.split_text(paper_text)
            new_chunks.extend(paper_text_split)
            new_chunks_sources.extend([uid] * len(paper_text_split))

    # Load existing chunks
    existing_chunks = []
    if os.path.exists(saved_chunks_file_path):
        existing_chunks = load_chunks(saved_chunks_file_path)

    # Load existing chunks
    existing_chunks_sources = []
    if os.path.exists(saved_chunk_sources_file_path):
        existing_chunks_sources = load_chunks(saved_chunk_sources_file_path)

    if new_chunks != []:
        all_chunks = existing_chunks + new_chunks
        create_chunks = all_chunks

        all_chunks_source = new_chunks_sources + existing_chunks_sources
        create_chunks_sources = all_chunks_source
    else:
        print("No new chunks to add.")

    if new_chunks != []:
        # Update paper hashes
        with open(PAPER_HASHES_PATH, "w") as f:
            json.dump(PAPER_HASHES, f, indent=2)
    return (create_chunks, create_chunks_sources)

def create_compressed_vector_database(temp_chunks):
    global new_chunks, saved_faiss_index_file_path, saved_chunks_file_path, saved_chunk_sources_file_path
    
    if not new_chunks:
        return

    model = load_model()
    embeddings = model.encode(temp_chunks[0]).astype('float32')
    d = embeddings.shape[1]

    # Use the Docker volume path explicitly
    temp_dir = "/data/temp"  # Dedicated temp subdirectory in volume
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Create temp files directly in the Docker volume
        with tempfile.NamedTemporaryFile(dir=temp_dir, delete=False, suffix=".index") as tmp_index, \
             tempfile.NamedTemporaryFile(dir=temp_dir, delete=False, suffix=".chunks") as tmp_chunks, \
             tempfile.NamedTemporaryFile(dir=temp_dir, delete=False, suffix=".sources") as tmp_sources:

            # 1. Create compressed index
            pq_index = faiss.IndexPQ(d, 32, 8)  # 32 subspaces, 8 bits
            pq_index.train(embeddings)
            pq_index.add(embeddings)
            faiss.write_index(pq_index, tmp_index.name)

            # 2. Write chunks and sources
            for chunk in temp_chunks[0]:
                tmp_chunks.write(f"{sanitize(chunk)}\n".encode('utf-8'))
            for source in temp_chunks[1]:
                tmp_sources.write(f"{sanitize(source)}\n".encode('utf-8'))

            # Flush all writes before moving files
            tmp_index.flush()
            tmp_chunks.flush()
            tmp_sources.flush()

            # Atomic file moves (same filesystem required)
            os.replace(tmp_index.name, saved_faiss_index_file_path)
            os.replace(tmp_chunks.name, saved_chunks_file_path)
            os.replace(tmp_sources.name, saved_chunk_sources_file_path)

            # Update version for cache invalidation
            update_version_file()

    finally:
        # Cleanup: Remove any remaining temp files
        for f in [tmp_index, tmp_chunks, tmp_sources]:
            try:
                if f and os.path.exists(f.name):
                    os.remove(f.name)
            except Exception as e:
                print(f"Error cleaning up temp file {f.name}: {str(e)}")


def create_vector_database(temp_chunks):
    global new_chunks, saved_faiss_index_file_path, saved_chunks_file_path, saved_chunk_sources_file_path
    
    if not new_chunks:
        return

    model = load_model()
    embeddings = model.encode(temp_chunks[0]).astype('float32')

    # Use Docker volume path for temp files
    temp_dir = "/data/temp"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Create temp files directly in Docker volume
        with tempfile.NamedTemporaryFile(dir=temp_dir, delete=False, suffix=".index") as tmp_index, \
             tempfile.NamedTemporaryFile(dir=temp_dir, delete=False, suffix=".chunks") as tmp_chunks, \
             tempfile.NamedTemporaryFile(dir=temp_dir, delete=False, suffix=".sources") as tmp_sources:

            # 1. Create flat index
            flat_index = faiss.IndexFlatL2(embeddings.shape[1])
            flat_index.add(embeddings)
            faiss.write_index(flat_index, tmp_index.name)

            # 2. Write chunks and sources
            for chunk in temp_chunks[0]:
                tmp_chunks.write(f"{sanitize(chunk)}\n".encode('utf-8'))
            for source in temp_chunks[1]:
                tmp_sources.write(f"{sanitize(source)}\n".encode('utf-8'))

            # Flush writes before moving files
            tmp_index.flush()
            tmp_chunks.flush()
            tmp_sources.flush()

            # Atomic file replacement
            os.replace(tmp_index.name, saved_faiss_index_file_path)
            os.replace(tmp_chunks.name, saved_chunks_file_path)
            os.replace(tmp_sources.name, saved_chunk_sources_file_path)

            # Update version for cache invalidation
            update_version_file()

    finally:
        # Cleanup any remaining temp files
        for f in [tmp_index, tmp_chunks, tmp_sources]:
            try:
                if f and os.path.exists(f.name):
                    os.remove(f.name)
            except Exception as e:
                print(f"Error cleaning up temp file {f.name}: {str(e)}")

def run_scraper():
    """Run the Scrapy spider via subprocess."""
    result = subprocess.run(["python3", "go-paper-spider.py"], capture_output=True)


def full_refresh_pipeline():
    run_scraper()
    temp_chunks = create_chunks()
    create_compressed_vector_database(temp_chunks)

def schedule_next_run(interval_hours: int = 24):
    """Run `full_refresh_pipeline` every `interval_hours`"""
    def run_and_reschedule():
        full_refresh_pipeline()
        schedule_next_run(interval_hours)  # Reschedule after each run

    #Schedule the next run after the specified interval
    delay = interval_hours * 3600
    timer = threading.Timer(delay, run_and_reschedule)
    timer.daemon = True  # Allows the program (e.g., Streamlit) to exit cleanly
    timer.start()

# Start the first scheduled run in a background thread
# Run once at startup
if "scheduler_started" not in st.session_state:
    st.session_state.scheduler_started = True
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
    """Retrieve top-k similar sentences with version check."""
    if "data_initialized" in st.session_state: 
        # Check for data updates
        check_and_reload_data()
        
    # Get from session state
    index = st.session_state.faiss_index
    chunks = st.session_state.chunks
    chunks_sources = st.session_state.chunks_sources
    
    query_embedding = model.encode(query_sentence).astype('float32').reshape(1, -1)
    distances, indices = index.search(query_embedding, k)
    
    similar_sentences = [(chunks[indices[0][i]], chunks_sources[indices[0][i]]) 
                         for i in range(min(k, len(indices[0])))]
    
    return similar_sentences, distances[0].tolist()

def find_url_for_sentence(pairs, sentence):
    for s, url in pairs:
        if s.strip() == sentence.strip():
            return url
    return "<unknown>"

def get_url_from_similar(similar_sentences, target_sentence):
    """Get first unique URL by matching normalized sentence text."""
    import re
    seen_urls = set()
    target_clean = re.sub(r'^\d+\.\s*', '', target_sentence.strip())
    
    for sent, url in similar_sentences:
        sent_clean = re.sub(r'^\d+\.\s*', '', sent.strip())
        if sent_clean == target_clean and url not in seen_urls:
            seen_urls.add(url)
            return url  # Return first unique match
            
    return "<unknown>"

def rerank_sentences(query, sentence_url_pairs):
    """
    Re-rank a list of (sentence, url) pairs by LLM relevance to the query.

    Args:
        query (str): The user's research query
        sentence_url_pairs (List[Tuple[str, str]]): Chunk text and associated URL

    Returns:
        List[Tuple[Tuple[str, str], float]]: [((sentence, url), score), ...]
    """
    # Extract just the sentence texts
    sentences = [s for s, _ in sentence_url_pairs]

    rerank_prompt = (
        "You are an AI tasked with ranking sentences based on their relevance to a query. "
        "For each sentence, provide a relevance score between 0 and 1 (where 1 is highly relevant) "
        "and a brief explanation. Return the results in this format exactly, do not bold the words:\n"
        "Sentence: <sentence>\nScore: <score>\nExplanation: <explanation>\n\n"
        "Query: '{}'\n\n"
        "Sentences to rank:\n{}"
    ).format(query, "\n".join([f"{i+1}. {s}" for i, s in enumerate(sentences)]))

    max_retries = 3
    retry_delay = 60  # seconds

    for attempt in range(max_retries):
        try:
            response = chat.invoke([HumanMessage(content=rerank_prompt)])
            lines = response.content.strip().split("\n")
            reranked = []

            # Parse response: every 3 lines = Sentence, Score, Explanation
            for i in range(0, len(lines), 3):
                try:
                    sentence_line = lines[i].replace("Sentence: ", "").strip()
                    # Remove leading numbering (e.g., "1. ", "2. ")
                    import re
                    sentence = re.sub(r'^\d+\.\s*', '', sentence_line)
                    score = float(lines[i+1].replace("Score: ", "").strip())
                    url = find_url_for_sentence(sentence_url_pairs, sentence)
                    reranked.append(((sentence_line, url), score))
                except (IndexError, ValueError):
                    continue

            # Sort by score, descending
            reranked.sort(key=lambda x: x[1], reverse=True)
            return reranked

        except Exception as e:
            if "rate_limit" in str(e).lower() and attempt < max_retries - 1:
                st.warning(f"Rate limit hit. Retrying in {retry_delay} seconds... (Attempt {attempt+1})")
                time.sleep(retry_delay)
                continue
            raise e

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
            top_sentence = ""
            top_url = ""

            # Retrieve and re-rank
            similar_sentences, _ = retrieve_similar_sentences(user_input, k=10)

            if not similar_sentences:
                context = "No context available."
                source_url = "<unknown>"
            else:    
                reranked = rerank_sentences(user_input, similar_sentences)
                threshold = 0.3

                if reranked and reranked[0][1] < threshold:
                    context = "Not enough context available."
                    source_url = "<unknown>"
                elif reranked:
                        top_sentence = reranked[0][0][0]
                        source_url = get_url_from_similar(similar_sentences, top_sentence)
                        context = top_sentence
                else:
                    context = "No context available."
                    source_url = "<unknown>"

            messages_to_send = st.session_state.messages + [
                SystemMessage(content=f"Context: {context}\n\nYou MUST respond with a concise answer limited to one paragraph.")
            ]
            response = chat.invoke(messages_to_send)
            ai_message_content = response.content

            # Replace the existing citation code with:
            if source_url not in ["<unknown>", "", None]:
                # Check if response already contains a reference
                if "Reference: [Source Link]" not in ai_message_content:
                    ai_message_content += f"\n\nReference: [Source Link]({source_url})"
            else:
                # Remove any existing invalid references from model response
                ai_message_content = ai_message_content.split("Reference: Source Link")[0].strip()


            # Save AI message
            st.session_state.messages.append(AIMessage(content=ai_message_content))
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
