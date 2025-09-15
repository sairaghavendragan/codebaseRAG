import streamlit as st
import requests
import re
from urllib.parse import urlparse
import os
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
#API_BASE_URL = "http://127.0.0.1:8000" # Make sure this matches your FastAPI server address
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

# --- Helper Functions for API Calls ---

def get_repo_list():
    """Fetches the list of available repositories from the backend."""
    try:
        response = requests.get(f"{API_BASE_URL}/repos")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching repositories: {e}")
        return []

def start_new_chat():
    """Starts a new chat session and gets a conversation ID."""
    try:
        response = requests.post(f"{API_BASE_URL}/chat/new-session")
        response.raise_for_status()
        return response.json().get("conversation_id")
    except requests.exceptions.RequestException as e:
        st.error(f"Error starting new chat session: {e}")
        return None

def post_query(repo_name, query, conversation_id=None, use_two_pass=True):
    """Posts a query to the backend and gets a response."""
    payload = {
        "repo_name": repo_name,
        "query": query,
        "conversation_id": conversation_id,
        "use_two_pass_rag": use_two_pass,
    }
    try:
        with st.spinner("Thinking..."):
            response = requests.post(f"{API_BASE_URL}/query-codebase", json=payload)
            response.raise_for_status()
            return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            st.error(f"Error: {e.response.json().get('detail', 'Repository or relevant information not found.')}")
        else:
            st.error(f"An HTTP error occurred: {e.response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"A connection error occurred: {e}")
    return None

def ingest_repo(repo_url, repo_name):
    """Sends a request to ingest a new repository."""
    payload = {"repo_url": repo_url, "repo_name": repo_name}
    try:
        response = requests.post(f"{API_BASE_URL}/ingest-repo", json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error ingesting repository: {e}")
        return None

def derive_repo_name_from_url(url: str) -> str:
    """Derives a clean repository name from a GitHub URL."""
    try:
        path = urlparse(url).path
        # Remove leading slash and trailing '.git' if it exists
        repo_name = path.lstrip('/').replace('.git', '')
        # Replace slashes with a safe character
        return repo_name.replace('/', '--')
    except Exception:
        return "default-repo-name"


# --- Streamlit UI ---

st.set_page_config(page_title="Codebase RAG Assistant", layout="wide")

# --- Session State Initialization ---
# This is crucial to maintain state across user interactions
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "repo_name" not in st.session_state:
    st.session_state.repo_name = None
if "repo_list" not in st.session_state:
    st.session_state.repo_list = get_repo_list()


# --- Sidebar for Controls ---
with st.sidebar:
    st.title("Codebase RAG Assistant")

    # --- Ingestion Section ---
    st.header("1. Ingest Repository")
    with st.form("ingest_form"):
        repo_url = st.text_input("GitHub URL", placeholder="https://github.com/tiangolo/fastapi")
        submitted = st.form_submit_button("Ingest")
        if submitted and repo_url:
            repo_name = derive_repo_name_from_url(repo_url)
            with st.spinner(f"Starting ingestion for `{repo_name}`... This may take a few minutes."):
                result = ingest_repo(repo_url, repo_name)
                if result:
                    st.success(f"Ingestion started for `{result['repo_name']}`! Refresh the repo list in a bit.")
                    # Refresh repo list after a short delay
                    st.session_state.repo_list = get_repo_list()

    # --- Repository Selection ---
    st.header("2. Select Repository")
    
    if st.button("Refresh Repo List"):
        st.session_state.repo_list = get_repo_list()

    if st.session_state.repo_list:
        st.session_state.repo_name = st.selectbox(
            "Available Repositories",
            options=st.session_state.repo_list,
            index=st.session_state.repo_list.index(st.session_state.repo_name) if st.session_state.repo_name in st.session_state.repo_list else 0,
            key="repo_selector"
        )
    else:
        st.warning("No repositories ingested yet. Ingest one to begin.")

    # --- Chat Mode and Options ---
    st.header("3. Chat Mode")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("New Chat Session", use_container_width=True):
            st.session_state.conversation_id = start_new_chat()
            st.session_state.messages = [] # Clear message history for new chat
            st.success("Started a new chat session!")

    with col2:
        if st.button("One-Shot Query", use_container_width=True):
            st.session_state.conversation_id = None # Clear conversation ID for one-shot
            st.session_state.messages = [] # Clear message history
            st.info("Switched to one-shot query mode.")
    
    use_two_pass_rag = st.toggle("Use Two-Pass RAG", value=True, help="Breaks down your query into sub-questions for more accurate retrieval. Better for complex questions.")


# --- Main Chat Interface ---
st.header(f"Chat with: `{st.session_state.repo_name}`")

if st.session_state.conversation_id:
    st.info(f"Mode: Conversational (ID: `{st.session_state.conversation_id[:8]}...`)")
else:
    st.info("Mode: One-Shot Query (No history is kept)")


# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.code(f"File: {source['file_path']}, Lines: {source['start_line']}-{source['end_line']}")


# Chat input box
if prompt := st.chat_input("Ask a question about the codebase...", disabled=not st.session_state.repo_name):
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get assistant response
    response = post_query(
        st.session_state.repo_name,
        prompt,
        st.session_state.conversation_id,
        use_two_pass_rag
    )

    # Add assistant response to chat history and display it
    if response:
        answer = response.get("answer", "Sorry, I couldn't find an answer.")
        sources = response.get("sources", [])
        
        # In conversational mode, the backend returns the conversation_id, update our state
        if response.get("conversation_id"):
             st.session_state.conversation_id = response["conversation_id"]

        with st.chat_message("assistant"):
            st.markdown(answer)
            if sources:
                with st.expander("Sources"):
                     for source in sources:
                        st.code(f"File: {source['file_path']}, Lines: {source['start_line']}-{source['end_line']}")

        assistant_message = {"role": "assistant", "content": answer, "sources": sources}
        st.session_state.messages.append(assistant_message)

elif not st.session_state.repo_name:
    st.warning("Please select a repository from the sidebar to start chatting.")