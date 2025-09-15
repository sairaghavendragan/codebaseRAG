 
# Codebase RAG Assistant  

An AI-powered assistant that allows you to "chat" with any public Git repository. Ingest a codebase, and then ask complex questions about its architecture, functionality, and implementation details.

This project leverages a sophisticated **Retrieval-Augmented Generation (RAG)** pipeline, using semantic code chunking with Tree-sitter, a ChromaDB vector store, and Google's Gemini LLM to provide accurate, source-cited answers.

 

## ✨ Features

-   **Ingest Any Public GitHub Repo**: Simply provide a URL to download, process, and index any public repository.
-   **Semantic Code Chunking**: Goes beyond simple text splitting. Uses **Tree-sitter** to parse code into meaningful chunks like functions, classes, and methods for highly relevant context retrieval.
-   **Multi-Language Support**: Thanks to Tree-sitter, it can intelligently parse Python, JavaScript, TypeScript, Go, Java, Rust, and more.
-   **Advanced Two-Pass RAG**: For complex queries, the system can first generate targeted sub-questions, retrieve context for each, and then synthesize a comprehensive final answer.
-   **Conversational Memory**: Maintains chat history within a session to understand follow-up questions and context.
-   **Source-Cited Answers**: Every response includes expandable references to the exact file paths and line numbers from which the information was sourced.
-   **Interactive Web UI**: A clean and simple interface built with Streamlit for easy interaction.
-   **Scalable Backend**: Built with FastAPI, featuring asynchronous background tasks for repository ingestion.

 

## 🛠️ Tech Stack

-   **Backend**: FastAPI, Uvicorn
-   **Frontend**: Streamlit
-   **LLM**: Google Gemini
-   **Vector Store**: ChromaDB
-   **Code Parsing**: Tree-sitter
-   **Data Validation**: Pydantic
-   **Configuration**: `python-dotenv`, `pydantic-settings`

## 🏁 Getting Started

Follow these steps to set up and run the project locally.

### 1. Prerequisites

-   Python 3.9+
-   Git
 

### 2. Clone the Repository

```bash
git clone https://github.com/sairaghavendragan/codebaseRAG
 
```

### 3. Set Up a Virtual Environment

It's highly recommended to use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 4. Install Dependencies

Install all required Python packages.Use UV if possible for better management. 

```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables

The application requires a Google Gemini API key.

1.  Create a file named `.env` in the project root.  
2.  Edit the `.env` file and add your Google API key:
    ```dotenv
    # .env
    GOOGLE_API_KEY="your_google_api_key_here" 
    ```

### 6. Run the Application

You need to run two processes in separate terminals: the backend API and the frontend UI.

1.  **Start the FastAPI Backend**:
    ```bash
    python -m api.main
    ```
    The API will be available at `http://127.0.0.1:8000`.

2.  **Start the Streamlit Frontend**:
    ```bash
    streamlit run frontend.py
    ```
    The web interface will open in your browser, typically at `http://localhost:8501`.

## 📖 How to Use

1.  **Ingest a Repository**: In the sidebar, enter the URL of a public GitHub repository (e.g., `https://github.com/tiangolo/fastapi`) and click "Ingest". Ingestion runs in the background and may take a few minutes.
2.  **Select a Repository**: After ingestion, click "Refresh Repo List". Your new repository should appear in the dropdown. Select it.
3.  **Choose a Chat Mode**:
    -   **New Chat Session**: Starts a conversational chat where the AI remembers previous questions.
    -   **One-Shot Query**: Each question is treated independently, with no memory of the past.
4.  **Ask a Question**: Type your question about the selected codebase in the chat input at the bottom and press Enter.
5.  **Review the Answer**: The assistant will provide an answer along with an expandable "Sources" section showing the code snippets used to generate the response.
6.  **(Optional) Use Two-Pass RAG**: For complex questions, enable the "Use Two-Pass RAG" toggle in the sidebar. This can improve accuracy by breaking your query into smaller, more specific searches.

## ⚙️ How It Works in Depth

### Ingestion Pipeline

The core of the retrieval system is its ability to process source code intelligently.

1.  **Download**: The `repo_downloader` clones the repository and filters for relevant source code files, excluding binaries, images, and `node_modules`.
2.  **Dispatch**: The `repo_processer` acts as a dispatcher. It inspects each file's type and sends it to the appropriate chunking strategy.
3.  **Semantic Chunking**:
    -   **Tree-sitter**: For supported programming languages, `TreeSitterCodeChunker` parses the file into an Abstract Syntax Tree (AST). It then creates chunks based on semantic boundaries like functions, classes, methods, and structs. This is far more effective than naive text splitting.
    -   **Markdown/Generic**: Markdown files are chunked by headings, and other text files use a fallback recursive character splitter.
4.  **Vectorization & Storage**: The `ChromaManager` takes these semantic chunks, generates vector embeddings using a sentence-transformer model, and stores them in a persistent ChromaDB collection named after the repository.

### RAG Query Pipeline

When a user asks a question, the `RAGPipeline` orchestrates the response.

1.  **Standard RAG**:
    -   The user's query is converted into a vector embedding.
    -   ChromaDB is queried to find the `top_k` most semantically similar code chunks.
    -   These chunks are combined with the original query into a detailed prompt for the Gemini LLM.
    -   Gemini generates a response based *only* on the provided context, citing sources as it goes.

2.  **Two-Pass RAG (for enhanced accuracy)**:
    -   **Pass 1**: The initial query retrieves a broad set of potentially relevant code chunks.
    -   **Sub-question Generation**: The original query and the initial chunks are sent to Gemini with a special prompt, asking it to generate a list of specific, atomic sub-questions. For example, "How does auth work?" might become "What library is used for password hashing?" and "Where are the login routes defined?".
    -   **Pass 2**: Each sub-question is used to perform its own retrieval from ChromaDB, gathering highly focused context.
    -   **Synthesis**: All retrieved chunks (from the original query and all sub-questions) are deduplicated and compiled into a final, rich context. This context is then used to generate the final answer, providing a much more detailed and accurate result.

## 📁 Project Structure

```
└── sairaghavendragan-codebaserag/
    ├── api/                # FastAPI backend logic
    │   └── main.py         # API endpoints, application lifecycle
    ├── config/             # Configuration management
    │   └── settings.py     # Pydantic settings model (loads .env)
    ├── ingestion/          # Logic for downloading and processing repos
    │   ├── repo_downloader.py # Clones Git repos and extracts files
    │   ├── repo_processer.py  # Orchestrates the chunking strategies
    │   └── chunking_strategies/ # Different methods for semantic chunking
    │       ├── basechunker.py
    │       ├── tree_sitter_code_chunker.py # Advanced AST-based chunking
    │       ├── python_chunker.py           # AST-based chunking specific to Python
    │       └── ...
    ├── rag_core/           # Core RAG pipeline and business logic
    │   ├── rag_pipeline.py # Main RAG execution flow (incl. two-pass logic)
    │   ├── gemini_client.py# Client for interacting with Google Gemini
    │   ├── prompt_builder.py# Constructs prompts for the LLM
    │   └── chat_manager.py # Manages conversational history
    ├── vector_store/       # Manages the ChromaDB vector store
    │   └── chroma_manager.py# Handles creating, adding, and querying collections
    ├── frontend.py         # Streamlit frontend application
    └── requirements.txt    # Project dependencies
```
 