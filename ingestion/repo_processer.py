# ingestion/repo_processor.py

from typing import List, Dict, Any
import logging

from ingestion.repo_downloader import download_and_extract_raw_files
from .chunking_strategies.basechunker import BaseChunker # Import the base class
from .chunking_strategies.generic_chunker import GenericChunker
from .chunking_strategies.python_chunker import PythonChunker
# from .chunking_strategies.python_chunker import PythonChunker # Will uncomment/add later
# from .chunking_strategies.javascript_chunker import JavascriptChunker # Will add later
from .chunking_strategies.markdown_chunker import MarkdownChunker # Will add later

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# You can configure default chunking parameters here, or make them configurable via FastAPI later.
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100

# --- Chunker Instances (or a factory to create them) ---
# For simplicity, we can instantiate them here. For more complex apps,
# you might use a dependency injection system or a factory pattern.
# We'll create a dictionary to map file types to chunker instances.

# Initialize GenericChunker
generic_chunker_instance = GenericChunker(
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP
)
python_chunker_instance = PythonChunker(
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP
)
markdown_chunker_instance = MarkdownChunker(
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP
)
# Placeholders for other chunkers (will be uncommented/initialized when ready)
# python_chunker_instance = PythonChunker(
#     chunk_size=DEFAULT_CHUNK_SIZE,
#     chunk_overlap=DEFAULT_CHUNK_OVERLAP
# )
# javascript_chunker_instance = JavascriptChunker(...)
# markdown_chunker_instance = MarkdownChunker(...)


# Dispatcher map for file types to chunker instances
# This map will evolve as we implement more chunkers
CHUNKER_MAP: Dict[str, BaseChunker] = {
    # Default to generic for all
    'default': generic_chunker_instance,
    'txt': generic_chunker_instance, # Plain text files
    'json': generic_chunker_instance, # JSON files
    'xml': generic_chunker_instance, # XML files
    'yaml': generic_chunker_instance, # YAML files
    'yml': generic_chunker_instance,
    'html': generic_chunker_instance, # HTML files
    'css': generic_chunker_instance, # CSS files
    'scss': generic_chunker_instance, # SASS files
    'py': python_chunker_instance, # Uncomment when PythonChunker is ready
    # 'js': javascript_chunker_instance, # Uncomment when JavascriptChunker is ready
    # 'ts': javascript_chunker_instance,
    # 'jsx': javascript_chunker_instance,
    # 'tsx': javascript_chunker_instance,
    'md': markdown_chunker_instance, # Uncomment when MarkdownChunker is ready
}


def process_repository_for_rag(repo_url: str, repo_name: str) -> List[Dict]:
    """
    Orchestrates the full ingestion process:
    1. Downloads and extracts raw files from a Git repository.
    2. Dispatches raw files to appropriate semantic chunking strategies (class-based).

    Args:
        repo_url (str): The URL of the Git repository.
        repo_name (str): The name to assign to the repository.

    Returns:
        List[Dict]: A list of all semantically chunked documents from the repository.
    """
    logging.info(f"Starting full RAG ingestion pipeline for {repo_name} from {repo_url}")
    all_semantic_chunks = []

    # Phase 1: Download and Extract Raw Files
    raw_file_docs = download_and_extract_raw_files(repo_url, repo_name)
    if not raw_file_docs:
        logging.warning(f"No raw files extracted for {repo_name}. Aborting chunking.")
        return []

    logging.info(f"Extracted {len(raw_file_docs)} raw files. Starting semantic chunking.")

    # Phase 2: Semantic Chunking
    for doc in raw_file_docs:
        file_content = doc['content']
        file_path = doc['meta']['file_path']
        file_type = doc['meta'].get('file_type', 'unknown') # Use .get() with a default

        # Determine which chunker to use
        # Use .get() with a fallback to 'default' key for robust lookup
        chunker_instance = CHUNKER_MAP.get(file_type, CHUNKER_MAP['default'])

        logging.debug(f"Dispatching {file_path} (type: {file_type}) to {type(chunker_instance).__name__}.")

        chunks_for_file = chunker_instance.chunk(doc) # Call the 'chunk' method of the instance

        if chunks_for_file:
            all_semantic_chunks.extend(chunks_for_file)
        else:
            logging.debug(f"No chunks generated for {file_path}")

    logging.info(f"Finished semantic chunking for {repo_name}. Total chunks: {len(all_semantic_chunks)}")
    return all_semantic_chunks