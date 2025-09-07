# ingestion/repo_processor.py

from typing import List, Dict, Any, Optional
import logging
from collections import defaultdict

from ingestion.repo_downloader import download_and_extract_raw_files
from .chunking_strategies.basechunker import BaseChunker
from .chunking_strategies.generic_chunker import GenericChunker
from .chunking_strategies.python_chunker import PythonChunker # Keeping this imported for potential explicit use, though Tree-sitter will be default for .py
from .chunking_strategies.markdown_chunker import MarkdownChunker
from .chunking_strategies.tree_sitter_code_chunker import TreeSitterCodeChunker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
DEFAULT_CHUNK_SIZE = 1500
DEFAULT_CHUNK_OVERLAP = 200

# Mapping of file extensions to Tree-sitter language names
# This is crucial for dynamic Tree-sitter chunker selection.
TREE_SITTER_LANGUAGE_MAP: Dict[str, str] = {
     
    'js': 'javascript',
    'ts': 'typescript',
    'jsx': 'javascript', # Treat JSX as JavaScript
    'tsx': 'typescript', # Treat TSX as TypeScript
    'go': 'go',
    'rb': 'ruby',
    'rs': 'rust',
    'java': 'java',
    'c': 'c',
    'cpp': 'cpp',
    'h': 'cpp', # C/C++ headers
    'hpp': 'cpp',
    'cs': 'c_sharp',
    'php': 'php',
    'dart': 'dart',
    'html': 'html', # Tree-sitter can also parse HTML
    'css': 'css',   # Tree-sitter can also parse CSS
    # Add other language extensions as you add support and grammars
}

# --- Chunker Instances and Cache ---

# Cache for Tree-sitter chunkers, dynamically created per language
tree_sitter_chunker_cache: Dict[str, TreeSitterCodeChunker] = {}

def get_tree_sitter_chunker(lang: str) -> Optional[TreeSitterCodeChunker]:
    """
    Retrieves or creates a TreeSitterCodeChunker instance for a given language.
    Returns None if the language parser cannot be loaded.
    """
    if lang not in tree_sitter_chunker_cache:
        try:
            # Instantiate with default parameters for now
            chunker = TreeSitterCodeChunker(
                language=lang,
                chunk_size=DEFAULT_CHUNK_SIZE,
                chunk_overlap=DEFAULT_CHUNK_OVERLAP
            )
            tree_sitter_chunker_cache[lang] = chunker
        except ImportError as e:
            logging.warning(f"Could not load Tree-sitter parser for language '{lang}': {e}. This language will be chunked generically.")
            return None # Indicate failure to load specific chunker
        except Exception as e:
            logging.error(f"Unexpected error initializing TreeSitterCodeChunker for '{lang}': {e}. Falling back to generic.")
            return None
    return tree_sitter_chunker_cache.get(lang) # Use .get() to safely return None if init failed

# Initialize common chunker instances
generic_chunker_instance = GenericChunker(
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP
)
# PythonChunker instance is kept, but Tree-sitter will be default for '.py'
python_chunker_instance = PythonChunker(
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP
)
markdown_chunker_instance = MarkdownChunker(
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP
)


# Dispatcher map for explicitly defined file types to chunker instances.
# Programming languages are handled dynamically by Tree-sitter.
CHUNKER_MAP: Dict[str, BaseChunker] = {
    'txt': generic_chunker_instance, # Plain text files
    'json': generic_chunker_instance, # JSON files
    'xml': generic_chunker_instance, # XML files
    'yaml': generic_chunker_instance, # YAML files
    'yml': generic_chunker_instance,
    'md': markdown_chunker_instance,
    'py': python_chunker_instance,
    # or fall back to generic.
}


def process_repository_for_rag(repo_url: str, repo_name: str) -> List[Dict]:
    """
    Orchestrates the full ingestion process:
    1. Downloads and extracts raw files from a Git repository.
    2. Dispatches raw files to appropriate semantic chunking strategies.

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

        chunker_instance: BaseChunker = generic_chunker_instance # Default fallback

        # 1. Check for explicitly defined chunkers (e.g., Markdown, plain text)
        if file_type in CHUNKER_MAP:
            chunker_instance = CHUNKER_MAP[file_type]
            logging.debug(f"Dispatching {file_path} (type: {file_type}) to explicit {type(chunker_instance).__name__}.")
        # 2. If not explicitly mapped, try to use Tree-sitter for known code languages
        elif file_type in TREE_SITTER_LANGUAGE_MAP:
            ts_language_name = TREE_SITTER_LANGUAGE_MAP[file_type]
            ts_chunker = get_tree_sitter_chunker(ts_language_name)
            if ts_chunker:
                chunker_instance = ts_chunker
                logging.debug(f"Dispatching {file_path} (type: {file_type}) to TreeSitterCodeChunker for '{ts_language_name}'.")
            else:
                # get_tree_sitter_chunker returned None, meaning it failed to load
                logging.debug(f"TreeSitterCodeChunker failed for '{ts_language_name}'. Falling back to generic for {file_path}.")
        # 3. If no specific chunker found and not a known Tree-sitter language, use generic (already set as default fallback)
        else:
            logging.debug(f"No specific chunker found for {file_path} (type: {file_type}). Falling back to {type(chunker_instance).__name__}.")

        chunks_for_file = chunker_instance.chunk(doc) # Call the 'chunk' method of the instance

        if chunks_for_file:
            all_semantic_chunks.extend(chunks_for_file)
        else:
            logging.debug(f"No chunks generated for {file_path}")

    logging.info(f"Finished semantic chunking for {repo_name}. Total chunks: {len(all_semantic_chunks)}")
    return all_semantic_chunks