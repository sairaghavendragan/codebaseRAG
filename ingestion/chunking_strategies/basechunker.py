 

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Unified Semantic Chunk Schema ---
# This is the expected structure for each chunk dictionary output by any chunker.
# All keys in 'meta' should be present, even if their values are None or empty for certain chunk types.
CHUNK_SCHEMA = {
    'content': '...',                     # The actual text content of the chunk
    'meta': {
        'repo_name': '...',               # Name of the repository
        'file_path': '...',               # Full path to the original file
        'start_line': ...,                # Starting line number of the chunk in the original file (1-indexed)
        'end_line': ...,                  # Ending line number of the chunk in the original file (1-indexed)
        'chunk_type': '...',              # e.g., 'text_block', 'function', 'class', 'method', 'heading', 'top_level_code'
        'name': '...',                    # (Optional) Name of the entity (e.g., function name, class name, heading title)
        'parent_name': '...',             # (Optional) Name of the parent entity (e.g., class name for a method)
        'section': '...',                 # (Optional) For Markdown, the heading path (e.g., "Installation/Dependencies")
        'language': '...',                # (Optional) Programming language (e.g., 'python', 'javascript', 'markdown', 'text')
        # Add other relevant metadata fields here if universally applicable:
        # 'start_char_idx': ...,          # Optional: Starting character index
        # 'end_char_idx': ...,            # Optional: Ending character index
    }
}

class BaseChunker(ABC):
    """
    Abstract base class for all semantic chunking strategies.
    Defines the common interface for processing a raw file document into chunks.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Initializes the base chunker with common parameters.
        Individual chunkers can override or extend these.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logging.debug(f"Initialized BaseChunker with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    @abstractmethod
    def chunk(self, raw_file_document: Dict[str, Any]) -> List[Dict]:
        """
        Abstract method to be implemented by concrete chunker classes.
        Takes a raw file document and returns a list of semantic chunks.

        Args:
            raw_file_document (Dict[str, Any]): A dictionary representing a raw file,
                                                 typically from Phase 1, with 'content'
                                                 and 'meta' keys.

        Returns:
            List[Dict]: A list of dictionaries, where each dictionary adheres to the
                        defined CHUNK_SCHEMA, representing a semantically meaningful chunk.
        """
        pass

    def _get_line_numbers(self, content_lines: List[str], start_char: int, end_char: int) -> tuple[int, int]:
        """
        Helper to calculate 1-indexed start and end line numbers from character indices.
        Assumes content_lines is a list of lines from the original content.
        """
        current_char_count = 0
        start_line = 1
        end_line = 1

        for i, line in enumerate(content_lines):
            line_length_with_newline = len(line) + 1 # +1 for the newline character

            # Check if start_char falls within this line
            if start_char >= current_char_count and start_char < current_char_count + line_length_with_newline:
                start_line = i + 1

            # Check if end_char falls within this line
            # We check (end_char - 1) because end_char is typically exclusive
            if (end_char - 1) >= current_char_count and (end_char - 1) < current_char_count + line_length_with_newline:
                end_line = i + 1

            current_char_count += line_length_with_newline
            if end_line != 1 and start_line != 1: # Optimization: break early if both found
                break
        
        # Fallback if content is very short or no newlines
        if start_line == 1 and end_line == 1 and len(content_lines) == 1:
            return 1, 1

        return start_line, end_line

    def _create_base_chunk_meta(self, raw_doc_meta: Dict[str, Any], chunk_type: str, language: str,
                                start_line: int, end_line: int,
                                name: Optional[str] = None, parent_name: Optional[str] = None,
                                section: Optional[str] = None) -> Dict[str, Any]:
        """Helper to create the common metadata dictionary for a chunk."""
        return {
            'repo_name': raw_doc_meta['repo_name'],
            'file_path': raw_doc_meta['file_path'],
            'start_line': start_line,
            'end_line': end_line,
            'chunk_type': chunk_type,
            'language': language,
            'name': name,
            'parent_name': parent_name,
            'section': section,
        }