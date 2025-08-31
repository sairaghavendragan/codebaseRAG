# ingestion/chunking_strategies/generic_chunker.py

from typing import List, Dict, Any
import logging

from .basechunker  import BaseChunker # Import the BaseChunker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GenericChunker(BaseChunker):
    """
    A generic text chunker that splits content into fixed-size blocks
    with optional overlap. Serves as a fallback for unsupported file types
    or for top-level content not handled by semantic chunkers.
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        """
        Initializes the GenericChunker.

        Args:
            chunk_size (int): The maximum number of characters per chunk.
            chunk_overlap (int): The number of characters to overlap between chunks.
        """
        super().__init__(chunk_size, chunk_overlap)
        logging.info(f"Initialized GenericChunker with chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}")

    def chunk(self, raw_file_document: Dict[str, Any]) -> List[Dict]:
        """
        Splits content into generic text blocks based on character count with overlap.

        Args:
            raw_file_document (Dict[str, Any]): A dictionary containing 'content' and 'meta'.
                                                 Example: {'content': '...', 'meta': {'repo_name': '...', 'file_path': '...', 'file_type': '...'}}

        Returns:
            List[Dict]: A list of generic text chunks with metadata adhering to CHUNK_SCHEMA.
        """
        content = raw_file_document['content']
        meta = raw_file_document['meta']
        file_path = meta['file_path']
        repo_name = meta['repo_name']
        file_type = meta.get('file_type', 'unknown') # Use .get() in case file_type is missing

        if not content.strip():
            logging.debug(f"Skipping empty content for generic chunking: {file_path}")
            return []

        chunks = []
        content_lines = content.splitlines() # For line number calculation
        current_char_offset = 0

        while current_char_offset < len(content):
            end_char_offset = min(current_char_offset + self.chunk_size, len(content))
            chunk_content = content[current_char_offset:end_char_offset].strip()

            if not chunk_content: # Skip empty chunks if they result from stripping
                current_char_offset += (self.chunk_size - self.chunk_overlap)
                continue

            # Calculate line numbers using the helper from BaseChunker
            start_line, end_line = self._get_line_numbers(content_lines, current_char_offset, end_char_offset)

            # Create chunk metadata using the helper from BaseChunker
            chunk_meta = self._create_base_chunk_meta(
                raw_doc_meta=meta,
                chunk_type='text_block', # Specific chunk_type for generic chunks
                language=file_type,     # Use the file_type from the raw document as the language for generic chunks
                start_line=start_line,
                end_line=end_line,
                name=None,              # No specific name for generic text blocks
                parent_name=None,
                section=None
            )

            chunks.append({
                'content': chunk_content,
                'meta': chunk_meta
            })

            # Move to the next chunk start point, considering overlap
            current_char_offset += (self.chunk_size - self.chunk_overlap)

            # Prevent infinite loop if chunk_size <= chunk_overlap
            if self.chunk_size <= self.chunk_overlap and current_char_offset < len(content):
                current_char_offset += 1

        logging.debug(f"Chunked {file_path} into {len(chunks)} generic text blocks.")
        return chunks