# ingestion/chunking_strategies/markdown_chunker.py

import re
from typing import List, Dict, Any, Optional
import logging

from .basechunker import BaseChunker
from .generic_chunker import GenericChunker # For unhandled text blocks if needed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MarkdownChunker(BaseChunker):
    """
    A semantic chunker for Markdown files.
    Splits content by Markdown headings (H1-H6) and includes the heading path in metadata.
    Handles code blocks to prevent misinterpreting headers within them.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        super().__init__(chunk_size, chunk_overlap)
        # MarkdownChunker might use GenericChunker for content between headings or very large non-heading sections
        self.generic_chunker = GenericChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        logging.info(f"Initialized MarkdownChunker with chunk_size={chunk_size}, overlap={chunk_overlap}.")

    def chunk(self, raw_file_document: Dict[str, Any]) -> List[Dict]:
        content = raw_file_document['content']
        meta = raw_file_document['meta']
        file_path = meta['file_path']
        repo_name = meta['repo_name']
        
        if not content.strip():
            logging.debug(f"Skipping empty content for Markdown chunking: {file_path}")
            return []

        all_chunks = []
        lines = content.splitlines()
        
        # Keep track of (level, text) for headers
        header_stack: List[tuple[int, str]] = []
        current_section_lines: List[str] = []
        current_section_start_line = 1 # 1-indexed line number where current section started
        
        in_code_block = False

        for i, line in enumerate(lines):
            line_number = i + 1 # Convert to 1-indexed

            # Toggle code block flag if we encounter triple backticks
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                current_section_lines.append(line)
                continue

            # If inside a code block, treat as regular text within the current section
            if in_code_block:
                current_section_lines.append(line)
                continue

            # Check for Markdown headers (outside code blocks)
            header_match = re.match(r"^(#+)\s(.*)", line)
            if header_match:
                # If there's content in the current section, chunk it before starting a new one
                if current_section_lines:
                    self._add_markdown_section_chunk(
                        all_chunks,
                        "\n".join(current_section_lines).strip(),
                        meta,
                        current_section_start_line,
                        line_number - 1, # End line of previous section
                        header_stack
                    )
                
                header_level = len(header_match.group(1))
                header_text = header_match.group(2).strip()

                # Adjust header stack based on the new header's level
                while header_stack and header_stack[-1][0] >= header_level:
                    header_stack.pop()
                header_stack.append((header_level, header_text))

                # Start a new section with the header line itself
                current_section_lines = [line]
                current_section_start_line = line_number
            else:
                # Add regular content to the current section
                current_section_lines.append(line)
        
        # Add the very last section if any content remains
        if current_section_lines:
             self._add_markdown_section_chunk(
                all_chunks,
                "\n".join(current_section_lines).strip(),
                meta,
                current_section_start_line,
                len(lines), # End line is the last line of the file
                header_stack
            )

        # Sort all chunks by their starting line number for a coherent order
        all_chunks.sort(key=lambda x: x['meta']['start_line'])

        logging.info(f"Chunked {file_path} into {len(all_chunks)} Markdown semantic chunks.")
        return all_chunks

    def _add_markdown_section_chunk(self,
        all_chunks: List[Dict],
        chunk_content: str,
        raw_doc_meta: Dict[str, Any],
        start_line: int,
        end_line: int,
        header_stack: List[tuple[int, str]]
    ):
        """Helper to create and add a Markdown section chunk."""
        if not chunk_content: # Don't add empty chunks
            return

        # Derive section path (e.g., "Level1/Level2/CurrentHeading")
        section_path_parts = [h[1] for h in header_stack]
        section_path = "/".join(section_path_parts) if section_path_parts else None
        
        # The 'name' of the chunk could be the deepest heading, or None for top-level paragraphs
        name = section_path_parts[-1] if section_path_parts else None

        # Determine chunk type
        chunk_type = 'heading_section'
        # If the content starts with a heading, the chunk_type could be 'heading'
        # if re.match(r"^(#+)\s", chunk_content.splitlines()[0]):
        #    chunk_type = 'heading'
        # else:
        #    chunk_type = 'markdown_text_block' # Or just 'text_block'


        chunk_meta = self._create_base_chunk_meta(
            raw_doc_meta=raw_doc_meta,
            chunk_type=chunk_type,
            language='markdown',
            start_line=start_line,
            end_line=end_line,
            name=name,
            parent_name=section_path_parts[-2] if len(section_path_parts) > 1 else None,
            section=section_path
        )
        all_chunks.append({'content': chunk_content, 'meta': chunk_meta})