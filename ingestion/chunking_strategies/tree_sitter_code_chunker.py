# ingestion/chunking_strategies/treesitter_code_chunker.py

import os
from typing import List, Dict, Any, Optional, Tuple
import logging
from collections import deque

from .basechunker import BaseChunker
from .generic_chunker import GenericChunker # For handling parts of large semantic chunks or top-level code

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    from tree_sitter import Parser, Language
    import tree_sitter_language_pack
except ImportError:
    logging.error("Please install tree_sitter and tree_sitter_language_pack: pip install tree_sitter tree_sitter_language_pack")
    raise


class TreeSitterCodeChunker(BaseChunker):
    """
    A semantic chunker for various programming languages using Tree-sitter.
    It identifies major semantic units (functions, classes, methods) and creates chunks
    for them. If a semantic unit is too large, it is further broken down.
    Uncovered code (imports, global statements) is handled by GenericChunker.
    """

    # Mapping of common Tree-sitter node types to our internal chunk_type schema
    NODE_TYPE_TO_CHUNK_TYPE: Dict[str, str] = {
        'function_definition': 'function',          # Python, C, C++, Go, Ruby, PHP, Rust, Java, C#
        'method_definition': 'method',              # Python, Java, C#
        'class_definition': 'class',                # Python
        'object_declaration': 'object',
        'class_specifier': 'class',              
        'namespace_declaration': 'module',          # C++, C#
        'function_signature': 'function',           # Dart
        'method_signature': 'method',               #Dart
        'interface_declaration': 'interface',       # Java, C#
        'function_declaration': 'function',        #  JS/TS support
        'class_declaration': 'class',              #   JS/TS support
        'method_declaration': 'method',             # Go
        "class": "class",                          # Ruby
        "singleton_class": "class",                # Ruby
        "method": "method",                        # Ruby
        "singleton_method": "method",              # Ruby
        "alias": "method",                        # Ruby
        "module": "module",                        # Ruby
        "function_item": "function",               # Rust
    }
    NODE_TYPE_TO_NAME_FIELD: Dict[str, str] = {
        'function_definition': 'name',          # Python, C, C++, Go, Ruby, PHP, Rust, Java, C#
        'method_definition': 'name',            # Python, Java, C#
        'class_definition': 'name',             # Python
        'object_declaration': 'name',           # Often uses 'name'
        'class_specifier': 'name',              # C++
        'namespace_declaration': 'name',        # C++, C#
        'function_signature': 'name',           # Dart (assuming full definition name, see note below)
        'method_signature': 'name',             # Dart (assuming full definition name, see note below)
        'interface_declaration': 'name',        # Java, C#
        'function_declaration': 'name',         # JS/TS
        'class_declaration': 'name',            # JS/TS
        'method_declaration': 'name',           # Go
        "class": "name",                        # Ruby
        "singleton_class": "name",              # Ruby
        "method": "name",                       # Ruby
        "singleton_method": "name",             # Ruby
        "alias": "name",                        # Ruby
        "module": "name",                       # Ruby
        "function_item": "name",                # Rust
        
    }

    # Common Tree-sitter node types that represent identifiers/names
    IDENTIFIER_NODE_TYPES = {
        'identifier', 'name', 'shorthand_property_identifier_pattern'
    }

    def __init__(self, language: str, chunk_size: int = 1500, chunk_overlap: int = 250):
        super().__init__(chunk_size, chunk_overlap)
        self.language_str = language
        self.generic_chunker = GenericChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        try:
            self.parser = tree_sitter_language_pack.get_parser(language)
            self.ts_language = tree_sitter_language_pack.get_language(language)
            logging.info(f"Initialized TreeSitterCodeChunker for language '{language}' with chunk_size={chunk_size}, overlap={chunk_overlap}.")
        except Exception as e:
            logging.error(f"Failed to load tree-sitter parser for language '{language}': {e}")
            raise ImportError(
                f"Could not load tree-sitter parser for language '{language}'. "
                f"Please ensure it's available in tree_sitter_language_pack. Error: {e}"
            )

    def _get_node_content(self, node: Any,  lines: List[str]) -> Tuple[str, int, int]:
        """
        Extracts the full text content for a given AST node, including any preceding
        decorators or docstrings, returning 1-indexed start/end lines.
        """
        start_line_0_indexed = node.start_point[0]
        end_line_0_indexed = node.end_point[0]

        # Extend start_line to include preceding comments/decorators for definitions
        if node.type in self.NODE_TYPE_TO_CHUNK_TYPE: 
            current_line_idx = start_line_0_indexed - 1
            while current_line_idx >= 0:
                line_content = lines[current_line_idx].strip()
                # Common decorator/comment indicators for various languages
                if not line_content or line_content.startswith(('@', '#', '//', '/*')):
                    start_line_0_indexed = current_line_idx
                    current_line_idx -= 1
                else:
                    break
        
        # Adjust end line for multi-line comments or ensure entire body is captured
        # Tree-sitter node.end_point is exclusive of the character at end_point.column.
        # So end_line_0_indexed correctly points to the last line containing the node's code.

        # Reconstruct content from adjusted line numbers
        actual_content_lines = lines[start_line_0_indexed : end_line_0_indexed + 1]
        
        # If the node's start_point.column is > 0, we might need to indent if content is taken from full lines
        # This is a simplification; for perfect accuracy, we'd need to cut content_bytes.
        # But for RAG, line-based content is often sufficient.
        
        return "\n".join(actual_content_lines), start_line_0_indexed + 1, end_line_0_indexed + 1 # 1-indexed

    def _get_node_name(self, node: Any) -> Optional[str]:
        """
        Attempts to extract the name (identifier) from a definition node.
        Prioritizes field-based extraction, then language-specific patterns,
        then a general child-iteration fallback.
        """
        # 1. Try using a pre-defined field name for the node type from our map
        name_field = self.NODE_TYPE_TO_NAME_FIELD.get(node.type)
        if name_field:
            name_child = node.child_by_field_name(name_field)
            if name_child and name_child.type in self.IDENTIFIER_NODE_TYPES:
                return name_child.text.decode('utf-8')

        # 2. Language-specific overrides (these handle nuances where a simple 'name' field
        #    might not be direct or require specific child pathing)
        if self.language_str == 'python' and node.type in ['function_definition', 'class_definition', 'method_definition']:
            name_child = node.child_by_field_name('name')
            if name_child and name_child.type == 'identifier': # Python's identifier is often directly a 'name' field
                return name_child.text.decode('utf-8')
        
        # JS/TS often have 'name' field but worth keeping specific if there are specific cases
        if self.language_str in ['javascript', 'typescript'] and node.type in ['function_declaration', 'class_declaration', 'method_definition']:
            name_child = node.child_by_field_name('name')
            if name_child and name_child.type == 'identifier':
                return name_child.text.decode('utf-8')

        # 3. Fallback: Iterate through direct children looking for common identifier types
        #    This is the least reliable but can catch names in grammars that don't use
        #    field names consistently or where the identifier is a direct, unlabeled child.
        for child in node.children:
            if child.type in self.IDENTIFIER_NODE_TYPES:
                return child.text.decode('utf-8')
        
        return None

    def chunk(self, raw_file_document: Dict[str, Any]) -> List[Dict]:
        content = raw_file_document['content']
        meta = raw_file_document['meta']
        file_path = meta['file_path']
        repo_name = meta['repo_name']
        
        if not content.strip():
            logging.debug(f"Skipping empty content for Tree-sitter chunking: {file_path}")
            return []

        all_chunks = []
        lines = content.splitlines()
        content_bytes = content.encode('utf-8')
        covered_lines = [False] * len(lines) # Tracks which lines have been covered by a semantic chunk (0-indexed)

        try:
            tree = self.parser.parse(content_bytes)
            # If the root node has an ERROR child, parsing failed or incomplete
            if tree.root_node.children and tree.root_node.children[0].type == 'ERROR':
                logging.warning(f"Tree-sitter parsing error for {file_path}. Falling back to generic chunking.")
                return self.generic_chunker.chunk(raw_file_document)

        except Exception as e:
            logging.warning(f"Error parsing {file_path} with tree-sitter ({self.language_str}): {e}. Falling back to generic chunking.")
            return self.generic_chunker.chunk(raw_file_document)

        # Nodes to process: tuple of (node, parent_name_for_this_node)
        # parent_name_for_this_node represents the immediate semantic parent (e.g., class name for a method)
        nodes_to_process: List[Tuple[Any, Optional[str]]] = [(tree.root_node, None)]
        processed_node_ids = set() # To avoid processing nodes multiple times

        while nodes_to_process:
            current_node, parent_name_for_this_node_context = nodes_to_process.pop(0) # Process in order
            
            # Skip if already processed or if it's an error node (error nodes can sometimes have recursive children)
            if id(current_node) in processed_node_ids or current_node.type == 'ERROR':
                continue
            processed_node_ids.add(id(current_node))

            # Check if this node represents a major semantic unit we want to chunk
            chunk_type_candidate = self.NODE_TYPE_TO_CHUNK_TYPE.get(current_node.type)

            if chunk_type_candidate:
                # This is a semantic unit, extract its content and name
                chunk_content, start_line, end_line = self._get_node_content(current_node, lines)
                node_name = self._get_node_name(current_node)

                # Mark lines as covered (0-indexed) so they are not picked up by the 'uncovered code' logic
                for i in range(start_line - 1, end_line):
                    if 0 <= i < len(covered_lines): # Boundary check
                        covered_lines[i] = True

                # The parent name for *this* chunk is the context passed down from its own parent
                effective_parent_name = parent_name_for_this_node_context
                
                # If the chunk content is too large, break it down further using the generic chunker
                if len(chunk_content) > self.chunk_size and self.chunk_size > 0:
                    temp_raw_doc_for_generic = {
                        'content': chunk_content,
                        'meta': {
                            'repo_name': repo_name,
                            'file_path': file_path,
                            'file_type': meta.get('file_type', self.language_str),
                        }
                    }
                    sub_chunks = self.generic_chunker.chunk(temp_raw_doc_for_generic)

                    for sub_chunk in sub_chunks:
                        # Adjust metadata for generic sub-chunks to reflect their position in the ORIGINAL file
                        sub_chunk['meta']['start_line'] += (start_line - 1) # Shift by the parent unit's start line offset
                        sub_chunk['meta']['end_line'] += (start_line - 1)
                        sub_chunk['meta']['chunk_type'] = f"{chunk_type_candidate}_part" if node_name else "code_block"
                        sub_chunk['meta']['language'] = self.language_str
                        sub_chunk['meta']['name'] = node_name
                        sub_chunk['meta']['parent_name'] = effective_parent_name
                        all_chunks.append(sub_chunk)
                else:
                    # Content fits within chunk_size, create a single semantic chunk
                    chunk_meta = self._create_base_chunk_meta(
                        raw_doc_meta=meta,
                        chunk_type=chunk_type_candidate,
                        language=self.language_str,
                        start_line=start_line,
                        end_line=end_line,
                        name=node_name,
                        parent_name=effective_parent_name,
                        section=None # Tree-sitter doesn't inherently define 'sections' like markdown
                    )
                    all_chunks.append({'content': chunk_content, 'meta': chunk_meta})

                # Determine the parent name context to pass to the *children* of the current node.
                # If the current node has a name, it becomes the parent for its children.
                # Otherwise, its children inherit the current node's parent context (their grandparent).
                parent_name_for_children = node_name if node_name else effective_parent_name
                for child in current_node.children:
                    nodes_to_process.append((child, parent_name_for_children))

            else:
                # This node is NOT a major semantic unit itself (e.g., a 'program' node, a 'module',
                # or just a block of statements that isn't a class/function definition itself, a comment
                # outside definitions, etc.).
                # We should continue to traverse its children, passing down the *same* parent context.
                for child in current_node.children:
                    nodes_to_process.append((child, parent_name_for_this_node_context))


        # --- Handle Uncovered (Top-Level) Code ---
        # This part catches any code not explicitly covered by the high-level semantic units.
        uncovered_blocks = []
        current_block_start_line_idx = -1 # 0-indexed line index

        for i, line_covered in enumerate(covered_lines):
            if not line_covered and lines[i].strip(): # Only consider non-empty lines as uncovered code
                if current_block_start_line_idx == -1: # Start of a new uncovered block
                    current_block_start_line_idx = i
            else: # If the line IS covered or empty, and we were in an uncovered block
                if current_block_start_line_idx != -1:
                    uncovered_blocks.append((current_block_start_line_idx, i - 1)) # End of the block
                    current_block_start_line_idx = -1 # Reset

        # Add the last block if it extends to the end of the file
        if current_block_start_line_idx != -1:
            uncovered_blocks.append((current_block_start_line_idx, len(lines) - 1))

        # Process these uncovered blocks using the internal GenericChunker
        for start_idx, end_idx in uncovered_blocks:
            block_content = "\n".join(lines[start_idx : end_idx + 1]).strip()
            
            if not block_content: # Skip empty blocks that might result from stripping
                continue

            temp_raw_doc_for_generic = {
                'content': block_content,
                'meta': {
                    'repo_name': repo_name,
                    'file_path': file_path,
                    'file_type': meta.get('file_type', self.language_str),
                }
            }
            generic_chunks_from_block = self.generic_chunker.chunk(temp_raw_doc_for_generic)

            # Adjust metadata for generic chunks to reflect their position in the ORIGINAL file
            for chunk in generic_chunks_from_block:
                chunk['meta']['start_line'] += start_idx # Shift by the block's start line offset
                chunk['meta']['end_line'] += start_idx   # Shift by the block's start line offset
                chunk['meta']['chunk_type'] = 'top_level_code' # More specific type for code, rather than just 'text_block'
                chunk['meta']['language'] = self.language_str
            
            all_chunks.extend(generic_chunks_from_block)

        # Sort all chunks by their starting line number for a coherent order in the final output
        all_chunks.sort(key=lambda x: x['meta']['start_line'])

        logging.info(f"Chunked {file_path} into {len(all_chunks)} Tree-sitter semantic chunks (language: {self.language_str}).")
        return all_chunks