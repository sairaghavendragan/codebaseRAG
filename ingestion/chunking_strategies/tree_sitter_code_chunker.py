# ingestion/chunking_strategies/tree_sitter_code_chunker.py

import os
from typing import List, Dict, Any, Optional
import logging

from .basechunker import BaseChunker
from .generic_chunker import GenericChunker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    from tree_sitter import Parser, Language, Node
    import tree_sitter_language_pack
except ImportError:
    logging.error("Please install tree_sitter and tree_sitter_language_pack")
    raise


class TreeSitterCodeChunker(BaseChunker):
    """
    A semantic chunker for various programming languages using Tree-sitter.
    This version uses a single-pass AST traversal to create cohesive chunks for
    major semantic units (functions, classes) and groups top-level code together.
    It also breaks down overly large semantic or top-level code blocks using a
    generic chunker to respect context window limits.
    """

    # --- EXPANDED MAPPINGS for better language support ---
    NODE_TYPE_TO_CHUNK_TYPE: Dict[str, str] = {
        # Python
        'function_definition': 'function',
        'class_definition': 'class',
        
        # JavaScript / TypeScript
        'function_declaration': 'function',
        'generator_function_declaration': 'function',
        'class_declaration': 'class',
        'method_definition': 'method',
        
        # Go
        'function_declaration': 'function',
        'method_declaration': 'method',
        'type_declaration': 'struct',
        
        # Java
        'class_declaration': 'class',
        'method_declaration': 'method',
        'interface_declaration': 'interface',
        'constructor_declaration': 'constructor',

        # C / C++
        'function_definition': 'function',
        'struct_specifier': 'struct',
        'class_specifier': 'class',
        
        # Rust
        'function_item': 'function',
        'struct_item': 'struct',
        'impl_item': 'implementation',
        'trait_item': 'trait',

        # Ruby
        'class': 'class',
        'method': 'method',
        'module': 'module',
    }

    NODE_TYPE_TO_NAME_FIELD: Dict[str, str] = {
        'function_definition': 'name', 'class_definition': 'name',
        'function_declaration': 'name', 'class_declaration': 'name',
        'method_definition': 'name', 'method_declaration': 'name',
        'type_declaration': 'name', 'interface_declaration': 'name',
        'constructor_declaration': 'name', 'function_item': 'name',
        'struct_item': 'name', 'impl_item': 'type', 'trait_item': 'name',
        'class': 'name', 'method': 'name', 'module': 'name',
        'struct_specifier': 'name', 'class_specifier': 'name',
    }
    
    IDENTIFIER_NODE_TYPES = {'identifier', 'type_identifier', 'shorthand_property_identifier'}

    def __init__(self, language: str, chunk_size: int = 1500, chunk_overlap: int = 250):
        super().__init__(chunk_size, chunk_overlap)
        self.language_str = language
        self.generic_chunker = GenericChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        try:
            self.parser = tree_sitter_language_pack.get_parser(language)
            logging.info(f"Initialized TreeSitterCodeChunker for '{language}'.")
        except Exception as e:
            raise ImportError(f"Could not load parser for '{language}': {e}")

    def _get_node_name(self, node: Node) -> Optional[str]:
        name_field = self.NODE_TYPE_TO_NAME_FIELD.get(node.type)
        if name_field:
            name_node = node.child_by_field_name(name_field)
            if name_node:
                return name_node.text.decode('utf-8')
        # Fallback for nodes where name is not a direct field
        for child in node.children:
            if child.type in self.IDENTIFIER_NODE_TYPES:
                return child.text.decode('utf-8')
        return None

    def chunk(self, raw_file_document: Dict[str, Any]) -> List[Dict]:
        content = raw_file_document['content']
        meta = raw_file_document['meta']
        file_path = meta['file_path']
        
        if not content.strip():
            return []

        all_chunks = []
        lines = content.splitlines()
        content_bytes = content.encode('utf-8')

        try:
            tree = self.parser.parse(content_bytes)
            if tree.root_node.has_error:
                logging.warning(f"Tree-sitter parsing error in {file_path}. Chunking may be suboptimal.")
        except Exception as e:
            logging.warning(f"Error parsing {file_path} with tree-sitter ({self.language_str}): {e}. Falling back to generic chunking.")
            return self.generic_chunker.chunk(raw_file_document)

        top_level_nodes_buffer = []

        def flush_top_level_buffer():
            if not top_level_nodes_buffer:
                return
            
            start_node = top_level_nodes_buffer[0]
            end_node = top_level_nodes_buffer[-1]
            start_line_0_idx = start_node.start_point[0]
            end_line_0_idx = end_node.end_point[0]
            
            block_content = "\n".join(lines[start_line_0_idx : end_line_0_idx + 1])

            if block_content.strip():
                # This aggregated top-level block might still be too large, so we pass it
                # to the generic chunker for size-based splitting.
                temp_doc = {'content': block_content, 'meta': meta}
                sub_chunks = self.generic_chunker.chunk(temp_doc)

                for sub_chunk in sub_chunks:
                    # The generic chunker returns line numbers relative to the block's content.
                    # We must adjust them to be relative to the original file.
                    sub_chunk['meta']['start_line'] += start_line_0_idx
                    sub_chunk['meta']['end_line'] += start_line_0_idx
                    sub_chunk['meta']['chunk_type'] = 'top_level_code'
                    sub_chunk['meta']['language'] = self.language_str
                    all_chunks.append(sub_chunk)
            
            top_level_nodes_buffer.clear()

        def process_node(node: Node, parent_name: Optional[str] = None):
            nonlocal top_level_nodes_buffer

            for child in node.children:
                if child.type == 'ERROR' or child.start_byte == child.end_byte:
                    continue

                chunk_type_candidate = self.NODE_TYPE_TO_CHUNK_TYPE.get(child.type)
                
                if chunk_type_candidate:
                    flush_top_level_buffer()

                    node_name = self._get_node_name(child)
                    start_line_0_idx = child.start_point[0]
                    end_line_0_idx = child.end_point[0]
                    chunk_content = "\n".join(lines[start_line_0_idx : end_line_0_idx + 1])

                    if len(chunk_content) > self.chunk_size:
                        # This semantic chunk is too large. Break it down.
                        temp_doc = {'content': chunk_content, 'meta': meta}
                        sub_chunks = self.generic_chunker.chunk(temp_doc)
                        for sub_chunk in sub_chunks:
                            sub_chunk['meta']['start_line'] += start_line_0_idx
                            sub_chunk['meta']['end_line'] += start_line_0_idx
                            # Label it as a part of the original semantic unit
                            sub_chunk['meta']['chunk_type'] = f"{chunk_type_candidate}_part"
                            sub_chunk['meta']['language'] = self.language_str
                            # Propagate the name and parent for context
                            sub_chunk['meta']['name'] = node_name
                            sub_chunk['meta']['parent_name'] = parent_name
                            all_chunks.append(sub_chunk)
                    else:
                        # This semantic chunk is a good size.
                        chunk_meta = self._create_base_chunk_meta(
                            raw_doc_meta=meta,
                            chunk_type=chunk_type_candidate,
                            language=self.language_str,
                            start_line=start_line_0_idx + 1,
                            end_line=end_line_0_idx + 1,
                            name=node_name,
                            parent_name=parent_name
                        )
                        all_chunks.append({'content': chunk_content, 'meta': chunk_meta})

                    # Recursively process children to find nested definitions
                    effective_parent_name = f"{parent_name}::{node_name}" if parent_name and node_name else node_name
                    process_node(child, parent_name=effective_parent_name)
                
                else:
                    # Not a major semantic unit, add to the buffer for top-level code.
                    top_level_nodes_buffer.append(child)

        process_node(tree.root_node)
        flush_top_level_buffer()

        all_chunks.sort(key=lambda x: x['meta']['start_line'])

        logging.info(f"Chunked {file_path} into {len(all_chunks)} Tree-sitter semantic chunks (language: {self.language_str}).")
        return all_chunks