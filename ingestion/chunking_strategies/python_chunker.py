# ingestion/chunking_strategies/python_chunker.py

import ast
from typing import List, Dict, Any, Optional
import logging

from .basechunker import BaseChunker
from .generic_chunker import GenericChunker # Import for handling top-level code

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PythonChunker(BaseChunker):
    """
    A semantic chunker for Python code.
    Uses Python's AST module to extract functions, classes, and methods as distinct chunks.
    Any code not covered by these structures (e.g., imports, global variables,
    top-level statements) is chunked using the GenericChunker.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        super().__init__(chunk_size, chunk_overlap)
        # PythonChunker will use an internal GenericChunker for unhandled top-level code
        self.generic_chunker = GenericChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        logging.info(f"Initialized PythonChunker. Internal generic_chunker configured with chunk_size={chunk_size}, overlap={chunk_overlap}.")

    def chunk(self, raw_file_document: Dict[str, Any]) -> List[Dict]:
        content = raw_file_document['content']
        meta = raw_file_document['meta']
        file_path = meta['file_path']
        repo_name = meta['repo_name']
        
        if not content.strip():
            logging.debug(f"Skipping empty content for Python chunking: {file_path}")
            return []

        all_chunks = []
        lines = content.splitlines() # Split by lines for easier line number mapping
        covered_lines = [False] * len(lines) # Tracks which lines have been covered by a semantic chunk (0-indexed)

        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            logging.warning(f"SyntaxError in Python file {file_path}: {e}. Falling back to full generic chunking for this file.")
            # If parsing fails, use generic chunker for the entire file
            return self.generic_chunker.chunk(raw_file_document)

        # --- AST Traversal: Custom NodeVisitor to extract semantic chunks ---
        class PythonASTVisitor(ast.NodeVisitor):
            def __init__(self, outer_chunker_instance: 'PythonChunker'):
                self.outer = outer_chunker_instance # Reference to the parent PythonChunker instance
                self.class_stack = [] # To keep track of parent class names for methods

            def _get_node_source_with_decorators(self, node: Any) -> tuple[str, int, int]:
                """
                Extracts the source code for an AST node, including decorators and docstrings.
                Assumes Python 3.8+ for node.end_lineno.
                Returns 1-indexed start_line, end_line.
                """
                start_line_0_indexed = node.lineno - 1 # Convert to 0-indexed for `lines` list

                # Find the true start line considering decorators
                if hasattr(node, 'decorator_list') and node.decorator_list:
                    # Decorators appear before the function/class definition
                    first_decorator_line_0_indexed = min(d.lineno for d in node.decorator_list) - 1
                    start_line_0_indexed = min(start_line_0_indexed, first_decorator_line_0_indexed)
                
                # Find the true end line considering body content
                # node.end_lineno is 1-indexed
                end_line_0_indexed = getattr(node, 'end_lineno', node.lineno) - 1

                # If the node has a body (like ClassDef, FunctionDef), ensure we capture the whole body
                # The last statement in the body determines the end.
                if hasattr(node, 'body') and node.body:
                    last_body_stmt = node.body[-1]
                    end_line_0_indexed = max(end_line_0_indexed, getattr(last_body_stmt, 'end_lineno', end_line_0_indexed + 1) - 1)
                
                # If it's a docstring at the module/class/function level, ensure it's captured with the entity
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    docstring_node = ast.get_docstring(node)
                    if docstring_node:
                        # ast.get_docstring does not give line info directly.
                        # This part is complex. For now, rely on overall node start/end.
                        # If docstring is separate Expr statement, it's captured by overall node bounds if properly structured.
                        pass

                # Ensure end_line_0_indexed does not exceed available lines
                end_line_0_indexed = min(end_line_0_indexed, len(lines) - 1)

                # Extract the content based on 0-indexed line numbers
                chunk_content = "\n".join(lines[start_line_0_indexed : end_line_0_indexed + 1]).strip()

                return chunk_content, start_line_0_indexed + 1, end_line_0_indexed + 1 # Return 1-indexed lines

            def _add_chunk(self, node: Any, chunk_type: str, name: str, parent_name: Optional[str] = None):
                chunk_content, start_line, end_line = self._get_node_source_with_decorators(node)

                # Mark lines as covered (0-indexed)
                for i in range(start_line - 1, end_line):
                    if i < len(covered_lines): # Boundary check
                        covered_lines[i] = True

                # Create chunk metadata using the helper from BaseChunker
                chunk_meta = self.outer._create_base_chunk_meta(
                    raw_doc_meta=meta,
                    chunk_type=chunk_type,
                    language='python',
                    start_line=start_line,
                    end_line=end_line,
                    name=name,
                    parent_name=parent_name,
                    section=None # Python code usually doesn't have "sections" like markdown
                )
                all_chunks.append({'content': chunk_content, 'meta': chunk_meta})
                
            # --- Visit methods for AST nodes ---
            def visit_ClassDef(self, node):
                class_name = node.name
                self._add_chunk(node, 'class', class_name)
                self.class_stack.append(class_name) # Push class name onto stack
                self.generic_visit(node) # Recursively visit child nodes (methods, nested classes)
                self.class_stack.pop() # Pop class name after visiting its children

            def visit_FunctionDef(self, node):
                function_name = node.name
                parent_name = self.class_stack[-1] if self.class_stack else None
                chunk_type = 'method' if parent_name else 'function'
                self._add_chunk(node, chunk_type, function_name, parent_name)
                self.generic_visit(node)

            def visit_AsyncFunctionDef(self, node):
                function_name = node.name
                parent_name = self.class_stack[-1] if self.class_stack else None
                chunk_type = 'async_method' if parent_name else 'async_function'
                self._add_chunk(node, chunk_type, function_name, parent_name)
                self.generic_visit(node)

            # You can add more visit methods here for other important Python structures
            # if they represent distinct semantic units you want to extract explicitly.

        visitor = PythonASTVisitor(self)
        visitor.visit(tree)

        # --- Handle Uncovered (Top-Level) Code ---
        # This part ensures that imports, global variables, module-level docstrings,
        # and standalone statements are also chunked.
        uncovered_blocks = []
        current_block_start_line_idx = -1 # 0-indexed line index

        for i, line_covered in enumerate(covered_lines):
            if not line_covered: # If the line is not covered by a semantic chunk
                if current_block_start_line_idx == -1: # Start of a new uncovered block
                    current_block_start_line_idx = i
            else: # If the line IS covered, and we were in an uncovered block
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

            # Create a temporary raw document for the generic chunker, adjusting line numbers
            temp_raw_doc_for_generic = {
                'content': block_content,
                'meta': {
                    'repo_name': repo_name,
                    'file_path': file_path,
                    'file_type': meta.get('file_type', 'py'), # Pass original file type
                }
            }
            generic_chunks_from_block = self.generic_chunker.chunk(temp_raw_doc_for_generic)

            # Adjust metadata for generic chunks to reflect their position in the ORIGINAL file
            for chunk in generic_chunks_from_block:
                chunk['meta']['start_line'] += start_idx # Shift by the block's start line offset
                chunk['meta']['end_line'] += start_idx   # Shift by the block's start line offset
                chunk['meta']['chunk_type'] = 'top_level_code' # More specific type than 'text_block'
                chunk['meta']['language'] = 'python' # Explicitly set for these
            
            all_chunks.extend(generic_chunks_from_block)

        # Sort all chunks by their starting line number for a coherent order in the final output
        all_chunks.sort(key=lambda x: x['meta']['start_line'])

        logging.info(f"Chunked {file_path} into {len(all_chunks)} Python semantic chunks (including top-level code).")
        return all_chunks
    