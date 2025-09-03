import re
import logging
from typing import List, Dict, Any, Callable

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class RAGPipeline:
    """
    Class-based implementation of a Retrieval-Augmented Generation (RAG) pipeline.
    """
    
    # Class-level regex for extracting citations from LLM responses
    CITATION_REGEX = re.compile(r"\[FILE:\s*(.+?),\s*LINES:\s*(\d+)-(\d+)\]")

    def __init__(
        self,
        chroma_manager: Any,
        gemini_generate_response_func: Callable[[str], str],
        prompt_builder_func: Callable[[str, List[Dict]], str]
    ):
        """
        Initializes the RAG pipeline with dependencies.

        Args:
            chroma_manager: An instance responsible for document retrieval.
            gemini_generate_response_func: Function to call LLM with prompt and return response.
            prompt_builder_func: Function to construct prompt from query and chunks.
        """
        self.chroma_manager = chroma_manager
        self.generate_response = gemini_generate_response_func
        self.build_prompt = prompt_builder_func

    def extract_sources_from_response(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Extracts source citations from the LLM's response.

        Args:
            response_text (str): LLM's raw response.

        Returns:
            List[Dict]: Unique source references.
        """
        found_sources = set()
        extracted_sources_list = []

        for match in self.CITATION_REGEX.finditer(response_text):
            file_path = match.group(1).strip()
            start_line = int(match.group(2))
            end_line = int(match.group(3))

            source_tuple = (file_path, start_line, end_line)
            if source_tuple not in found_sources:
                found_sources.add(source_tuple)
                extracted_sources_list.append({
                    'file_path': file_path,
                    'start_line': start_line,
                    'end_line': end_line
                })

        logging.debug(f"Extracted {len(extracted_sources_list)} unique sources from LLM response.")
        return extracted_sources_list

    def expand_context_chunks(
        self,
        repo_name: str,
        initial_chunks: List[Dict],
        context_expansion_factor: int = 0
    ) -> List[Dict]:
        """
        Placeholder for context expansion logic. Currently returns initial chunks as-is.

        Args:
            repo_name (str): Name of the repository.
            initial_chunks (List[Dict]): Chunks retrieved initially.
            context_expansion_factor (int): Number of neighboring chunks to include.

        Returns:
            List[Dict]: Expanded chunks.
        """
        if context_expansion_factor <= 0 or not initial_chunks:
            return initial_chunks

        logging.warning("Context expansion is configured but not fully implemented. Returning initial chunks.")
        return initial_chunks

    def run(
        self,
        repo_name: str,
        query: str,
        top_k_retrieval: int = 3,
        context_expansion_factor: int = 0
    ) -> Dict[str, Any]:
        """
        Runs the RAG pipeline: retrieval, prompt building, LLM call, and source extraction.

        Args:
            repo_name (str): Repository name to search.
            query (str): User's natural language query.
            top_k_retrieval (int): Number of top relevant chunks to retrieve.
            context_expansion_factor (int): Context window size for neighboring chunks.

        Returns:
            Dict[str, Any]: Final result with answer and sources.
        """
        logging.info(f"Running RAG query for repo '{repo_name}' with query: '{query[:100]}...' (top_k={top_k_retrieval})")

        # Step 1: Retrieve from vector DB
        try:
            retrieved_chunks = self.chroma_manager.query_collection(
                repo_name, query_text=query, top_k=top_k_retrieval
            )
            if not retrieved_chunks:
                logging.warning("No relevant chunks found.")
                return {
                    'query': query,
                    'answer': "I could not find any relevant information in the codebase for your query.",
                    'sources': []
                }
            logging.debug(f"Retrieved {len(retrieved_chunks)} chunks.")
        except RuntimeError as e:
            logging.error(f"Retrieval error: {e}")
            return {'query': query, 'answer': f"Retrieval error: {e}", 'sources': []}

        # Step 2: Expand context (if needed)
        context_chunks = self.expand_context_chunks(repo_name, retrieved_chunks, context_expansion_factor)

        # Step 3: Build prompt
        try:
            prompt = self.build_prompt(query, context_chunks)
        except Exception as e:
            logging.error(f"Prompt building failed: {e}")
            return {'query': query, 'answer': f"Prompt building failed: {e}", 'sources': []}

        # Step 4: Generate response
        try:
            response_text = self.generate_response(prompt)
        except RuntimeError as e:
            logging.error(f"LLM generation error: {e}")
            return {'query': query, 'answer': f"LLM generation error: {e}", 'sources': []}
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            return {'query': query, 'answer': f"Unexpected error: {e}", 'sources': []}

        # Step 5: Extract sources
        sources = self.extract_sources_from_response(response_text)

        # Final response
        return {
            'query': query,
            'answer': response_text,
            'sources': sources
        }
