# rag_core/rag_pipeline.py

import re
import logging
from typing import List, Dict, Any, Callable, Set, Optional # Added Set, Optional
from pydantic import BaseModel, Field # NEW: For structured sub-questions

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# NEW: Pydantic model for structured sub-question output from Gemini
class Subquestions(BaseModel):
    subquestions: List[str] = Field(
        ...,
        description="A list of specific and focused sub-questions derived from the original query and provided context. "
                    "Generate at least 2, and up to 5 sub-questions. Ensure they are directly answerable from code documentation "
                    "or code structure. If the original query is simple and direct, it's acceptable to generate only 1-2 focused sub-questions."
    )

class RAGPipeline:
    """
    Class-based implementation of a Retrieval-Augmented Generation (RAG) pipeline.
    """
    
    # Class-level regex for extracting citations from LLM responses
    CITATION_REGEX = re.compile(r"\[FILE:\s*(.+?),\s*LINES:\s*(\d+)-(\d+)\]")

    # NEW: System template for generating sub-questions
    SUBQUESTION_GENERATION_SYSTEM_PROMPT = """
You are an AI assistant tasked with breaking down a complex user query about a codebase into more specific and focused sub-questions.
You will be provided with the original user query and an initial set of relevant code snippets.
Your goal is to generate 2-5 sub-questions that, if answered, would collectively provide a comprehensive answer to the original query.
The sub-questions should be specific, actionable, and geared towards finding information within a codebase.

Follow these rules:
1.  **Focus on Codebase:** The sub-questions should aim to extract information typically found in source code, documentation, or configuration files.
2.  **Use Context:** Leverage the provided "Initial Code Context" to refine your sub-questions, making them more specific and relevant to the actual codebase content.
3.  **Output Format:** Respond ONLY with a JSON object conforming to the following Pydantic schema:
    ```json
    {
      "subquestions": ["sub-question 1", "sub-question 2", ...]
    }
    ```
    Do NOT include any conversational text, explanations, or other formatting outside the JSON object.
"""

    def __init__(
        self,
        chroma_manager: Any,
        gemini_client: Any,
        prompt_builder: Any
    ):
        """
        Initializes the RAG pipeline with dependencies.

        Args:
            chroma_manager: An instance responsible for document retrieval.
            gemini_client: An instance for interacting with the Gemini LLM.
            prompt_builder: An instance for constructing prompts.
        """
        self.chroma_manager = chroma_manager
        self.gemini_client = gemini_client
        self.prompt_builder = prompt_builder

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

    # NEW: Helper method to generate sub-questions using Gemini's structured output
    def _generate_subquestions(self, original_query: str, initial_chunks: List[Dict]) -> List[str]:
        """
        Uses Gemini to generate structured sub-questions based on the original query
        and an initial set of retrieved chunks.
        """
        logging.info(f"Generating sub-questions for query: '{original_query[:100]}...'")
        
        # Build the prompt for sub-question generation using the new prompt_builder method
        subquestion_prompt_text = self.prompt_builder.build_subquestion_prompt(
            original_query=original_query,
            initial_chunks=initial_chunks,
            system_template=self.SUBQUESTION_GENERATION_SYSTEM_PROMPT
        )

        # Call Gemini with the structured response schema
        structured_response: Optional[Subquestions] = self.gemini_client.generate_structured_response(
            prompt=subquestion_prompt_text,
            response_schema=Subquestions # Pass the Pydantic model
        )

        if structured_response and structured_response.subquestions:
            logging.info(f"Generated {len(structured_response.subquestions)} sub-questions: {structured_response.subquestions}")
            return structured_response.subquestions
        else:
            logging.warning("Failed to generate sub-questions or received an empty list. Proceeding with original query only.")
            return [] # Return empty list if generation fails or no sub-questions

    # NEW: Helper method to deduplicate chunks based on a unique identifier
    def _deduplicate_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Deduplicates a list of chunks based on a unique identifier derived from their metadata.
        A chunk is considered unique by its (repo_name, file_path, start_line, end_line).
        """
        seen_identifiers: Set[tuple] = set()
        deduplicated_list: List[Dict] = []

        for chunk in chunks:
            meta = chunk['meta']
            identifier = (
                meta.get('repo_name'),
                meta.get('file_path'),
                meta.get('start_line'),
                meta.get('end_line')
            )
            if identifier not in seen_identifiers:
                seen_identifiers.add(identifier)
                deduplicated_list.append(chunk)
            else:
                logging.debug(f"Deduplicated chunk: {identifier}")
        
        logging.info(f"Deduplicated chunks: {len(chunks)} original, {len(deduplicated_list)} unique.")
        return deduplicated_list

    def run(
        self,
        repo_name: str,
        query: str,
        top_k_retrieval: int = 3,
        context_expansion_factor: int = 0,
        use_two_pass_rag: bool = False # NEW: Parameter to enable/disable two-pass
    ) -> Dict[str, Any]:
        """
        Runs the RAG pipeline: retrieval, prompt building, LLM call, and source extraction.
        Optionally uses a two-pass strategy for complex queries.

        Args:
            repo_name (str): Repository name to search.
            query (str): User's natural language query.
            top_k_retrieval (int): Number of top relevant results to retrieve per query.
            context_expansion_factor (int): Context window size for neighboring chunks.
            use_two_pass_rag (bool): If True, enables the two-pass RAG strategy.

        Returns:
            Dict[str, Any]: Final result with answer and sources.
        """
        logging.info(f"Running RAG query for repo '{repo_name}' with query: '{query[:100]}...' (top_k={top_k_retrieval}, two_pass={use_two_pass_rag})")

        all_retrieved_chunks: List[Dict] = []
        effective_queries: List[str] = [query] # Start with the original query

        # --- Pass 1: Initial Retrieval ---
        try:
            initial_chunks = self.chroma_manager.query_collection(
                repo_name, query_text=query, top_k=top_k_retrieval
            )
            if not initial_chunks:
                logging.warning("No relevant chunks found in initial retrieval.")
                if not use_two_pass_rag: # If not using two-pass, and no initial chunks, we're done.
                    return {
                        'query': query,
                        'answer': "I could not find any relevant information in the codebase for your query.",
                        'sources': []
                    }
            all_retrieved_chunks.extend(initial_chunks)
            logging.debug(f"Retrieved {len(initial_chunks)} chunks in initial pass.")
        except RuntimeError as e:
            logging.error(f"Initial retrieval error: {e}")
            return {'query': query, 'answer': f"Retrieval error: {e}", 'sources': []}


        # --- Conditional Pass 2: Sub-question Generation and Retrieval ---
        if use_two_pass_rag:
            logging.info("Two-pass RAG enabled. Generating sub-questions.")
            # Generate sub-questions using Gemini, informed by initial chunks
            subquestions = self._generate_subquestions(query, initial_chunks)
            
            if subquestions:
                # Add sub-questions to the list of queries to run
                effective_queries.extend(subquestions)

                # For each sub-question, perform another retrieval
                for sq_idx, sub_q in enumerate(subquestions):
                    logging.info(f"Retrieving for sub-question {sq_idx+1}/{len(subquestions)}: '{sub_q[:100]}...'")
                    try:
                        sub_q_chunks = self.chroma_manager.query_collection(
                            repo_name, query_text=sub_q, top_k=top_k_retrieval
                        )
                        if sub_q_chunks:
                            all_retrieved_chunks.extend(sub_q_chunks)
                            logging.debug(f"Retrieved {len(sub_q_chunks)} chunks for sub-question '{sub_q[:50]}...'")
                        else:
                            logging.debug(f"No chunks found for sub-question '{sub_q[:50]}...'")
                    except RuntimeError as e:
                        logging.error(f"Sub-question retrieval error for '{sub_q[:50]}...': {e}")
            else:
                logging.warning("No sub-questions generated. Proceeding with initial chunks only for final prompt.")
        
        # --- Aggregation and Deduplication ---
        if not all_retrieved_chunks:
            logging.warning("No relevant chunks found after all retrieval passes.")
            return {
                'query': query,
                'answer': "I could not find any relevant information in the codebase for your query.",
                'sources': []
            }

        final_chunks_for_llm = self._deduplicate_chunks(all_retrieved_chunks)
        
        # Step 2: Expand context (if needed) - (Existing Placeholder)
        context_chunks = self.expand_context_chunks(repo_name, final_chunks_for_llm, context_expansion_factor)

        # Step 3: Build final prompt
        try:
            prompt = self.prompt_builder.build_rag_prompt(query, context_chunks)
        except Exception as e:
            logging.error(f"Final prompt building failed: {e}")
            return {'query': query, 'answer': f"Prompt building failed: {e}", 'sources': []}

        # Step 4: Generate final response
        try:
            response_text = self.gemini_client.generate_response(prompt)
        except RuntimeError as e:
            logging.error(f"LLM generation error: {e}")
            return {'query': query, 'answer': f"LLM generation error: {e}", 'sources': []}
        except Exception as e:
            logging.error(f"Unexpected error during final generation: {e}")
            return {'query': query, 'answer': f"Unexpected error: {e}", 'sources': []}

        # Step 5: Extract sources from final response
        sources = self.extract_sources_from_response(response_text)

        # Final response
        return {
            'query': query,
            'answer': response_text,
            'sources': sources
        }