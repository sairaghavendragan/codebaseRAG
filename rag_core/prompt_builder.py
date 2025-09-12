import logging
from typing import List, Dict,Tuple,Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PromptBuilder:
    DEFAULT_SYSTEM_TEMPLATE = """
You are an AI assistant designed to answer questions about a codebase.
You can be comprehensive and provide detailed answers to complex questions.
**You might also be provided with the conversation history.** Use this history to understand the ongoing context and refine your answer to be relevant to the current turn of conversation.
You will be provided with a user's question and several relevant code snippets from the repository.
Your task is to answer the question accurately and concisely, relying solely on the provided context.

Follow these rules:
1. **Do NOT** use your own general knowledge. If the answer is not in the provided code snippets, state that you don't have enough information.
2. **Cite Sources:** For every piece of information you extract from the code snippets, explicitly reference the source file and line numbers. The format for citations should be: [FILE: <file_path>, LINES: <start_line>-<end_line>].
3. **Code Formatting:** Use markdown code blocks when showing code (e.g., ```python).
4. **Clarity & Synthesis:** Provide a clear, step-by-step answer if the question implies a process.When multiple relevant contexts are provided, synthesize the information logically to form a comprehensive answer.**
5. **Focus:** Directly address the user's question.
"""

    DEFAULT_CHUNK_TEMPLATE = """
<DOC_START>
File: {file_path}
Language: {language}
Type: {chunk_type} {name} (Parent: {parent_name})
Lines: {start_line}-{end_line}

{content}
<DOC_END>
"""

    def __init__(
        self,
        system_template: str = None,
        chunk_template: str = None,
        max_prompt_tokens: int = 15000
    ):
        self.system_template = system_template or self.DEFAULT_SYSTEM_TEMPLATE
        self.chunk_template = chunk_template or self.DEFAULT_CHUNK_TEMPLATE
        self.max_prompt_tokens = max_prompt_tokens

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token estimation: ~1 token ≈ 4 characters."""
        return len(text) // 4
    def _format_chat_history(self, chat_history: List[Tuple[str, str]]) -> str:
        """Formats chat history into a string for the prompt."""
        if not chat_history:
            return ""
        
        formatted_history = ["Conversation History:"]
        for i, (q, a) in enumerate(chat_history):
            formatted_history.append(f"User (Turn {i+1}): {q}")
            formatted_history.append(f"Assistant (Turn {i+1}): {a}")
        return "\n".join(formatted_history) + "\n\n"

    def build_rag_prompt(self, query: str, retrieved_chunks: List[Dict],chat_history: Optional[List[Tuple[str, str]]] = None) -> str:
        formatted_chunks = []
        current_tokens = self.estimate_tokens(self.system_template) + self.estimate_tokens(f"User Query: {query}\n\nRelevant Code Context:")
        formatted_history_str = ""
        if chat_history:
            formatted_history_str = self._format_chat_history(chat_history)
        logging.debug(f"Constructing prompt for query: {query[:100]}... with {len(retrieved_chunks)} chunks.")

        for chunk in retrieved_chunks:
            meta = chunk['meta']
            formatted_chunk = self.chunk_template.format(
                file_path=meta.get('file_path', 'N/A'),
                language=meta.get('language', 'unknown'),
                chunk_type=meta.get('chunk_type', 'text'),
                name=meta.get('name', ''),
                parent_name=meta.get('parent_name', ''),
                start_line=meta.get('start_line', 'N/A'),
                end_line=meta.get('end_line', 'N/A'),
                content=chunk.get('content', '')
            )
            chunk_tokens = self.estimate_tokens(formatted_chunk)

            if current_tokens + chunk_tokens < self.max_prompt_tokens:
                formatted_chunks.append(formatted_chunk)
                current_tokens += chunk_tokens
            else:
                logging.warning("Stopped adding chunks due to max token limit.")
                break

        context_string = "\n\n".join(formatted_chunks)

        final_prompt = f"""{self.system_template}

---
{formatted_history_str}User Query: {query}

Relevant Code Context:
{context_string}

---
Based on the provided context and your instructions, please answer the user's query.
"""

        logging.info(f"Final prompt built (estimated {current_tokens} tokens, {len(formatted_chunks)} chunks included).")
        return final_prompt.strip()
    

    def build_subquestion_prompt(self, original_query: str, initial_chunks: List[Dict], system_template: str, chat_history: Optional[List[Tuple[str, str]]] = None) -> str:
        """
        Builds a prompt for Gemini to generate sub-questions, providing the original query
        and an initial set of relevant chunks as context.
        Optionally includes chat history for contextual awareness.

        Args:
            original_query (str): The initial user query.
            initial_chunks (List[Dict]): A list of chunks retrieved from the first pass.
            system_template (str): The system instructions for sub-question generation,
                                   including the JSON schema requirement.
            chat_history (Optional[List[Tuple[str, str]]]): Previous conversation turns.

        Returns:
            str: The fully constructed prompt for sub-question generation.
        """
        formatted_chunks = []
        # Estimate tokens for the system template and the query part
        current_tokens = self.estimate_tokens(system_template) + self.estimate_tokens(f"Original User Query: {original_query}\n\nInitial Code Context:")

        # Add chat history if available
        formatted_history_str = ""
        if chat_history:
            formatted_history_str = self._format_chat_history(chat_history)
            current_tokens += self.estimate_tokens(formatted_history_str)

        logging.debug(f"Constructing sub-question prompt for query: {original_query[:100]}... with {len(initial_chunks)} initial chunks. History present: {bool(chat_history)}.")

        for chunk in initial_chunks:
            meta = chunk['meta']
            # Use the default chunk template for formatting initial context
            formatted_chunk = self.chunk_template.format(
                file_path=meta.get('file_path', 'N/A'),
                language=meta.get('language', 'unknown'),
                chunk_type=meta.get('chunk_type', 'text'),
                name=meta.get('name', ''),
                parent_name=meta.get('parent_name', ''),
                start_line=meta.get('start_line', 'N/A'),
                end_line=meta.get('end_line', 'N/A'),
                content=chunk.get('content', '')
            )
            chunk_tokens = self.estimate_tokens(formatted_chunk)

            if current_tokens + chunk_tokens < self.max_prompt_tokens:
                formatted_chunks.append(formatted_chunk)
                current_tokens += chunk_tokens
            else:
                logging.warning(f"Stopped adding initial chunks to sub-question prompt due to max token limit. Max: {self.max_prompt_tokens}, Current: {current_tokens}, Chunk size: {chunk_tokens}")
                break
        
        context_string = "\n\n".join(formatted_chunks)

        subquestion_prompt = f"""{system_template}

---
{formatted_history_str}Original User Query: {original_query}

Initial Code Context:
{context_string}

---
Based on the above, generate sub-questions as a JSON object.
"""
        logging.info(f"Sub-question prompt built (estimated {current_tokens} tokens, {len(formatted_chunks)} chunks included).")
        return subquestion_prompt.strip()