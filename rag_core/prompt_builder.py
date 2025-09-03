import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PromptBuilder:
    DEFAULT_SYSTEM_TEMPLATE = """
You are an AI assistant designed to answer questions about a codebase.
You will be provided with a user's question and several relevant code snippets from the repository.
Your task is to answer the question accurately and concisely, relying solely on the provided context.

Follow these rules:
1. **Do NOT** use your own general knowledge. If the answer is not in the provided code snippets, state that you don't have enough information.
2. **Cite Sources:** For every piece of information you extract from the code snippets, explicitly reference the source file and line numbers. The format for citations should be: [FILE: <file_path>, LINES: <start_line>-<end_line>].
3. **Code Formatting:** Use markdown code blocks when showing code (e.g., ```python).
4. **Clarity:** Provide a clear, step-by-step answer if the question implies a process.
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

    def build_rag_prompt(self, query: str, retrieved_chunks: List[Dict]) -> str:
        formatted_chunks = []
        current_tokens = self.estimate_tokens(self.system_template) + self.estimate_tokens(f"User Query: {query}\n\nRelevant Code Context:")

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
User Query: {query}

Relevant Code Context:
{context_string}

---
Based on the provided context and your instructions, please answer the user's query.
"""

        logging.info(f"Final prompt built (estimated {current_tokens} tokens, {len(formatted_chunks)} chunks included).")
        return final_prompt.strip()
