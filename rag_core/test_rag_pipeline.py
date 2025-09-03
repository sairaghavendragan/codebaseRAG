import os
import sys
import shutil
import logging
from dotenv import load_dotenv  # pip install python-dotenv

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables (especially GOOGLE_API_KEY)
load_dotenv()

# Import all necessary components
from ingestion.repo_processer import process_repository_for_rag # NEW IMPORT!
from vector_store.chroma_manager import ChromaManager
from rag_core.gemini_client import GeminiClient
from rag_core.prompt_builder import PromptBuilder
from rag_core.rag_pipeline import run_rag_query
 


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration for Testing ---
TEST_CHROMA_DB_DIR = "data/test_chroma_db_rag_pipeline"  # Separate DB dir for this test
TEST_REPO_URL = "https://github.com/sairaghavendragan/Kosha"
TEST_REPO_NAME = "kosha-rag-test-repo"
TEST_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TEST_GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # Load from .env

# RAG specific parameters
TEST_TOP_K_RETRIEVAL = 6  # Increase slightly to get more context
TEST_CONTEXT_EXPANSION_FACTOR = 0  # Not yet implemented

def build_prompt_wrapper(query, chunks):
    # Using the PromptBuilder class as is (or convert to functional if you want)
    prompt_builder = PromptBuilder()
    return prompt_builder.build_rag_prompt(query, chunks)

def run_rag_pipeline_test():
    print(f"--- Running Phase 4 RAG Pipeline Test ---")

    if not TEST_GOOGLE_API_KEY:
        print("ERROR: GOOGLE_API_KEY not found in .env file. Please set it to run this test.")
        return

    # Clean up previous test data if it exists
    if os.path.exists(TEST_CHROMA_DB_DIR):
        logging.info(f"Cleaning up previous RAG pipeline test DB at {TEST_CHROMA_DB_DIR}")
        shutil.rmtree(TEST_CHROMA_DB_DIR)

    # --- Phase 1 & 2: Ingest and Chunk Repository ---
    print(f"\n--- Step 1: Ingesting and Chunking Repository ({TEST_REPO_URL}) ---")
    try:
        semantic_chunks = process_repository_for_rag(TEST_REPO_URL, TEST_REPO_NAME)
        if not semantic_chunks:
            print("No semantic chunks generated. Aborting RAG pipeline test.")
            return
        print(f"Successfully generated {len(semantic_chunks)} semantic chunks.")
    except Exception as e:
        print(f"Error during ingestion and chunking: {e}")
        import traceback; traceback.print_exc()
        return

    # --- Phase 3: Initialize ChromaDB and Add Chunks ---
    print("\n--- Step 2: Initializing ChromaDB and Adding Chunks ---")
    try:
        chroma_manager = ChromaManager(
            persist_directory=TEST_CHROMA_DB_DIR,
            embedding_model_name=TEST_EMBEDDING_MODEL
        )
        chroma_manager.add_chunks(TEST_REPO_NAME, semantic_chunks)
        print(f"Added {len(semantic_chunks)} chunks to ChromaDB.")
    except Exception as e:
        print(f"Error during ChromaDB initialization or adding chunks: {e}")
        import traceback; traceback.print_exc()
        return

    # --- Phase 4: Initialize RAG Components and Run Query ---
    print("\n--- Step 3: Initializing RAG Components and Running Query ---")
    try:
        gemini_client = GeminiClient(api_key=TEST_GOOGLE_API_KEY)
        user_query = "How does one use reminder function."
        print(f"User Query: '{user_query}'")

        rag_result = run_rag_query(
            repo_name=TEST_REPO_NAME,
            query=user_query,
            chroma_manager=chroma_manager,
            gemini_generate_response_func=gemini_client.generate_response,
            prompt_builder_func=build_prompt_wrapper,
            top_k_retrieval=TEST_TOP_K_RETRIEVAL,
            context_expansion_factor=TEST_CONTEXT_EXPANSION_FACTOR
        )

        print("\n--- RAG Pipeline Result ---")
        print(f"Query: {rag_result['query']}")
        print(f"\nAnswer:\n{rag_result['answer']}")
        print(f"\nSources ({len(rag_result['sources'])} unique):")
        if rag_result['sources']:
            for src in rag_result['sources']:
                print(f"  - FILE: {src['file_path']}, LINES: {src['start_line']}-{src['end_line']}")
        else:
            print("  No sources extracted.")

    except Exception as e:
        print(f"Error during RAG pipeline execution: {e}")
        import traceback; traceback.print_exc()
        return
    finally:
        # Clean up test data
        print(f"\n--- Cleaning up test ChromaDB directory: {TEST_CHROMA_DB_DIR} ---")
        if os.path.exists(TEST_CHROMA_DB_DIR):
            shutil.rmtree(TEST_CHROMA_DB_DIR)
            print("Cleanup complete.")

    print("\n--- Phase 4 RAG Pipeline Test Complete ---")

if __name__ == "__main__":
    run_rag_pipeline_test()
