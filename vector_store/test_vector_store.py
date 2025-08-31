 

import os
import sys
import shutil
import logging

# Ensure the project root is in sys.path to allow imports like 'ingestion.repo_processor'
# This assumes you are running the script from the project root directory using 'python -m'
# e.g., python -m vector_store.test_vector_store
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ingestion.repo_processer import process_repository_for_rag # NEW IMPORT!
from vector_store.chroma_manager import ChromaManager
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration for Integrated Testing ---
TEST_CHROMA_DB_DIR = "data/test_chroma_db_integrated" # Use a distinct directory for this test
TEST_REPO_URL = "https://github.com/sairaghavendragan/Kosha" # Use a repo with Python and Markdown
# TEST_REPO_URL = "https://github.com/tiangolo/fastapi" # Another good option
TEST_REPO_NAME = "kosha-test-repo" # A clean name for the ingested repo in ChromaDB
TEST_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def run_integrated_vector_store_test():
    print(f"--- Running Integrated Phase 1 + 2 + 3 Test ---")
    print(f"Ingesting and chunking repository: {TEST_REPO_URL}")

    # Clean up previous test data if it exists
    if os.path.exists(TEST_CHROMA_DB_DIR):
        logging.info(f"Cleaning up previous integrated test DB at {TEST_CHROMA_DB_DIR}")
        shutil.rmtree(TEST_CHROMA_DB_DIR)

    try:
        # 1. Full Ingestion & Semantic Chunking (Phase 1 & 2)
        print("\n--- Starting Full Ingestion and Chunking ---")
        semantic_chunks = process_repository_for_rag(TEST_REPO_URL, TEST_REPO_NAME)

        if not semantic_chunks:
            print("No semantic chunks generated. Aborting vector store test.")
            return

        print(f"\nSuccessfully generated {len(semantic_chunks)} semantic chunks from {TEST_REPO_URL}.")
        print("--- Finished Full Ingestion and Chunking ---")

        # 2. Initialize ChromaManager
        chroma_manager = ChromaManager(
            persist_directory=TEST_CHROMA_DB_DIR,
            embedding_model_name=TEST_EMBEDDING_MODEL
        )

        # 3. Add chunks to ChromaDB. Embeddings are generated automatically by Chroma.
        print(f"\nAdding {len(semantic_chunks)} chunks to ChromaDB for repo '{TEST_REPO_NAME}'...")
        chroma_manager.add_chunks(TEST_REPO_NAME, semantic_chunks)
        print("Chunks added successfully.")

        # 4. Perform a sample query based on the ingested content
        # Choose a query that should match content in the Kosha repo
        query_string = "How do I configure the Telegram bot and what AI model is used for summaries?"
        top_k = 3

        print(f"\nQuerying ChromaDB for '{query_string}' (top {top_k} results)...")
        retrieved_results = chroma_manager.query_collection(TEST_REPO_NAME, query_string, top_k)

        print("\n--- Retrieved Results ---")
        if retrieved_results:
            for i, result in enumerate(retrieved_results):
                print(f"Result {i+1}:")
                print(f"  Content (first 200 chars): {result['content'][:200]}...")
                print(f"  Meta: {result['meta']}")
                print("-" * 30)
        else:
            print("No results retrieved.")
        
        # Optional: Verify collection listing
        print("\n--- Listing Collections ---")
        collections = chroma_manager.list_collections()
        print(f"Available collections: {collections}")

    except Exception as e:
        print(f"\nAn error occurred during the integrated test: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean up test data after verification, even if an error occurred
        print(f"\nCleaning up integrated test ChromaDB directory: {TEST_CHROMA_DB_DIR}")
        if os.path.exists(TEST_CHROMA_DB_DIR):
            shutil.rmtree(TEST_CHROMA_DB_DIR)
            print("Cleanup complete.")

    print("\n--- Integrated Test Complete ---")

if __name__ == "__main__":
    run_integrated_vector_store_test()