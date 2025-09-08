# ingestion/test_ingestion.py (remains the same as before)

# --- Configuration for Testing ---
TEST_REPO_URL = "https://github.com/highcharts/highcharts"
TEST_REPO_NAME = "test-highcharts-repo"

# ingestion/test_ingestion.py (updated to call repo_processor)

import os
from  ingestion.repo_processer import process_repository_for_rag

def run_full_ingestion_test():
    print(f"--- Running Full Phase 2 Ingestion Test ---")
    print(f"Attempting to process repository: {TEST_REPO_URL}")

    try:
        semantic_chunks =  process_repository_for_rag(TEST_REPO_URL, TEST_REPO_NAME)

        if semantic_chunks:
            print(f"\nSuccessfully generated {len(semantic_chunks)} semantic chunks.")
            print("\n--- Sample Semantic Chunks (First 5) ---")
            for i, chunk in enumerate(semantic_chunks[:5]):
                print(f"Chunk {i+1}:")
                print(f"  Repo: {chunk['meta']['repo_name']}")
                print(f"  Path: {chunk['meta']['file_path']}")
                print(f"  Type: {chunk['meta']['chunk_type']}")
                if 'name' in chunk['meta'] and chunk['meta']['name']:
                    print(f"  Name: {chunk['meta']['name']}")
                if 'parent_name' in chunk['meta'] and chunk['meta']['parent_name']:
                    print(f"  Parent: {chunk['meta']['parent_name']}")
                print(f"  Lines: {chunk['meta']['start_line']}-{chunk['meta']['end_line']}")
                print(f"  Language: {chunk['meta'].get('language', 'N/A')}")
                print(f"  Content (first 200 chars):\n{chunk['content'][:200]}...")
                print("-" * 30)

            from collections import defaultdict
            chunk_type_counts = defaultdict(int)
            for chunk in semantic_chunks:
                chunk_type_counts[chunk['meta']['chunk_type']] += 1
            print("\n--- Chunk Type Counts ---")
            for ctype, count in sorted(chunk_type_counts.items()):
                print(f"  {ctype}: {count} chunks")

        else:
            print("\nNo semantic chunks generated or an error occurred. Check logs above.")
    except Exception as e:
        print(f"\nAn error occurred during full ingestion test: {e}")
        import traceback
        traceback.print_exc()


    print("\n--- Test Complete ---")

if __name__ == "__main__":
    run_full_ingestion_test()
