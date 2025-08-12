 
import os
import re
from typing import List, Dict
from gitingest import ingest  
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_and_extract_raw_files(repo_url: str, repo_name: str) -> List[Dict]:
    """
    Clones a Git repository and extracts relevant file contents into a structured format.
    Leverages gitingest's built-in filtering.

    Args:
        repo_url (str): The URL of the Git repository.
        repo_name (str): The name to assign to the repository.

    Returns:
        List[Dict]: A list of "raw file documents", each a dictionary like:
            {'content': '...', 'meta': {'repo_name': '...', 'file_path': '...', 'file_type': '...'}}
    """
    logging.info(f"Starting ingestion for repository: {repo_url} (name: {repo_name})")
    extracted_files = []

    # --- Configuration for Filtering (Leveraging gitingest's built-in filters) ---
    # These patterns will be passed directly to gitingest.ingest()
    # Adjust include/exclude patterns and max_file_size as needed for your project's scope.
    # We are aiming to include common source code files and exclude typical non-code/binary/config files.
    INCLUDE_PATTERNS = [
        "*.py", "*.js", "*.ts", "*.jsx", "*.tsx", "*.java", "*.c", "*.cpp", "*.h", "*.hpp",
        "*.go", "*.rb", "*.php", "*.cs", "*.swift", "*.kt", "*.rs", "*.vue", "*.svelte",
        "*.html", "*.css", "*.scss", "*.less", "*.json", "*.xml", "*.yml", "*.yaml", # Often contain config/structure
        "*.md", "*.txt", # Can contain valuable documentation
        "Dockerfile", "Makefile", "requirements.txt", "package.json", # Important config files
    ]
    EXCLUDE_PATTERNS = [
        "node_modules/*", ".git/*", "__pycache__/*", "dist/*", "build/*", ".venv/*", "venv/*",
        "site-packages/*", "*.log", "*.lock", "*.zip", "*.tar.gz", "*.rar", "*.7z", "*.bin",
        "*.dll", "*.exe", "*.jar", "*.class", "*.so", "*.obj", "*.o", # Compiled/binary files
        "*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp", "*.tiff", "*.ico", "*.svg", # Image files
        "*.mp3", "*.wav", "*.ogg", "*.flac", "*.mp4", "*.avi", "*.mov", "*.mkv", # Media files
        "*.sqlite", "*.db", # Database files
        "*.pdf", "*.doc", "*.docx", "*.xls", "*.xlsx", "*.ppt", "*.pptx", # Document files
        "*.min.js", "*.map", # Minified JS, source maps
    ]
    MAX_FILE_SIZE = 512 * 1024 # 512 KB per file (adjust as needed, large files consume more tokens)


    # Regex to split files based on gitingest's output format:
    # "================================================\nFILE: path/to/file.py\n================================================\n"
    # This regex captures the file path from the header.
    FILE_DELIMITER_REGEX = r"================================================\s*FILE:\s*(.*?)\s*================================================\s*"

    try:
        # Call gitingest.ingest with built-in filtering
        summary_str, tree_str, content_str = ingest(
            repo_url,
            include_patterns=INCLUDE_PATTERNS,
            exclude_patterns=EXCLUDE_PATTERNS,
            max_file_size=MAX_FILE_SIZE
        )
        logging.info(f"Repository {repo_url} fetched successfully by gitingest. Analyzing {len(content_str)} chars of content.")

        # Split the content string using the regex delimiter
        # re.split will return a list where:
        # - The first item is usually empty (if delimiter is at the start)
        # - Subsequent items alternate between captured groups (file path) and the content that follows.
        parts = re.split(FILE_DELIMITER_REGEX, content_str)

        # The first part is usually empty or contains text before the first file.
        # We start processing from the first actual file path and content.
        # Parts will look like: ['', 'path/to/file1.py', 'content of file1', 'path/to/file2.py', 'content of file2', ...]
        # So we iterate taking two steps at a time, skipping the first empty element.
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts): # Ensure there's both a path and content
                file_path = parts[i].strip()
                file_content = parts[i+1].strip()

                if not file_path or not file_content:
                    continue # Skip empty paths or contents resulting from split

                file_extension = os.path.splitext(file_path)[1].lower()
                # Remove leading dot from extension
                file_type = file_extension.lstrip('.') if file_extension else 'unknown'
                # For Dockerfile, Makefile, requirements.txt, etc. which have no extension
                if not file_type and '.' not in os.path.basename(file_path):
                    file_type = os.path.basename(file_path).lower()


                extracted_files.append({
                    'content': file_content,
                    'meta': {
                        'repo_name': repo_name,
                        'file_path': file_path,
                        'file_type': file_type,
                    }
                })

        logging.info(f"Finished extracting {len(extracted_files)} structured files from {repo_name}.")
        return extracted_files

    except Exception as e:
        logging.error(f"Error ingesting or processing repository {repo_url}: {e}", exc_info=True)
        # Re-raising the exception might be better for API endpoints to catch specific errors.
        # For a function meant to return data, returning an empty list upon critical failure is also acceptable.
        raise # Re-raise for now to ensure errors are propagated