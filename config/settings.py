# config/settings.py

from pydantic_settings import BaseSettings, SettingsConfigDict # pip install pydantic-settings
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AppSettings(BaseSettings):
    """
    Application settings loaded from environment variables or a .env file.
    """
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # API Keys
    google_api_key: str

    # ChromaDB Settings
    chroma_db_path: str = "data/chroma_db"
    embedding_model_name: str = "all-MiniLM-L6-v2"

   
    # RAG Pipeline Settings
    default_top_k_retrieval: int = 3
    context_expansion_factor: int = 0
    use_two_pass_rag_default: bool = True  
    
    # FastAPI specific settings
    # You might want to add host/port here, but uvicorn takes them as args generally
    # app_host: str = "0.0.0.0"
    # app_port: int = 8000

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Validate critical settings after loading
        if not self.google_api_key:
            logging.error("GOOGLE_API_KEY is not set in environment or .env file.")
            raise ValueError("GOOGLE_API_KEY must be set.")
        
        logging.info("Application settings loaded.")
        logging.debug(f"Chroma DB Path: {self.chroma_db_path}")
        logging.debug(f"Embedding Model: {self.embedding_model_name}")
        logging.debug(f"Default Top-K Retrieval: {self.default_top_k_retrieval}")
        logging.debug(f"Context Expansion Factor: {self.context_expansion_factor}")
        logging.debug(f"Use Two-Pass RAG Default: {self.use_two_pass_rag_default}")

# Instantiate settings once to validate on load (optional, or instantiate on app startup)
# try:
#     settings = AppSettings()
# except ValueError as e:
#     logging.critical(f"Failed to load application settings: {e}")
#     # Exit or handle appropriately if settings are critical
#     raise