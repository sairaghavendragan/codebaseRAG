# api/main.py

import os
import sys
import logging
from contextlib import asynccontextmanager # For FastAPI lifecycle events
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field # For request/response models
import uvicorn # For running the app

# Add the project root to sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our custom modules
from config.settings import AppSettings # Our settings
from ingestion.repo_processer import process_repository_for_rag # Phase 1 & 2
from vector_store.chroma_manager import ChromaManager # Phase 3
from rag_core.gemini_client import GeminiClient         # Phase 4
from rag_core.prompt_builder import PromptBuilder     # Phase 4
from rag_core.rag_pipeline import RagPipeline         # Phase 4

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- FastAPI App Lifecycle Management ---

# Use asynccontextmanager for more modern FastAPI lifecycle management (FastAPI >= 0.95.0)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles FastAPI startup and shutdown events.
    Initializes RAG components and stores them in app.state.
    """
    logger.info("Application startup: Initializing RAG components...")
    
    # Load settings
    try:
        settings = AppSettings()
        app.state.settings = settings
        logger.info("App settings loaded.")
    except Exception as e:
        logger.critical(f"Failed to load application settings: {e}")
        # In a real app, you might want a more graceful exit or fallback
        raise RuntimeError(f"Application failed to start due to configuration errors: {e}")

    # Initialize ChromaManager
    try:
        app.state.chroma_manager = ChromaManager(
            persist_directory=settings.chroma_db_path,
            embedding_model_name=settings.embedding_model_name
        )
        logger.info("ChromaManager initialized.")
    except Exception as e:
        logger.critical(f"Failed to initialize ChromaManager: {e}")
        raise

    # Initialize GeminiClient
    try:
        app.state.gemini_client = GeminiClient(google_api_key=settings.google_api_key)
        logger.info("GeminiClient initialized.")
    except Exception as e:
        logger.critical(f"Failed to initialize GeminiClient: {e}")
        raise

    # Initialize PromptBuilder
    try:
        app.state.prompt_builder = PromptBuilder() # Using default templates for now
        logger.info("PromptBuilder initialized.")
    except Exception as e:
        logger.critical(f"Failed to initialize PromptBuilder: {e}")
        raise

    # Initialize RagPipeline
    try:
        app.state.rag_pipeline = RagPipeline(
            chroma_manager=app.state.chroma_manager,
            gemini_client=app.state.gemini_client,
            prompt_builder=app.state.prompt_builder,
            top_k_retrieval=settings.default_top_k_retrieval,
            context_expansion_factor=settings.context_expansion_factor
        )
        logger.info("RagPipeline initialized.")
    except Exception as e:
        logger.critical(f"Failed to initialize RagPipeline: {e}")
        raise

    logger.info("All RAG components initialized. Application ready.")
    yield # Application is ready to receive requests
    
    # --- Shutdown ---
    logger.info("Application shutdown: Cleaning up resources...")
    # Currently, ChromaDB PersistentClient doesn't require an explicit close.
    # If other resources needed cleanup (e.g., database connections), they'd go here.
    logger.info("Application shutdown complete.")


app = FastAPI(
    title="Codebase RAG Assistant API",
    description="An AI-powered assistant that answers questions about Git repository source code.",
    version="0.1.0",
    lifespan=lifespan # Link the lifespan context manager to the app
)

# --- Pydantic Models for API Request/Response Bodies ---

class IngestRepoRequest(BaseModel):
    repo_url: str = Field(..., example="https://github.com/tiangolo/fastapi")
    repo_name: str = Field(..., example="fastapi-repo-docs")

class IngestRepoResponse(BaseModel):
    status: str = Field("success", example="success")
    message: str = Field(..., example="Repository ingestion started/completed.")
    repo_name: str = Field(..., example="fastapi-repo-docs")

class QueryCodebaseRequest(BaseModel):
    repo_name: str = Field(..., example="fastapi-repo-docs")
    query: str = Field(..., example="How do I define a path operation with a Pydantic model?")
    top_k: Optional[int] = Field(None, ge=1, description="Number of top chunks to retrieve. Defaults to configured value.")

class SourceReference(BaseModel):
    file_path: str = Field(..., example="src/main.py")
    start_line: int = Field(..., example=10)
    end_line: int = Field(..., example=25)

class QueryCodebaseResponse(BaseModel):
    query: str = Field(..., example="How do I define a path operation with a Pydantic model?")
    answer: str = Field(..., example="You can define a path operation with a Pydantic model by...")
    sources: List[SourceReference] = Field(..., example=[{"file_path": "src/main.py", "start_line": 10, "end_line": 25}])


# --- API Endpoints ---

@app.post("/ingest-repo", response_model=IngestRepoResponse, status_code=status.HTTP_202_ACCEPTED)
async def ingest_repo_endpoint(request: IngestRepoRequest):
    """
    Ingest a Git repository into the system for RAG queries.
    This process can take a significant amount of time for large repositories.
    """
    logger.info(f"Received ingestion request for repo: {request.repo_url} (name: {request.repo_name})")
    try:
        # Phase 1 & 2: Process repository into semantic chunks
        semantic_chunks = process_repository_for_rag(request.repo_url, request.repo_name)

        if not semantic_chunks:
            logger.warning(f"No semantic chunks generated for {request.repo_name} from {request.repo_url}.")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No usable code or documentation found in the repository, or an error occurred during chunking."
            )

        # Phase 3: Add chunks to ChromaDB
        app.state.chroma_manager.add_chunks(request.repo_name, semantic_chunks)

        logger.info(f"Successfully ingested {len(semantic_chunks)} chunks for repo: {request.repo_name}.")
        return IngestRepoResponse(
            status="success",
            message="Repository ingestion completed successfully.",
            repo_name=request.repo_name
        )
    except Exception as e:
        logger.error(f"Error during repository ingestion for {request.repo_url}: {e}", exc_info=True)
        # More specific error handling could differentiate between 400 (bad input) and 500 (internal error)
        if "Invalid URL" in str(e) or "Repository not found" in str(e): # Heuristic for gitingest errors
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid repository URL or access issue: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error during ingestion: {e}")


@app.post("/query-codebase", response_model=QueryCodebaseResponse)
async def query_codebase_endpoint(request: QueryCodebaseRequest):
    """
    Query an ingested codebase for information.
    """
    logger.info(f"Received query request for repo '{request.repo_name}': '{request.query[:100]}...'")
    try:
        # Check if repository exists in ChromaDB first to give a clearer error
        if request.repo_name not in app.state.chroma_manager.list_collections():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Repository '{request.repo_name}' not found. Please ingest it first."
            )

        # Phase 4: Run RAG pipeline
        rag_result = app.state.rag_pipeline.run_rag_query(
            repo_name=request.repo_name,
            query=request.query,
            top_k=request.top_k
        )

        # If the answer explicitly says no info, return 404
        if "I could not find any relevant information" in rag_result['answer']:
             raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=rag_result['answer'] # Use the LLM's polite no-info message
            )

        return QueryCodebaseResponse(
            query=rag_result['query'],
            answer=rag_result['answer'],
            sources=rag_result['sources'] # This will be empty for now as per your decision
        )
    except HTTPException as e: # Catch already raised HTTPExceptions
        raise
    except Exception as e:
        logger.error(f"Error during codebase query for repo '{request.repo_name}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error during query: {e}")

@app.get("/repos", response_model=List[str])
async def list_repos_endpoint():
    """
    List all currently ingested repositories.
    """
    logger.info("Received request to list repositories.")
    try:
        collections = app.state.chroma_manager.list_collections()
        return collections
    except Exception as e:
        logger.error(f"Error listing repositories: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error: {e}")

@app.delete("/repo/{repo_name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_repo_endpoint(repo_name: str):
    """
    Delete an ingested repository and all its data.
    """
    logger.info(f"Received request to delete repository: {repo_name}")
    try:
        # Check if repository exists before attempting to delete for clearer response
        if repo_name not in app.state.chroma_manager.list_collections():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Repository '{repo_name}' not found."
            )
        app.state.chroma_manager.delete_collection(repo_name)
        logger.info(f"Repository '{repo_name}' deleted successfully.")
        return # 204 No Content response
    except HTTPException as e:
        raise
    except Exception as e:
        logger.error(f"Error deleting repository '{repo_name}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error during deletion: {e}")


# --- Entry point for running the FastAPI app with Uvicorn ---
if __name__ == "__main__":
    # Ensure uvicorn, httptools, and watchfiles are installed for --reload (pip install uvicorn[standard])
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)