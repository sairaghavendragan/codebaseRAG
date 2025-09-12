# api/main.py

import os
import sys
import logging
from contextlib import asynccontextmanager  # Changed from asynccontextmanager
from typing import List, Dict, Any, Optional,Tuple

from fastapi import FastAPI, HTTPException, status,BackgroundTasks
from pydantic import BaseModel, Field
import uvicorn

 

# Import our custom modules
from config.settings import AppSettings
from ingestion.repo_processer import process_repository_for_rag
from vector_store.chroma_manager import ChromaManager
from rag_core.gemini_client import GeminiClient
from rag_core.prompt_builder import PromptBuilder
from rag_core.rag_pipeline import RAGPipeline
from rag_core.chat_manager import ChatManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- FastAPI App Lifecycle Management ---

 
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup: Initializing RAG components...")

    try:
        settings = AppSettings()
        app.state.settings = settings
        logger.info("App settings loaded.")
    except Exception as e:
        logger.critical(f"Failed to load application settings: {e}")
        raise RuntimeError(f"Application failed to start due to configuration errors: {e}")

    try:
        app.state.chroma_manager = ChromaManager(
            persist_directory=settings.chroma_db_path,
            embedding_model_name=settings.embedding_model_name
        )
        logger.info("ChromaManager initialized.")
    except Exception as e:
        logger.critical(f"Failed to initialize ChromaManager: {e}")
        raise

    try:
        app.state.gemini_client = GeminiClient(google_api_key=settings.google_api_key)
        logger.info("GeminiClient initialized.")
    except Exception as e:
        logger.critical(f"Failed to initialize GeminiClient: {e}")
        raise

    try:
        app.state.prompt_builder = PromptBuilder()
        logger.info("PromptBuilder initialized.")
    except Exception as e:
        logger.critical(f"Failed to initialize PromptBuilder: {e}")
        raise
    try:
      
        app.state.chat_manager = ChatManager()
        logger.info("ChatManager initialized.")
    except Exception as e:
        logger.critical(f"Failed to initialize ChatManager: {e}")
        raise     
    try:
        app.state.rag_pipeline = RAGPipeline(
            chroma_manager=app.state.chroma_manager,
            gemini_client=app.state.gemini_client,
            prompt_builder=app.state.prompt_builder
        )
        logger.info("RagPipeline initialized.")
    except Exception as e:
        logger.critical(f"Failed to initialize RagPipeline: {e}")
        raise

    logger.info("All RAG components initialized. Application ready.")

    yield  # 🔄 This tells FastAPI the app is ready to serve requests

    # --- Shutdown logic ---
    logger.info("Application shutdown: Cleaning up resources...")
    logger.info("Application shutdown complete.")

app = FastAPI(
    title="Codebase RAG Assistant API",
    description="An AI-powered assistant that answers questions about Git repository source code.",
    version="0.1.0",
    lifespan=lifespan 
)
 

# --- Pydantic Models ---

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
    top_k: Optional[int] = Field(None, ge=3, description="Number of top chunks to retrieve.")
    use_two_pass_rag: bool = Field(True, description="Whether to use the two-pass RAG strategy " \
                                    "for enhanced query understanding and retrieval.") # NEW FIELD
    conversation_id: Optional[str] = Field(None, description="Optional ID for maintaining conversation history.")

class SourceReference(BaseModel):
    file_path: str = Field(..., example="src/main.py")
    start_line: int = Field(..., example=10)
    end_line: int = Field(..., example=25)

class QueryCodebaseResponse(BaseModel):
    query: str = Field(..., example="How do I define a path operation with a Pydantic model?")
    answer: str = Field(..., example="You can define a path operation with a Pydantic model by...")
    sources: List[SourceReference] = Field(...)
    conversation_id: Optional[str] = Field(None, description="The ID of the conversation this response belongs to.")

class NewChatSessionResponse(BaseModel):
    conversation_id: str = Field(..., example="a1b2c3d4-e5f6-7890-1234-567890abcdef")
    message: str = Field("New chat session created.", example="New chat session created.")    


# --- API Endpoints (all made synchronous) ---

def background_ingest_repo(repo_url: str, repo_name: str, chroma_manager: ChromaManager):
    logger.info(f"[Background] Starting ingestion for repo: {repo_url} (name: {repo_name})")
    try:
        semantic_chunks = process_repository_for_rag(repo_url, repo_name)

        if not semantic_chunks:
            logger.warning(f"[Background] No semantic chunks generated for {repo_name}.")
            return

        chroma_manager.add_chunks(repo_name, semantic_chunks)
        logger.info(f"[Background] Successfully ingested {len(semantic_chunks)} chunks for repo: {repo_name}.")
    except Exception as e:
        logger.error(f"[Background] Error during ingestion for {repo_url}: {e}", exc_info=True)



@app.post("/ingest-repo", response_model=IngestRepoResponse, status_code=status.HTTP_202_ACCEPTED)
def ingest_repo_endpoint(request: IngestRepoRequest, background_tasks: BackgroundTasks):
    logger.info(f"Received ingestion request for repo: {request.repo_url} (name: {request.repo_name})")
    try:
        if not hasattr(app.state, "chroma_manager"):
            raise HTTPException(status_code=500, detail="ChromaManager not initialized.")

        background_tasks.add_task(
            background_ingest_repo,
            repo_url=request.repo_url,
            repo_name=request.repo_name,
            chroma_manager=app.state.chroma_manager
        )

        return IngestRepoResponse(
            status="success",
            message="Repository ingestion started.",
            repo_name=request.repo_name
        )
    except Exception as e:
        logger.error(f"Failed to schedule background task: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to schedule ingestion: {e}")


@app.post("/query-codebase", response_model=QueryCodebaseResponse)
def query_codebase_endpoint(request: QueryCodebaseRequest):
    logger.info(f"Received query for repo '{request.repo_name}': '{request.query[:100]}...' (Two-pass RAG: {request.use_two_pass_rag}, Conversation ID: {request.conversation_id})") # Updated log with conversation ID
    try:
        if request.repo_name not in app.state.chroma_manager.list_collections():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Repository '{request.repo_name}' not found. Please ingest it first."
            )

        # NEW: Retrieve chat history if conversation_id is provided
        chat_history: Optional[List[Tuple[str, str]]] = None
        if request.conversation_id:
            chat_history = app.state.chat_manager.get_history(request.conversation_id)
            logger.debug(f"Retrieved chat history for '{request.conversation_id}': {len(chat_history)} entries.")

        rag_result = app.state.rag_pipeline.run(
            repo_name=request.repo_name,
            query=request.query,
            top_k_retrieval=request.top_k or app.state.settings.default_top_k_retrieval, 
            use_two_pass_rag=request.use_two_pass_rag,
            chat_history=chat_history  
        )

        if "I could not find any relevant information" in rag_result['answer']:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=rag_result['answer']
            )

        # NEW: Add current query and answer to chat history
        if request.conversation_id:
            app.state.chat_manager.add_message(request.conversation_id, request.query, rag_result['answer'])


        return QueryCodebaseResponse(
            query=rag_result['query'],
            answer=rag_result['answer'],
            sources=rag_result['sources'],
            conversation_id=request.conversation_id # Return the conversation ID
        )
    except HTTPException as e:
        raise
    except Exception as e:
        logger.error(f"Query error for repo '{request.repo_name}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Query error: {e}")


@app.get("/repos", response_model=List[str])
def list_repos_endpoint():
    logger.info("Received request to list repositories.")
    try:
        return app.state.chroma_manager.list_collections()
    except Exception as e:
        logger.error(f"Error listing repositories: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"List error: {e}")


@app.delete("/repo/{repo_name}", status_code=status.HTTP_204_NO_CONTENT)
def delete_repo_endpoint(repo_name: str):
    logger.info(f"Received request to delete repository: {repo_name}")
    try:
        if repo_name not in app.state.chroma_manager.list_collections():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Repository '{repo_name}' not found."
            )
        app.state.chroma_manager.delete_collection(repo_name)
        logger.info(f"Repository '{repo_name}' deleted.")
        return  # 204 No Content
    except HTTPException as e:
        raise
    except Exception as e:
        logger.error(f"Delete error for repo '{repo_name}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Delete error: {e}")

@app.post("/chat/new-session", response_model=NewChatSessionResponse, summary="Create a new chat conversation session")
def create_new_chat_session_endpoint():
    """
    Creates a new, unique conversation ID and initializes an empty history for it on the backend.
    This ID should be used by the client for subsequent queries in this conversation.
    """
    try:
        conversation_id = app.state.chat_manager.create_session()
        return NewChatSessionResponse(conversation_id=conversation_id)
    except Exception as e:
        logger.error(f"Error creating new chat session: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error creating new chat session: {e}")  

@app.delete("/chat/clear/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Clear a specific conversation history")
def clear_chat_history_endpoint(conversation_id: str):
    logger.info(f"Received request to clear chat history for conversation: {conversation_id}")
    try:
        app.state.chat_manager.clear_history(conversation_id)
        return
    except Exception as e:
        logger.error(f"Error clearing chat history for '{conversation_id}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error clearing chat history: {e}")

@app.get("/chat/list", response_model=List[str], summary="List all active conversation IDs")
def list_chat_conversations_endpoint():
    logger.info("Received request to list active chat conversations.")
    try:
        return app.state.chat_manager.list_conversations()
    except Exception as e:
        logger.error(f"Error listing chat conversations: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error listing chat conversations: {e}")    


# --- Uvicorn entry point ---

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
