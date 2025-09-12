# rag_core/chat_manager.py

import logging
from typing import Dict, List, Tuple, Optional
import uuid

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChatManager:
    """
    Manages conversation history for multiple chat sessions.
    Each session is identified by a unique conversation_id.
    History is stored as a list of (query, response) tuples.
    """
    def __init__(self):
        self.conversations: Dict[str, List[Tuple[str, str]]] = {}
        logger.info("ChatManager initialized.")

    def create_session(self) -> str:
        """
        Generates a new unique conversation ID and initializes an empty history for it.
        Returns the new conversation ID.
        """
        new_id = str(uuid.uuid4())
        self.conversations[new_id] = []
        logger.info(f"Created new conversation session: '{new_id}'")
        return new_id    

    def get_history(self, conversation_id: str) -> List[Tuple[str, str]]:
        """
        Retrieves the conversation history for a given ID.
        Returns an empty list if the ID does not exist.
        """
        history = self.conversations.get(conversation_id, [])
        logger.debug(f"Retrieved history for conversation_id '{conversation_id}': {len(history)} entries.")
        return history

    def add_message(self, conversation_id: str, query: str, response: str):
        """
        Adds a new query-response pair to the conversation history.
        Creates a new conversation if the ID does not exist.
        """
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
            logger.info(f"Created new conversation session: '{conversation_id}'")
        
        self.conversations[conversation_id].append((query, response))
        logger.debug(f"Added message to conversation '{conversation_id}'. Total entries: {len(self.conversations[conversation_id])}")

    def clear_history(self, conversation_id: str):
        """
        Clears the conversation history for a specific ID.
        """
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            logger.info(f"Cleared history for conversation_id: '{conversation_id}'")
        else:
            logger.warning(f"Attempted to clear history for non-existent conversation_id: '{conversation_id}'")

    def list_conversations(self) -> List[str]:
        """
        Lists all active conversation IDs.
        """
        return list(self.conversations.keys())