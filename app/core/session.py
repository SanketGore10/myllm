"""
Session management for conversation history.

Handles creating/loading sessions, managing message history,
and truncating context to fit within model's window.
"""

import uuid
from typing import List, Optional

from app.storage.database import get_db
from app.storage.database import Message as DBMessage
from app.models.schemas import Message
from app.engine.tokenizer import create_tokenizer
from app.utils.errors import SessionNotFoundError, ContextWindowExceededError
from app.utils.logging import get_logger

logger = get_logger(__name__)


class SessionManager:
    """Manages conversation sessions and history."""
    
    def __init__(self):
        """Initialize session manager."""
        self.db = get_db()
    
    async def create_session(self, model_name: str) -> str:
        """
        Create a new conversation session.
        
        Args:
            model_name: Name of the model for this session
        
        Returns:
            Session ID
        """
        session_id = await self.db.create_session(model_name)
        logger.info(f"Created new session: {session_id} (model={model_name})")
        return session_id
    
    async def get_session(self, session_id: str) -> Optional[str]:
        """
        Verify session exists and return model name.
        
        Args:
            session_id: Session ID
        
        Returns:
            Model name or None if session not found
        
        Raises:
            SessionNotFoundError: If session doesn't exist
        """
        session = await self.db.get_session_with_messages(session_id)
        
        if not session:
            raise SessionNotFoundError(session_id)
        
        return session.model_name
    
    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        tokens: Optional[int] = None,
    ) -> str:
        """
        Add a message to session history.
        
        Args:
            session_id: Session ID
            role: Message role
            content: Message content
            tokens: Token count (optional)
        
        Returns:
            Message ID
        """
        message_id = await self.db.add_message(session_id, role, content, tokens)
        logger.debug(f"Added {role} message to session {session_id}")
        return message_id
    
    async def get_messages(
        self,
        session_id: str,
        max_tokens: Optional[int] = None,
        template: str = "chatml",
    ) -> List[Message]:
        """
        Get session messages with optional truncation.
        
        Args:
            session_id: Session ID
            max_tokens: Maximum token budget (triggers truncation)
            template: Prompt template for token counting
        
        Returns:
            List of messages (possibly truncated)
        """
        # Get all messages from database
        db_messages = await self.db.get_session_messages(session_id)
        
        # Convert to schema messages
        messages = [
            Message(role=m.role, content=m.content)
            for m in db_messages
        ]
        
        logger.debug(f"Retrieved {len(messages)} messages from session {session_id}")
        
        # Apply truncation if needed
        if max_tokens and messages:
            tokenizer = create_tokenizer(session_id)
            messages = tokenizer.estimate_trimmed_messages(
                messages=messages,
                max_tokens=max_tokens,
                template=template,
            )
            logger.debug(f"Truncated to {len(messages)} messages (max_tokens={max_tokens})")
        
        return messages
    
    async def get_messages_with_new(
        self,
        session_id: str,
        new_messages: List[Message],
        max_tokens: int,
        template: str = "chatml",
    ) -> List[Message]:
        """
        Get session history combined with new messages, with truncation.
        
        This is the key method for chat: combines existing history with
        new user message, then truncates to fit context window.
        
        Args:
            session_id: Session ID
            new_messages: New messages to append (not yet saved)
            max_tokens: Maximum token budget
            template: Prompt template
        
        Returns:
            Combined and truncated message list
        """
        # Get existing history
        existing_messages = await self.get_messages(session_id, max_tokens=None)
        
        # Combine with new messages
        all_messages = existing_messages + new_messages
        
        # Truncate combined history
        tokenizer = create_tokenizer(session_id)
        truncated_messages = tokenizer.estimate_trimmed_messages(
            messages=all_messages,
            max_tokens=max_tokens,
            template=template,
        )
        
        logger.debug(
            f"Combined history: {len(existing_messages)} existing + "
            f"{len(new_messages)} new = {len(all_messages)} total -> "
            f"{len(truncated_messages)} after truncation"
        )
        
        return truncated_messages
    
    async def save_conversation_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        user_tokens: Optional[int] = None,
        assistant_tokens: Optional[int] = None,
    ) -> None:
        """
        Save a complete conversation turn (user + assistant).
        
        Args:
            session_id: Session ID
            user_message: User's message
            assistant_message: Assistant's response
            user_tokens: User message token count
            assistant_tokens: Assistant message token count
        """
        await self.add_message(session_id, "user", user_message, user_tokens)
        await self.add_message(session_id, "assistant", assistant_message, assistant_tokens)
        
        logger.info(f"Saved conversation turn to session {session_id}")
    
    async def clear_session_history(self, session_id: str) -> None:
        """
        Clear all messages from a session (keep session itself).
        
        Args:
            session_id: Session ID
        """
        # Get session to verify it exists
        await self.get_session(session_id)
        
        # Delete all messages
        db_messages = await self.db.get_session_messages(session_id)
        
        async with self.db.get_session() as db_session:
            for msg in db_messages:
                await db_session.delete(msg)
        
        logger.info(f"Cleared history for session {session_id}")


def create_session_manager() -> SessionManager:
    """
    Create a session manager instance.
    
    Returns:
        SessionManager instance
    """
    return SessionManager()
