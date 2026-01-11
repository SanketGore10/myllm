"""
Database layer with SQLAlchemy ORM models.

Provides SQLite storage for conversation sessions and messages.
Handles database initialization and CRUD operations.
"""

import uuid
from datetime import datetime
from typing import List, Optional
from contextlib import asynccontextmanager

from sqlalchemy import Column, String, Integer, Text, DateTime, ForeignKey, Index, create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, selectinload
from sqlalchemy.future import select

from app.core.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

Base = declarative_base()


class Session(Base):
    """Conversation session model."""
    
    __tablename__ = "sessions"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    model_name = Column(String(255), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationship to messages
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan", lazy="selectin")
    
    def __repr__(self) -> str:
        return f"<Session(id={self.id}, model={self.model_name})>"


class Message(Base):
    """Message model for conversation history."""
    
    __tablename__ = "messages"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(20), nullable=False)  # "system", "user", "assistant"
    content = Column(Text, nullable=False)
    tokens = Column(Integer, nullable=True)  # Token count for this message
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationship to session
    session = relationship("Session", back_populates="messages")
    
    # Index for efficient session + timestamp queries
    __table_args__ = (
        Index("idx_session_created", "session_id", "created_at"),
    )
    
    def __repr__(self) -> str:
        return f"<Message(id={self.id}, role={self.role}, session={self.session_id})>"


class Database:
    """Database manager with async support."""
    
    def __init__(self, db_url: str):
        """
        Initialize database.
        
        Args:
            db_url: Database URL (sqlite+aiosqlite:///path/to/db.db)
        """
        self.db_url = db_url
        self.engine = create_async_engine(db_url, echo=False)
        self.async_session_maker = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
    
    async def init_db(self):
        """Create all tables if they don't exist."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database initialized")
    
    async def close(self):
        """Close database connection."""
        await self.engine.dispose()
        logger.info("Database connection closed")
    
    @asynccontextmanager
    async def get_session(self):
        """Get async database session (context manager)."""
        async with self.async_session_maker() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
    
    async def create_session(self, model_name: str) -> str:
        """
        Create a new conversation session.
        
        Args:
            model_name: Name of the model for this session
        
        Returns:
            Session ID
        """
        async with self.get_session() as db:
            session = Session(model_name=model_name)
            db.add(session)
            await db.flush()
            logger.info(f"Created session {session.id} for model {model_name}")
            return session.id
    
    async def get_session_with_messages(self, session_id: str) -> Optional[Session]:
        """
        Get session with all messages eagerly loaded.
        
        Args:
            session_id: Session ID
        
        Returns:
            Session object or None if not found
        """
        async with self.get_session() as db:
            result = await db.execute(
                select(Session)
                .options(selectinload(Session.messages))
                .where(Session.id == session_id)
            )
            session = result.scalar_one_or_none()
            
            if session:
                # Sort messages by creation time
                session.messages.sort(key=lambda m: m.created_at)
            
            return session
    
    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        tokens: Optional[int] = None,
    ) -> str:
        """
        Add a message to a session.
        
        Args:
            session_id: Session ID
            role: Message role ("system", "user", "assistant")
            content: Message content
            tokens: Token count (optional)
        
        Returns:
            Message ID
        """
        async with self.get_session() as db:
            message = Message(
                session_id=session_id,
                role=role,
                content=content,
                tokens=tokens,
            )
            db.add(message)
            await db.flush()
            
            # Update session timestamp
            await db.execute(
                select(Session).where(Session.id == session_id)
            )
            result = await db.execute(
                select(Session).where(Session.id == session_id)
            )
            session = result.scalar_one_or_none()
            if session:
                session.updated_at = datetime.utcnow()
            
            logger.debug(f"Added {role} message to session {session_id}")
            return message.id
    
    async def get_session_messages(self, session_id: str) -> List[Message]:
        """
        Get all messages for a session, ordered by creation time.
        
        Args:
            session_id: Session ID
        
        Returns:
            List of Message objects
        """
        async with self.get_session() as db:
            result = await db.execute(
                select(Message)
                .where(Message.session_id == session_id)
                .order_by(Message.created_at)
            )
            messages = result.scalars().all()
            return list(messages)
    
    async def delete_old_sessions(self, days: int) -> int:
        """
        Delete sessions older than specified days.
        
        Args:
            days: Age threshold in days
        
        Returns:
            Number of deleted sessions
        """
        from datetime import timedelta
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        async with self.get_session() as db:
            result = await db.execute(
                select(Session).where(Session.updated_at < cutoff_date)
            )
            sessions_to_delete = result.scalars().all()
            count = len(sessions_to_delete)
            
            for session in sessions_to_delete:
                await db.delete(session)
            
            logger.info(f"Deleted {count} old sessions")
            return count


# Global database instance
_db: Optional[Database] = None


def get_db() -> Database:
    """
    Get global database instance.
    
    Returns:
        Database instance
    """
    global _db
    if _db is None:
        settings = get_settings()
        db_url = f"sqlite+aiosqlite:///{settings.db_path}"
        _db = Database(db_url)
    return _db


async def init_database():
    """Initialize database (call on app startup)."""
    db = get_db()
    await db.init_db()


async def close_database():
    """Close database (call on app shutdown)."""
    global _db
    if _db:
        await _db.close()
        _db = None
