"""
Runtime manager orchestrating all components.

High-level orchestrator that coordinates session management, prompt building,
and inference execution for chat and generation requests.
"""

import uuid
from typing import Iterator, List, Optional

from app.models.schemas import Message, InferenceOptions
from app.models.registry import get_registry
from app.core.session import create_session_manager
from app.core.prompt import create_prompt_builder
from app.services.inference import get_inference_service
from app.services.embeddings import get_embeddings_service
from app.utils.logging import get_logger, PerformanceLogger
from app.utils.errors import ContextWindowExceededError

logger = get_logger(__name__)


class RuntimeManager:
    """High-level runtime orchestrator for LLM operations."""
    
    def __init__(self):
        """Initialize runtime manager."""
        self.session_manager = create_session_manager()
        self.inference_service = get_inference_service()
        self.embeddings_service = get_embeddings_service()
        self.registry = get_registry()
        
        logger.info("RuntimeManager initialized")
    
    async def chat(
        self,
        model_name: str,
        messages: List[Message],
        session_id: Optional[str] = None,
        options: Optional[InferenceOptions] = None,
        stream: bool = True,
    ) -> tuple[Iterator[str], str]:
        """
        Execute a chat request with conversation history.
        
        Flow:
        1. Create or load session
        2. Get session history + new messages
        3. Truncate to fit context window
        4. Build prompt from messages
        5. Execute inference
        6. Save assistant response to session
        
        Args:
            model_name: Name of model to use
            messages: New messages (typically just user message)
            session_id: Existing session ID (None = create new)
            options: Generation options
            stream: Enable streaming
        
        Returns:
            (token_generator, session_id) tuple
        """
        with PerformanceLogger(logger, f"Chat request (model={model_name}, stream={stream})"):
            # Step 1: Create or verify session
            if session_id:
                logger.debug(f"Using existing session: {session_id}")
                await self.session_manager.get_session(session_id)
            else:
                session_id = await self.session_manager.create_session(model_name)
                logger.debug(f"Created new session: {session_id}")
            
            # Step 2: Get model config for family, template, and context size
            model_config = self.registry.get_model_config(model_name)
            family = model_config.family  # CRITICAL: Use family not template name
            context_size = model_config.context_size
            
            # Reserve tokens for generation (max_tokens + buffer)
            max_tokens_generation = options.max_tokens if options and options.max_tokens else 512
            prompt_max_tokens = context_size - max_tokens_generation - 100  # 100 token safety buffer
            
            logger.debug(
                f"Context budget: total={context_size}, "
                f"prompt_max={prompt_max_tokens}, "
                f"generation={max_tokens_generation}"
            )
            
            # Step 3: Build prompt builder from family (explicit template registry)
            prompt_builder = create_prompt_builder(family)
            stop_tokens = prompt_builder.get_stop_tokens()  # Get stop tokens from template
            
            logger.debug(f"Using family '{family}' with stop tokens: {stop_tokens}")
            
            # Step 4: Get combined history (existing + new messages) with truncation
            all_messages = await self.session_manager.get_messages_with_new(
                session_id=session_id,
                new_messages=messages,
                max_tokens=prompt_max_tokens,
                template=family,  # Pass family for tokenization
            )
            
            if not all_messages:
                raise ContextWindowExceededError(0, context_size)
            
            # Step 5: Build prompt from truncated messages
            prompt = prompt_builder.build_prompt(all_messages)
            
            logger.debug(f"Built prompt: {len(prompt)} chars, {len(all_messages)} messages")
            
            # Step 6: Execute inference with stop tokens
            token_generator = self.inference_service.infer(
                model_name=model_name,
                prompt=prompt,
                stop_tokens=stop_tokens,  # CRITICAL: Pass stop tokens
                options=options,
                stream=stream,
            )
            
            # Return generator and session ID
            # Note: Caller is responsible for consuming generator and saving the response
            return token_generator, session_id
    
    async def save_assistant_response(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """
        Save conversation turn to session.
        
        Call this after consuming the token generator from chat().
        
        Args:
            session_id: Session ID
            user_message: User's message content
            assistant_message: Assistant's complete response
        """
        await self.session_manager.save_conversation_turn(
            session_id=session_id,
            user_message=user_message,
            assistant_message=assistant_message,
        )
        
        logger.debug(f"Saved conversation turn to session {session_id}")
    
    def generate(
        self,
        model_name: str,
        prompt: str,
        options: Optional[InferenceOptions] = None,
        stream: bool = True,
    ) -> Iterator[str]:
        """
        Execute a single-shot generation request (no session management).
        
        Flow:
        1. Build prompt (no history, just user prompt)
        2. Execute inference
        3. Return generator
        
        Args:
            model_name: Name of model to use
            prompt: Input prompt
            options: Generation options
            stream: Enable streaming
        
        Returns:
            Token generator
        """
        with PerformanceLogger(logger, f"Generate request (model={model_name}, stream={stream})"):
            # Get model config for template
            model_config = self.registry.get_model_config(model_name)
            template = model_config.template
            
            # For generate endpoint, treat prompt as-is (no template formatting)
            # This gives users more control for single-shot use cases
            
            # Execute inference
            token_generator = self.inference_service.infer(
                model_name=model_name,
                prompt=prompt,
                options=options,
                stream=stream,
            )
            
            return token_generator
    
    def embed(self, model_name: str, text: str) -> List[float]:
        """
        Generate embedding for text.
        
        Args:
            model_name: Name of model to use
            text: Input text
        
        Returns:
            Embedding vector
        """
        with PerformanceLogger(logger, f"Embedding request (model={model_name})"):
            embedding = self.embeddings_service.generate_embedding(model_name, text)
            return embedding


def get_runtime() -> RuntimeManager:
    """
    Get runtime manager instance.
    
    Returns:
        RuntimeManager instance
    """
    return RuntimeManager()
