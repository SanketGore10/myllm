"""
Runtime manager coordinating inference and chat sessions.
"""

from typing import Iterator, Optional, Dict, List, Tuple
from threading import Lock
import uuid

from app.services.inference import InferenceService
from app.models.schemas import InferenceOptions, Message
from app.core.prompt import create_prompt_builder
from app.models.registry import get_registry
from app.utils.logging import get_logger, PerformanceLogger

logger = get_logger(__name__)

_runtime = None
_runtime_lock = Lock()


def get_runtime() -> "RuntimeManager":
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            _runtime = RuntimeManager()
        return _runtime


class RuntimeManager:
    """Central runtime coordinator."""

    def __init__(self):
        self.inference_service = InferenceService()
        logger.info("RuntimeManager initialized")

    # -------------------------------------------------
    # STATELESS GENERATE
    # -------------------------------------------------
    def generate(
        self,
        model_name: str,
        prompt: str,
        options: Optional[InferenceOptions] = None,
        stream: bool = True,
    ) -> Iterator[str]:
        with PerformanceLogger(
            logger, f"Generate (model={model_name}, stream={stream})"
        ):
            return self.inference_service.infer(
                model_name=model_name,
                prompt=prompt,
                options=options,
                stream=stream,
            )

    # -------------------------------------------------
    # CHAT (STATEFUL PROMPT)
    # -------------------------------------------------
    async def chat(
        self,
        model_name: str,
        messages: List[Message],
        session_id: Optional[str] = None,
        options: Optional[InferenceOptions] = None,
        stream: bool = True,
    ) -> Tuple[Iterator[str], str]:
        """
        Chat with a model using conversation history.
        """
        with PerformanceLogger(
            logger, f"Chat (model={model_name}, stream={stream})"
        ):
            registry = get_registry()
            model_config = registry.get_model_config(model_name)
            family = model_config.family

            # Build prompt
            prompt_builder = create_prompt_builder(family)
            
            normalized_messages = [
                m if isinstance(m, dict) else {"role": m.role, "content": m.content}
                for m in messages
            ]

            prompt = prompt_builder.build_prompt(normalized_messages)

            stop_tokens = prompt_builder.get_stop_tokens()

            logger.debug(
                f"Chat prompt built: family={family}, "
                f"stop_tokens={stop_tokens}"
            )

            # Generate session_id if new
            if not session_id:
                session_id = str(uuid.uuid4())

            token_generator = self.inference_service.infer(
                model_name=model_name,
                prompt=prompt,
                stop_tokens=stop_tokens,
                options=options,
                stream=stream,
            )

            return token_generator, session_id

    # -------------------------------------------------
    # USAGE
    # -------------------------------------------------
    def get_last_usage(self) -> Optional[Dict[str, int]]:
        return self.inference_service.get_last_usage()

    # -------------------------------------------------
    # PERSISTENCE HOOK (NO-OP SAFE)
    # -------------------------------------------------
    async def save_assistant_response(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """
        Persist conversation turn.
        Safe no-op if storage is disabled.
        """
        try:
            from app.storage.database import save_conversation_turn

            await save_conversation_turn(
                session_id=session_id,
                user_message=user_message,
                assistant_message=assistant_message,
            )
        except Exception:
            # Persistence must never break chat
            logger.debug("Conversation persistence skipped")





# """
# Runtime manager coordinating inference.
# """

# from typing import Iterator, Optional, Dict
# from threading import Lock

# from app.services.inference import InferenceService
# from app.models.schemas import InferenceOptions
# from app.utils.logging import get_logger, PerformanceLogger

# logger = get_logger(__name__)

# _runtime = None
# _runtime_lock = Lock()


# def get_runtime() -> "RuntimeManager":
#     global _runtime
#     with _runtime_lock:
#         if _runtime is None:
#             _runtime = RuntimeManager()
#         return _runtime


# class RuntimeManager:
#     def __init__(self):
#         self.inference_service = InferenceService()
#         logger.info("RuntimeManager initialized")

#     def generate(
#         self,
#         model_name: str,
#         prompt: str,
#         options: Optional[InferenceOptions] = None,
#         stream: bool = True,
#     ) -> Iterator[str]:
#         with PerformanceLogger(
#             logger, f"Generate (model={model_name}, stream={stream})"
#         ):
#             return self.inference_service.infer(
#                 model_name=model_name,
#                 prompt=prompt,
#                 options=options,
#                 stream=stream,
#             )

#     def get_last_usage(self) -> Optional[Dict[str, int]]:
#         """Expose last inference usage."""
#         return self.inference_service.get_last_usage()
