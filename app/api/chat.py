"""
Chat API endpoint with streaming support.
"""

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.models.schemas import ChatRequest, ChatResponse, ChatStreamChunk, Message
from app.core.runtime import get_runtime
from app.engine.streaming import stream_tokens_as_sse, StreamAccumulator
from app.utils.logging import get_logger
from app.utils.errors import MyLLMError, ModelNotFoundError, InferenceError

import json

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat")
async def chat(request: ChatRequest):
    logger.info(
        f"Chat request: model={request.model}, "
        f"messages={len(request.messages)}, "
        f"stream={request.stream}"
    )

    try:
        runtime = get_runtime()

        token_generator, session_id = await runtime.chat(
            model_name=request.model,
            messages=request.messages,
            session_id=request.session_id,
            options=request.options,
            stream=request.stream,
        )

        user_messages = [m for m in request.messages if m.role == "user"]
        user_message_content = user_messages[-1].content if user_messages else ""

        # -------------------------------------------------
        # STREAMING MODE
        # -------------------------------------------------
        if request.stream:

            async def generate_sse():
                accumulator = StreamAccumulator()

                try:
                    async for sse_event in stream_tokens_as_sse(
                        token_generator, session_id
                    ):
                        if sse_event.startswith("data: "):
                            payload = json.loads(sse_event[6:])
                            token = payload.get("token")
                            if token:
                                accumulator.add_token(token)

                        yield sse_event

                    # Save conversation after streaming ends
                    if user_message_content:
                        await runtime.save_assistant_response(
                            session_id=session_id,
                            user_message=user_message_content,
                            assistant_message=accumulator.get_full_text(),
                        )

                    # Emit FINAL usage event
                    usage = runtime.get_last_usage()

                    done_event = ChatStreamChunk(
                        done=True,
                        session_id=session_id,
                        usage=usage,
                    )

                    yield f"data: {done_event.model_dump_json()}\n\n"

                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    error_event = ChatStreamChunk(done=True, error=str(e))
                    yield f"data: {error_event.model_dump_json()}\n\n"

            return EventSourceResponse(generate_sse())

        # -------------------------------------------------
        # NON-STREAMING MODE
        # -------------------------------------------------
        tokens = list(token_generator)
        assistant_message = "".join(tokens)

        if user_message_content:
            await runtime.save_assistant_response(
                session_id=session_id,
                user_message=user_message_content,
                assistant_message=assistant_message,
            )

        usage = runtime.get_last_usage()

        return ChatResponse(
            message=Message(role="assistant", content=assistant_message),
            session_id=session_id,
            usage=usage
            or {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        )

    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    except InferenceError as e:
        raise HTTPException(status_code=500, detail=str(e))

    except MyLLMError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.exception("Unexpected error in chat endpoint")
        raise HTTPException(status_code=500, detail="Internal server error")
