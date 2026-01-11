"""
Chat API endpoint with streaming support.

Handles POST /api/chat for conversational requests with session management.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from app.models.schemas import ChatRequest, ChatResponse, ChatStreamChunk, Message
from app.core.runtime import get_runtime
from app.engine.streaming import stream_tokens_as_sse, StreamAccumulator
from app.utils.logging import get_logger
from app.utils.errors import MyLLMError, ModelNotFoundError, InferenceError

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat")
async def chat(request: ChatRequest):
    """
    Chat with a model, optionally continuing a conversation.
    
    Supports streaming (SSE) and non-streaming modes.
    Automatically creates or continues sessions for conversation history.
    """
    logger.info(
        f"Chat request: model={request.model}, "
        f"messages={len(request.messages)}, "
        f"stream={request.stream}"
    )
    
    try:
        runtime = get_runtime()
        
        # Execute chat request
        token_generator, session_id = await runtime.chat(
            model_name=request.model,
            messages=request.messages,
            session_id=request.session_id,
            options=request.options,
            stream=request.stream,
        )
        
        # Get user message for saving later
        user_messages = [m for m in request.messages if m.role == "user"]
        user_message_content = user_messages[-1].content if user_messages else ""
        
        if request.stream:
            # Streaming mode: return SSE stream
            async def generate_sse():
                """Generator for SSE events."""
                accumulator = StreamAccumulator()
                
                try:
                    # Stream tokens
                    async for sse_event in stream_tokens_as_sse(token_generator, session_id):
                        # Parse to check if done
                        import json
                        if sse_event.startswith("data: "):
                            data = json.loads(sse_event[6:])
                            if data.get("token"):
                                accumulator.add_token(data["token"])
                        
                        yield sse_event
                    
                    # Save conversation turn after streaming completes
                    if user_message_content:
                        await runtime.save_assistant_response(
                            session_id=session_id,
                            user_message=user_message_content,
                            assistant_message=accumulator.get_full_text(),
                        )
                
                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    error_event = ChatStreamChunk(done=True, error=str(e))
                    yield f"data: {error_event.model_dump_json()}\n\n"
            
            return EventSourceResponse(generate_sse())
        
        else:
            # Non-streaming mode: collect full response
            tokens = list(token_generator)
            assistant_message = "".join(tokens)
            
            # Save conversation turn
            if user_message_content:
                await runtime.save_assistant_response(
                    session_id=session_id,
                    user_message=user_message_content,
                    assistant_message=assistant_message,
                )
            
            # Return complete response
            response = ChatResponse(
                message=Message(role="assistant", content=assistant_message),
                session_id=session_id,
                usage={
                    "prompt_tokens": 0,  # TODO: implement accurate counting
                    "completion_tokens": len(tokens),
                    "total_tokens": len(tokens),
                },
            )
            
            return response
    
    except ModelNotFoundError as e:
        logger.error(f"Model not found: {e.model_name}")
        raise HTTPException(status_code=404, detail=str(e))
    
    except InferenceError as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    except MyLLMError as e:
        logger.error(f"MyLLM error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
