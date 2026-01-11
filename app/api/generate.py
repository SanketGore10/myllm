"""
Generate API endpoint for single-shot text generation.

Handles POST /api/generate for stateless generation without conversation history.
"""

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.models.schemas import GenerateRequest, GenerateResponse, GenerateStreamChunk
from app.core.runtime import get_runtime
from app.engine.streaming import stream_tokens_as_sse, StreamAccumulator
from app.utils.logging import get_logger
from app.utils.errors import MyLLMError, ModelNotFoundError, InferenceError

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["generate"])


@router.post("/generate")
async def generate(request: GenerateRequest):
    """
    Generate text from a prompt (stateless, no conversation history).
    
    Supports streaming (SSE) and non-streaming modes.
    """
    logger.info(
        f"Generate request: model={request.model}, "
        f"prompt_len={len(request.prompt)}, "
        f"stream={request.stream}"
    )
    
    try:
        runtime = get_runtime()
        
        # Execute generation
        token_generator = runtime.generate(
            model_name=request.model,
            prompt=request.prompt,
            options=request.options,
            stream=request.stream,
        )
        
        if request.stream:
            # Streaming mode: return SSE stream
            async def generate_sse():
                """Generator for SSE events."""
                try:
                    # Stream tokens without session_id
                    async for sse_event in stream_tokens_as_sse(token_generator, session_id=None):
                        yield sse_event
                
                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    error_event = GenerateStreamChunk(done=True, error=str(e))
                    yield f"data: {error_event.model_dump_json()}\n\n"
            
            return EventSourceResponse(generate_sse())
        
        else:
            # Non-streaming mode: collect full response
            tokens = list(token_generator)
            generated_text = "".join(tokens)
            
            # Return complete response
            response = GenerateResponse(
                text=generated_text,
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
        logger.error(f"Unexpected error in generate endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
