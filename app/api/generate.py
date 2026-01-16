"""
Generate API endpoint for single-shot text generation.
"""

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.models.schemas import GenerateRequest, GenerateResponse, GenerateStreamChunk
from app.core.runtime import get_runtime
from app.engine.streaming import stream_tokens_as_sse
from app.utils.logging import get_logger
from app.utils.errors import MyLLMError, ModelNotFoundError, InferenceError

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["generate"])


@router.post("/generate")
async def generate(request: GenerateRequest):
    """Stateless text generation (streaming or non-streaming)."""
    logger.info(
        f"Generate request: model={request.model}, "
        f"prompt_len={len(request.prompt)}, "
        f"stream={request.stream}"
    )

    try:
        runtime = get_runtime()

        token_generator = runtime.generate(
            model_name=request.model,
            prompt=request.prompt,
            options=request.options,
            stream=request.stream,
        )

        if request.stream:
            async def generate_sse():
                try:
                    async for event in stream_tokens_as_sse(
                        token_generator, session_id=None
                    ):
                        yield event
                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    error_event = GenerateStreamChunk(done=True, error=str(e))
                    yield f"data: {error_event.model_dump_json()}\n\n"

            return EventSourceResponse(generate_sse())

        # Non-streaming mode
        tokens = list(token_generator)
        generated_text = "".join(tokens)

        model = runtime.get_model(request.model)
        usage = model.get_last_usage()

        return GenerateResponse(
            text=generated_text,
            usage=usage
            or {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        )

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
