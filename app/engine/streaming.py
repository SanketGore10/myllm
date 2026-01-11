"""
Streaming utilities for converting generator to SSE format.

Handles Server-Sent Events (SSE) formatting and error handling
for streaming responses.
"""

import json
from typing import AsyncIterator, Iterator, Optional, Dict, Any

from app.utils.logging import get_logger

logger = get_logger(__name__)


async def stream_tokens_as_sse(
    token_generator: Iterator[str],
    session_id: Optional[str] = None,
) -> AsyncIterator[str]:
    """
    Convert token generator to SSE format.
    
    Yields SSE-formatted events with tokens and completion status.
    Accumulates full response for final event.
    
    Args:
        token_generator: Iterator yielding tokens
        session_id: Optional session ID to include in final event
    
    Yields:
        SSE-formatted strings: "data: {json}\n\n"
    """
    accumulated_text = ""
    token_count = 0
    
    try:
        for token in token_generator:
            accumulated_text += token
            token_count += 1
            
            # Yield token event
            event_data = {
                "token": token,
                "done": False,
            }
            
            yield f"data: {json.dumps(event_data)}\n\n"
        
        # Yield completion event
        completion_event = {
            "done": True,
            "full_text": accumulated_text,
            "token_count": token_count,
        }
        
        if session_id:
            completion_event["session_id"] = session_id
        
        yield f"data: {json.dumps(completion_event)}\n\n"
        
        logger.debug(f"Streaming completed: {token_count} tokens")
    
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        
        # Yield error event
        error_event = {
            "done": True,
            "error": str(e),
        }
        
        yield f"data: {json.dumps(error_event)}\n\n"


async def stream_response_non_sse(
    token_generator: Iterator[str],
) -> Dict[str, Any]:
    """
    Collect full response from token generator (non-streaming mode).
    
    Args:
        token_generator: Iterator yielding tokens
    
    Returns:
        Dictionary with complete response
    """
    accumulated_text = ""
    token_count = 0
    
    try:
        for token in token_generator:
            accumulated_text += token
            token_count += 1
        
        logger.debug(f"Generation completed: {token_count} tokens")
        
        return {
            "text": accumulated_text,
            "token_count": token_count,
        }
    
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise


def sse_event(data: Dict[str, Any]) -> str:
    """
    Format data as SSE event.
    
    Args:
        data: Data dictionary to send
    
    Returns:
        SSE-formatted string
    """
    return f"data: {json.dumps(data)}\n\n"


class StreamAccumulator:
    """Helper class to accumulate streamed tokens."""
    
    def __init__(self):
        self.tokens: list[str] = []
        self.token_count = 0
    
    def add_token(self, token: str) -> None:
        """Add a token to accumulator."""
        self.tokens.append(token)
        self.token_count += 1
    
    def get_full_text(self) -> str:
        """Get accumulated full text."""
        return "".join(self.tokens)
    
    def reset(self) -> None:
        """Reset accumulator."""
        self.tokens.clear()
        self.token_count = 0
