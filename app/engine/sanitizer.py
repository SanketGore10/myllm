"""
Output sanitization layer - removes control tokens and artifacts.

CRITICAL: Runs in ENGINE layer, not UI. Both CLI and API benefit.

"""

import re
from typing import List, Optional


class OutputSanitizer:
    """
    Sanitizes model output before display.
    
    Removes:
    - Stop tokens (</s>, <|im_end|>, etc.)
    - Role markers (<|im_start|>, [INST], etc.)
    - Duplicate assistant prefixes
    - Control sequences
    """
    
    def __init__(self, stop_tokens: List[str]):
        """
        Initialize sanitizer with model-specific stop tokens.
        
        Args:
            stop_tokens: Stop tokens from model template
        """
        self.stop_tokens = stop_tokens
        self.buffer = []
        
        # Patterns to strip (order matters - most specific first)
        self.strip_patterns = [
            # Stop tokens (escaped for regex)
            *[(re.escape(token), "") for token in stop_tokens],
            
            # ChatML tokens
            (r"<\|im_start\|>\s*(user|assistant|system)\s*", ""),
            (r"<\|im_end\|>", ""),
            
            # LLaMA tokens
            (r"\[INST\]", ""),
            (r"\[/INST\]", ""),
            (r"<<SYS>>", ""),
            (r"<</SYS>>", ""),
            (r"<s>", ""),
            (r"</s>", ""),
            
            # LLaMA 3 tokens
            (r"<\|begin_of_text\|>", ""),
            (r"<\|end_of_text\|>", ""),
            (r"<\|start_header_id\|>\s*(user|assistant|system)\s*<\|end_header_id\|>", ""),
            (r"<\|eot_id\|>", ""),
            
            # Alpaca/Phi tokens
            (r"###\s*(Instruction|Response|System):\s*", ""),
            
            # Duplicate role markers
            (r"\b(assistant|user|system)\s+\1\b", r"\1"),  # "assistant assistant" → "assistant"
            
            # Standalone role words at start (but not in content)
            (r"^\s*(assistant|user|system)\s*:?\s*", ""),
        ]
    
    def sanitize(self, text: str) -> str:
        """
        Sanitize complete text (non-streaming).
        
        Args:
            text: Raw model output
        
        Returns:
            Cleaned text
        """
        for pattern, replacement in self.strip_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Clean up extra whitespace
        text = re.sub(r'\n\n\n+', '\n\n', text)  # Max 2 newlines
        text = text.strip()
        
        return text
    
    def should_stop(self, token: str) -> bool:
        """
        Check if token contains a stop sequence.
        
        Args:
            token: Token to check
        
        Returns:
            True if generation should stop
        """
        # Add to buffer for lookahead
        self.buffer.append(token)
        combined = "".join(self.buffer[-20:])  # Last 20 tokens
        
        # Check if any stop token appears in buffer
        for stop in self.stop_tokens:
            if stop in combined:
                return True
        
        return False
    
    def sanitize_token(self, token: str) -> Optional[str]:
        """
        Sanitize single token (streaming mode).
        
        Args:
            token: Single token from stream
        
        Returns:
            - Cleaned token if safe to emit
            - None if token should be suppressed
        """
        # Check for stop tokens first (highest priority)
        if self.should_stop(token):
            return None  # Stop generation
        
        # Check if token is purely a control sequence
        for pattern, _ in self.strip_patterns:
            if re.fullmatch(pattern, token, flags=re.IGNORECASE):
                return None  # Suppress this token entirely
        
        # Partial cleanup (but don't over-strip content)
        cleaned = token
        for pattern, replacement in self.strip_patterns:
            # Only apply if pattern is complete in this token
            if re.search(pattern, cleaned, flags=re.IGNORECASE):
                cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
        
        # If token was completely stripped, don't emit
        if not cleaned.strip() and token.strip():
            return None
        
        return cleaned
    
    def reset(self):
        """Reset buffer for new generation."""
        self.buffer = []
