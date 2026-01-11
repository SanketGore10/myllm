"""Unit tests for app/engine/sanitizer.py - Output sanitization."""

import pytest

from app.engine.sanitizer import OutputSanitizer


class TestOutputSanitizer:
    """Test OutputSanitizer class."""
    
    def test_strips_stop_tokens(self):
        """Test sanitizer removes stop tokens."""
        sanitizer = OutputSanitizer(stop_tokens=["</s>", "<|im_end|>"])
        
        dirty = "Hello world</s>"
        clean = sanitizer.sanitize(dirty)
        
        assert "</s>" not in clean
        assert clean == "Hello world"
    
    def test_strips_chatml_tokens(self):
        """Test sanitizer removes ChatML tokens."""
        sanitizer = OutputSanitizer(stop_tokens=["<|im_end|>"])
        
        dirty = "<|im_start|>user\nHello<|im_end|>"
        clean = sanitizer.sanitize(dirty)
        
        assert "<|im_start|>" not in clean
        assert "<|im_end|>" not in clean
        assert "Hello" in clean
    
    def test_strips_llama_tokens(self):
        """Test sanitizer removes LLaMA tokens."""
        sanitizer = OutputSanitizer(stop_tokens=["</s>"])
        
        dirty = "<s>[INST] Question [/INST] Answer</s>"
        clean = sanitizer.sanitize(dirty)
        
        assert "[INST]" not in clean
        assert "[/INST]" not in clean
        assert "<s>" not in clean
        assert "Question" in clean
        assert "Answer" in clean
    
    def test_strips_phi_tokens(self):
        """Test sanitizer removes Phi/Alpaca tokens."""
        sanitizer = OutputSanitizer(stop_tokens=["###"])
        
        dirty = "### Instruction:\nTest### Response:\nAnswer"
        clean = sanitizer.sanitize(dirty)
        
        assert "### Instruction:" not in clean
        assert "### Response:" not in clean
        assert "Test" in clean
        assert "Answer" in clean
    
    def test_removes_duplicate_role_markers(self):
        """Test sanitizer fixes duplicate role markers."""
        sanitizer = OutputSanitizer(stop_tokens=[])
        
        dirty = "assistant assistant hello"
        clean = sanitizer.sanitize(dirty)
        
        # Should reduce to single "assistant"
        assert clean.count("assistant") == 1
    
    def test_stops_on_token_detection(self):
        """Test should_stop detects stop tokens."""
        sanitizer = OutputSanitizer(stop_tokens=["</s>", "STOP"])
        
        # Build up tokens
        assert not sanitizer.should_stop("Hello")
        assert not sanitizer.should_stop(" world")
        assert sanitizer.should_stop("</s>")  # Should detect stop
    
    def test_sanitize_token_suppresses_control_sequences(self):
        """Test sanitize_token suppresses control tokens."""
        sanitizer = OutputSanitizer(stop_tokens=["</s>"])
        
        # Control token should be suppressed
        result = sanitizer.sanitize_token("<|im_start|>")
        assert result is None or result == ""
        
        # Regular text should pass through
        result = sanitizer.sanitize_token("Hello")
        assert result == "Hello"
    
    def test_sanitize_token_stops_on_eos(self):
        """Test sanitize_token returns None on stop token."""
        sanitizer = OutputSanitizer(stop_tokens=["</s>"])
        
        # Add some normal tokens
        sanitizer.sanitize_token("Hello")
        sanitizer.sanitize_token(" world")
        
        # Stop token should return None
        result = sanitizer.sanitize_token("</s>")
        assert result is None
    
    def test_cleans_extra_whitespace(self):
        """Test sanitizer cleans up excessive whitespace."""
        sanitizer = OutputSanitizer(stop_tokens=[])
        
        dirty = "Hello\n\n\n\n\nworld"
        clean = sanitizer.sanitize(dirty)
        
        # Should reduce to max 2 newlines
        assert "\n\n\n" not in clean
    
    def test_reset_clears_buffer(self):
        """Test reset clears internal buffer."""
        sanitizer = OutputSanitizer(stop_tokens=["</s>"])
        
        # Add tokens to buffer
        sanitizer.sanitize_token("Hello")
        sanitizer.sanitize_token("world")
        assert len(sanitizer.buffer) > 0
        
        # Reset should clear
        sanitizer.reset()
        assert len(sanitizer.buffer) == 0
