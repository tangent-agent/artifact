"""Token usage tracking for LLM calls.

Tracks input tokens, output tokens, and cached tokens across all LLM calls
in a session.
"""

from dataclasses import dataclass, field
from typing import Dict, Any
import threading


@dataclass
class TokenUsage:
    """Token usage statistics."""
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cached_output_tokens: int = 0
    total_calls: int = 0
    cached_calls: int = 0
    
    def add(self, other: 'TokenUsage') -> None:
        """Add another TokenUsage to this one."""
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cached_input_tokens += other.cached_input_tokens
        self.cached_output_tokens += other.cached_output_tokens
        self.total_calls += other.total_calls
        self.cached_calls += other.cached_calls
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cached_output_tokens": self.cached_output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "total_calls": self.total_calls,
            "cached_calls": self.cached_calls,
            "cache_hit_rate": f"{(self.cached_calls / self.total_calls * 100):.1f}%" if self.total_calls > 0 else "0%",
        }
    
    def format_summary(self) -> str:
        """Format a human-readable summary."""
        total_tokens = self.input_tokens + self.output_tokens
        cache_hit_rate = (self.cached_calls / self.total_calls * 100) if self.total_calls > 0 else 0
        
        lines = [
            "=" * 80,
            "Token Usage Summary",
            "=" * 80,
            f"Total LLM Calls: {self.total_calls}",
            f"Cached Calls: {self.cached_calls} ({cache_hit_rate:.1f}% cache hit rate)",
        ]
        
        # Add cache explanation if no cache hits
        if self.cached_calls == 0 and self.total_calls > 0:
            lines.extend([
                "",
                "Note: 0% cache hit rate is normal on first run with unique test methods.",
                "      Cache will be effective on subsequent runs or with similar tests.",
            ])
        
        lines.extend([
            "",
            "Token Counts:",
            f"  Input Tokens:         {self.input_tokens:,}",
            f"  Output Tokens:        {self.output_tokens:,}",
            f"  Total Tokens:         {total_tokens:,}",
        ])
        
        if self.cached_input_tokens > 0 or self.cached_output_tokens > 0:
            lines.extend([
                "",
                "Cached Tokens (from provider-level caching):",
                f"  Cached Input Tokens:  {self.cached_input_tokens:,}",
                f"  Cached Output Tokens: {self.cached_output_tokens:,}",
            ])
        
        lines.append("=" * 80)
        return "\n".join(lines)


class TokenCounter:
    """Global token counter for tracking usage across all LLM calls."""
    
    def __init__(self):
        self._usage = TokenUsage()
        self._lock = threading.Lock()
    
    def record_call(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_input_tokens: int = 0,
        cached_output_tokens: int = 0,
        from_cache: bool = False
    ) -> None:
        """Record a single LLM call."""
        with self._lock:
            self._usage.input_tokens += input_tokens
            self._usage.output_tokens += output_tokens
            self._usage.cached_input_tokens += cached_input_tokens
            self._usage.cached_output_tokens += cached_output_tokens
            self._usage.total_calls += 1
            if from_cache:
                self._usage.cached_calls += 1
    
    def get_usage(self) -> TokenUsage:
        """Get current usage statistics."""
        with self._lock:
            return TokenUsage(
                input_tokens=self._usage.input_tokens,
                output_tokens=self._usage.output_tokens,
                cached_input_tokens=self._usage.cached_input_tokens,
                cached_output_tokens=self._usage.cached_output_tokens,
                total_calls=self._usage.total_calls,
                cached_calls=self._usage.cached_calls,
            )
    
    def reset(self) -> None:
        """Reset the counter."""
        with self._lock:
            self._usage = TokenUsage()
    
    def print_summary(self) -> None:
        """Print a formatted summary."""
        usage = self.get_usage()
        print("\n" + usage.format_summary())


# Global token counter instance
_global_counter = TokenCounter()


def get_token_counter() -> TokenCounter:
    """Get the global token counter instance."""
    return _global_counter


def record_llm_call(
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_input_tokens: int = 0,
    cached_output_tokens: int = 0,
    from_cache: bool = False
) -> None:
    """Record an LLM call in the global counter."""
    _global_counter.record_call(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_input_tokens,
        cached_output_tokens=cached_output_tokens,
        from_cache=from_cache
    )


