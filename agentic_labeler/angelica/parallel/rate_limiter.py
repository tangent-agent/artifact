"""Token bucket rate limiter for API calls.

This module implements a simple token bucket algorithm to limit the rate of
API calls to respect provider rate limits.
"""

import time
from typing import Optional


class RateLimiter:
    """Token bucket rate limiter for controlling API request rates.
    
    This implementation uses a token bucket algorithm where:
    - Tokens are added at a constant rate (refill_rate)
    - Each request consumes one token
    - Requests block if no tokens are available
    
    Args:
        requests_per_minute: Maximum number of requests allowed per minute.
                           If None or 0, rate limiting is disabled.
    
    Example:
        >>> limiter = RateLimiter(requests_per_minute=60)
        >>> limiter.acquire()  # Blocks if rate limit exceeded
    """
    
    def __init__(self, requests_per_minute: Optional[int] = None):
        """Initialize the rate limiter.
        
        Args:
            requests_per_minute: Maximum requests per minute. None or 0 disables limiting.
        """
        self.enabled = requests_per_minute is not None and requests_per_minute > 0
        
        if self.enabled and requests_per_minute is not None:
            # Convert requests per minute to tokens per second
            self.refill_rate = requests_per_minute / 60.0
            self.capacity = float(requests_per_minute)
            self.tokens = self.capacity
            self.last_refill = time.time()
        else:
            self.refill_rate = 0.0
            self.capacity = 0.0
            self.tokens = 0.0
            self.last_refill = 0.0
    
    def _refill(self) -> None:
        """Refill tokens based on elapsed time since last refill."""
        if not self.enabled:
            return
            
        now = time.time()
        elapsed = now - self.last_refill
        
        # Add tokens based on elapsed time
        self.tokens = min(
            self.capacity,
            self.tokens + elapsed * self.refill_rate
        )
        self.last_refill = now
    
    def acquire(self, tokens: int = 1) -> None:
        """Acquire tokens, blocking if necessary.
        
        This method will block until the requested number of tokens are available.
        
        Args:
            tokens: Number of tokens to acquire (default: 1)
        """
        if not self.enabled:
            return
        
        while True:
            self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return
            
            # Calculate how long to wait for tokens to refill
            tokens_needed = tokens - self.tokens
            wait_time = tokens_needed / self.refill_rate
            
            # Sleep for a portion of the wait time to avoid busy waiting
            time.sleep(min(wait_time, 0.1))
    
    def try_acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens without blocking.
        
        Args:
            tokens: Number of tokens to acquire (default: 1)
            
        Returns:
            True if tokens were acquired, False otherwise
        """
        if not self.enabled:
            return True
        
        self._refill()
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        
        return False
    
    def reset(self) -> None:
        """Reset the rate limiter to full capacity."""
        if self.enabled:
            self.tokens = self.capacity
            self.last_refill = time.time()


