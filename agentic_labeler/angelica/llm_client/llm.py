from __future__ import annotations

"""LLM + Embeddings adapters with caching support.

This project uses LangChain's OpenAI-compatible clients (`ChatOpenAI`, `OpenAIEmbeddings`).

Environment variables:
- API_KEY    : provider key (OpenRouter or any OpenAI-compatible server)
- BASE_URL   : provider base URL (default: https://openrouter.ai/api/v1)
- LLM_MODEL  : default chat model (default: openai/gpt-4o)
- EMBEDDINGS_MODEL : embedding model (default: text-embedding-3-small)
- ENABLE_LLM_CACHE : enable LLM response caching (default: true)
- LLM_CACHE_DIR : directory for cache storage (default: .llm_cache)

Optional model overrides:
- LABELER_A_MODEL
- LABELER_B_MODEL
- ADJUDICATOR_MODEL

Caching:
- LLM responses are cached using SQLite to avoid redundant API calls
- Embeddings are cached separately
- Cache can be disabled by setting ENABLE_LLM_CACHE=false

If your provider does not support embeddings on BASE_URL, switch make_embeddings()
to a supported provider, or use a local embedding model.
"""

import os
from typing import Optional, Type

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain.callbacks.base import BaseCallbackHandler
from pydantic import BaseModel
from typing import Any, Dict, List

from angelica.llm_client.token_counter import record_llm_call


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read an environment variable with a default."""
    v = os.getenv(name)
    return v if v else default


def _setup_cache() -> None:
    """Setup LLM caching automatically based on environment variables."""
    enable_cache_str = _env("ENABLE_LLM_CACHE", "true")
    enable_cache = enable_cache_str.lower() in ("true", "1", "yes") if enable_cache_str else True
    
    if not enable_cache:
        return
    
    cache_dir = _env("LLM_CACHE_DIR", ".llm_cache") or ".llm_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "langchain.db")
    
    try:
        set_llm_cache(SQLiteCache(database_path=cache_path))
        print(f"✓ LLM caching enabled: {cache_path}")
    except Exception as e:
        print(f"Warning: Could not enable LLM cache: {e}")


class TokenTrackingCallback(BaseCallbackHandler):
    """Callback to track token usage from LLM calls."""
    
    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Called when LLM finishes."""
        try:
            # Extract token usage from response
            if hasattr(response, 'llm_output') and response.llm_output:
                token_usage = response.llm_output.get('token_usage', {})
                if token_usage:
                    input_tokens = token_usage.get('prompt_tokens', 0)
                    output_tokens = token_usage.get('completion_tokens', 0)
                    cached_input = token_usage.get('prompt_tokens_details', {}).get('cached_tokens', 0)
                    
                    record_llm_call(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cached_input_tokens=cached_input,
                        from_cache=False
                    )
        except Exception:
            # Silently fail if token tracking fails
            pass


# Global callback instance
_token_callback = TokenTrackingCallback()


# Setup cache automatically on module import
_setup_cache()


def make_chat_llm(model_env: str, temperature: float = 0.1) -> ChatOpenAI:
    """Create an OpenAI-compatible chat model with caching support.

    Parameters
    - model_env: environment variable name that may override the model (e.g., LABELER_A_MODEL)
    - temperature: sampling temperature (0..2 typical)
    
    Caching:
    - Standard caching: Caches based on full prompt (via set_llm_cache)
    - Output caching: Can be enabled via model_kwargs for providers that support it
    """
    api_key = _env("API_KEY")
    if not api_key:
        raise ValueError("API_KEY is not set.")

    base_url = _env("BASE_URL", "https://openrouter.ai/api/v1")
    model = _env(model_env, _env("LLM_MODEL", "openai/gpt-4o"))
    
    # Check if output caching should be enabled (for compatible providers)
    enable_output_cache_str = _env("ENABLE_OUTPUT_CACHE", "false")
    enable_output_cache = enable_output_cache_str.lower() in ("true", "1", "yes") if enable_output_cache_str else False
    
    model_kwargs: dict = {}
    if enable_output_cache:
        # For OpenAI-compatible providers that support output caching
        # This is provider-specific and may need adjustment
        model_kwargs["cache_control"] = {"type": "ephemeral"}

    return ChatOpenAI(
        base_url=base_url,
        api_key=api_key,  # type: ignore[arg-type]
        model=model or "openai/gpt-4o",
        temperature=temperature,
        model_kwargs=model_kwargs or {},
        callbacks=[_token_callback],
    )


def make_structured_llm(model_env: str, schema: Type[BaseModel], temperature: float = 0.1):
    """Create a chat model that returns structured output matching `schema`."""
    return make_chat_llm(model_env, temperature).with_structured_output(schema)


def make_embeddings() -> OpenAIEmbeddings:
    """Create an embeddings client.

    This uses OpenAI-compatible embeddings. Your BASE_URL/provider must support embeddings.
    """
    api_key = _env("API_KEY")
    if not api_key:
        raise ValueError("API_KEY is not set.")

    base_url = _env("BASE_URL", "https://openrouter.ai/api/v1")
    emb_model = _env("EMBEDDINGS_MODEL", "text-embedding-3-small")

    return OpenAIEmbeddings(
        base_url=base_url,
        api_key=api_key,  # type: ignore[arg-type]
        model=emb_model or "text-embedding-3-small",
    )
