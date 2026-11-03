"""LLM calls with lossless logging.

One client (client.LLMClient) for Ollama's native /api/chat and its OpenAI-compatible /v1, and one event
schema (hbench.llm_call/3) for every transport, including LiteLLM-mediated calls (runlog.litellm_cb).
"""

from .client import ENDPOINTS, LLMClient, LLMResult  # noqa: F401
from .normalize import path_class  # noqa: F401
from .record import LLM_CALL_SCHEMA, load_tokens, record_llm_call, rederive  # noqa: F401
