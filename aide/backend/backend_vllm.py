import logging
import re
import time
import os
from typing import Optional, Dict, Any, Tuple
import openai
from omegaconf import OmegaConf

from aide.backend.utils import OutputType, opt_messages_to_list, backoff_create

logger = logging.getLogger("aide")

_client: openai.OpenAI = None  # type: ignore
_vllm_config: dict = {
    "base_url": os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
    "api_key": os.getenv("VLLM_API_KEY", "EMPTY"),
}

# Define vLLM/OpenAI exceptions
VLLM_API_EXCEPTIONS = (
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIError,
    openai.InternalServerError,
)

def _setup_vllm_client():
    """Sets up the OpenAI client for vLLM server."""
    global _client
    if _client is None:
        logger.info(f"Setting up vLLM client with base_url: {_vllm_config['base_url']}")
        try:
            _client = openai.OpenAI(
                base_url=_vllm_config["base_url"],
                api_key=_vllm_config["api_key"],
                max_retries=0,  # Rely on backoff
            )
        except Exception as e:
            logger.error(f"Failed to setup vLLM client: {e}")
            raise

def set_vllm_config(cfg: OmegaConf):
    """Update vLLM config from OmegaConf."""
    global _vllm_config
    if cfg.get("vllm"):
        _vllm_config.update({
            "base_url": cfg.vllm.get("base_url", _vllm_config["base_url"]),
            "api_key": cfg.vllm.get("api_key", _vllm_config["api_key"]),
        })
    logger.info(f"Updated vLLM config: base_url={_vllm_config['base_url']}")

def _split_prompt(system_message: Optional[str], user_message: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Split a long system_message into system and user parts if user_message is None."""
    if user_message or not system_message:
        return system_message, user_message
    task_match = re.search(r"(# Task description[\s\S]*?)(# Instructions|$)", system_message, re.DOTALL)
    if task_match:
        system_part = system_message[:task_match.start()].strip()
        user_part = task_match.group(1).strip()
        logger.info("Split system_message into system and user parts")
        return system_part, user_part
    logger.warning("Could not split system_message; treating as system prompt only")
    return system_message, None

def query(
    system_message: Optional[str] = None,
    user_message: Optional[str] = None,
    model: str = "Qwen/Qwen2-0.5B-Instruct",
    temperature: float = 0.7,
    max_new_tokens: int = 200,
    **model_kwargs: Any,
) -> Tuple[OutputType, float, int, int, Dict[str, Any]]:
    """
    Query a vLLM-hosted model using OpenAI-compatible API.

    Args:
        system_message: Optional system prompt (e.g., role instructions).
        user_message: Optional user prompt (e.g., coding task).
        model: Model name (e.g., 'Qwen/Qwen2-0.5B-Instruct').
        temperature: Sampling temperature (default: 0.7).
        max_new_tokens: Maximum new tokens to generate (default: 200).
        **model_kwargs: Additional generation arguments.

    Returns:
        Tuple: (response, latency, input_tokens, output_tokens, metadata)
    """
    _setup_vllm_client()

    # Split prompt if needed
    system_message, user_message = _split_prompt(system_message, user_message)

    # Prepare messages
    messages = opt_messages_to_list(system_message, user_message, convert_system_to_user=False)

    # Filter kwargs and map to vLLM-compatible names
    filtered_kwargs = {
        k: v for k, v in model_kwargs.items()
        if k in {"temperature", "max_tokens", "top_p", "top_k", "stop"}
    }
    filtered_kwargs["model"] = model
    filtered_kwargs["temperature"] = temperature
    filtered_kwargs["max_tokens"] = max_new_tokens
    if model_kwargs != filtered_kwargs:
        logger.warning(f"Ignored invalid model_kwargs: {set(model_kwargs) - set(filtered_kwargs)}")

    # Perform API call
    t0 = time.time()
    try:
        completion = backoff_create(
            _client.chat.completions.create,
            VLLM_API_EXCEPTIONS,
            messages=messages,
            **filtered_kwargs,
        )
    except Exception as e:
        logger.error(f"vLLM query failed: {e}")
        raise ValueError(f"vLLM query failed: {str(e)}")

    req_time = time.time() - t0

    # Process response
    choice = completion.choices[0]
    output = choice.message.content or ""

    # Token counts (vLLM may not always return usage)
    input_tokens = completion.usage.prompt_tokens if completion.usage else 0
    output_tokens = completion.usage.completion_tokens if completion.usage else 0

    info = {
        "model": completion.model,
        "finish_reason": choice.finish_reason,
    }

    return output, req_time, input_tokens, output_tokens, info