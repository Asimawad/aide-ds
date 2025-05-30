"""Backend for Ollama model API."""

import json
import logging
import time
from funcy import notnone, once, select_values
import openai
from dotenv import load_dotenv
import os

from aide.backend.utils import (
    FunctionSpec,
    OutputType,
    opt_messages_to_list,
    backoff_create,
)

logger = logging.getLogger("aide")
# load environemtn variables
load_dotenv()
_client: openai.OpenAI = None  # type: ignore

OLLAMA_API_EXCEPTIONS = (
    openai.AuthenticationError,  # Raised for authentication issues
    openai.APIConnectionError,  # Raised for connection issues
    openai.RateLimitError,  # Raised when rate limits are exceeded
    openai.APIError,  # Raised for general API errors
    openai.InternalServerError,
)


@once
def _setup_ollama_client():
    global _client
    _client = openai.OpenAI(
        base_url="http://localhost:11434/v1/", api_key="ollama", max_retries=0
    )


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    excute:bool=False,
    step_identifier = None,
    top_p :float = 0.9,
    num_responses:int=1,
    top_k:int=50,
    convert_system_to_user: bool = False,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    _setup_ollama_client()
    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore
    print(filtered_kwargs)
    # {'temperature': 0.6, 'top_p': 0.9, 'top_k': 50, 'num_responses': 1, 'model': 'deepseek-r1:latest'}
    messages = opt_messages_to_list(
        system_message, user_message, convert_system_to_user=convert_system_to_user
    )
    func_spec = None # edit this if you find a model that supports using tools
    t0 = time.time()
    if func_spec is not None:
        _tools = [
            {
                "type": "function",
                "function": {
                    "name": func_spec.name,
                    "description": func_spec.description,
                    "parameters": func_spec.json_schema,
                },
                "strict": True,
            }
        ]

        # force the model the use the function
        _tool_choice = [{"type": "function", "function": {"name": func_spec.name}}]
        #  generate output with function call
        completion = backoff_create(
            _client.chat.completions.create,
            OLLAMA_API_EXCEPTIONS,
            messages=messages,
            tools=_tools,
            tool_choice=_tool_choice,
            **filtered_kwargs,
        )
        choice = completion.choices[0]

        assert (
            choice.message.tool_calls
        ), f"function_call is empty, it is not a function call: {choice.message}"
        assert (
            choice.message.tool_calls[0].function.name == func_spec.name
        ), "Function name mismatch"
        try:
            output = json.loads(choice.message.tool_calls[0].function.arguments)
        except json.JSONDecodeError as e:
            logger.error(
                f"Error decoding the function arguments: {choice.message.tool_calls[0].function.arguments}"
            )
            raise e
    else:
        completion = backoff_create(
            _client.chat.completions.create,
            OLLAMA_API_EXCEPTIONS,
            messages=messages,
            **filtered_kwargs,
        )
        req_time = time.time() - t0

        choice = completion.choices[0]
        output = choice.message.content
        with open("output.json", "w") as f:
            json.dump(messages[0], f, indent=4)
        logger.info(f"output: {output}")

    in_tokens = completion.usage.prompt_tokens
    out_tokens = completion.usage.completion_tokens

    info = {
        "system_fingerprint": completion.system_fingerprint,
        "model": completion.model,
        "created": completion.created,
    }

    return output, req_time, in_tokens, out_tokens, info