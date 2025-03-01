"""Backend for Ollama model API."""

import json
import logging
import time
from funcy import notnone, once, select_values
import openai


from aide.backend.utils import (FunctionSpec,    OutputType,    opt_messages_to_list,    backoff_create)

logger = logging.getLogger("aide")

_client: openai.OpenAI = None  # type: ignore
_client = openai.OpenAI(base_url= 'http://localhost:11434/v1/' , api_key='ollama',max_retries=0)


OLLAMA_API_EXCEPTIONS = (
    # openai.InvalidRequestError,  # Raised for invalid API requests
    openai.AuthenticationError,  # Raised for authentication issues
    openai.APIConnectionError,   # Raised for connection issues
    openai.RateLimitError,       # Raised when rate limits are exceeded
    openai.APIError,             # Raised for general API errors
    openai.InternalServerError
)

@once
def _setup_ollama_client():
    global _client
    client = openai.OpenAI(base_url= 'http://localhost:11434/v1/' , api_key='ollama',max_retries=0)



def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    convert_system_to_user: bool = False,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    _setup_ollama_client()
    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore

    messages = opt_messages_to_list(system_message, user_message, convert_system_to_user=convert_system_to_user)
    
    # func_spec = None # edit this if you find a model that supports using tools
    if func_spec is not None:
        _tools = [{
            "type": "function",
            "function": {
                "name": func_spec.name,
                "description": func_spec.description,
                "parameters": func_spec.json_schema,
            },
            "strict": True}]
        # [func_spec.as_openai_tool_dict]
        print("tooooooools" , _tools)

        # force the model the use the function
        _tool_choice = [{
            "type": "function",
            "function": {"name": func_spec.name}} ]
        # [func_spec.openai_tool_choice_dict]
        print("______________________________________________________________","\n",_tool_choice)


    t0 = time.time()
    if func_spec is not None:
        # with open('out.txt', 'wb') as f:
            
        #     f.write(messages)
        print(messages)
        completion = backoff_create(
        _client.chat.completions.create,
        OLLAMA_API_EXCEPTIONS,
        messages=messages,
        tools = _tools,
        tool_choice=_tool_choice,
        **filtered_kwargs,
    )
    else:
        completion = backoff_create(
        _client.chat.completions.create,
        OLLAMA_API_EXCEPTIONS,
        messages=messages,
        **filtered_kwargs,
    )
    req_time = time.time() - t0

    choice = completion.choices[0]
    print(choice)
    if func_spec is None:
        output = choice.message.content
        print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOo")
        print(messages,"\n",output) # remove later
        with open("output.json", "w") as f:
            json.dump(messages[0], f, indent=4)
        
        print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOo")
    else:
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

    in_tokens = completion.usage.prompt_tokens
    out_tokens = completion.usage.completion_tokens

    info = {
        "system_fingerprint": completion.system_fingerprint,
        "model": completion.model,
        "created": completion.created,
    }

    return output, req_time, in_tokens, out_tokens, info


def fun():
# """Backend for HF API."""

# import json
# import logging
# import time

# from aide.backend.utils import (
#     FunctionSpec,
#     OutputType,
#     opt_messages_to_list,
#     backoff_create,
# )
# from funcy import notnone, once, select_values
# import openai

# logger = logging.getLogger("aide")

# _client: openai.OpenAI = None  # type: ignore


# OPENAI_TIMEOUT_EXCEPTIONS = (
#     openai.RateLimitError,
#     openai.APIConnectionError,
#     openai.APITimeoutError,
#     openai.InternalServerError,
# )

# @once
# def _setup_openai_client():
#     global _client
#     # Build the kwargs for the client; if any unsupported keys (e.g. "proxies") exist, remove them.
#     client_kwargs = {
#         "base_url": "https://router.huggingface.co/hf-inference/v1",
#         "api_key": "HF_TOKEN",
#     }
#     # Ensure no "proxies" key is passed.
#     client_kwargs.pop("proxies", None)
#     _client = openai.OpenAI(**client_kwargs)

# def query(
#     system_message: str | None,
#     user_message: str | None,
#     func_spec: FunctionSpec | None = None,
#     convert_system_to_user: bool = False,
#     **model_kwargs,
# ) -> tuple[OutputType, float, int, int, dict]:
#     _setup_openai_client()
#     filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore

#     messages = opt_messages_to_list(system_message, user_message, convert_system_to_user=convert_system_to_user)

#     t0 = time.time()

#     completion = backoff_create(
#         _client.chat.completions.create,
#         OPENAI_TIMEOUT_EXCEPTIONS,
#         messages=messages,
#         **filtered_kwargs,
#     )
#     req_time = time.time() - t0

#     choice = completion.choices[0]
#     output = choice.message.content

#     in_tokens = completion.usage.prompt_tokens
#     out_tokens = completion.usage.completion_tokens

#     info = {
#         "system_fingerprint": completion.system_fingerprint,
#         "model": completion.model,
#         "created": completion.created,
#     }

#     return output, req_time, in_tokens, out_tokens, info
    pass