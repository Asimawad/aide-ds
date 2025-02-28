import logging
import time
from funcy import notnone, once, select_values
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

_client : pipeline = None
@once
def _setup_local_client(model_name):
    """Loads the model and tokenizer once."""
    print("Loading model... This may take a while.")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/deepseek-llm-7b-chat",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-chat")

    _client = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128
    )
    
    print("Model loaded successfully!")
    return _client

"""Backend for HF API."""

from aide.backend.utils import (
    FunctionSpec,
    OutputType,
    opt_messages_to_list,
    backoff_create,
)


logger = logging.getLogger("aide")

def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    convert_system_to_user: bool = False,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    _setup_local_client()
    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore

    messages = opt_messages_to_list(system_message, user_message, convert_system_to_user=convert_system_to_user)
    t0 = time.time()
    # edit this
    completion = _client(messages, max_new_tokens=512, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1, **model_kwargs)
    
    req_time = time.time() - t0

    choice = completion.choices[0]
    output = choice.message.content
    # Token counts (approximate)
    # in_tokens = len(input_text.split())  # Token count estimation for input
    out_tokens = len(output.split())  # Token count estimation for output

    info = {
        "model": "deepseek-ai/deepseek-llm-7b-chat",  # Hardcoded for now, replace if needed
        "created": time.time(),
    }

    # return output, req_time, in_tokens, out_tokens, info
