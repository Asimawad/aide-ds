import os
import logging
import torch
from typing import Optional, Dict, Any, Tuple
import sys
import shutil
from pprint import pprint
from pathlib import Path
import time
from aide.backend.utils import _split_prompt,opt_messages_to_list

# 1. Turn off HF‑hub / safetensors tqdm bars entirely
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from rich.console import Console
from rich.syntax import Syntax



try:
    from aide.interpreter import Interpreter, ExecutionResult
    from aide.utils.response import extract_code, format_code, wrap_code
    from aide.utils import serialize # For saving ExecutionResult
except ImportError as e:
    print(f"Error importing AIDE modules: {e}")
    print("Please ensure AIDE is installed (e.g., `pip install -e .` from aide-ds root) or paths are correct.")
    sys.exit(1)
    
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))


logger = logging.getLogger("aide")
console = Console()

class LocalLLMManager:
    _cache = {}  # Cache to store loaded models

    @classmethod
    def get_model(cls, model_name: str,load_in_4bit:bool = True) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        
        """Load or retrieve a model from cache with/without 4-bit quantization."""
        if model_name not in cls._cache:
            cls.clear_cache()
            logger.info(f"Loading local model: {model_name}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
                # Set padding token to avoid attention mask issues
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")

                tokenizer.padding_side = "left" # Important for batch generation if ever implemented
                quantization_config = None
                if load_in_4bit and torch.cuda.is_available():
                    quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    )
                    logger.info("Using 4-bit quantization (BitsAndBytesConfig).")

                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config, #load unquantized model
                    device_map="auto",
                    torch_dtype=torch.bfloat16 if quantization_config else None,
                    trust_remote_code=True,
                )
                if load_in_4bit:
                    logger.info(f"Quantized (4-bit) local model '{model_name}' loaded successfully.")
                else:
                    logger.info(f"Unquantized local model '{model_name}' loaded successfully.")
                # Move model to GPU if available
                cls._cache[model_name] = (tokenizer, model)
            except Exception as e:
                logger.error(f"Failed to load local model {model_name}: {e}")
                raise
        return cls._cache[model_name]

    @classmethod
    def clear_cache(cls, model_name: Optional[str] = None) -> None:
        """Clear specific model or entire cache to free memory."""
        if model_name:
            cls._cache.pop(model_name, None)
            logger.info(f"Cleared cache for model: {model_name}")
        else:
            cls._cache.clear()
            logger.info("Cleared entire model cache")
    @classmethod
    def generate_response(
        cls,
        model_name: str,
        prompt:str,
        tokenizer :AutoTokenizer ,
        model:AutoModelForCausalLM,
        system_message: Optional[str] = None,
        user_message: Optional[str] = None,
        num_responses:int = 1,
        **gen_kwargs: Any,
    ) -> Optional[str]:
        """Generate response with proper system/user prompt handling."""
        
        # Check prompt length against model’s max context
        inputs = tokenizer(prompt, return_tensors="pt",return_attention_mask=True)
        input_ids= inputs['input_ids'].to(model.device)
        attention_mask = inputs['attention_mask'].to(model.device)
        prompt_length = input_ids.shape[1]        
        
        gen_kwargs = {
            "temperature": gen_kwargs.get("temperature", 0.6),
            "max_new_tokens": gen_kwargs.get("max_new_tokens", 512),
            "top_p": gen_kwargs.get("top_p",0.9),
            "top_k": gen_kwargs.get("top_k",50),
            "repetition_penalty": gen_kwargs.get("repetition_penalty"),
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "do_sample": gen_kwargs.get("temperature", 0.6) > 0.0,
            "num_return_sequences": num_responses,
        }
        # Filter out None values
        gen_kwargs      = {k: v for k, v in gen_kwargs.items() if v is not None}
        
        t0 = time.time()
        outputs = []
        console.rule(f"[bold blue]Generating Responses")

        try:
            # Generate with attention mask
            with torch.no_grad():
                generated_outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs,
                )
            logger.info("finished generation")
            # Decode each generated sequence, removing the prompt part
            for i in range(num_responses):
                # generated_outputs shape is (num_responses, seq_len)
                output_ids = generated_outputs[i,prompt_length:]
                output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
                outputs.append(output_text.strip())

        except Exception as e:
            logger.error(f"Error generating response for {model_name}: {e}")
            raise ValueError(f"Failed to generate response: {str(e)}")

        t1 = time.time()
        latency = t1-t0
        logger.info(f"Batch generation of {num_responses} responses and {len(output_ids)} tokens took {t1-t0:.2f} seconds.")
        return outputs,prompt_length,len(output_ids),latency
def query(
    system_message: Optional[str] = None,
    user_message: Optional[str] = None,
    model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    func_spec=None,
    convert_system_to_user=False,
    num_responses:int = 1,
    **model_kwargs: Any,
) -> Tuple[Optional[str], float, int, int, Dict[str, Any]]:
    """
    Query the local model with system and user messages.

    Args:
        system_message: Optional system prompt (e.g., role instructions).
        user_message: Optional user prompt (e.g., coding task).
        model: Model name (e.g., 'HuggingFaceTB/SmolLM-135M-Instruct').
        temperature: Sampling temperature (default: 0.7).
        max_new_tokens: Maximum new tokens to generate (default: 200).
        **model_kwargs: Additional generation arguments.

    Returns:
        Tuple: (response, latency, input_tokens, output_tokens, metadata)
    """
    print(f"Models Kwargs are {model_kwargs}")
    tokenizer, model_instance = LocalLLMManager.get_model(model)
    # console.rule(f"[bold blue] Stage 1 ")
    console.rule(f"[bold red]System Prompt")
    logger.info(f"{system_message} ")
    console.rule(f"[bold red]User Prompt")
    logger.info(f" {user_message} ")
        
    messages = opt_messages_to_list(
        system_message, user_message)

    # Apply chat template
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        logger.info("Applied chat template to prompt.")
    else:
        prompt = f"{system_message or ''}\n\n{user_message or ''}".strip()

    logger.debug(f"Generating with params:  num_return_sequences = {num_responses}, {model_kwargs}")
    logger.debug(f"_______________________________________________________________________________")
    # logger.info(f"Input prompt (start):\n{prompt}...")
    try:
        responses,input_len,output_len,latency = LocalLLMManager.generate_response(
            model_name=model,
            tokenizer=tokenizer, 
            model=model_instance,
            prompt=prompt,
            num_responses =num_responses,
            do_sample=False,
            **model_kwargs,
        )
        first_response = responses[0] 
 
        for i,r in enumerate(responses):
            console.rule(f"[bold red]Response {i}")
            logger.info(f"Generated response: {r}")
        return first_response, latency, input_len,output_len, {}
    except Exception as e:
        logger.error(f"Query failed for model {model}: {e}")
        return None, latency, input_len,output_len, {}