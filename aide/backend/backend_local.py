import logging
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger("aide")

class LocalLLMManager:
    _cache = {}  # Cache to store loaded models

    @classmethod
    def get_model(cls, model_name: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        """Load or retrieve a model from cache with 4-bit quantization."""
        if model_name not in cls._cache:
            logger.info(f"Loading local model: {model_name}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                # Set padding token to avoid attention mask issues
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                )
                logger.info(f"Quantized (4-bit) local model '{model_name}' loaded successfully.")
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
    def _split_prompt(cls, system_message: Optional[str], user_message: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        """Split a long system_message into system and user parts if user_message is None."""
        if user_message or not system_message:
            return system_message, user_message

        # Heuristic: Split on '# Task description' or similar to isolate task
        task_match = re.search(r"(# Task description[\s\S]*?)(# Instructions|$)", system_message, re.DOTALL)
        if task_match:
            system_part = system_message[:task_match.start()]  # Up to task description
            user_part = task_match.group(1).strip()  # Task description and beyond
            logger.info("Split system_message into system and user parts")
            return system_part.strip(), user_part
        else:
            # Fallback: Use system_message as-is, warn about potential issues
            logger.warning("Could not split system_message; treating as system prompt only")
            return system_message, None

    @classmethod
    def generate_response(
        cls,
        model_name: str,
        system_message: Optional[str] = None,
        user_message: Optional[str] = None,
        temperature: float = 0.7,
        max_new_tokens: int = 200,
        **gen_kwargs: Any,
    ) -> Optional[str]:
        """Generate response with proper system/user prompt handling."""
        tokenizer, model = cls.get_model(model_name)
        try:
            # Split prompt if needed
            system_message, user_message = cls._split_prompt(system_message, user_message)

            # Construct messages
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            if user_message:
                messages.append({"role": "user", "content": user_message})
            elif not messages:
                raise ValueError("At least one of system_message or user_message must be provided")

            # Apply chat template
            if hasattr(tokenizer, "apply_chat_template"):
                prompt = tokenizer.apply_chat_template(messages, tokenize=False)
            else:
                prompt = f"{system_message or ''}\n\n{user_message or ''}".strip()

            # Check prompt length against model’s max context
            input_ids = tokenizer.encode(prompt, return_tensors="pt")
            prompt_length = input_ids.size(1)
            max_model_length = getattr(model.config, "max_position_embeddings", 2048)
            if prompt_length > max_model_length:
                raise ValueError(
                    f"Prompt length ({prompt_length} tokens) exceeds model’s max context "
                    f"({max_model_length} tokens). Use a larger model (e.g., Qwen2-0.5B) or shorten the prompt in agent.py."
                )

            # Move to device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            input_ids = input_ids.to(device)

            # Filter valid kwargs
            valid_generate_kwargs = {
                "max_length", "max_new_tokens", "min_length", "min_new_tokens",
                "do_sample", "temperature", "top_k", "top_p", "typical_p",
                "repetition_penalty", "no_repeat_ngram_size", "num_beams",
                "num_beam_groups", "diversity_penalty", "length_penalty",
                "early_stopping", "renormalize_logits", "logits_processor",
                "output_scores", "return_dict_in_generate", "forced_bos_token_id",
                "forced_eos_token_id", "output_hidden_states", "output_attentions",
                "num_return_sequences", "pad_token_id", "bos_token_id",
                "eos_token_id", "attention_mask", "use_cache"
            }
            filtered_kwargs = {k: v for k, v in gen_kwargs.items() if k in valid_generate_kwargs}
            if gen_kwargs != filtered_kwargs:
                logger.warning(f"Ignored invalid model_kwargs: {set(gen_kwargs) - set(filtered_kwargs)}")

            # Generate with attention mask
            attention_mask = input_ids.ne(tokenizer.pad_token_id).to(device)
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                **filtered_kwargs,
            )
            response = tokenizer.decode(output[0], skip_special_tokens=True)
            return response
        except Exception as e:
            logger.error(f"Error generating response for {model_name}: {e}")
            raise ValueError(f"Failed to generate response: {str(e)}")

def query(
    system_message: Optional[str] = None,
    user_message: Optional[str] = None,
    model: str = "HuggingFaceTB/SmolLM-135M-Instruct",
    temperature: float = 0.7,
    max_new_tokens: int = 2024,
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
    try:
        response = LocalLLMManager.generate_response(
            model_name=model,
            system_message=system_message,
            user_message=user_message,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            **model_kwargs,
        )
        logger.info(f"Generated response: {response[:100]}...")  # Truncate for logging
        return response, 0.0, 0, 0, {}
    except Exception as e:
        logger.error(f"Query failed for model {model}: {e}")
        return None, 0.0, 0, 0, {}