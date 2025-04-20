import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import shutil # Import shutil for file operations
from pprint import pprint # Import pprint for pretty-printing

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
    TextGenerationPipeline,
)
from rich.console import Console
from rich.syntax import Syntax

# import AIDE components -
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root)) # Add aide-ds root to path

try:
    from aide.interpreter import Interpreter, ExecutionResult
    from aide.utils.response import extract_code, format_code, wrap_code
    from aide.utils import serialize # For saving ExecutionResult
except ImportError as e:
    print(f"Error importing AIDE modules: {e}")
    print("Please ensure AIDE is installed (e.g., `pip install -e .` from aide-ds root) or paths are correct.")
    sys.exit(1)

# --- Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("PromptPlayground")
console = Console()

# Add a file handler to save logs to a file
log_file_path = project_root / "playground.log"
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s'))
logger.addHandler(file_handler)

# --- Model Loading ---
MODEL_CACHE: Dict[str, Tuple[Any, Any]] = {} # model_name -> (model, tokenizer)

def load_model_and_tokenizer(model_id: str, load_in_4bit: bool = True) -> Tuple[Any, Any]:
    """Loads model and tokenizer, caching them."""
    if model_id in MODEL_CACHE:
        logger.info(f"Using cached model/tokenizer for {model_id}")
        return MODEL_CACHE[model_id]

    logger.info(f"Loading model and tokenizer: {model_id} (4-bit: {load_in_4bit})")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Set tokenizer pad_token to eos_token: {tokenizer.eos_token}")
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
        elif load_in_4bit:
            logger.warning("CUDA not available, cannot load in 4-bit. Loading in default precision.")

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16 if quantization_config else None,
            trust_remote_code=True,
        )
        logger.info(f"Model '{model_id}' loaded successfully.")
        MODEL_CACHE[model_id] = (model, tokenizer)
        return model, tokenizer
    except Exception as e:
        logger.exception(f"Failed to load model {model_id}")
        raise
    return None, None # Should not be reached if exception is raised

# --- Prompt Parsing ---
def parse_prompt_file(filepath: Path) -> Tuple[Optional[str], str]:
    """Parses prompt file into system and user parts."""
    try:
        content = filepath.read_text()
        separator = "---USER---"
        if separator in content:
            system_prompt, user_prompt = content.split(separator, 1)
            return system_prompt.strip(), user_prompt.strip()
        else:
            logger.warning(f"Separator '{separator}' not found in {filepath}. Using entire content as user prompt.")
            return None, content.strip()
    except FileNotFoundError:
        logger.error(f"Prompt file not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error reading prompt file {filepath}: {e}")
        raise

# --- Code Generation ---
def generate_code(
    model: Any,
    tokenizer: Any,
    system_prompt: Optional[str],
    user_prompt: str,
    generation_params: Dict[str, Any]
) -> str:
    """Generates code using the loaded model and tokenizer."""
    # Use chat template if available and makes sense
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    try:
        # Use apply_chat_template for instruction/chat models
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        logger.info("Applied chat template to prompt.")
        pprint(prompt_text)
    except Exception as e:
        logger.warning(f"Could not apply chat template (model might not support it or error: {e}). Using basic concatenation.")
        prompt_text = (system_prompt + "\n\n" if system_prompt else "") + user_prompt

    logger.debug(f"Generating with params: {generation_params}")
    logger.info(f"Input prompt (start):\n{prompt_text[:300]}...")

    # Use pipeline for easier generation handling
    # Note: pipeline might re-load model if not managed carefully, but direct .generate is fine too.
    # Using direct generate here for more control matching backend_local
    inputs = tokenizer(prompt_text, return_tensors="pt", return_attention_mask=True)
    
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    gen_kwargs = {
        "temperature": generation_params.get("temperature", 0.2),
        "max_new_tokens": generation_params.get("max_new_tokens", 2048),
        "top_p": generation_params.get("top_p"),
        "top_k": generation_params.get("top_k"),
        "repetition_penalty": generation_params.get("repetition_penalty"),
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": generation_params.get("temperature", 0.2) > 0.0,
    }
    # Filter out None values
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs
        )
    
    t1 = time.time()
    logger.info(f"Generation took {t1-t0:.2f} seconds.")

    # Decode only the newly generated tokens
    output_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    return output_text.strip()

# --- Code Generation (Batched) ---
def generate_code_batch(
    model: Any,
    tokenizer: Any,
    system_prompt: Optional[str],
    user_prompt: str,
    num_responses: int,
    generation_params: Dict[str, Any]
) -> List[str]:
    """Generates a batch of code responses using the loaded model."""
    # Prepare the single prompt text using chat template if possible
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    try:
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        logger.info("Applied chat template to prompt for batch generation.")
    except Exception as e:
        logger.warning(f"Could not apply chat template for batch generation (error: {e}). Using basic concatenation.")
        prompt_text = (system_prompt + "\n\n" if system_prompt else "") + user_prompt

    logger.info(f"Input prompt (start) for batch generation:\n{prompt_text[:300]}...")

    # Tokenize the single prompt
    inputs = tokenizer(prompt_text, return_tensors="pt", padding=True, truncation=False) # No truncation here
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    prompt_token_length = input_ids.shape[1]

    # Prepare generation arguments for batch generation
    do_sample = generation_params.get("temperature", 0.2) > 0.0 or generation_params.get("top_p", None) is not None or generation_params.get("top_k", None) is not None
    if num_responses > 1 and not do_sample:
        logger.warning("Generating multiple responses requested, but temperature is 0 and top_p/top_k are not set. Responses will likely be identical. Enabling sampling with default temperature.")
        generation_params["temperature"] = 0.2 # Force some sampling
        do_sample = True

    gen_kwargs = {
        "temperature": generation_params.get("temperature", 0.2),
        "max_new_tokens": generation_params.get("max_new_tokens", 2048),
        "top_p": generation_params.get("top_p"),
        "top_k": generation_params.get("top_k"),
        "repetition_penalty": generation_params.get("repetition_penalty"),
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
        "num_return_sequences": num_responses, # Generate multiple sequences
    }
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None} # Filter None

    logger.info(f"Generating {num_responses} responses with config: {gen_kwargs}")

    t0 = time.time()
    outputs = []
    try:
        with torch.no_grad():
            generated_outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs
            )

        # Decode each generated sequence, removing the prompt part
        for i in range(num_responses):
            # generated_outputs shape is (num_responses, seq_len)
            output_ids = generated_outputs[i][prompt_token_length:]
            output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
            outputs.append(output_text.strip())

    except Exception as e:
        logger.exception("Error during batch generation")
        # Return list of error messages or empty list? Let's return errors.
        return [f"Error during generation: {e}"] * num_responses

    t1 = time.time()
    logger.info(f"Batch generation of {num_responses} responses took {t1-t0:.2f} seconds.")
    return outputs


# --- Main Execution Logic ---
def run_playground(args):
    """Runs the prompt testing and code execution loop."""
    model, tokenizer = load_model_and_tokenizer(args.model_id, not args.no_4bit)
    system_prompt, user_prompt = parse_prompt_file(Path(args.prompt_file))

    output_base_dir = Path(args.output_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    generation_params = {
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
    }

    console.rule(f"[bold blue]Generating {args.num_responses} Responses (Batched)")
    all_raw_responses = generate_code_batch(
        model, tokenizer, system_prompt, user_prompt,
        args.num_responses, generation_params
    )

    if not all_raw_responses:
        logger.error("Batch generation failed to produce any responses.")
        return

    for i, raw_response in enumerate(all_raw_responses):
        console.rule(f"[bold blue]Processing Response {i+1}/{args.num_responses}")
        response_dir = output_base_dir / f"response_{i}"
        response_dir.mkdir(exist_ok=True)

        # Setup workspace for this response's execution
        workspace_dir = response_dir / "workspace"
        if workspace_dir.exists(): # Clean up previous run for this index if any
            shutil.rmtree(workspace_dir)
        workspace_dir.mkdir(exist_ok=True)
        input_dir = workspace_dir / "input"
        input_dir.mkdir(exist_ok=True)
        # Create dummy input files for basic testing
        (input_dir / "train.csv").write_text("id,feature1,feature2,target\n0,1,1,0\n1,2,2,1")
        (input_dir / "test.csv").write_text("id,feature1,feature2\n100,3,3\n101,4,4")
        (input_dir / "sample_submission.csv").write_text("id,target\n100,0\n101,0")

        submission_dir = workspace_dir / "submission" # Interpreter expects this structure
        submission_dir.mkdir(exist_ok=True)
        working_dir = workspace_dir / "working"
        working_dir.mkdir(exist_ok=True)

        # Save Raw Response
        raw_response_path = response_dir / "raw_response.txt"
        raw_response_path.write_text(raw_response)
        logger.info(f"Raw response saved to: {raw_response_path}")
        console.print(f"[bold cyan]Raw Response {i+1}:[/bold cyan]")
        console.print(raw_response[:1000] + ("..." if len(raw_response) > 1000 else ""))
        console.print("-" * 20)

        # Extract and Save Code
        extracted_code = extract_code(raw_response)
        if not extracted_code:
            logger.error(f"Response {i+1}: Could not extract valid Python code.")
            console.print(f"[bold red]Response {i+1}: Code extraction FAILED.[/bold red]")
            # Create an empty execution log
            exec_log_path = response_dir / "execution_log.json"
            error_result = ExecutionResult(term_out=["Code extraction failed."], exec_time=0, exc_type="ExtractionError")
            try:
                 serialize.dump_json(error_result, exec_log_path)
            except Exception as e:
                 exec_log_path.with_suffix(".txt").write_text(str(error_result))
            continue # Skip execution for this response

        formatted_extracted_code = format_code(extracted_code)
        code_path = response_dir / "extracted_code.py"
        code_path.write_text(formatted_extracted_code)
        logger.info(f"Extracted code saved to: {code_path}")
        console.print(f"[bold green]Extracted Code {i+1}:[/bold green]")
        console.print(Syntax(formatted_extracted_code, "python", theme="default", line_numbers=True))
        console.print("-" * 20)

        # Execute code
        logger.info(f"Executing code for response {i+1}...")
        interpreter = Interpreter(
            working_dir=workspace_dir,
            timeout=args.execution_timeout
        )
        try:
            exec_result: ExecutionResult = interpreter.run(formatted_extracted_code, reset_session=True)
        except Exception as e:
             logger.error(f"Interpreter failed for response {i+1}: {e}", exc_info=True)
             exec_result = ExecutionResult(term_out=[f"Interpreter Error: {e}"], exec_time=0, exc_type=type(e).__name__)
        finally:
            # Ensure interpreter process is cleaned up even if run fails
            try:
                interpreter.cleanup_session()
            except Exception as e_clean:
                 logger.error(f"Error cleaning up interpreter for response {i+1}: {e_clean}")

        # Save execution result
        exec_log_path = response_dir / "execution_log.json"
        try:
             serialize.dump_json(exec_result, exec_log_path)
             logger.info(f"Execution log saved to: {exec_log_path}")
        except Exception as e:
             logger.error(f"Failed to save execution log: {e}")
             exec_log_path.with_suffix(".txt").write_text(str(exec_result))

        # Print execution summary
        console.print(f"[bold magenta]Execution Result {i+1}:[/bold magenta]")
        console.print(f"  Success: {exec_result.exc_type is None}")
        console.print(f"  Execution Time: {exec_result.exec_time:.2f}s")
        if exec_result.exc_type:
            console.print(f"  Error Type: [bold red]{exec_result.exc_type}[/bold red]")
        console.print(f"  Terminal Output (preview):")
        # Ensure term_out is treated as a list of strings
        term_out_str = "\n".join(exec_result.term_out) if isinstance(exec_result.term_out, list) else str(exec_result.term_out)
        console.print("[dim]" + term_out_str[:500] + ("\n..." if len(term_out_str) > 500 else "") + "[/dim]")
        console.print(f"  Full output/logs saved in: {response_dir}")
        console.print("-" * 20)

    logger.info("Playground run finished.")

# --- Argument Parser ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIDE Prompt Playground with Batched Inference")
    parser.add_argument("--model_id", type=str, required=True, help="Hugging Face model ID (e.g., 'deepseek-ai/deepseek-coder-7b-instruct-v1.5')")
    parser.add_argument("--prompt_file", type=str, required=True, help="Path to the prompt text file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs for each response.")
    parser.add_argument("--num_responses", type=int, default=1, help="Number of responses to generate IN ONE BATCH.") # Clarified help
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature (must be >0 for multiple different responses).")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Max new tokens to generate.")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p (nucleus) sampling.")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling.")
    parser.add_argument("--repetition_penalty", type=float, default=None, help="Repetition penalty.")
    parser.add_argument("--execution_timeout", type=int, default=60, help="Timeout for code execution in seconds.")
    parser.add_argument("--no_4bit", action="store_true", help="Disable 4-bit quantization.")

    args = parser.parse_args()

    if args.num_responses > 1 and args.temperature == 0.0 and args.top_p is None and args.top_k is None:
         logger.warning("num_responses > 1 but temperature is 0 and top_p/top_k are not set. Forcing temperature to 0.2 to enable sampling diverse responses.")
         args.temperature = 0.2

    run_playground(args)

# END OF MODIFIED FILE aide-ds/prompt_playground/prompt_playground.py

"""python prompt_playground/prompt_playground.py --model_id='deepseek-ai/deepseek-coder-7b-instruct-v1.5' --prompt_file=prompt_playground/real_prompt.txt --output_dir="prompt_playground"
"""
# python prompt_playground/prompt_playground.py --model_id='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B' --prompt_file=prompt_playground/real_prompt.txt --output_dir="prompt_playground"
