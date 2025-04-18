#!/bin/bash

# Start Ollama server in the background
echo "Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!
echo "Ollama server started with PID: $OLLAMA_PID"

# Start vLLM server in the background
echo "Starting vLLM server..."
.aide-ds/bin/python -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --port 8000 \
    --dtype bfloat16 \
    --device cuda &
VLLM_PID=$!
echo "vLLM server started with PID: $VLLM_PID"

# Wait for servers to initialize
echo "Waiting for servers to initialize..."
sleep 5

# Debug: Print current working directory and list files
echo "Current working directory: $(pwd)"
echo "Files in current directory:"
ls -la

# Debug: Print the command to be executed
echo "Executing command: $@"

# Execute the command passed to the container
exec "$@"