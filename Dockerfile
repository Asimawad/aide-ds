# Stage 1: Install VS Code CLI
FROM alpine/curl AS vscode-installer
RUN mkdir /aichor && \
    curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output /aichor/vscode_cli.tar.gz && \
    tar -xf /aichor/vscode_cli.tar.gz -C /aichor

# Stage 2: Main Project Setup
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04 AS base

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-distutils python3-pip git \
    build-essential pkg-config libhdf5-dev curl sudo nano wget unzip software-properties-common ca-certificates 

# Ensure `python` points to Python 3.11
RUN ln -sf /usr/bin/python3.11 /usr/bin/python

# Install `uv` (a wrapper for pip)
COPY --from=ghcr.io/astral-sh/uv:0.4.20 /uv /bin/uv
ENV UV_SYSTEM_PYTHON=1

# Set environment variables
ENV TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126

RUN git clone https://github.com/Asimawad/aide-ds aide-ds

# Set working directory
WORKDIR /aide-ds
RUN echo "Current working directory: $(pwd)"


# Create and use virtual environment
RUN uv venv .aide-ds --python 3.11 && \
    export DEBIAN_FRONTEND=noninteractive && \ 
    /bin/uv pip install --python .aide-ds/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu126 && \
    /bin/uv pip install --python .aide-ds/bin/python transformers bitsandbytes s3fs accelerate && \
    /bin/uv pip install --python .aide-ds/bin/python -e .

# Install Ollama using the official install script
RUN curl -fsSL https://ollama.com/install.sh | sh
EXPOSE 11434


 # Copy your project files into the container
COPY . .


# Copy the VS Code CLI binary from the first stage
COPY --from=vscode-installer /aichor /aichor

# Copy the entrypoint script that starts Ollama
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
RUN chmod +x ./run.sh

# Set the entrypoint to start Ollama -or anything and then run the container’s command
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]


# Default command: launch an interactive bash shell - in case of using vscode cli
CMD ["/bin/bash"]