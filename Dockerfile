ARG CUDA_VERSION=12.1.0
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu22.04

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY src/ ./src/
COPY api/ ./api/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Install dependencies with CUDA support
RUN python3.11 -m venv .venv && \
    .venv/bin/pip install --upgrade pip && \
    .venv/bin/pip install -e ".[api]" && \
    .venv/bin/pip install -e ".[cuda12]" --extra-index-url https://download.pytorch.org/whl/cu121

# Download model (optional - can be mounted as volume instead)
# RUN python -c "from transformers import AutoModelForTokenClassification, AutoTokenizer; \
#     AutoModelForTokenClassification.from_pretrained('mistralai/Mistral-7B-v0.3'); \
#     AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.3')"

# Expose port
EXPOSE 8000

# Run API server
CMD [".venv/bin/python", "-m", "uvicorn", "api.serve:app", "--host", "0.0.0.0", "--port", "8000"]