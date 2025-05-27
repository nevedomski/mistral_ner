# Setup Guide for mistral-ner

This guide helps you set up the mistral-ner project for GPU-accelerated training.

## Requirements

- NVIDIA GPU with CUDA support
- CUDA toolkit installed (11.x or 12.x)
- Python 3.11

## Quick Start

### 1. Detect Your Environment

Run the environment detection script:

```bash
python setup_env.py
```

This will detect your CUDA version and recommend the appropriate installation command.

### 2. Installation Options

#### Option A: Using UV (Recommended)

Based on your CUDA version, use one of these commands:

**CUDA 11.x:**

```bash
uv pip install .[cuda11]
```

**CUDA 12.x:**

```bash
uv pip install .[cuda12]
```

**Generic CUDA (auto-detect):**

```bash
uv pip install .[cuda]
```

#### Option B: Using Conda

For conda environments, use the appropriate environment file:

**CUDA 11.x:**

```bash
conda env create -f environment-cuda11.yml
conda activate mistral-ner
```

**CUDA 12.x:**

```bash
conda env create -f environment-cuda12.yml
conda activate mistral-ner
```

## GPU Requirements

This project requires a GPU with CUDA support for:

- 8-bit quantization via BitsAndBytes
- Efficient training with mixed precision (FP16/BF16)
- Memory-efficient fine-tuning with LoRA

## Verifying Your Installation

After installation, verify your setup:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import bitsandbytes; print('BitsAndBytes: Installed')"
```

## Training

Once installed, start training:

```bash
# Single GPU
python finetune_conll2023.py

# Multi-GPU
accelerate launch finetune_conll2023.py
# or
torchrun --nproc_per_node=NUM_GPUS finetune_conll2023.py
```

## Troubleshooting

### CUDA Version Mismatch

- Re-run `python setup_env.py` to detect the correct version
- Use the conda environment files for better isolation

### Out of Memory (OOM) Errors

- Reduce batch size in `finetune_conll2023.py`
- Increase gradient accumulation steps
- Use a GPU with more VRAM (minimum 16GB recommended)

### BitsAndBytes Import Error

- Ensure you have a compatible CUDA version
- Check that your GPU supports INT8 operations

## Advanced Configuration

### Custom CUDA Versions

For specific CUDA versions not covered, modify `pyproject.toml`:

```toml
[project.optional-dependencies]
cuda_custom = [
    "torch>=2.7.0+cuXXX",  # Replace XXX with your CUDA version
    "torchaudio>=2.7.0+cuXXX",
    "torchvision>=0.22.0+cuXXX",
    "bitsandbytes>=0.41.0",
]
```

Then install with:

```bash
uv pip install .[cuda_custom] --extra-index-url https://download.pytorch.org/whl/cuXXX
```

### Docker Support

For consistent environments across systems, consider using Docker:

```dockerfile
# For CUDA 12.1
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3.11 python3-pip
COPY . /app
WORKDIR /app
RUN pip install uv && uv pip install .[cuda12]
```

## GPU Memory Requirements

- Minimum: 16GB VRAM (with 8-bit quantization)
- Recommended: 24GB+ VRAM for comfortable training
- Multi-GPU: Scales linearly with number of GPUs