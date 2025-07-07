# Installation Guide

## System Requirements

- Python 3.11
- CUDA 11.8+ or 12.0+ (for GPU support)
- 16GB+ RAM
- 20GB+ GPU memory (for full model) or 12GB+ (with 8-bit quantization)

## Installation Methods

### Using pip (Recommended)

```bash
# For CUDA 12.x
pip install -e ".[cuda12]"

# For CUDA 11.x  
pip install -e ".[cuda11]"

# For CPU only
pip install -e .

# For development
pip install -e ".[dev]"

# For documentation
pip install -e ".[docs]"
```

### Using UV (Faster)

```bash
# Install UV first
curl -LsSf https://astral.sh/uv/install.sh | sh

# Then install the package
uv pip install -e ".[cuda12]"
```

## Verify Installation

```bash
# Check installation
python -c "from src import __version__; print(f'Mistral NER v{__version__}')"

# Run tests
pytest tests/
```

## Troubleshooting

See our [Troubleshooting Guide](../reference/troubleshooting.md) for common installation issues.