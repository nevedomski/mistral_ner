# Mistral NER Fine-tuning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Fine-tune Mistral-7B for Named Entity Recognition (NER) using 8-bit quantization and LoRA for memory-efficient training. Supports multi-dataset training with 9 built-in datasets and flexible label mapping for unified entity schemas.

## Features

- **Memory-efficient training**: 8-bit quantization + LoRA reduces memory usage by ~75%
- **Multi-dataset training**: Train on multiple NER datasets simultaneously with automatic label unification
- **Flexible label mapping**: Profile-based, file-based, or inline label mappings for dataset compatibility
- **9 built-in datasets**: Support for CoNLL-2003, OntoNotes, WNUT-17, Few-NERD, WikiNER, and 4 PII datasets
- **Modular architecture**: Clean separation of concerns for easy customization
- **WandB integration**: Optional experiment tracking (can be disabled)
- **Checkpoint resumption**: Resume training from any checkpoint
- **Multi-GPU support**: Distributed training with Accelerate
- **REST API**: FastAPI server for model serving
- **Comprehensive configuration**: YAML-based config with CLI overrides

## Project Structure

```
mistral_ner/
├── src/                    # Core modules
│   ├── config.py          # Configuration management
│   ├── data.py            # Data loading and processing
│   ├── model.py           # Model setup and LoRA
│   ├── training.py        # Training loop and callbacks
│   ├── evaluation.py      # Metrics and evaluation
│   └── utils.py           # Utilities and helpers
├── scripts/               # Executable scripts
│   ├── train.py          # Main training script
│   └── inference.py      # Inference script
├── configs/               # Configuration files
│   └── default.yaml      # Default configuration
├── api/                   # API server
│   └── serve.py          # FastAPI endpoint
├── tests/                 # Unit tests
└── notebooks/            # Example notebooks
```

## Installation

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/nevedomski/mistral_ner.git
cd mistral_ner

# Run automated setup (detects CUDA version)
python setup_env.py
```

### Manual Installation

Using UV (recommended):
```bash
# For CUDA 12.x
uv pip install -e ".[cuda12]"

# For CUDA 11.x
uv pip install -e ".[cuda11]"

# For CPU only
uv pip install -e .
```

Using pip:
```bash
pip install -e ".[cuda12]"  # or cuda11
```

### Optional Dependencies

```bash
# For API server
pip install -e ".[api]"

# For development
pip install -e ".[dev]"
```

## Usage

### Training

Basic training:
```bash
python scripts/train.py
```

With custom configuration:
```bash
python scripts/train.py \
    --config configs/default.yaml \
    --num-train-epochs 10 \
    --batch-size 8 \
    --learning-rate 1e-4
```

Disable WandB:
```bash
python scripts/train.py --no-wandb
```

Resume from checkpoint:
```bash
python scripts/train.py --resume-from-checkpoint ./mistral-ner-finetuned/checkpoint-500
```

Multi-GPU training:
```bash
accelerate launch scripts/train.py
```

### Multi-Dataset Training

Train on multiple datasets with automatic label unification:
```bash
python scripts/train.py --config configs/bank_pii.yaml
```

Example multi-dataset configuration:
```yaml
data:
  multi_dataset:
    enabled: true
    dataset_names: ["conll2003", "ontonotes", "gretel_pii", "ai4privacy"]
    dataset_weights: [0.15, 0.25, 0.3, 0.3]
    label_mapping_profile: "bank_pii"  # Use predefined mappings
```

### Inference

Command line inference:
```bash
python scripts/inference.py \
    --model-path ./mistral-ner-finetuned-final \
    --text "John Smith works at Microsoft in Seattle."
```

Batch inference from file:
```bash
python scripts/inference.py \
    --model-path ./mistral-ner-finetuned-final \
    --file input.txt \
    --output predictions.txt
```

### API Server

Start the server:
```bash
python -m api.serve --host 0.0.0.0 --port 8000
```

Make predictions:
```bash
curl -X POST "http://localhost:8000/predict" \
    -H "Content-Type: application/json" \
    -d '{"texts": ["John Smith works at Microsoft in Seattle."]}'
```

## Configuration

The project uses a hierarchical configuration system:

1. **Default configuration**: `configs/default.yaml`
2. **Environment variables**: Via `.env` file
3. **Command-line arguments**: Override any setting

Example configuration:
```yaml
model:
  model_name: "mistralai/Mistral-7B-v0.3"
  load_in_8bit: true
  lora_r: 16
  lora_alpha: 32

training:
  num_train_epochs: 5
  per_device_train_batch_size: 4
  learning_rate: 2e-4
  
logging:
  use_wandb: true
  wandb_project: "mistral-ner"
```

## Memory Requirements

With 8-bit quantization and LoRA:
- **Minimum**: 16GB VRAM (batch size 1-2)
- **Recommended**: 24GB VRAM (batch size 4-8)
- **Optimal**: 40GB+ VRAM (batch size 16+)

## Performance

On a single A100 GPU:
- Training time: ~2 hours for 5 epochs
- Inference: ~50 samples/second
- F1 Score: ~92% on CoNLL-2003 test set

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size: `--batch-size 2`
- Enable gradient checkpointing (enabled by default)
- Use CPU offloading in config

### WandB Issues
- Disable with: `--no-wandb`
- Use offline mode: Set `WANDB_MODE=offline`

### Import Errors
- Ensure you're in the project directory
- Check CUDA version matches installation

## Contributing

1. Fork the repository
2. Create your feature branch
3. Add tests for new functionality
4. Ensure all tests pass: `pytest tests/`
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Citation

If you use this code in your research, please cite:

```bibtex
@software{mistral_ner,
  author = {Sergei Nevedomski},
  title = {Mistral NER: Efficient Fine-tuning for Named Entity Recognition},
  year = {2024},
  url = {https://github.com/nevedomski/mistral_ner}
}
```