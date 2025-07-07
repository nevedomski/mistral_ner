# Quick Start Guide

## 5-Minute Setup

### 1. Install Mistral NER

```bash
pip install -e ".[cuda12]"
```

### 2. Download Pre-trained Model (Optional)

```bash
# Download a pre-trained model
python scripts/download_model.py --model mistral-ner-conll2003
```

### 3. Run Inference

```bash
# On a single text
python scripts/inference.py \
    --model-path mistral-ner-conll2003 \
    --text "Apple Inc. CEO Tim Cook announced new products in Cupertino."

# Output:
# Apple Inc. [ORG] CEO Tim Cook [PER] announced new products in Cupertino [LOC].
```

### 4. Train Your Own Model

```bash
# Using default configuration
python scripts/train.py

# Monitor training
tensorboard --logdir logs/
```

## Next Steps

- [First Training Tutorial](first-training.md) - Train your first model
- [Configuration Guide](../user-guide/configuration.md) - Customize settings
- [API Documentation](../api-reference/overview.md) - Use programmatically