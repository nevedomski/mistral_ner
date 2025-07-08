# Checkpoint Saving and Loading Guide

## Overview

This guide explains how models, checkpoints, and configurations are saved and loaded in the Mistral NER project.

## Model Saving Behavior

### During Training

1. **Regular Checkpoints**: Saved to `output_dir` (default: `./mistral-ner-finetuned/`)
   - Contains: LoRA adapter weights only
   - Files: `adapter_config.json`, `adapter_model.safetensors`
   - Size: Small (~10-50MB)

2. **Final Model**: Saved to `final_output_dir` (default: `./mistral-ner-finetuned-final/`)
   - Contains: LoRA adapter weights
   - Also includes: Tokenizer and config.yaml

3. **Merged Model** (NEW): If `merge_adapters_on_save: true` (default)
   - Saved to: `./mistral-ner-finetuned-final-merged/`
   - Contains: Complete model with LoRA weights merged
   - Size: Large (~14GB for Mistral-7B)
   - Ready for deployment without base model

### File Structure

```
# Checkpoint directory
mistral-ner-finetuned/checkpoint-500/
├── adapter_config.json      # LoRA configuration
├── adapter_model.safetensors # LoRA weights only
├── config.yaml              # Training configuration
├── tokenizer_config.json
├── special_tokens_map.json
└── tokenizer.json

# Final model directory (adapters only)
mistral-ner-finetuned-final/
├── adapter_config.json
├── adapter_model.safetensors
├── config.yaml
├── tokenizer_config.json
├── special_tokens_map.json
└── tokenizer.json

# Merged model directory (full model)
mistral-ner-finetuned-final-merged/
├── config.json              # Model configuration
├── model.safetensors        # Complete merged weights
├── config.yaml              # Training configuration
├── tokenizer_config.json
├── special_tokens_map.json
└── tokenizer.json
```

## Loading Models

### For Inference

The inference script automatically detects the model type:

```python
# Loading adapter-only model (requires base model)
python scripts/inference.py \
    --model-path ./mistral-ner-finetuned-final \
    --base-model mistralai/Mistral-7B-v0.3 \
    --text "John works at Microsoft"

# Loading merged model (standalone)
python scripts/inference.py \
    --model-path ./mistral-ner-finetuned-final-merged \
    --text "John works at Microsoft"
```

### Programmatically

```python
from src.model import load_model_from_checkpoint, setup_model

# Load adapter model
model, tokenizer = load_model_from_checkpoint(
    checkpoint_path="./mistral-ner-finetuned-final",
    config=config,
    base_model_name="mistralai/Mistral-7B-v0.3"
)

# Load merged model
model, tokenizer = setup_model(
    model_name="./mistral-ner-finetuned-final-merged",
    config=config
)
```

## Configuration Options

### Enable/Disable Adapter Merging

In your config file:

```yaml
training:
  merge_adapters_on_save: true  # Default: true
```

Or via command line:

```bash
python scripts/train.py --merge-adapters-on-save false
```

### Resume Training

```bash
# Resume from specific checkpoint
python scripts/train.py --resume-from-checkpoint ./mistral-ner-finetuned/checkpoint-500

# Resume from last checkpoint (if configured)
python scripts/train.py
```

## Best Practices

1. **For Development**: Use adapter-only models (smaller, faster to save/load)
2. **For Deployment**: Use merged models (no dependency on base model)
3. **For Fine-tuning**: Keep adapter format for continued training
4. **For Inference**: Merged models are slightly faster

## Storage Requirements

- Base Mistral-7B model: ~14GB
- LoRA adapters: ~10-50MB
- Merged model: ~14GB
- Training checkpoints: ~50MB each

## Troubleshooting

### Issue: Model too large to save
- Solution: Disable adapter merging with `merge_adapters_on_save: false`

### Issue: Can't load model for inference
- Check if it's an adapter model (needs base model) or merged model
- For adapters: Provide `--base-model` parameter

### Issue: Resume training fails
- Ensure checkpoint contains required files
- Check that config versions match

## Technical Details

### What Gets Saved

1. **Model Weights**:
   - Adapters: Only LoRA matrices (Q, K, V, O projections)
   - Merged: Complete model weights

2. **Tokenizer**:
   - Vocabulary
   - Special tokens
   - Model max length (from config)

3. **Configuration**:
   - Training hyperparameters
   - Model architecture
   - Data settings

### Loading Process

1. **Checkpoint Detection**: Check for `adapter_config.json`
2. **Base Model Loading**: Load with quantization if configured
3. **Adapter Application**: Apply LoRA weights to base model
4. **Tokenizer Setup**: Configure padding and max length
5. **Validation**: Ensure all components are compatible