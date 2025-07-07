# First Training Tutorial

This comprehensive guide will walk you through training your first NER model with Mistral NER.

## Prerequisites

Before starting, ensure you have:
- Completed the [installation](installation.md)
- Reviewed the [quick start guide](quickstart.md)
- At least 16GB GPU memory (or 12GB with quantization)

## Step 1: Prepare Your Environment

```bash
# Create a working directory
mkdir mistral-ner-tutorial
cd mistral-ner-tutorial

# Verify installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Step 2: Download the Dataset

The CoNLL-2003 dataset will be automatically downloaded on first use:

```bash
# Test data loading
python -c "from src.data import load_conll2003_dataset; dataset = load_conll2003_dataset(); print(f'Dataset loaded: {len(dataset)} splits')"
```

## Step 3: Basic Training

Start with the default configuration:

```bash
# Run training with default settings
python scripts/train.py

# Monitor training progress
# In another terminal:
tensorboard --logdir logs/
```

## Step 4: Customize Your Training

Create a custom configuration:

```yaml
# my_config.yaml
model:
  model_name: "mistralai/Mistral-7B-v0.3"
  load_in_4bit: true  # Memory efficient
  lora_r: 16
  lora_alpha: 32

training:
  num_train_epochs: 3
  learning_rate: 3e-4
  per_device_train_batch_size: 4

logging:
  wandb_project: "my-first-ner"
```

Run with custom config:

```bash
python scripts/train.py --config my_config.yaml
```

## Step 5: Evaluate Your Model

After training completes:

```bash
# Run evaluation
python scripts/train.py --eval-only --model-name ./mistral-ner-finetuned-final

# Test on custom text
python scripts/inference.py \
    --model-path ./mistral-ner-finetuned-final \
    --text "Apple Inc. was founded by Steve Jobs in Cupertino."
```

## Expected Results

With default settings, you should achieve:
- F1 Score: ~85-90% on validation set
- Training time: ~2-3 hours on single GPU
- Memory usage: ~12GB with 4-bit quantization

## Common Issues and Solutions

### Out of Memory Error

```yaml
# Reduce batch size and enable more memory optimizations
training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 16
  gradient_checkpointing: true
```

### Slow Training

Enable mixed precision:

```yaml
training:
  fp16: true  # or bf16 on newer GPUs
```

### Poor Performance

Try different loss functions:

```yaml
training:
  loss_type: "focal"
  focal_gamma: 3.0  # Increase for harder datasets
```

## Next Steps

- Explore [different datasets](../user-guide/datasets.md)
- Try [hyperparameter optimization](../user-guide/hyperparameter-tuning.md)
- Experiment with [loss functions](../user-guide/loss-functions.md)
- Learn about [multi-dataset training](../user-guide/datasets.md#multi-dataset-training)

## Tips for Success

1. **Start small**: Use a subset of data for initial experiments
2. **Monitor metrics**: Watch validation loss for overfitting
3. **Save checkpoints**: Enable checkpoint saving for long runs
4. **Use wandb**: Track experiments for reproducibility

---

!!! success "Congratulations!"
    You've successfully trained your first NER model with Mistral NER. 
    Continue exploring the documentation to unlock more advanced features.