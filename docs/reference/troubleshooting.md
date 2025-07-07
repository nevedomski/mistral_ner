# Troubleshooting Guide

This guide covers common issues and their solutions when using Mistral NER.

## Installation Issues

### CUDA Not Available

**Symptom**: `torch.cuda.is_available()` returns False

**Solutions**:
1. Verify CUDA installation:
   ```bash
   nvidia-smi
   nvcc --version
   ```

2. Install correct PyTorch version:
   ```bash
   # For CUDA 12.x
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   
   # For CUDA 11.x
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### BitsAndBytes Installation Failed

**Symptom**: Error when installing or importing bitsandbytes

**Solutions**:
1. Install from source:
   ```bash
   pip install bitsandbytes --no-binary bitsandbytes
   ```

2. For Windows users:
   ```bash
   pip install bitsandbytes-windows
   ```

## Training Issues

### Out of Memory (OOM) Errors

**Symptom**: `CUDA out of memory` error during training

**Solutions**:

1. **Enable quantization**:
   ```yaml
   model:
     load_in_4bit: true  # Uses ~6GB instead of 24GB
   ```

2. **Reduce batch size**:
   ```yaml
   training:
     per_device_train_batch_size: 2
     gradient_accumulation_steps: 16
   ```

3. **Enable gradient checkpointing**:
   ```yaml
   training:
     gradient_checkpointing: true
   ```

4. **Clear cache periodically**:
   ```yaml
   training:
     clear_cache_steps: 50
   ```

### Training Extremely Slow

**Symptom**: Each epoch takes hours

**Solutions**:

1. **Enable mixed precision**:
   ```yaml
   training:
     fp16: true  # or bf16 for A100 GPUs
   ```

2. **Check data loading**:
   ```yaml
   data:
     preprocessing_num_workers: 4
   ```

3. **Disable unnecessary logging**:
   ```yaml
   training:
     logging_steps: 100  # Increase from default 10
   ```

### Model Not Learning (Loss Not Decreasing)

**Symptom**: Loss plateaus or increases

**Solutions**:

1. **Adjust learning rate**:
   ```yaml
   training:
     learning_rate: 1e-4  # Try different values
     warmup_ratio: 0.1
   ```

2. **Change loss function**:
   ```yaml
   training:
     loss_type: "focal"
     focal_gamma: 3.0  # Increase for imbalanced data
   ```

3. **Check data quality**:
   ```python
   # Visualize label distribution
   from src.visualization import plot_label_distribution
   plot_label_distribution(train_dataset)
   ```

## Quantization Issues

### Quantization Not Working

**Symptom**: Model loads in full precision despite quantization settings

**Solutions**:

1. **Check bitsandbytes installation**:
   ```python
   import bitsandbytes as bnb
   print(f"BitsAndBytes version: {bnb.__version__}")
   ```

2. **Verify configuration**:
   ```yaml
   model:
     load_in_8bit: true   # Only one should be true
     load_in_4bit: false
   ```

3. **Check GPU compatibility**:
   - Requires GPU with compute capability >= 3.5
   - Run `nvidia-smi` to check GPU model

### 8-bit vs 4-bit Quantization

**When to use each**:
- **8-bit**: Better accuracy, ~10GB VRAM
- **4-bit**: More memory efficient, ~6GB VRAM

## Inference Issues

### Predictions All "O" (No Entities)

**Symptom**: Model only predicts non-entity labels

**Solutions**:

1. **Check threshold if using confidence filtering**
2. **Verify model loaded correctly**:
   ```python
   # Check if LoRA weights are loaded
   print(model.peft_config)
   ```

3. **Use different checkpoint**:
   ```bash
   # Try best checkpoint instead of final
   python scripts/inference.py --model-path ./mistral-ner-finetuned/checkpoint-best
   ```

### Tokenization Misalignment

**Symptom**: Entity boundaries don't match original text

**Solutions**:

1. **Enable proper tokenizer settings**:
   ```python
   tokenizer = AutoTokenizer.from_pretrained(
       model_name,
       add_prefix_space=True  # Important for proper alignment
   )
   ```

2. **Check max_length setting**:
   ```yaml
   data:
     max_length: 256  # Increase if truncating
   ```

## WandB Issues

### WandB Offline Mode Not Working

**Symptom**: Runs not syncing after coming online

**Solutions**:

1. **Sync manually**:
   ```bash
   wandb sync ./wandb_logs/offline-run-*
   ```

2. **Check environment**:
   ```bash
   export WANDB_MODE=offline
   export WANDB_DIR=./wandb_logs
   ```

See [WandB Offline Mode](wandb-offline.md) for detailed guide.

## Multi-Dataset Issues

### Label Mismatch Errors

**Symptom**: Error about incompatible label sets

**Solutions**:

1. **Check label mappings**:
   ```python
   for dataset_config in config.dataset_configs:
       loader = registry.get_loader(dataset_config.name)
       print(f"{dataset_config.name}: {loader.get_labels()}")
   ```

2. **Use unified label schema**:
   ```yaml
   data:
     use_unified_labels: true
   ```

## Performance Issues

### Low F1 Score

**Common causes and solutions**:

1. **Class imbalance**: Use focal loss
2. **Insufficient training**: Increase epochs
3. **Poor hyperparameters**: Use hyperopt
4. **Dataset quality**: Check annotation consistency

### Inconsistent Results

**Solutions**:

1. **Set seeds**:
   ```yaml
   training:
     seed: 42
     data_seed: 42
   ```

2. **Disable non-deterministic ops**:
   ```python
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False
   ```

## Debug Mode

Enable comprehensive debugging:

```yaml
training:
  debug: true
  
logging:
  log_level: "debug"
```

This will show:
- Detailed model loading info
- Tokenization examples
- Batch composition
- Memory usage

## Getting Help

If these solutions don't resolve your issue:

1. **Check logs**: Look for error messages in `./logs/`
2. **Run validation**: `python scripts/validate.py`
3. **Create minimal example**: Isolate the problem
4. **Report issue**: Include config, error message, and environment info

## Environment Debugging

Collect system information:

```python
# save as debug_env.py
import torch
import transformers
import platform

print(f"Python: {platform.python_version()}")
print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

---

!!! tip "Prevention is Better"
    Most issues can be prevented by:
    - Starting with default configurations
    - Testing on small data subsets first
    - Monitoring resource usage
    - Keeping dependencies updated