# Configuration Reference

## Overview

Mistral NER uses a comprehensive YAML-based configuration system that allows fine-grained control over all aspects of training, evaluation, and inference. The configuration is structured into logical sections, each controlling different components of the system.

## Configuration Structure

```yaml
# Main configuration sections
model:        # Model architecture and LoRA settings
data:         # Dataset configuration and preprocessing
training:     # Training hyperparameters and settings
logging:      # Logging and experiment tracking
hyperopt:     # Hyperparameter optimization settings (optional)
```

## Complete Configuration Schema

### Model Configuration

Controls the base model, quantization, and LoRA adapter settings.

```yaml
model:
  # Base model settings
  model_name: "mistralai/Mistral-7B-v0.3"  # HuggingFace model ID
  num_labels: 9                             # Number of NER labels (including O)
  
  # Quantization settings
  load_in_8bit: false                       # Enable 8-bit quantization
  load_in_4bit: true                        # Enable 4-bit quantization (QLoRA)
  # Note: Only one quantization method can be active at a time
  # If both are true, 4-bit takes precedence as it's more memory efficient
  device_map: "auto"                        # Automatic device placement
  trust_remote_code: true                   # Allow custom model code
  use_cache: false                          # Disable KV cache during training
  
  # LoRA configuration
  lora_r: 16                                # LoRA rank (attention dimension)
  lora_alpha: 32                            # LoRA scaling parameter
  lora_dropout: 0.1                         # LoRA dropout rate
  lora_bias: "none"                         # LoRA bias: none, all, lora_only
  
  # Target modules for LoRA
  target_modules:
    - "q_proj"                              # Query projection
    - "k_proj"                              # Key projection
    - "v_proj"                              # Value projection
    - "o_proj"                              # Output projection
    # Optional: Add feed-forward layers
    # - "gate_proj"
    # - "up_proj"
    # - "down_proj"
  
  task_type: "TOKEN_CLS"                    # Task type for PEFT
  
  # Model-specific optimizations
  attention_dropout: 0.1                    # Attention dropout rate
  hidden_dropout: 0.1                       # Hidden layer dropout
```

#### Key Parameters Explained

- **`load_in_8bit`** / **`load_in_4bit`**: Enable quantization for memory-efficient training. Use only one at a time:
  - 8-bit: Better performance, uses ~10GB VRAM for Mistral-7B
  - 4-bit: More memory efficient, uses ~6GB VRAM for Mistral-7B
- **`lora_r`**: Controls the rank of LoRA decomposition. Higher values = more parameters but better capacity
- **`lora_alpha`**: Scaling factor for LoRA. Common practice: alpha = 2 * r
- **`target_modules`**: Which transformer layers to apply LoRA to

### Data Configuration

Handles dataset loading, preprocessing, and multi-dataset training.

```yaml
data:
  # Single dataset configuration
  dataset_name: "conll2003"                 # Dataset identifier
  
  # OR Multi-dataset configuration
  dataset_configs:
    - name: conll2003
      split: train
      language: en                          # For multilingual datasets
      weight: 1.0                           # Sampling weight
    
    - name: ontonotes
      split: train
      subset: english_v4
      weight: 1.5
    
    - name: gretel_pii
      split: train
      max_examples: 10000                   # Limit dataset size
  
  # Mixing strategy for multiple datasets
  mixing_strategy: "interleave"             # concatenate, interleave, weighted
  interleave_probs: [0.6, 0.4]             # Dataset sampling probabilities
  sampling_temperature: 1.0                 # Temperature for weighted sampling
  
  # Tokenization settings
  max_length: 256                           # Maximum sequence length
  label_all_tokens: false                   # Label only first subword token
  return_entity_level_metrics: true         # Compute per-entity metrics
  
  # Data augmentation (optional)
  lowercase_prob: 0.1                       # Random lowercase probability
  mask_prob: 0.05                           # Token masking probability
  
  # Advanced tokenization
  stride: 128                               # Stride for long sequences
  truncation_strategy: "longest_first"      # How to truncate long sequences
  
  # Label configuration
  label_names:
    - "O"
    - "B-PER"
    - "I-PER"
    - "B-ORG"
    - "I-ORG"
    - "B-LOC"
    - "I-LOC"
    - "B-MISC"
    - "I-MISC"
  
  # Processing settings
  preprocessing_num_workers: 4              # Parallel preprocessing workers
  cache_dir: "~/.cache/mistral_ner"        # Dataset cache directory
  streaming: false                          # Use dataset streaming
```

### Training Configuration

Controls all training hyperparameters and optimization settings.

```yaml
training:
  # Output directories
  output_dir: "./mistral-ner-finetuned"     # Checkpoint directory
  final_output_dir: "./mistral-ner-final"   # Final model directory
  
  # Basic training settings
  num_train_epochs: 5                       # Number of training epochs
  per_device_train_batch_size: 4            # Batch size per GPU
  per_device_eval_batch_size: 8             # Evaluation batch size
  gradient_accumulation_steps: 8            # Gradient accumulation steps
  
  # Learning rate and optimization
  learning_rate: 2e-4                       # Initial learning rate
  warmup_ratio: 0.03                        # Warmup ratio (0-1)
  warmup_steps: 0                           # OR fixed warmup steps
  weight_decay: 0.01                        # Weight decay (L2 regularization)
  adam_beta1: 0.9                           # Adam beta1
  adam_beta2: 0.999                         # Adam beta2
  adam_epsilon: 1e-8                        # Adam epsilon
  max_grad_norm: 1.0                        # Gradient clipping
  
  # Learning rate scheduler
  lr_scheduler_type: "cosine"               # linear, cosine, polynomial, constant
  lr_scheduler_kwargs:                      # Additional scheduler arguments
    num_cycles: 0.5                         # For cosine scheduler
  
  # Loss function configuration
  loss_type: "focal"                        # focal, cross_entropy, label_smoothing, etc.
  focal_gamma: 2.0                          # Focal loss gamma parameter
  focal_alpha: null                         # Auto-compute from class frequencies
  use_class_weights: true                   # Enable class weighting
  class_weight_type: "inverse"              # inverse, inverse_sqrt, effective
  class_weight_smoothing: 1.0               # Smoothing for class weights
  
  # Evaluation and checkpointing
  eval_strategy: "steps"                    # no, steps, epoch
  eval_steps: 50                            # Evaluation frequency
  save_strategy: "steps"                    # no, steps, epoch
  save_steps: 50                            # Checkpoint frequency
  save_total_limit: 3                       # Maximum checkpoints to keep
  load_best_model_at_end: true              # Load best model after training
  metric_for_best_model: "eval_f1"          # Metric for model selection
  greater_is_better: true                   # Whether higher metric is better
  
  # Early stopping
  early_stopping_patience: 3                # Patience for early stopping
  early_stopping_threshold: 0.0             # Minimum improvement threshold
  
  # Training optimizations
  fp16: true                                # Mixed precision training
  bf16: false                               # BFloat16 training (A100 only)
  gradient_checkpointing: true              # Memory-efficient training
  optim: "paged_adamw_32bit"               # Optimizer (memory efficient)
  
  # Advanced settings
  max_steps: -1                             # Maximum training steps (-1 = disabled)
  dataloader_drop_last: false               # Drop incomplete batches
  dataloader_num_workers: 0                 # DataLoader workers
  past_index: -1                            # For using past key values
  run_name: null                            # Custom run name
  disable_tqdm: false                       # Disable progress bars
  remove_unused_columns: true               # Remove unused dataset columns
  label_names: null                         # Custom label names
  
  # Debugging
  debug: false                              # Debug mode
  prediction_loss_only: false               # Only compute loss (no metrics)
  
  # Memory management
  max_memory_mb: null                       # Maximum memory usage
  clear_cache_every_n_steps: 100            # Clear CUDA cache frequency
```

### Logging Configuration

Controls experiment tracking and logging behavior.

```yaml
logging:
  # Logging levels
  log_level: "info"                         # debug, info, warning, error, critical
  log_level_replica: "warning"              # Log level for replicas
  logging_first_step: false                 # Log first training step
  logging_steps: 10                         # Training logging frequency
  logging_nan_inf_filter: true              # Filter NaN/Inf values
  
  # Experiment tracking
  report_to: ["wandb", "tensorboard"]      # Tracking backends
  wandb_project: "mistral-ner"              # WandB project name
  wandb_entity: null                        # WandB entity (username/team)
  wandb_run_name: null                      # Custom WandB run name
  wandb_tags: ["ner", "mistral", "lora"]   # WandB tags
  wandb_mode: "online"                      # online, offline, disabled
  
  # TensorBoard settings
  tensorboard_dir: "./logs"                 # TensorBoard log directory
  
  # Model card
  push_to_hub: false                        # Push model to HuggingFace Hub
  hub_model_id: null                        # HuggingFace model ID
  hub_strategy: "every_save"                # Hub push strategy
  hub_token: null                           # HuggingFace API token
  
  # Logging behavior
  log_on_each_node: false                   # Log on all nodes (distributed)
  logging_dir: null                         # Override logging directory
```

### Hyperparameter Optimization Configuration

Optional section for automated hyperparameter tuning.

```yaml
hyperopt:
  enabled: false                            # Enable hyperparameter optimization
  strategy: "combined"                      # optuna, asha, combined, random
  num_trials: 50                            # Number of trials to run
  max_concurrent: 4                         # Maximum parallel trials
  timeout: 3600                             # Maximum time in seconds
  
  # Optimization objective
  metric: "eval_f1"                         # Metric to optimize
  mode: "max"                               # max or min
  
  # Search space definition
  search_space:
    learning_rate:
      type: "loguniform"
      low: 1e-5
      high: 1e-3
    
    lora_r:
      type: "choice"
      choices: [8, 16, 32, 64]
    
    per_device_train_batch_size:
      type: "choice"
      choices: [4, 8, 16]
    
    warmup_ratio:
      type: "uniform"
      low: 0.0
      high: 0.1
  
  # Strategy-specific settings
  optuna_sampler: "TPE"                     # TPE, CMA, Random
  optuna_pruner: "median"                   # median, hyperband, none
  optuna_n_startup_trials: 10               # Random trials before optimization
  
  asha_max_t: 100                           # ASHA maximum iterations
  asha_grace_period: 10                     # ASHA grace period
  asha_reduction_factor: 3                  # ASHA reduction factor
  asha_brackets: 1                          # ASHA brackets
  
  # Storage and resources
  results_dir: "./hyperopt_results"         # Results directory
  study_name: "mistral_ner_study"           # Study name
  log_to_file: true                         # Log to file
  ray_address: null                         # Ray cluster address
  resources_per_trial:                      # Resources per trial
    cpu: 2.0
    gpu: 1.0
```

## Environment Variables

The configuration system supports environment variable substitution:

```yaml
model:
  model_name: ${MODEL_NAME:-mistralai/Mistral-7B-v0.3}
  
logging:
  wandb_api_key: ${WANDB_API_KEY}
  hub_token: ${HF_TOKEN}
  
data:
  cache_dir: ${CACHE_DIR:-~/.cache/mistral_ner}
```

## Configuration Loading

### From YAML File

```python
from src.config import Config

# Load from file
config = Config.from_yaml("configs/default.yaml")

# Access nested values
print(config.model.lora_r)
print(config.training.learning_rate)
```

### With CLI Overrides

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --learning-rate 1e-4 \
    --num-train-epochs 10 \
    --lora-r 32
```

### Programmatic Configuration

```python
from src.config import Config, ModelConfig, TrainingConfig

# Create config programmatically
config = Config(
    model=ModelConfig(
        model_name="mistralai/Mistral-7B-v0.3",
        lora_r=32,
        lora_alpha=64
    ),
    training=TrainingConfig(
        learning_rate=2e-4,
        num_train_epochs=5,
        per_device_train_batch_size=4
    )
)
```

## Configuration Examples

### Minimal Configuration

```yaml
# configs/minimal.yaml
model:
  model_name: "mistralai/Mistral-7B-v0.3"

data:
  dataset_name: "conll2003"

training:
  output_dir: "./output"
  num_train_epochs: 3
```

### High Performance Configuration

```yaml
# configs/high_performance.yaml
model:
  model_name: "mistralai/Mistral-7B-v0.3"
  lora_r: 32
  lora_alpha: 64
  lora_dropout: 0.05
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

data:
  dataset_configs:
    - name: conll2003
    - name: ontonotes
  mixing_strategy: "interleave"
  max_length: 256

training:
  learning_rate: 3e-4
  num_train_epochs: 10
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 4
  warmup_ratio: 0.05
  loss_type: "focal"
  focal_gamma: 2.0
  fp16: true
  gradient_checkpointing: true
  eval_steps: 100
  save_steps: 100

logging:
  report_to: ["wandb"]
  wandb_project: "mistral-ner-high-perf"
```

### Multi-Dataset PII Configuration

```yaml
# configs/pii_detection.yaml
model:
  model_name: "mistralai/Mistral-7B-v0.3"
  lora_r: 64
  lora_alpha: 128

data:
  dataset_configs:
    - name: gretel_pii
      weight: 2.0
    - name: ai4privacy
      weight: 2.0
    - name: conll2003
      weight: 0.5
  mixing_strategy: "weighted"
  sampling_temperature: 0.5

training:
  loss_type: "batch_balanced_focal"
  batch_balance_beta: 0.995
  focal_gamma: 3.0
  learning_rate: 1e-4
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 8
```

### Memory-Efficient Configuration

```yaml
# configs/memory_efficient.yaml
model:
  model_name: "mistralai/Mistral-7B-v0.3"
  load_in_8bit: true
  lora_r: 8
  lora_alpha: 16

data:
  max_length: 128
  preprocessing_num_workers: 2

training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 16
  gradient_checkpointing: true
  fp16: true
  optim: "paged_adamw_8bit"
  clear_cache_every_n_steps: 50
```

## Validation Rules

The configuration system enforces several validation rules:

1. **Required Fields**: `model.model_name`, `data.dataset_name` (or `dataset_configs`), `training.output_dir`

2. **Value Constraints**:
   - `lora_r` must be positive and typically powers of 2
   - `learning_rate` must be positive
   - `num_train_epochs` must be positive integer
   - `batch_size` must be positive integer

3. **Logical Constraints**:
   - If `warmup_ratio` is set, `warmup_steps` should be 0
   - `focal_alpha` length must match `num_labels` if provided as list
   - `interleave_probs` must sum to 1.0

4. **Type Checking**:
   - All fields are type-checked at runtime
   - Invalid types will raise configuration errors

## Best Practices

1. **Start with Defaults**: Use the provided default configuration and modify only what you need

2. **Version Control**: Keep your configuration files in version control for reproducibility

3. **Use Environment Variables**: For sensitive information like API keys

4. **Document Changes**: Add comments to explain non-standard settings

5. **Validate Early**: Test configuration with a dry run before full training

## Troubleshooting

### Common Issues

**Issue**: Configuration validation error
```
Solution: Check required fields and value constraints
```

**Issue**: Memory overflow
```
Solution: Reduce batch_size, enable gradient_checkpointing, use 8-bit optimizer
```

**Issue**: Slow training
```
Solution: Increase batch_size, reduce logging_steps, disable unused features
```

**Issue**: Quantization not working (model loads in full precision)
```
Solution: 
1. Check if bitsandbytes is installed: pip install bitsandbytes>=0.41.0
2. Ensure only one quantization method is enabled (load_in_8bit OR load_in_4bit)
3. Check logs for "BitsAndBytes version" message
4. For 8-bit: set load_in_8bit=true and load_in_4bit=false
5. For 4-bit: set load_in_8bit=false and load_in_4bit=true
```

### Debug Mode

Enable debug mode for detailed configuration information:

```yaml
training:
  debug: true
  
logging:
  log_level: "debug"
```

---

!!! tip "Next Steps"
    Now that you understand the configuration system, explore our [API Reference](../api-reference/overview.md) 
    to learn how to use Mistral NER programmatically.