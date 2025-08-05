# AGENTS.md - configs/ Directory

## Overview
Hydra configuration files for the Mistral NER v2 framework. Uses YAML format with structured composition.

## Directory Structure

### Root Configurations
- **default.yaml**: Base configuration with all defaults
- **multi_dataset.yaml**: Multi-dataset training configuration
- **multi_dataset_extended.yaml**: Extended multi-dataset with more datasets
- **offline.yaml**: Configuration for offline training (no internet)
- **hyperopt.yaml**: Hyperparameter optimization configuration
- **validation_config.yaml**: Validation-only configuration
- **8bit_quantization.yaml**: 8-bit quantization settings
- **bank_pii.yaml**: Banking-specific PII detection configuration

### Subdirectories

#### data/
Dataset-specific configurations:
- **conll2003.yaml**: CoNLL-2003 NER dataset
- **multi_dataset.yaml**: Multi-dataset configuration
- **multi_dataset_extended.yaml**: Extended dataset list
- **pii_dataset.yaml**: PII detection datasets

##### data/datasets/
Individual dataset configurations:
- **ai4privacy_pii.yaml**: AI4Privacy PII dataset
- **conll_pii.yaml**: CoNLL with PII labels
- **gretel_pii.yaml**: Gretel synthetic PII
- **ontonotes_pii.yaml**: OntoNotes with PII mapping

#### hyperopt/
Hyperparameter search configurations:
- **combined.yaml**: Combined hyperopt settings

#### logging/
Logging configurations:
- **info_wandb.yaml**: Info-level logging with WandB
- **info_wandb_off.yaml**: Info-level logging without WandB
- **info_wandb_offline.yaml**: Offline WandB logging

#### model/
Model configurations:
- **mistral7b03_qlora.yaml**: Mistral 7B v0.3 with QLoRA

#### training/
Training configurations:
- **focal_fp16_tf32.yaml**: Focal loss with mixed precision

## Configuration Composition

### How It Works
1. Start with default.yaml as base
2. Override with specific configs (e.g., multi_dataset.yaml)
3. Apply command-line overrides last

### Example Composition
```bash
# Uses default.yaml + multi_dataset.yaml
python main.py --config-name=multi_dataset

# Uses default.yaml + custom overrides
python main.py data.dataset_name=conll2003 training.num_epochs=5
```

## Key Configuration Sections

### Model Configuration
- Model name/path
- Quantization settings (4-bit, 8-bit)
- LoRA parameters (rank, alpha, dropout)
- Target modules for fine-tuning

### Data Configuration
- Dataset selection
- Preprocessing options
- Tag scheme settings
- Multi-dataset mixing strategies

### Training Configuration
- Learning rate and schedule
- Batch size and accumulation
- Mixed precision settings
- Evaluation strategy
- Checkpoint settings

### Logging Configuration
- Log levels and directories
- WandB integration
- Metric tracking

## Best Practices

1. **Modularity**: Keep configurations focused and composable
2. **Defaults**: Set sensible defaults in default.yaml
3. **Documentation**: Comment complex settings in YAML
4. **Validation**: Configs are validated by Pydantic models
5. **Override Hierarchy**: CLI > Composed configs > Defaults

## Common Patterns

### Multi-Dataset Training
```yaml
data:
  dataset_names: ["conll2003", "ontonotes", "wnut17"]
  dataset_weights: [0.5, 0.3, 0.2]
  mixing_strategy: "weighted"
```

### Quantization Settings
```yaml
model:
  load_in_4bit: true
  bnb_4bit_compute_dtype: "bfloat16"
  bnb_4bit_quant_type: "nf4"
```

### Mixed Precision Training
```yaml
training:
  bf16: true
  tf32: true
  fp16: false
```

## Troubleshooting

1. **Configuration Not Found**: Check --config-path parameter
2. **Override Syntax**: Use dots for nested values (e.g., model.lora_r=16)
3. **Type Errors**: Ensure override types match schema (strings need quotes)
4. **Composition Order**: Later configs override earlier ones