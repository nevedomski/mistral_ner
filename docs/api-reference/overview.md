# API Reference Overview

Mistral NER provides multiple interfaces for integration:

- **Python API**: Direct programmatic access
- **REST API**: HTTP endpoints for inference
- **CLI**: Command-line tools

## Python API

### Basic Usage

```python
from src.config import Config
from src.model import setup_model
from src.data import prepare_datasets

# Load configuration
config = Config.from_yaml("configs/default.yaml")

# Setup model
model, tokenizer = setup_model(config.model.model_name, config)

# Inference
from scripts.inference import NERInference

ner = NERInference(model, tokenizer)
entities = ner.predict("Apple Inc. CEO Tim Cook announced new products.")
print(entities)
# Output: [("Apple Inc.", "ORG"), ("Tim Cook", "PER")]
```

### Core Modules

#### Configuration (`src.config`)

```python
from src.config import Config, ModelConfig, TrainingConfig

# Load from YAML
config = Config.from_yaml("path/to/config.yaml")

# Create programmatically
config = Config(
    model=ModelConfig(
        model_name="mistralai/Mistral-7B-v0.3",
        load_in_4bit=True
    ),
    training=TrainingConfig(
        learning_rate=2e-4,
        num_train_epochs=5
    )
)

# Update from arguments
config.update_from_args(args)
```

#### Model Management (`src.model`)

```python
from src.model import (
    setup_model,
    create_bnb_config,
    save_model_checkpoint,
    load_model_for_inference
)

# Setup quantized model
bnb_config = create_bnb_config(load_in_8bit=True)
model = load_base_model(model_name, config, bnb_config)

# Save checkpoint
save_model_checkpoint(model, tokenizer, output_dir, is_final=True)
```

#### Data Processing (`src.data`)

```python
from src.data import (
    load_conll2003_dataset,
    prepare_datasets,
    create_data_collator
)

# Load dataset
dataset = load_conll2003_dataset()

# Prepare for training
train_dataset, eval_dataset, test_dataset, data_collator = prepare_datasets(
    tokenizer=tokenizer,
    config=config,
    dataset=dataset
)
```

#### Training (`src.training`)

```python
from src.training import (
    setup_trainer,
    run_training_pipeline
)

# Run complete pipeline
trainer, best_checkpoint = run_training_pipeline(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    config=config
)
```

#### Evaluation (`src.evaluation`)

```python
from src.evaluation import evaluate_model, compute_metrics

# Evaluate model
metrics = evaluate_model(
    model=model,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    label_names=config.data.label_names
)

print(f"F1 Score: {metrics['eval_f1']:.4f}")
```

### Advanced Features

#### Multi-Dataset Support

```python
from src.datasets import DatasetRegistry, DatasetMixer

# Load multiple datasets
registry = DatasetRegistry()
datasets = [
    registry.get_loader("conll2003").load(),
    registry.get_loader("ontonotes").load()
]

# Mix datasets
mixer = DatasetMixer(strategy="interleave")
mixed_dataset = mixer.mix(datasets)
```

#### Custom Loss Functions

```python
from src.losses import create_loss_function

# Create custom loss
loss_fn = create_loss_function(
    loss_type="focal",
    num_labels=9,
    focal_gamma=3.0,
    class_frequencies=frequencies
)
```

#### Hyperparameter Optimization

```python
from src.hyperopt import HyperparameterOptimizer

# Run optimization
with HyperparameterOptimizer(config.hyperopt) as optimizer:
    results = optimizer.optimize(
        objective_func,
        search_space,
        config
    )
```

## REST API

See [REST API Reference](rest-api.md) for HTTP endpoint documentation.

## CLI Reference

### Training Commands

```bash
# Basic training
python scripts/train.py

# With custom config
python scripts/train.py --config configs/my_config.yaml

# Override parameters
python scripts/train.py \
    --learning-rate 1e-4 \
    --num-train-epochs 10 \
    --batch-size 8
```

### Inference Commands

```bash
# Single text
python scripts/inference.py \
    --model-path ./model \
    --text "Your text here"

# Batch file
python scripts/inference.py \
    --model-path ./model \
    --file input.txt \
    --output predictions.txt

# Interactive mode
python scripts/inference.py \
    --model-path ./model \
    --interactive
```

### Utility Commands

```bash
# Validate setup
python scripts/validate.py

# Benchmark model
python scripts/benchmark.py --model-path ./model

# Export model
python scripts/export.py \
    --model-path ./model \
    --output-format onnx
```

## Type Annotations

All modules use Python 3.11+ type hints:

```python
from typing import Optional, Union, List, Dict, Tuple
from pathlib import Path
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

def predict(
    text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: Optional[torch.device] = None
) -> List[Tuple[str, str]]:
    """Predict entities in text."""
    ...
```

## Error Handling

All API methods include proper error handling:

```python
try:
    model, tokenizer = setup_model(model_name, config)
except FileNotFoundError as e:
    logger.error(f"Model not found: {e}")
    raise
except torch.cuda.OutOfMemoryError:
    logger.error("GPU OOM - try enabling quantization")
    raise
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise
```

## Next Steps

- Explore the [Python API](python-api.md) in detail
- Learn about the [REST API](rest-api.md)
- See [examples](https://github.com/nevedomski/mistral_ner/tree/main/examples) for real-world usage