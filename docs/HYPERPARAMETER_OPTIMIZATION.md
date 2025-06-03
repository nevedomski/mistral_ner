# Hyperparameter Optimization for Mistral NER

This document describes the hyperparameter optimization feature that combines **Bayesian Optimization** (via Optuna) with **Hyperband** (via Ray Tune) for efficient and intelligent hyperparameter search.

## 🎯 Overview

The hyperparameter optimization system provides multiple strategies for finding optimal configurations:

- **🧠 Bayesian Optimization**: Intelligent search using Optuna's TPE algorithm
- **⚡ Hyperband**: Efficient early stopping using Ray Tune's ASHA scheduler  
- **🔥 Combined Strategy**: Best of both worlds - intelligent search + smart stopping
- **🎲 Random Search**: Baseline comparison strategy

## 🚀 Quick Start

### 1. Enable Hyperparameter Optimization

Add to your config YAML:

```yaml
hyperopt:
  enabled: true
  strategy: "combined"  # Recommended: Bayesian + Hyperband
  num_trials: 50
  search_space:
    learning_rate:
      type: "loguniform"
      low: 1e-5
      high: 1e-3
    lora_r:
      type: "choice"
      choices: [8, 16, 32, 64]
```

### 2. Run Optimization

```bash
# Using the main training script
python scripts/train.py --config configs/hyperopt.yaml

# Or using the dedicated hyperopt script
python scripts/hyperopt.py --config configs/hyperopt.yaml
```

### 3. Get Results

The optimizer will:
1. 🔍 Search the hyperparameter space intelligently
2. ⚡ Stop poor-performing trials early
3. 📊 Report the best configuration
4. 🏆 Train a final model with optimal hyperparameters

## 📋 Configuration Guide

### Strategy Options

```yaml
hyperopt:
  strategy: "combined"    # Recommended
  # Options: "optuna", "asha", "combined", "random"
```

#### Strategy Comparison

| Strategy | Search Method | Early Stopping | Best For |
|----------|---------------|-----------------|----------|
| `combined` | Bayesian (TPE) | ASHA | **Recommended** - Best performance |
| `optuna` | Bayesian (TPE) | None | Small search spaces |
| `asha` | Random | ASHA | Large search spaces |
| `random` | Random | None | Baseline comparison |

### Search Space Definition

```yaml
search_space:
  # Continuous parameters
  learning_rate:
    type: "loguniform"
    low: 1e-5
    high: 1e-3
  
  warmup_ratio:
    type: "uniform"
    low: 0.0
    high: 0.1
  
  # Discrete parameters
  lora_r:
    type: "choice"
    choices: [8, 16, 32, 64]
  
  per_device_train_batch_size:
    type: "choice"
    choices: [4, 8, 16]
  
  # Integer parameters
  num_epochs:
    type: "int"
    low: 1
    high: 10
```

#### Parameter Types

- `uniform`: Continuous uniform distribution
- `loguniform`: Log-uniform distribution (good for learning rates)
- `choice`: Discrete categorical choices
- `int`: Integer range
- `logint`: Log-scale integer range

### Trial Settings

```yaml
hyperopt:
  num_trials: 50           # Total trials to run
  max_concurrent: 4        # Parallel trials
  timeout: 3600           # Max time in seconds
  
  # Metric to optimize
  metric: "eval_f1"
  mode: "max"             # "max" or "min"
```

### Advanced Configuration

```yaml
hyperopt:
  # Optuna settings
  optuna_sampler: "TPE"           # TPE, CMA, Random, GPSampler
  optuna_pruner: "median"         # median, hyperband, none
  optuna_n_startup_trials: 10
  
  # ASHA settings
  asha_max_t: 100                 # Max training iterations
  asha_grace_period: 10           # Min iterations before pruning
  asha_reduction_factor: 3        # Aggressiveness of pruning
  
  # Storage and logging
  results_dir: "./hyperopt_results"
  study_name: "mistral_ner_optimization"
  log_to_file: true
```

## 🖥️ Usage Examples

### Command Line Interface

```bash
# Basic optimization
python scripts/hyperopt.py --config configs/hyperopt.yaml

# Override strategy
python scripts/hyperopt.py --strategy combined --num-trials 20

# Save best configuration
python scripts/hyperopt.py --save-best-config best_hyperparams.yaml

# Dry run (show config and exit)
python scripts/hyperopt.py --dry-run

# Debug mode
python scripts/hyperopt.py --debug
```

### Programmatic Usage

```python
from src.config import Config
from src.hyperopt import HyperparameterOptimizer, create_objective_function
from src.hyperopt.utils import create_ray_tune_search_space

# Load configuration
config = Config.from_yaml("configs/hyperopt.yaml")

# Create optimizer
with HyperparameterOptimizer(config.hyperopt) as optimizer:
    # Create objective function
    objective_func = create_objective_function(
        base_config=config,
        hyperopt_config=config.hyperopt,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Create search space
    search_space = create_ray_tune_search_space(config.hyperopt)
    
    # Run optimization
    results = optimizer.optimize(objective_func, search_space, config)
    
    # Get best result
    best_result = results.get_best_result()
    print(f"Best F1: {best_result.metrics['eval_f1']:.4f}")
    print(f"Best config: {best_result.config}")
```

## 📊 Results and Analysis

### Best Configuration

After optimization completes, you'll see:

```
==========================================
HYPERPARAMETER OPTIMIZATION COMPLETED
==========================================
Total trials completed: 50
Best eval_f1: 0.8742
Best hyperparameters:
  learning_rate: 0.00034
  lora_r: 32
  per_device_train_batch_size: 8
  warmup_ratio: 0.045
  weight_decay: 0.0021
==========================================
```

### Saving Results

```bash
# Save best configuration to file
python scripts/hyperopt.py --save-best-config best_config.yaml
```

### Results Directory Structure

```
hyperopt_results/
├── study_name/
│   ├── trial_*/
│   │   ├── checkpoint.json
│   │   └── result.json
│   └── experiment_state.json
├── hyperopt.log
└── best_config.yaml
```

## 🔧 Distributed Optimization

### Ray Cluster Setup

```bash
# Start Ray cluster
ray start --head --port=6379

# In other machines
ray start --address="head_node_ip:6379"
```

### Configuration for Distributed

```yaml
hyperopt:
  ray_address: "ray://head_node_ip:10001"
  max_concurrent: 16        # More parallel trials
  resources_per_trial:
    cpu: 2.0
    gpu: 1.0
```

## 🎛️ Strategy Details

### Combined Strategy (Recommended)

The combined strategy uses:
- **Optuna's TPE** for intelligent hyperparameter suggestions
- **Ray Tune's ASHA** for efficient early stopping

```yaml
hyperopt:
  strategy: "combined"
  optuna_enabled: true
  asha_enabled: true
  
  # TPE will suggest promising hyperparameters
  optuna_sampler: "TPE"
  
  # ASHA will stop poor trials early
  asha_grace_period: 1
  asha_reduction_factor: 3
```

**Benefits:**
- 🧠 Smart hyperparameter exploration
- ⚡ 50-80% reduction in computation time
- 🎯 Better final performance than random search
- 📈 Efficient convergence

### Optuna-Only Strategy

```yaml
hyperopt:
  strategy: "optuna"
  optuna_sampler: "TPE"    # Tree-structured Parzen Estimator
```

**Best for:**
- Small search spaces
- When you want to run all trials to completion
- CPU-only environments

### ASHA-Only Strategy

```yaml
hyperopt:
  strategy: "asha"
  asha_max_t: 100
  asha_grace_period: 10
```

**Best for:**
- Large search spaces
- When early stopping is most important
- Quick exploration

## 🐛 Troubleshooting

### Common Issues

#### Ray Connection Issues
```bash
# Check Ray status
ray status

# Reset Ray
ray stop
ray start --head
```

#### Memory Issues
```yaml
hyperopt:
  max_concurrent: 2        # Reduce parallel trials
  resources_per_trial:
    gpu: 0.5              # Share GPU memory
```

#### Slow Convergence
```yaml
hyperopt:
  optuna_n_startup_trials: 20    # More random trials first
  asha_grace_period: 5          # Allow more training before stopping
```

### Debug Mode

```bash
# Enable debug logging
python scripts/hyperopt.py --debug

# Check specific trial logs
tail -f hyperopt_results/study_name/trial_*/logs.txt
```

## 📈 Performance Tips

### 1. Efficient Search Space Design

```yaml
# Good: Log-uniform for learning rates
learning_rate:
  type: "loguniform"
  low: 1e-5
  high: 1e-3

# Good: Discrete choices for model architecture
lora_r:
  type: "choice"
  choices: [8, 16, 32, 64]

# Avoid: Too large search spaces
# batch_size:
#   type: "int"
#   low: 1
#   high: 1000  # Too large!
```

### 2. Optimal Trial Settings

```yaml
# For development
num_trials: 20
max_concurrent: 4

# For production
num_trials: 100
max_concurrent: 8
```

### 3. Early Stopping Configuration

```yaml
# Aggressive early stopping (faster, may miss good configs)
asha_grace_period: 1
asha_reduction_factor: 4

# Conservative early stopping (slower, more thorough)
asha_grace_period: 5
asha_reduction_factor: 2
```

## 🔗 Integration with Existing Workflow

### With Regular Training

```bash
# Run hyperparameter optimization first
python scripts/hyperopt.py --save-best-config best.yaml

# Then train with best configuration
python scripts/train.py --config best.yaml
```

### With WandB Integration

The optimizer automatically:
- 🚫 Disables WandB for individual trials (reduces clutter)
- ✅ Enables WandB for final training with best hyperparameters
- 📊 Logs optimization summary to main project

### With Existing Configs

```yaml
# Your existing config
model:
  model_name: "mistralai/Mistral-7B-v0.3"
  # ... other settings

training:
  num_train_epochs: 5
  # ... other settings

# Add hyperopt section
hyperopt:
  enabled: true
  strategy: "combined"
  # ... hyperopt settings
```

## 📚 References

- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Ray Tune Documentation](https://docs.ray.io/en/latest/tune/)
- [ASHA Paper](https://arxiv.org/abs/1810.05934)
- [TPE Algorithm](https://papers.nips.cc/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html)