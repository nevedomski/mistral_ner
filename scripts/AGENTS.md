# AGENTS.md - scripts/ Directory

## Overview
Entry point scripts for training and data loading operations.

## Scripts

### train.py
- **Purpose**: Main training script orchestrator
- **Functionality**: 
  - Initializes Hydra configuration
  - Sets up model, data, and training components
  - Manages the training loop
  - Handles checkpointing and evaluation

### load_data.py
- **Purpose**: Standalone data loading and inspection script
- **Functionality**:
  - Test dataset loading without full training
  - Inspect dataset statistics
  - Validate data preprocessing

## Usage Patterns

### Standard Training
```bash
uv run python scripts/train.py
```

### Custom Configuration
```bash
uv run python scripts/train.py --config-name=multi_dataset
```

### Override Parameters
```bash
uv run python scripts/train.py data.dataset_name=conll2003 training.num_epochs=5
```

## Integration with Main Module
- Scripts use Hydra's @hydra.main decorator
- Configuration is loaded from configs/ directory
- All heavy lifting delegated to src/ modules

## Best Practices
1. Keep scripts thin - logic belongs in src/
2. Use Hydra for all configuration management
3. Add proper error handling and logging
4. Document command-line usage in docstrings

## Common Issues
- Hydra working directory changes - use absolute paths
- Configuration composition order matters
- Override syntax must match config structure exactly