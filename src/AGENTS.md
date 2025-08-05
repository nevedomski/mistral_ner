# AGENTS.md - src/ Directory

## Overview
Core source code directory containing the main implementation of the Mistral NER v2 framework.

## Architecture

### Configuration System
- **config.py**: Main configuration aggregator using Hydra + Pydantic
- **config_*.py**: Modular configuration components (model, training, datasets, logging, hyperopt)
- **Type-safe validation**: All configs use Pydantic with comprehensive validators

### Core Components
- **model.py**: Model initialization and configuration (Mistral-7B with QLoRA)
- **training.py**: Training loop implementation with memory management
- **evaluation.py**: Evaluation metrics and seqeval integration
- **data.py**: Data loading and tokenization pipeline
- **utils.py**: Utility functions (logging, GPU management, etc.)

### Incomplete Components (Prefixed with _)
- **_batch_balancing.py**: Batch balancing strategies (not integrated)
- **_losses.py**: Advanced loss functions (focal loss, class balancing)
- **_schedulers.py**: Learning rate schedulers (not integrated)
- **_visualization.py**: Visualization utilities (not integrated)

## Critical Issues

### 1. Configuration Coupling
- **Location**: config.py line 77-85
- **Issue**: setup_wandb validator modifies training.report_to based on logging settings
- **Impact**: Creates tight coupling between configuration modules
- **DOCDEV-TODO**: Refactor to use factory pattern or separate resolution phase

### 2. Data Processing Validation
- **Location**: data.py line 83-96
- **Issue**: Missing validation for edge cases in tokenization
- **Impact**: Potential silent data corruption
- **DOCDEV-TODO**: Add validation for empty word_ids and label length mismatches

### 3. Incomplete Multi-Dataset Support
- **Location**: config_datasets.py line 138-148
- **Issue**: Core multi-dataset feature only supports single dataset
- **Impact**: Cannot use advertised multi-dataset training
- **DOCDEV-TODO**: Implement concat and weighted mixing strategies

## Best Practices

### Adding New Configurations
1. Create new config_*.py file with Pydantic dataclass
2. Add type hints and Field descriptions
3. Include validators for complex logic
4. Update Config class in config.py to include new component

### Error Handling
- Use structured logging with appropriate levels
- Provide clear error messages with context
- Handle GPU OOM errors gracefully
- Validate all external inputs

## Performance Considerations
- GPU memory is actively managed with cache clearing
- Mixed precision training supported (fp16/bf16)
- Gradient accumulation for large effective batch sizes
- Memory monitoring callbacks available

## Security Considerations
- All dataset inputs should be validated
- Network requests need proper timeouts and SSL verification
- Avoid dynamic code execution (eval, exec)
- Sanitize all user-provided configurations

## Future Improvements
1. Complete integration of underscore-prefixed modules
2. Decouple configuration validation logic
3. Add comprehensive unit testing
4. Implement streaming dataset support
5. Add distributed training capabilities