# WandB Offline Training Workflow

This document explains how to use Weights & Biases (WandB) in offline mode for training environments without internet connectivity.

## Overview

The enhanced WandB integration supports three modes:
- **online**: Default mode, requires internet and WandB API key
- **offline**: Stores logs locally for later syncing
- **disabled**: Completely disables WandB logging

## Quick Start - Offline Training

### 1. Use Offline Configuration

```bash
# Train with offline configuration
python scripts/train.py --config configs/offline.yaml

# Or override mode in any config
python scripts/train.py --wandb-mode offline
```

### 2. Monitor Offline Runs

```bash
# List all offline runs
python scripts/sync_wandb.py --list

# Check specific run
ls -la ./wandb/offline-run-*
```

### 3. Sync to WandB Cloud

```bash
# Set your API key
export WANDB_API_KEY=your_api_key_here

# Sync all offline runs
python scripts/sync_wandb.py

# Sync specific run
python scripts/sync_wandb.py --run-id offline-run-20231120_123456-abc123
```

## Configuration Options

### YAML Configuration

```yaml
logging:
  use_wandb: true
  wandb_mode: "offline"  # online, offline, disabled
  wandb_dir: "./wandb"   # Directory for offline storage
  wandb_project: "mistral-ner"
  wandb_name: "offline-experiment"
  wandb_tags: ["offline", "experiment"]
  wandb_notes: "Offline training run"
  wandb_resume: "allow"  # allow, must, never, auto
  wandb_run_id: null     # For resuming specific runs
```

### Command Line Override

```bash
# Basic offline training
python scripts/train.py --wandb-mode offline

# Offline with custom settings
python scripts/train.py \
    --wandb-mode offline \
    --wandb-name "my-offline-experiment" \
    --wandb-tags "offline" "baseline" \
    --wandb-notes "Testing offline functionality"

# Disable WandB completely
python scripts/train.py --no-wandb
```

## Environment Variables

All WandB settings can be controlled via environment variables:

```bash
export WANDB_MODE=offline
export WANDB_DIR=./wandb
export WANDB_PROJECT=mistral-ner
export WANDB_NAME=offline-experiment
export WANDB_TAGS=offline,experiment
export WANDB_NOTES="Offline training run"
```

## Offline Workflow

### 1. Training Phase (No Internet Required)

```bash
# Start offline training
python scripts/train.py --config configs/offline.yaml

# Training runs normally, logs stored locally in ./wandb/
# No internet connection required
```

### 2. Management Phase

```bash
# List offline runs with details
python scripts/sync_wandb.py --list

# Expected output:
# Found 2 offline runs:
#   - offline-run-20231120_123456-abc123 (45.2 MB)
#   - offline-run-20231120_134567-def456 (38.7 MB)

# Preview what would be synced
python scripts/sync_wandb.py --dry-run
```

### 3. Sync Phase (Internet Required)

```bash
# Set API key (get from https://wandb.ai/settings)
export WANDB_API_KEY=your_api_key_here

# Sync all runs
python scripts/sync_wandb.py

# Or sync specific run
python scripts/sync_wandb.py --run-id offline-run-20231120_123456-abc123
```

## Resuming Offline Runs

### Resume from Checkpoint

```bash
# Resume specific run
python scripts/train.py \
    --wandb-mode offline \
    --wandb-run-id abc123def456 \
    --wandb-resume must \
    --resume-from-checkpoint ./mistral-ner-finetuned/checkpoint-100
```

### Continue Offline Series

```bash
# Start new run with continuation tags
python scripts/train.py \
    --wandb-mode offline \
    --wandb-name "experiment-part-2" \
    --wandb-tags "offline" "continuation" \
    --wandb-notes "Continuing from previous offline run"
```

## Troubleshooting

### No API Key Warning

If you see this warning during online mode:
```
WANDB_API_KEY not found and no api key provided. Switching to offline mode for this session.
```

**Solution**: The system automatically falls back to offline mode. Set your API key when ready to sync:
```bash
export WANDB_API_KEY=your_api_key_here
```

### Sync Failures

```bash
# Check run integrity
ls -la ./wandb/offline-run-*/

# Try syncing with verbose output
wandb sync ./wandb/offline-run-20231120_123456-abc123 --verbose

# Clean corrupted run (last resort)
rm -rf ./wandb/offline-run-corrupted-id
```

### Large Offline Runs

For runs with large artifacts:

```bash
# Check sizes before syncing
python scripts/sync_wandb.py --list

# Sync specific large runs one at a time
python scripts/sync_wandb.py --run-id large-run-id
```

## Best Practices

### 1. Directory Management

```bash
# Keep offline runs organized
export WANDB_DIR=/data/wandb-offline

# Or use date-based directories
export WANDB_DIR=./wandb/$(date +%Y-%m-%d)
```

### 2. Tagging Strategy

```yaml
wandb_tags:
  - "offline"           # Always tag offline runs
  - "experiment-v1"     # Experiment version
  - "gpu-count-4"       # Hardware info
  - "batch-size-32"     # Key hyperparameters
```

### 3. Regular Sync Schedule

```bash
# Create sync script for regular intervals
#!/bin/bash
# sync-daily.sh
export WANDB_API_KEY=your_key
python scripts/sync_wandb.py
echo "Sync completed at $(date)"
```

### 4. Storage Cleanup

```bash
# After successful sync, clean old runs
# (Make sure sync was successful first!)
find ./wandb -name "offline-run-*" -mtime +7 -exec rm -rf {} \;
```

## Integration Examples

### Slurm Job Script

```bash
#!/bin/bash
#SBATCH --job-name=mistral-ner-offline
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

# No internet on compute nodes - use offline mode
python scripts/train.py \
    --config configs/offline.yaml \
    --wandb-name "slurm-job-${SLURM_JOB_ID}"

# Sync after job completes (if login node has internet)
if [ $? -eq 0 ]; then
    python scripts/sync_wandb.py
fi
```

### Docker Container

```dockerfile
# Dockerfile
FROM python:3.11

# Copy offline config
COPY configs/offline.yaml /app/configs/
COPY scripts/ /app/scripts/

# Run offline by default
CMD ["python", "scripts/train.py", "--config", "configs/offline.yaml"]
```

### Jupyter Notebook

```python
# notebook_training.ipynb
import os
os.environ['WANDB_MODE'] = 'offline'

from src.config import Config
config = Config.from_yaml('configs/offline.yaml')
config.setup_wandb()

# Training proceeds normally, logs stored locally
```

## Advanced Features

### Custom Offline Directory Structure

```python
# Custom directory naming
import datetime
wandb_dir = f"./experiments/{datetime.date.today()}/wandb"
config.logging.wandb_dir = wandb_dir
```

### Batch Sync Operations

```bash
# Sync runs matching pattern
for run in ./wandb/offline-run-*experiment*; do
    python scripts/sync_wandb.py --run-id $(basename $run)
done
```

### Monitoring Disk Usage

```bash
# Monitor offline storage usage
du -sh ./wandb/
python scripts/sync_wandb.py --list | grep MB
```

This workflow enables seamless ML experimentation in offline environments while maintaining full WandB integration capabilities.