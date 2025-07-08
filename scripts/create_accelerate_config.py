#!/usr/bin/env python3
"""Create a minimal accelerate config that avoids duplication with training config."""

import sys
from pathlib import Path

import yaml

# Add parent directory to path to import project modules
sys.path.append(str(Path(__file__).parent.parent))

from src.config import Config


def determine_mixed_precision(config: Config) -> str:
    """Determine mixed precision setting from project config."""
    if config.training.bf16:
        return "bf16"
    elif config.training.fp16:
        return "fp16"
    else:
        return "no"


def create_minimal_accelerate_config(config_file: str = "configs/default.yaml"):
    """Create a minimal accelerate config that doesn't duplicate settings."""

    # Load project configuration
    try:
        project_config = Config.from_yaml(config_file)
        mixed_precision = determine_mixed_precision(project_config)
        print(f"Detected mixed precision from project config: {mixed_precision}")
    except Exception as e:
        print(f"Warning: Could not load project config from {config_file}: {e}")
        print("Using default mixed precision: no")
        mixed_precision = "no"

    # Detect available GPUs
    import torch

    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPU(s)")

    # Determine distributed type based on GPU count
    if num_gpus == 0:
        distributed_type = "NO"
        use_cpu = True
        print("No GPUs detected, will use CPU")
    elif num_gpus == 1:
        distributed_type = "NO"  # Single GPU doesn't need distributed
        use_cpu = False
        print("Single GPU detected, using non-distributed mode")
    else:
        distributed_type = "MULTI_GPU"
        use_cpu = False
        print(f"Multiple GPUs detected ({num_gpus}), using distributed mode")

    # Minimal config that lets training script handle most settings
    config = {
        "compute_environment": "LOCAL_MACHINE",
        "distributed_type": distributed_type,  # Auto-detected based on GPU count
        "machine_rank": 0,
        "main_training_function": "main",
        "num_machines": 1,
        "mixed_precision": mixed_precision,  # Set from project config
        # Don't specify these - let them be auto-detected:
        # - num_processes (auto-detected based on available GPUs)
        # - gpu_ids (auto-detected)
        "use_cpu": use_cpu,
        # PyTorch 2.0+ optimization backend
        # Options: 'no', 'eager', 'aot_eager', 'inductor', 'aot_ts_nvfuser', 'nvprims_nvfuser', 'cudagraphs', 'ofi', 'fx2trt', 'onnxrt', 'ipex'
        "dynamo_backend": "no",  # Disable by default, can be overridden via CLI
        # Explicitly disable deprecated IPEX settings to avoid warnings
        "ipex_config": {},  # Empty dict instead of {"enabled": false} to avoid deprecation warning
    }

    # Save to project directory
    config_path = Path("accelerate_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\nCreated minimal accelerate config at: {config_path}")
    print("\nConfig contents:")
    print(yaml.dump(config, default_flow_style=False, sort_keys=False))

    print("\nUsage:")
    print("  accelerate launch --config_file accelerate_config.yaml scripts/train.py")
    print("\nOr override settings:")
    print("  accelerate launch --config_file accelerate_config.yaml --mixed_precision fp16 scripts/train.py")
    print("  accelerate launch --config_file accelerate_config.yaml --num_processes 4 scripts/train.py")
    print("  accelerate launch --config_file accelerate_config.yaml --dynamo_backend inductor scripts/train.py")

    print("\nNote: This config is designed to work with your project's training configuration")
    print("and avoid duplication of settings.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create minimal accelerate config")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to project config file (default: configs/default.yaml)",
    )
    args = parser.parse_args()

    create_minimal_accelerate_config(args.config)
