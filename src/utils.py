"""Utility functions for Mistral NER fine-tuning."""

from __future__ import annotations

import gc
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import wandb
from peft import PeftModel
from transformers import PreTrainedModel, PreTrainedTokenizerBase

if TYPE_CHECKING:
    from .config import Config, LoggingConfig


def validate_wandb_config(logging_config: LoggingConfig) -> None:
    """Validate WandB configuration and auto-adjust if needed."""
    import warnings
    
    # Validate mode
    valid_modes = {"online", "offline", "disabled"}
    if logging_config.wandb_mode not in valid_modes:
        raise ValueError(f"wandb_mode must be one of {valid_modes}, got {logging_config.wandb_mode}")
    
    # Validate resume strategy
    valid_resume = {"allow", "must", "never", "auto"}
    if logging_config.wandb_resume not in valid_resume:
        raise ValueError(f"wandb_resume must be one of {valid_resume}, got {logging_config.wandb_resume}")
    
    # Auto-fallback to offline if no API key and online mode
    if logging_config.wandb_mode == "online":
        api_key = logging_config.wandb_api_key or os.getenv("WANDB_API_KEY")
        if not api_key:
            warnings.warn(
                "WANDB_API_KEY not found and no api key provided. "
                "Switching to offline mode for this session.",
                UserWarning,
                stacklevel=2
            )
            logging_config.wandb_mode = "offline"


def setup_logging(log_level: str = "info", log_dir: str = "./logs") -> logging.Logger:
    """Setup logging configuration."""
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    # Convert log level string to logging constant
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    # Create logger
    logger = logging.getLogger("mistral_ner")
    logger.setLevel(numeric_level)

    # Remove existing handlers
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_dir_path / "training.log")
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)

    return logger


def print_trainable_parameters(model: PreTrainedModel) -> dict[str, Any]:
    """Print and return the number of trainable parameters in the model."""
    trainable_params = 0
    all_param = 0

    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    trainable_percent = 100 * trainable_params / all_param

    result = {"trainable_params": trainable_params, "all_params": all_param, "trainable_percent": trainable_percent}

    print(f"trainable params: {trainable_params:,} || all params: {all_param:,} || trainable%: {trainable_percent:.4f}")

    return result


def clear_gpu_cache() -> None:
    """Clear GPU cache to free up memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def check_gpu_memory() -> dict[str, Any]:
    """Check GPU memory usage."""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    memory_info = {}
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(i) / 1024**3  # GB
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB

        memory_info[f"gpu_{i}"] = {
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "total_gb": round(total, 2),
            "free_gb": round(total - allocated, 2),
            "utilization_percent": round((allocated / total) * 100, 2),
        }

    return memory_info


def detect_mixed_precision_support() -> dict[str, bool]:
    """Detect which mixed precision training is supported."""
    support = {"fp16": False, "bf16": False, "tf32": False}

    if torch.cuda.is_available():
        # Check FP16 support
        support["fp16"] = True  # Generally supported on all CUDA GPUs

        # Check BF16 support (Ampere and newer)
        device_capability = torch.cuda.get_device_capability()
        if device_capability[0] >= 8:  # Ampere (SM 8.0) and newer
            support["bf16"] = True

        # Check TF32 support
        support["tf32"] = torch.cuda.is_bf16_supported()

    return support


def save_model_and_tokenizer(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, output_dir: str | Path, is_peft: bool = True
) -> None:
    """Save model and tokenizer to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save model
    if is_peft:
        model.save_pretrained(output_dir)
    else:
        model.save_pretrained(output_dir)

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)

    print(f"Model and tokenizer saved to {output_dir}")


def load_checkpoint(
    checkpoint_path: str | Path,
    model_class: type[PreTrainedModel],
    tokenizer_class: type[PreTrainedTokenizerBase],
    config: Config,
    device_map: str = "auto",
) -> tuple[PreTrainedModel | PeftModel, PreTrainedTokenizerBase]:
    """Load model and tokenizer from checkpoint."""
    from .model import setup_model

    logger = logging.getLogger("mistral_ner")
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    # Load tokenizer
    tokenizer = tokenizer_class.from_pretrained(checkpoint_path)

    # Check if it's a PEFT model
    peft_config_path = Path(checkpoint_path) / "adapter_config.json"

    if peft_config_path.exists():
        # Load PEFT model
        model, _ = setup_model(model_name=config.model.model_name, config=config, device_map=device_map)
        model = PeftModel.from_pretrained(model, checkpoint_path)
    else:
        # Load regular model
        model = model_class.from_pretrained(
            checkpoint_path, device_map=device_map, trust_remote_code=config.model.trust_remote_code
        )

    return model, tokenizer


def setup_wandb_logging(config: Config) -> None:
    """Setup Weights & Biases logging with enhanced offline support."""
    if config.logging.use_wandb and config.logging.wandb_mode != "disabled":
        # Validate configuration first
        validate_wandb_config(config.logging)
        
        wandb_config = {
            "model_name": config.model.model_name,
            "num_train_epochs": config.training.num_train_epochs,
            "per_device_train_batch_size": config.training.per_device_train_batch_size,
            "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
            "learning_rate": config.training.learning_rate,
            "warmup_ratio": config.training.warmup_ratio,
            "weight_decay": config.training.weight_decay,
            "lora_r": config.model.lora_r,
            "lora_alpha": config.model.lora_alpha,
            "lora_dropout": config.model.lora_dropout,
            "max_length": config.data.max_length,
        }

        # Prepare wandb.init parameters
        init_params = {
            "project": config.logging.wandb_project,
            "entity": config.logging.wandb_entity,
            "name": config.logging.wandb_name,
            "tags": config.logging.wandb_tags,
            "notes": config.logging.wandb_notes,
            "config": wandb_config,
            "mode": config.logging.wandb_mode,  # type: ignore[arg-type]
            "dir": config.logging.wandb_dir,
            "resume": config.logging.wandb_resume,
        }
        
        # Add run_id if specified for resuming
        if config.logging.wandb_run_id:
            init_params["id"] = config.logging.wandb_run_id
            
        wandb.init(**init_params)


def list_offline_runs(wandb_dir: str = "./wandb") -> list[dict[str, Any]]:
    """List all offline WandB runs in the specified directory."""
    wandb_path = Path(wandb_dir)
    offline_runs = []
    
    if not wandb_path.exists():
        return offline_runs
    
    # Look for offline run directories
    for run_dir in wandb_path.iterdir():
        if run_dir.is_dir() and run_dir.name.startswith("offline-run-"):
            run_info = {
                "run_id": run_dir.name,
                "path": str(run_dir),
                "created": run_dir.stat().st_ctime,
                "size_mb": sum(f.stat().st_size for f in run_dir.rglob("*") if f.is_file()) / (1024 * 1024)
            }
            offline_runs.append(run_info)
    
    return sorted(offline_runs, key=lambda x: x["created"], reverse=True)


def sync_offline_run(run_path: str) -> bool:
    """Sync a specific offline run to WandB servers."""
    try:
        # Use wandb sync command
        import subprocess
        _ = subprocess.run(
            ["wandb", "sync", run_path],
            capture_output=True,
            text=True,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        logging.getLogger("mistral_ner").error(f"Failed to sync run {run_path}: {e}")
        return False


def sync_all_offline_runs(wandb_dir: str = "./wandb") -> dict[str, Any]:
    """Sync all offline runs to WandB servers."""
    offline_runs = list_offline_runs(wandb_dir)
    results = {"synced": [], "failed": []}
    
    for run in offline_runs:
        if sync_offline_run(run["path"]):
            results["synced"].append(run["run_id"])
        else:
            results["failed"].append(run["run_id"])
    
    return results


def get_compute_dtype() -> torch.dtype:
    """Get the appropriate compute dtype based on hardware support."""
    if torch.cuda.is_available():
        # Check for BF16 support
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        else:
            return torch.float16
    return torch.float32


def estimate_memory_usage(
    model_size_gb: float = 14.0,  # Mistral-7B is ~14GB in float32
    batch_size: int = 4,
    sequence_length: int = 256,
    use_8bit: bool = True,
    use_lora: bool = True,
) -> dict[str, float]:
    """Estimate memory usage for training."""
    # Base model memory
    model_memory = model_size_gb * 0.25 if use_8bit else model_size_gb  # 8-bit reduces to ~25%

    # LoRA memory overhead (minimal)
    lora_memory = 0.1 if use_lora else 0  # ~100MB for LoRA adapters

    # Activation memory (rough estimate)
    activation_memory = (batch_size * sequence_length * 4096 * 4) / 1024**3  # ~4GB per batch

    # Optimizer memory (AdamW stores 2 states per parameter)
    optimizer_memory = lora_memory * 2 if use_lora else model_memory * 2

    # Gradient memory
    gradient_memory = model_memory if not use_lora else lora_memory

    total_memory = model_memory + lora_memory + activation_memory + optimizer_memory + gradient_memory

    return {
        "model_memory_gb": round(model_memory, 2),
        "lora_memory_gb": round(lora_memory, 2),
        "activation_memory_gb": round(activation_memory, 2),
        "optimizer_memory_gb": round(optimizer_memory, 2),
        "gradient_memory_gb": round(gradient_memory, 2),
        "total_memory_gb": round(total_memory, 2),
        "recommended_gpu_memory_gb": round(total_memory * 1.2, 2),  # 20% overhead
    }
