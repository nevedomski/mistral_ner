"""Tests for utility functions."""

import pytest
import sys
from pathlib import Path
import torch
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import (
    setup_logging,
    detect_mixed_precision_support,
    estimate_memory_usage,
    get_compute_dtype
)


def test_setup_logging(tmp_path):
    """Test logging setup."""
    log_dir = tmp_path / "logs"
    logger = setup_logging(log_level="debug", log_dir=str(log_dir))
    
    assert isinstance(logger, logging.Logger)
    assert logger.level == logging.DEBUG
    assert log_dir.exists()
    assert (log_dir / "training.log").exists()
    
    # Test logging
    logger.info("Test message")
    
    # Check log file contains message
    with open(log_dir / "training.log", "r") as f:
        content = f.read()
        assert "Test message" in content


def test_detect_mixed_precision_support():
    """Test mixed precision detection."""
    support = detect_mixed_precision_support()
    
    assert isinstance(support, dict)
    assert "fp16" in support
    assert "bf16" in support
    assert "tf32" in support
    
    # All values should be boolean
    for key, value in support.items():
        assert isinstance(value, bool)


def test_estimate_memory_usage():
    """Test memory usage estimation."""
    # Test with default parameters
    estimate = estimate_memory_usage(
        model_size_gb=14.0,
        batch_size=4,
        sequence_length=256,
        use_8bit=True,
        use_lora=True
    )
    
    assert isinstance(estimate, dict)
    assert "model_memory_gb" in estimate
    assert "lora_memory_gb" in estimate
    assert "total_memory_gb" in estimate
    assert "recommended_gpu_memory_gb" in estimate
    
    # With 8-bit and LoRA, model memory should be reduced
    assert estimate["model_memory_gb"] < 14.0
    assert estimate["lora_memory_gb"] < 1.0
    
    # Test without optimizations
    estimate_full = estimate_memory_usage(
        model_size_gb=14.0,
        batch_size=4,
        sequence_length=256,
        use_8bit=False,
        use_lora=False
    )
    
    # Full precision should use more memory
    assert estimate_full["total_memory_gb"] > estimate["total_memory_gb"]
    assert estimate_full["model_memory_gb"] == 14.0


def test_get_compute_dtype():
    """Test compute dtype selection."""
    dtype = get_compute_dtype()
    
    assert dtype in [torch.float32, torch.float16, torch.bfloat16]
    
    # If CUDA is available, should not be float32
    if torch.cuda.is_available():
        assert dtype != torch.float32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])