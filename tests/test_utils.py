"""Tests for utility functions."""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import Config, DataConfig, LoggingConfig, ModelConfig, TrainingConfig
from src.utils import (
    check_gpu_memory,
    clear_gpu_cache,
    detect_mixed_precision_support,
    estimate_memory_usage,
    get_compute_dtype,
    list_offline_runs,
    load_checkpoint,
    print_trainable_parameters,
    save_model_and_tokenizer,
    setup_logging,
    setup_wandb_logging,
    sync_all_offline_runs,
    sync_offline_run,
    validate_wandb_config,
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
    with open(log_dir / "training.log") as f:
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
    for value in support.values():
        assert isinstance(value, bool)


def test_estimate_memory_usage():
    """Test memory usage estimation."""
    # Test with default parameters
    estimate = estimate_memory_usage(
        model_size_gb=14.0, batch_size=4, sequence_length=256, use_8bit=True, use_lora=True
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
        model_size_gb=14.0, batch_size=4, sequence_length=256, use_8bit=False, use_lora=False
    )

    # Full precision should use more memory
    assert estimate_full["total_memory_gb"] > estimate["total_memory_gb"]
    assert estimate_full["model_memory_gb"] == 14.0


def test_get_compute_dtype():
    """Test compute dtype selection."""
    dtype = get_compute_dtype()

    assert dtype in [torch.float32, torch.float16, torch.bfloat16]

    # Test with mocked CUDA availability
    with patch("torch.cuda.is_available", return_value=False):
        dtype_no_cuda = get_compute_dtype()
        assert dtype_no_cuda == torch.float32

    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.is_bf16_supported", return_value=True):
            dtype_bf16 = get_compute_dtype()
            assert dtype_bf16 == torch.bfloat16

        with patch("torch.cuda.is_bf16_supported", return_value=False):
            dtype_fp16 = get_compute_dtype()
            assert dtype_fp16 == torch.float16


def test_validate_wandb_config():
    """Test WandB configuration validation."""
    # Test valid config
    config = LoggingConfig()
    config.wandb_mode = "online"
    config.wandb_resume = "allow"
    config.wandb_api_key = "test_key"

    # Should not raise
    validate_wandb_config(config)

    # Test invalid mode
    config.wandb_mode = "invalid_mode"
    with pytest.raises(ValueError, match="wandb_mode must be one of"):
        validate_wandb_config(config)

    # Test invalid resume
    config.wandb_mode = "online"
    config.wandb_resume = "invalid_resume"
    with pytest.raises(ValueError, match="wandb_resume must be one of"):
        validate_wandb_config(config)

    # Test auto-fallback to offline when no API key
    config.wandb_resume = "allow"
    config.wandb_api_key = None

    with patch.dict(os.environ, {}, clear=True), pytest.warns(UserWarning, match="WANDB_API_KEY not found"):
        validate_wandb_config(config)
        assert config.wandb_mode == "offline"


def test_setup_logging_invalid_level():
    """Test setup_logging with invalid log level."""
    with pytest.raises(ValueError, match="Invalid log level"):
        setup_logging(log_level="invalid_level")


def test_print_trainable_parameters():
    """Test parameter counting function."""
    # Create a mock model with parameters
    mock_model = Mock()

    # Create mock parameters
    param1 = Mock()
    param1.numel.return_value = 1000
    param1.requires_grad = True

    param2 = Mock()
    param2.numel.return_value = 500
    param2.requires_grad = False

    param3 = Mock()
    param3.numel.return_value = 300
    param3.requires_grad = True

    mock_model.named_parameters.return_value = [
        ("layer1", param1),
        ("layer2", param2),
        ("layer3", param3),
    ]

    with patch("builtins.print") as mock_print:
        result = print_trainable_parameters(mock_model)

    assert result["trainable_params"] == 1300  # param1 + param3
    assert result["all_params"] == 1800  # all params
    assert result["trainable_percent"] == pytest.approx(72.2222, rel=1e-3)

    mock_print.assert_called_once()


def test_clear_gpu_cache():
    """Test GPU cache clearing."""
    with (
        patch("gc.collect") as mock_gc,
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.empty_cache") as mock_empty,
        patch("torch.cuda.synchronize") as mock_sync,
    ):
        clear_gpu_cache()

        mock_gc.assert_called_once()
        mock_empty.assert_called_once()
        mock_sync.assert_called_once()

    # Test without CUDA
    with patch("gc.collect") as mock_gc, patch("torch.cuda.is_available", return_value=False):
        clear_gpu_cache()
        mock_gc.assert_called_once()


def test_check_gpu_memory():
    """Test GPU memory checking."""
    # Test without CUDA
    with patch("torch.cuda.is_available", return_value=False):
        result = check_gpu_memory()
        assert result == {"error": "CUDA not available"}

    # Test with CUDA
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.device_count", return_value=2),
        patch("torch.cuda.memory_allocated") as mock_alloc,
        patch("torch.cuda.memory_reserved") as mock_reserved,
        patch("torch.cuda.get_device_properties") as mock_props,
    ):
        # Mock device properties
        mock_device_props = Mock()
        mock_device_props.total_memory = 24 * (1024**3)  # 24GB
        mock_props.return_value = mock_device_props

        mock_alloc.side_effect = [8 * (1024**3), 4 * (1024**3)]  # 8GB, 4GB
        mock_reserved.side_effect = [10 * (1024**3), 6 * (1024**3)]  # 10GB, 6GB

        result = check_gpu_memory()

        assert "gpu_0" in result
        assert "gpu_1" in result
        assert result["gpu_0"]["allocated_gb"] == 8.0
        assert result["gpu_0"]["total_gb"] == 24.0
        assert result["gpu_0"]["free_gb"] == 16.0


def test_detect_mixed_precision_support_detailed():
    """Test detailed mixed precision detection."""
    # Test without CUDA
    with patch("torch.cuda.is_available", return_value=False):
        support = detect_mixed_precision_support()
        assert support == {"fp16": False, "bf16": False, "tf32": False}

    # Test with CUDA but old GPU (capability < 8.0)
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.get_device_capability", return_value=(7, 5)),
        patch("torch.cuda.is_bf16_supported", return_value=False),
    ):
        support = detect_mixed_precision_support()
        assert support["fp16"] is True
        assert support["bf16"] is False
        assert support["tf32"] is False

    # Test with modern GPU (capability >= 8.0)
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.get_device_capability", return_value=(8, 0)),
        patch("torch.cuda.is_bf16_supported", return_value=True),
    ):
        support = detect_mixed_precision_support()
        assert support["fp16"] is True
        assert support["bf16"] is True
        assert support["tf32"] is True


def test_save_model_and_tokenizer(tmp_path):
    """Test model and tokenizer saving."""
    mock_model = Mock()
    mock_tokenizer = Mock()

    output_dir = tmp_path / "model_output"

    with patch("builtins.print") as mock_print:
        save_model_and_tokenizer(mock_model, mock_tokenizer, str(output_dir), is_peft=True)

    assert output_dir.exists()
    mock_model.save_pretrained.assert_called_once_with(str(output_dir))
    mock_tokenizer.save_pretrained.assert_called_once_with(str(output_dir))
    mock_print.assert_called_once_with(f"Model and tokenizer saved to {output_dir}")

    # Test with is_peft=False
    mock_model.reset_mock()
    mock_tokenizer.reset_mock()

    save_model_and_tokenizer(mock_model, mock_tokenizer, str(output_dir), is_peft=False)
    mock_model.save_pretrained.assert_called_once_with(str(output_dir))


def test_load_checkpoint(tmp_path):
    """Test checkpoint loading."""
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()

    # Create mock adapter config for PEFT model
    adapter_config = {"base_model_name_or_path": "test_model"}
    (checkpoint_path / "adapter_config.json").write_text(json.dumps(adapter_config))

    mock_config = Mock()
    mock_config.model.model_name = "test_model"
    mock_config.model.trust_remote_code = True

    mock_model_class = Mock()
    mock_tokenizer_class = Mock()
    mock_tokenizer = Mock()
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

    with (
        patch("src.model.setup_model") as mock_setup,
        patch("src.utils.PeftModel") as mock_peft,
        patch("logging.getLogger"),
    ):
        mock_base_model = Mock()
        mock_setup.return_value = (mock_base_model, None)
        mock_peft_model = Mock()
        mock_peft.from_pretrained.return_value = mock_peft_model

        model, tokenizer = load_checkpoint(str(checkpoint_path), mock_model_class, mock_tokenizer_class, mock_config)

        assert model == mock_peft_model
        assert tokenizer == mock_tokenizer
        mock_setup.assert_called_once()
        mock_peft.from_pretrained.assert_called_once()

    # Test loading regular model (no adapter_config.json)
    (checkpoint_path / "adapter_config.json").unlink()

    mock_regular_model = Mock()
    mock_model_class.from_pretrained.return_value = mock_regular_model

    with patch("logging.getLogger"):
        model, tokenizer = load_checkpoint(str(checkpoint_path), mock_model_class, mock_tokenizer_class, mock_config)

        assert model == mock_regular_model
        mock_model_class.from_pretrained.assert_called_once_with(
            str(checkpoint_path), device_map="auto", trust_remote_code=True
        )


def test_setup_wandb_logging():
    """Test WandB logging setup."""
    # Create mock config
    config = Config(model=ModelConfig(), data=DataConfig(), training=TrainingConfig(), logging=LoggingConfig())
    config.logging.use_wandb = True
    config.logging.wandb_mode = "offline"
    config.logging.wandb_project = "test_project"
    config.logging.wandb_run_id = "test_run_id"

    with patch("src.utils.wandb") as mock_wandb, patch("src.utils.validate_wandb_config") as mock_validate:
        setup_wandb_logging(config)

        mock_validate.assert_called_once_with(config.logging)
        mock_wandb.init.assert_called_once()

        # Check that run_id was included in init params
        call_args = mock_wandb.init.call_args[1]
        assert "id" in call_args
        assert call_args["id"] == "test_run_id"

    # Test without WandB
    config.logging.use_wandb = False
    with patch("src.utils.wandb") as mock_wandb:
        setup_wandb_logging(config)
        mock_wandb.init.assert_not_called()

    # Test with disabled mode
    config.logging.use_wandb = True
    config.logging.wandb_mode = "disabled"
    with patch("src.utils.wandb") as mock_wandb:
        setup_wandb_logging(config)
        mock_wandb.init.assert_not_called()


def test_list_offline_runs(tmp_path):
    """Test listing offline WandB runs."""
    wandb_dir = tmp_path / "wandb"
    wandb_dir.mkdir()

    # Create some offline run directories
    run1 = wandb_dir / "offline-run-20231201_120000-abc123"
    run1.mkdir()
    (run1 / "files" / "wandb-metadata.json").parent.mkdir(parents=True)
    (run1 / "files" / "wandb-metadata.json").write_text('{"test": "data"}')

    run2 = wandb_dir / "offline-run-20231202_130000-def456"
    run2.mkdir()
    (run2 / "files" / "config.yaml").parent.mkdir(parents=True)
    (run2 / "files" / "config.yaml").write_text("test: config")

    # Create a non-offline directory (should be ignored)
    (wandb_dir / "online-run-xyz").mkdir()

    runs = list_offline_runs(str(wandb_dir))

    assert len(runs) == 2
    run_ids = [run["run_id"] for run in runs]
    assert "offline-run-20231201_120000-abc123" in run_ids
    assert "offline-run-20231202_130000-def456" in run_ids

    # Check run info structure
    for run in runs:
        assert "run_id" in run
        assert "path" in run
        assert "created" in run
        assert "size_mb" in run

    # Test with non-existent directory
    empty_runs = list_offline_runs(str(tmp_path / "nonexistent"))
    assert empty_runs == []


def test_sync_offline_run():
    """Test syncing offline WandB run."""
    with patch("subprocess.run") as mock_run:
        # Test successful sync
        mock_run.return_value = Mock()
        result = sync_offline_run("/path/to/run")
        assert result is True
        mock_run.assert_called_once_with(["wandb", "sync", "/path/to/run"], capture_output=True, text=True, check=True)

    with patch("subprocess.run") as mock_run, patch("logging.getLogger") as mock_logger:
        # Test failed sync
        mock_run.side_effect = subprocess.CalledProcessError(1, "wandb")
        result = sync_offline_run("/path/to/run")
        assert result is False
        mock_logger.return_value.error.assert_called_once()


def test_sync_all_offline_runs():
    """Test syncing all offline runs."""
    mock_runs = [
        {"run_id": "run1", "path": "/path/to/run1"},
        {"run_id": "run2", "path": "/path/to/run2"},
        {"run_id": "run3", "path": "/path/to/run3"},
    ]

    with (
        patch("src.utils.list_offline_runs", return_value=mock_runs),
        patch("src.utils.sync_offline_run") as mock_sync,
    ):
        # Mock some successes and failures
        mock_sync.side_effect = [True, False, True]

        result = sync_all_offline_runs("/test/wandb")

        assert result["synced"] == ["run1", "run3"]
        assert result["failed"] == ["run2"]
        assert mock_sync.call_count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
