"""Tests for utility functions."""

import json
import logging
import os
import subprocess
import sys
import warnings
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


def test_setup_logging_invalid_level_duplicate(tmp_path):
    """Test logging setup with invalid log level."""
    log_dir = tmp_path / "logs"
    with pytest.raises(ValueError, match="Invalid log level"):
        setup_logging(log_level="invalid", log_dir=str(log_dir))


def test_validate_wandb_config_valid():
    """Test validate_wandb_config with valid configuration."""
    mock_config = Mock()
    mock_config.wandb_mode = "online"
    mock_config.wandb_resume = "allow"
    mock_config.wandb_api_key = "test_key"

    # Should not raise any exceptions
    validate_wandb_config(mock_config)


def test_validate_wandb_config_invalid_mode():
    """Test validate_wandb_config with invalid mode."""
    mock_config = Mock()
    mock_config.wandb_mode = "invalid"
    mock_config.wandb_resume = "allow"

    with pytest.raises(ValueError, match="wandb_mode must be one of"):
        validate_wandb_config(mock_config)


def test_validate_wandb_config_invalid_resume():
    """Test validate_wandb_config with invalid resume strategy."""
    mock_config = Mock()
    mock_config.wandb_mode = "online"
    mock_config.wandb_resume = "invalid"

    with pytest.raises(ValueError, match="wandb_resume must be one of"):
        validate_wandb_config(mock_config)


def test_validate_wandb_config_missing_api_key():
    """Test validate_wandb_config with missing API key triggers fallback."""
    mock_config = Mock()
    mock_config.wandb_mode = "online"
    mock_config.wandb_resume = "allow"
    mock_config.wandb_api_key = None

    with patch.dict("os.environ", {}, clear=True), warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validate_wandb_config(mock_config)
        assert len(w) == 1
        assert "WANDB_API_KEY not found" in str(w[0].message)
        assert mock_config.wandb_mode == "offline"


def test_print_trainable_parameters_duplicate():
    """Test print_trainable_parameters function."""
    # Create a mock model with parameters
    mock_model = Mock()
    mock_param1 = Mock()
    mock_param1.numel.return_value = 1000
    mock_param1.requires_grad = True

    mock_param2 = Mock()
    mock_param2.numel.return_value = 500
    mock_param2.requires_grad = False

    mock_model.named_parameters.return_value = [
        ("layer1.weight", mock_param1),
        ("layer2.bias", mock_param2),
    ]

    with patch("builtins.print") as mock_print:
        result = print_trainable_parameters(mock_model)

        expected_result = {"trainable_params": 1000, "all_params": 1500, "trainable_percent": 66.6667}

        assert result["trainable_params"] == expected_result["trainable_params"]
        assert result["all_params"] == expected_result["all_params"]
        assert abs(result["trainable_percent"] - expected_result["trainable_percent"]) < 0.01
        mock_print.assert_called_once()


@patch("src.utils.torch.cuda.is_available")
@patch("src.utils.torch.cuda.empty_cache")
@patch("src.utils.torch.cuda.synchronize")
@patch("src.utils.gc.collect")
def test_clear_gpu_cache_with_cuda(mock_gc, mock_sync, mock_empty, mock_available):
    """Test clear_gpu_cache when CUDA is available."""
    mock_available.return_value = True

    clear_gpu_cache()

    mock_gc.assert_called_once()
    mock_empty.assert_called_once()
    mock_sync.assert_called_once()


@patch("src.utils.torch.cuda.is_available")
@patch("src.utils.gc.collect")
def test_clear_gpu_cache_without_cuda(mock_gc, mock_available):
    """Test clear_gpu_cache when CUDA is not available."""
    mock_available.return_value = False

    clear_gpu_cache()

    mock_gc.assert_called_once()


@patch("src.utils.torch.cuda.is_available")
def test_check_gpu_memory_without_cuda(mock_available):
    """Test check_gpu_memory when CUDA is not available."""
    mock_available.return_value = False

    result = check_gpu_memory()

    assert result == {"error": "CUDA not available"}


@patch("src.utils.torch.cuda.is_available")
@patch("src.utils.torch.cuda.device_count")
@patch("src.utils.torch.cuda.memory_allocated")
@patch("src.utils.torch.cuda.memory_reserved")
@patch("src.utils.torch.cuda.get_device_properties")
def test_check_gpu_memory_with_cuda(mock_props, mock_reserved, mock_allocated, mock_count, mock_available):
    """Test check_gpu_memory when CUDA is available."""
    mock_available.return_value = True
    mock_count.return_value = 2
    mock_allocated.side_effect = [2 * 1024**3, 4 * 1024**3]  # 2GB, 4GB
    mock_reserved.side_effect = [3 * 1024**3, 5 * 1024**3]  # 3GB, 5GB

    mock_device_props = Mock()
    mock_device_props.total_memory = 8 * 1024**3  # 8GB
    mock_props.return_value = mock_device_props

    result = check_gpu_memory()

    assert "gpu_0" in result
    assert "gpu_1" in result
    assert result["gpu_0"]["allocated_gb"] == 2.0
    assert result["gpu_0"]["total_gb"] == 8.0
    assert result["gpu_0"]["free_gb"] == 6.0


@patch("src.utils.torch.cuda.is_available")
@patch("src.utils.torch.cuda.get_device_capability")
@patch("src.utils.torch.cuda.is_bf16_supported")
def test_detect_mixed_precision_support_with_cuda(mock_bf16, mock_capability, mock_available):
    """Test detect_mixed_precision_support with CUDA and various capabilities."""
    mock_available.return_value = True
    mock_capability.return_value = (8, 0)  # Ampere architecture
    mock_bf16.return_value = True

    result = detect_mixed_precision_support()

    assert result["fp16"] is True
    assert result["bf16"] is True
    assert result["tf32"] is True


@patch("src.utils.torch.cuda.is_available")
def test_detect_mixed_precision_support_without_cuda(mock_available):
    """Test detect_mixed_precision_support without CUDA."""
    mock_available.return_value = False

    result = detect_mixed_precision_support()

    assert result["fp16"] is False
    assert result["bf16"] is False
    assert result["tf32"] is False


def test_save_model_and_tokenizer_duplicate(tmp_path):
    """Test save_model_and_tokenizer function."""
    mock_model = Mock()
    mock_tokenizer = Mock()
    output_dir = tmp_path / "model_output"

    with patch("builtins.print") as mock_print:
        save_model_and_tokenizer(mock_model, mock_tokenizer, str(output_dir))

        mock_model.save_pretrained.assert_called_once_with(str(output_dir))
        mock_tokenizer.save_pretrained.assert_called_once_with(str(output_dir))
        mock_print.assert_called_once()
        assert output_dir.exists()


@patch("src.utils.logging.getLogger")
def test_load_checkpoint_peft_model(mock_logger_get, tmp_path):
    """Test load_checkpoint with PEFT model."""
    from src.utils import PeftModel

    mock_logger = Mock()
    mock_logger_get.return_value = mock_logger

    # Create actual directory structure
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    (checkpoint_path / "adapter_config.json").write_text('{"model_type": "test"}')

    mock_tokenizer_class = Mock()
    mock_tokenizer = Mock()
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

    mock_model_class = Mock()
    mock_config = Mock()
    mock_config.model.model_name = "test_model"

    with patch("src.model.setup_model") as mock_setup, patch.object(PeftModel, "from_pretrained") as mock_peft_load:
        mock_base_model = Mock()
        mock_setup.return_value = (mock_base_model, None)
        mock_peft_model = Mock()
        mock_peft_load.return_value = mock_peft_model

        result_model, result_tokenizer = load_checkpoint(
            str(checkpoint_path), mock_model_class, mock_tokenizer_class, mock_config
        )

        assert result_model == mock_peft_model
        assert result_tokenizer == mock_tokenizer
        mock_logger.info.assert_called_once()


@patch("src.utils.logging.getLogger")
def test_load_checkpoint_regular_model(mock_logger_get, tmp_path):
    """Test load_checkpoint with regular model."""
    mock_logger = Mock()
    mock_logger_get.return_value = mock_logger

    # Create directory without adapter_config.json
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()

    mock_tokenizer_class = Mock()
    mock_tokenizer = Mock()
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

    mock_model_class = Mock()
    mock_model = Mock()
    mock_model_class.from_pretrained.return_value = mock_model

    mock_config = Mock()
    mock_config.model.trust_remote_code = True

    result_model, result_tokenizer = load_checkpoint(
        str(checkpoint_path), mock_model_class, mock_tokenizer_class, mock_config
    )

    assert result_model == mock_model
    assert result_tokenizer == mock_tokenizer
    mock_logger.info.assert_called_once()


@patch("src.utils.wandb.init")
def test_setup_wandb_logging_duplicate(mock_wandb_init):
    """Test setup_wandb_logging function."""
    mock_config = Mock()
    mock_config.logging.use_wandb = True
    mock_config.logging.wandb_mode = "online"
    mock_config.logging.wandb_project = "test_project"
    mock_config.logging.wandb_entity = "test_entity"
    mock_config.logging.wandb_name = "test_run"
    mock_config.logging.wandb_tags = ["test"]
    mock_config.logging.wandb_notes = "test notes"
    mock_config.logging.wandb_dir = "./wandb"
    mock_config.logging.wandb_resume = "allow"
    mock_config.logging.wandb_run_id = "test_id"
    mock_config.logging.wandb_api_key = "test_key"

    # Model config
    mock_config.model.model_name = "test_model"
    mock_config.model.lora_r = 16
    mock_config.model.lora_alpha = 32
    mock_config.model.lora_dropout = 0.1

    # Training config
    mock_config.training.num_train_epochs = 3
    mock_config.training.per_device_train_batch_size = 4
    mock_config.training.gradient_accumulation_steps = 8
    mock_config.training.learning_rate = 2e-4
    mock_config.training.warmup_ratio = 0.1
    mock_config.training.weight_decay = 0.01

    # Data config
    mock_config.data.max_length = 256

    with patch("src.utils.validate_wandb_config"):
        setup_wandb_logging(mock_config)

        mock_wandb_init.assert_called_once()
        call_args = mock_wandb_init.call_args[1]
        assert call_args["project"] == "test_project"
        assert call_args["id"] == "test_id"


def test_list_offline_runs_no_directory(tmp_path):
    """Test list_offline_runs when wandb directory doesn't exist."""
    non_existent_dir = tmp_path / "non_existent"
    result = list_offline_runs(str(non_existent_dir))
    assert result == []


def test_list_offline_runs_with_runs(tmp_path):
    """Test list_offline_runs with actual offline run directories."""
    wandb_dir = tmp_path / "wandb"
    wandb_dir.mkdir()

    # Create mock offline run directories
    run1_dir = wandb_dir / "offline-run-123"
    run1_dir.mkdir()
    (run1_dir / "test_file.txt").write_text("test content")

    run2_dir = wandb_dir / "offline-run-456"
    run2_dir.mkdir()
    (run2_dir / "test_file2.txt").write_text("test content 2")

    # Create non-offline run directory (should be ignored)
    other_dir = wandb_dir / "some-other-dir"
    other_dir.mkdir()

    result = list_offline_runs(str(wandb_dir))

    assert len(result) == 2
    run_ids = [run["run_id"] for run in result]
    assert "offline-run-123" in run_ids
    assert "offline-run-456" in run_ids

    # Check structure
    for run in result:
        assert "run_id" in run
        assert "path" in run
        assert "created" in run
        assert "size_mb" in run


@patch("subprocess.run")
@patch("src.utils.logging.getLogger")
def test_sync_offline_run_success(mock_logger_get, mock_subprocess):
    """Test sync_offline_run successful execution."""
    mock_logger = Mock()
    mock_logger_get.return_value = mock_logger

    mock_subprocess.return_value = Mock()

    result = sync_offline_run("/path/to/run")

    assert result is True
    mock_subprocess.assert_called_once_with(
        ["wandb", "sync", "/path/to/run"], capture_output=True, text=True, check=True
    )


@patch("subprocess.run")
@patch("src.utils.logging.getLogger")
def test_sync_offline_run_failure(mock_logger_get, mock_subprocess):
    """Test sync_offline_run with subprocess failure."""
    mock_logger = Mock()
    mock_logger_get.return_value = mock_logger

    mock_subprocess.side_effect = subprocess.CalledProcessError(1, "wandb")

    result = sync_offline_run("/path/to/run")

    assert result is False
    mock_logger.error.assert_called_once()


@patch("src.utils.sync_offline_run")
@patch("src.utils.list_offline_runs")
def test_sync_all_offline_runs_duplicate(mock_list_runs, mock_sync_run):
    """Test sync_all_offline_runs function."""
    mock_runs = [
        {"run_id": "run1", "path": "/path/to/run1"},
        {"run_id": "run2", "path": "/path/to/run2"},
        {"run_id": "run3", "path": "/path/to/run3"},
    ]
    mock_list_runs.return_value = mock_runs

    # Mock sync results: first two succeed, third fails
    mock_sync_run.side_effect = [True, True, False]

    result = sync_all_offline_runs("/test/wandb")

    assert result["synced"] == ["run1", "run2"]
    assert result["failed"] == ["run3"]
    assert mock_sync_run.call_count == 3


@patch("src.utils.torch.cuda.is_available")
@patch("src.utils.torch.cuda.is_bf16_supported")
def test_get_compute_dtype_with_bf16(mock_bf16, mock_available):
    """Test get_compute_dtype with BF16 support."""
    mock_available.return_value = True
    mock_bf16.return_value = True

    result = get_compute_dtype()
    assert result == torch.bfloat16


@patch("src.utils.torch.cuda.is_available")
@patch("src.utils.torch.cuda.is_bf16_supported")
def test_get_compute_dtype_with_fp16(mock_bf16, mock_available):
    """Test get_compute_dtype with only FP16 support."""
    mock_available.return_value = True
    mock_bf16.return_value = False

    result = get_compute_dtype()
    assert result == torch.float16


@patch("src.utils.torch.cuda.is_available")
def test_get_compute_dtype_cpu_only(mock_available):
    """Test get_compute_dtype with CPU only."""
    mock_available.return_value = False

    result = get_compute_dtype()
    assert result == torch.float32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
