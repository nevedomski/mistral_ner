"""Tests for training module."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
from transformers import TrainerControl, TrainerState, TrainingArguments

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import Config
from src.training import (
    CustomWandbCallback,
    MemoryCallback,
    TrainingManager,
    create_custom_trainer_class,
    run_training_pipeline,
)


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return Config()


@pytest.fixture
def mock_training_args():
    """Mock training arguments."""
    return Mock(spec=TrainingArguments)


@pytest.fixture
def mock_trainer_state():
    """Mock trainer state."""
    state = Mock(spec=TrainerState)
    state.global_step = 100
    return state


@pytest.fixture
def mock_trainer_control():
    """Mock trainer control."""
    return Mock(spec=TrainerControl)


class TestMemoryCallback:
    """Test MemoryCallback functionality."""

    @patch("src.training.clear_gpu_cache")
    @patch("src.training.check_gpu_memory")
    def test_memory_callback_on_step_end(
        self, mock_check_memory, mock_clear_cache, mock_training_args, mock_trainer_state, mock_trainer_control
    ):
        """Test memory callback on step end."""
        mock_check_memory.return_value = {"gpu_0": {"allocated_gb": 2.5, "utilization_percent": 75}}
        callback = MemoryCallback(clear_cache_steps=50)

        # Test when step is multiple of clear_cache_steps
        mock_trainer_state.global_step = 100
        callback.on_step_end(mock_training_args, mock_trainer_state, mock_trainer_control)

        mock_clear_cache.assert_called_once()
        mock_check_memory.assert_called_once()

    @patch("src.training.clear_gpu_cache")
    @patch("src.training.check_gpu_memory")
    def test_memory_callback_on_step_end_skip(
        self, mock_check_memory, mock_clear_cache, mock_training_args, mock_trainer_state, mock_trainer_control
    ):
        """Test memory callback skips when not on clear_cache_steps."""
        callback = MemoryCallback(clear_cache_steps=50)

        # Test when step is not multiple of clear_cache_steps
        mock_trainer_state.global_step = 99
        callback.on_step_end(mock_training_args, mock_trainer_state, mock_trainer_control)

        mock_clear_cache.assert_not_called()
        mock_check_memory.assert_not_called()

    @patch("src.training.clear_gpu_cache")
    def test_memory_callback_on_epoch_end(
        self, mock_clear_cache, mock_training_args, mock_trainer_state, mock_trainer_control
    ):
        """Test memory callback on epoch end."""
        callback = MemoryCallback()

        callback.on_epoch_end(mock_training_args, mock_trainer_state, mock_trainer_control)

        mock_clear_cache.assert_called_once()

    def test_memory_callback_initialization(self):
        """Test memory callback initialization."""
        callback = MemoryCallback(clear_cache_steps=100)
        assert callback.clear_cache_steps == 100

        # Test default value
        callback_default = MemoryCallback()
        assert callback_default.clear_cache_steps == 50


class TestCustomWandbCallback:
    """Test CustomWandbCallback functionality."""

    @patch("transformers.integrations.integration_utils.is_wandb_available", return_value=True)
    @patch("src.training.check_gpu_memory")
    def test_custom_wandb_callback_on_log_with_wandb(
        self, mock_check_memory, mock_wandb_available, mock_training_args, mock_trainer_state, mock_trainer_control
    ):
        """Test custom WandB callback logging with active WandB."""
        mock_check_memory.return_value = {
            "gpu_0": {"allocated_gb": 2.5, "utilization_percent": 75},
            "gpu_1": {"allocated_gb": 1.8, "utilization_percent": 60},
        }

        callback = CustomWandbCallback()
        callback._wandb = Mock()  # Simulate active WandB

        logs = {"train_loss": 0.5}
        mock_model = Mock()

        with patch.object(callback, "on_log", wraps=callback.on_log):
            # Mock the super().on_log call to avoid actual WandB interaction
            with patch("src.training.WandbCallback.on_log"):
                callback.on_log(
                    mock_training_args, mock_trainer_state, mock_trainer_control, model=mock_model, logs=logs
                )

            # Verify GPU memory stats were added to logs
            assert "gpu/gpu_0/memory_used_gb" in logs
            assert "gpu/gpu_0/memory_util_percent" in logs
            assert "gpu/gpu_1/memory_used_gb" in logs
            assert "gpu/gpu_1/memory_util_percent" in logs
            assert logs["gpu/gpu_0/memory_used_gb"] == 2.5
            assert logs["gpu/gpu_0/memory_util_percent"] == 75

    @patch("transformers.integrations.integration_utils.is_wandb_available", return_value=True)
    @patch("src.training.check_gpu_memory")
    def test_custom_wandb_callback_on_log_without_wandb(
        self, mock_check_memory, mock_wandb_available, mock_training_args, mock_trainer_state, mock_trainer_control
    ):
        """Test custom WandB callback logging without active WandB."""
        callback = CustomWandbCallback()
        callback._wandb = None  # No active WandB

        logs = {"train_loss": 0.5}

        with patch("src.training.WandbCallback.on_log"):
            callback.on_log(mock_training_args, mock_trainer_state, mock_trainer_control, logs=logs)

        # Verify no GPU memory stats were added
        assert "gpu/gpu_0/memory_used_gb" not in logs
        mock_check_memory.assert_not_called()

    @patch("transformers.integrations.integration_utils.is_wandb_available", return_value=True)
    @patch("src.training.check_gpu_memory")
    def test_custom_wandb_callback_on_log_with_error(
        self, mock_check_memory, mock_wandb_available, mock_training_args, mock_trainer_state, mock_trainer_control
    ):
        """Test custom WandB callback handling memory check errors."""
        mock_check_memory.return_value = {"error": "GPU not available"}

        callback = CustomWandbCallback()
        callback._wandb = Mock()

        logs = {"train_loss": 0.5}

        with patch("src.training.WandbCallback.on_log"):
            callback.on_log(mock_training_args, mock_trainer_state, mock_trainer_control, logs=logs)

        # Verify no GPU memory stats were added due to error
        assert "gpu/gpu_0/memory_used_gb" not in logs


class TestTrainingManager:
    """Test TrainingManager functionality."""

    @patch("src.training.load_seqeval_metric")
    def test_training_manager_initialization(self, mock_load_seqeval, sample_config):
        """Test training manager initialization."""
        mock_metric = Mock()
        mock_load_seqeval.return_value = mock_metric

        manager = TrainingManager(sample_config)

        assert manager.config == sample_config
        assert manager.seqeval_metric == mock_metric
        mock_load_seqeval.assert_called_once()

    @patch("src.training.detect_mixed_precision_support")
    @patch("src.training.Path")
    def test_create_training_arguments(self, mock_path, mock_detect_mp, sample_config):
        """Test training arguments creation."""
        mock_detect_mp.return_value = {"fp16": True, "bf16": True, "tf32": True}
        mock_path.return_value.mkdir = Mock()

        manager = TrainingManager(sample_config)

        with patch("src.training.TrainingArguments") as mock_training_args:
            mock_args = Mock()
            mock_training_args.return_value = mock_args

            result = manager.create_training_arguments()

            assert result == mock_args
            mock_training_args.assert_called_once()

            # Verify key arguments were passed
            call_kwargs = mock_training_args.call_args[1]
            assert call_kwargs["output_dir"] == sample_config.training.output_dir
            assert call_kwargs["num_train_epochs"] == sample_config.training.num_train_epochs
            assert call_kwargs["learning_rate"] == sample_config.training.learning_rate

    @patch("src.training.detect_mixed_precision_support")
    @patch("src.training.Path")
    def test_create_training_arguments_mixed_precision_priority(self, mock_path, mock_detect_mp, sample_config):
        """Test that bf16 takes priority over fp16."""
        mock_detect_mp.return_value = {"fp16": True, "bf16": True, "tf32": True}
        mock_path.return_value.mkdir = Mock()

        sample_config.training.fp16 = True
        sample_config.training.bf16 = True

        manager = TrainingManager(sample_config)

        with patch("src.training.TrainingArguments") as mock_training_args:
            manager.create_training_arguments()

            call_kwargs = mock_training_args.call_args[1]
            assert call_kwargs["bf16"] is True
            assert call_kwargs["fp16"] is False  # Should be disabled when bf16 is true

    @patch("src.training.detect_mixed_precision_support")
    @patch("src.training.Path")
    def test_create_training_arguments_no_wandb(self, mock_path, mock_detect_mp, sample_config):
        """Test training arguments creation without WandB."""
        mock_detect_mp.return_value = {"fp16": False, "bf16": False, "tf32": False}
        mock_path.return_value.mkdir = Mock()

        sample_config.logging.use_wandb = False

        manager = TrainingManager(sample_config)

        with patch("src.training.TrainingArguments") as mock_training_args:
            manager.create_training_arguments()

            call_kwargs = mock_training_args.call_args[1]
            assert call_kwargs["report_to"] == []

    @patch("src.training.compute_metrics_factory")
    @patch("src.training.EarlyStoppingCallback")
    @patch("src.training.CustomWandbCallback")
    @patch("src.training.MemoryCallback")
    @patch("src.training.Trainer")
    def test_create_trainer(
        self,
        mock_trainer_class,
        mock_memory_callback,
        mock_wandb_callback,
        mock_early_stopping,
        mock_compute_metrics_factory,
        sample_config,
    ):
        """Test trainer creation."""
        # Setup mocks
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_train_dataset = Mock()
        mock_eval_dataset = Mock()
        mock_data_collator = Mock()
        mock_trainer = Mock()
        mock_trainer_class.return_value = mock_trainer

        mock_memory_cb = Mock()
        mock_memory_callback.return_value = mock_memory_cb
        mock_wandb_cb = Mock()
        mock_wandb_callback.return_value = mock_wandb_cb
        mock_early_cb = Mock()
        mock_early_stopping.return_value = mock_early_cb

        mock_compute_metrics = Mock()
        mock_compute_metrics_factory.return_value = mock_compute_metrics

        manager = TrainingManager(sample_config)

        with patch.object(manager, "create_training_arguments") as mock_create_args:
            mock_training_args = Mock()
            mock_create_args.return_value = mock_training_args

            result = manager.create_trainer(
                model=mock_model,
                tokenizer=mock_tokenizer,
                train_dataset=mock_train_dataset,
                eval_dataset=mock_eval_dataset,
                data_collator=mock_data_collator,
            )

            assert result == mock_trainer

            # Verify trainer was created with correct arguments
            mock_trainer_class.assert_called_once()
            call_kwargs = mock_trainer_class.call_args[1]
            assert call_kwargs["model"] == mock_model
            assert call_kwargs["tokenizer"] == mock_tokenizer
            assert call_kwargs["train_dataset"] == mock_train_dataset
            assert call_kwargs["eval_dataset"] == mock_eval_dataset
            assert call_kwargs["data_collator"] == mock_data_collator
            assert call_kwargs["compute_metrics"] == mock_compute_metrics

            # Verify callbacks were added
            callbacks = call_kwargs["callbacks"]
            assert mock_memory_cb in callbacks
            assert mock_early_cb in callbacks
            assert mock_wandb_cb in callbacks

    @patch("src.training.compute_metrics_factory")
    @patch("src.training.MemoryCallback")
    @patch("src.training.Trainer")
    def test_create_trainer_custom_compute_metrics(
        self, mock_trainer_class, mock_memory_callback, mock_compute_metrics_factory, sample_config
    ):
        """Test trainer creation with custom compute_metrics."""
        # Disable WandB to avoid the callback issue
        sample_config.logging.use_wandb = False

        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_train_dataset = Mock()
        mock_eval_dataset = Mock()
        mock_data_collator = Mock()
        mock_custom_compute_metrics = Mock()

        manager = TrainingManager(sample_config)

        with patch.object(manager, "create_training_arguments"):
            manager.create_trainer(
                model=mock_model,
                tokenizer=mock_tokenizer,
                train_dataset=mock_train_dataset,
                eval_dataset=mock_eval_dataset,
                data_collator=mock_data_collator,
                compute_metrics=mock_custom_compute_metrics,
            )

            # Verify custom compute_metrics was used instead of factory
            mock_compute_metrics_factory.assert_not_called()
            call_kwargs = mock_trainer_class.call_args[1]
            assert call_kwargs["compute_metrics"] == mock_custom_compute_metrics

    @patch("src.training.clear_gpu_cache")
    @patch("src.training.wandb")
    def test_train_success(self, mock_wandb, mock_clear_cache, sample_config):
        """Test successful training."""
        mock_wandb.run = Mock()
        mock_wandb.finish = Mock()

        manager = TrainingManager(sample_config)
        mock_trainer = Mock()

        # Mock train result
        mock_train_result = Mock()
        mock_train_result.metrics = {"train_loss": 0.5, "eval_f1": 0.85}
        mock_trainer.train.return_value = mock_train_result

        result = manager.train(mock_trainer)

        assert result == mock_train_result.metrics
        mock_trainer.train.assert_called_once_with(resume_from_checkpoint=None)
        mock_trainer.save_model.assert_called_once_with(sample_config.training.final_output_dir)
        mock_trainer.save_state.assert_called_once()
        mock_clear_cache.assert_called_once()
        mock_wandb.finish.assert_called_once()

    @patch("src.training.clear_gpu_cache")
    @patch("src.training.wandb")
    def test_train_with_resume_checkpoint(self, mock_wandb, mock_clear_cache, sample_config):
        """Test training with resume from checkpoint."""
        mock_wandb.run = None  # No active run

        manager = TrainingManager(sample_config)
        mock_trainer = Mock()

        mock_train_result = Mock()
        mock_train_result.metrics = {"train_loss": 0.5}
        mock_trainer.train.return_value = mock_train_result

        result = manager.train(mock_trainer, resume_from_checkpoint="/path/to/checkpoint")

        assert result == mock_train_result.metrics
        mock_trainer.train.assert_called_once_with(resume_from_checkpoint="/path/to/checkpoint")

    @patch("src.training.clear_gpu_cache")
    @patch("src.training.wandb")
    @patch("src.training.os.path.join")
    def test_train_keyboard_interrupt(self, mock_join, mock_wandb, mock_clear_cache, sample_config):
        """Test training with keyboard interrupt."""
        mock_wandb.run = None
        mock_join.return_value = "/output/interrupted_checkpoint"

        manager = TrainingManager(sample_config)
        mock_trainer = Mock()
        mock_trainer.train.side_effect = KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            manager.train(mock_trainer)

        # Verify checkpoint was saved
        mock_trainer.save_model.assert_called_with("/output/interrupted_checkpoint")
        mock_trainer.save_state.assert_called_once()
        mock_clear_cache.assert_called_once()

    @patch("src.training.clear_gpu_cache")
    @patch("src.training.wandb")
    def test_train_cuda_oom(self, mock_wandb, mock_clear_cache, sample_config):
        """Test training with CUDA OOM error."""
        mock_wandb.run = None

        manager = TrainingManager(sample_config)
        mock_trainer = Mock()
        mock_trainer.train.side_effect = torch.cuda.OutOfMemoryError()

        with pytest.raises(torch.cuda.OutOfMemoryError):
            manager.train(mock_trainer)

        mock_clear_cache.assert_called_once()

    @patch("src.training.clear_gpu_cache")
    @patch("src.training.wandb")
    def test_train_generic_exception(self, mock_wandb, mock_clear_cache, sample_config):
        """Test training with generic exception."""
        mock_wandb.run = None

        manager = TrainingManager(sample_config)
        mock_trainer = Mock()
        mock_trainer.train.side_effect = RuntimeError("Training failed")

        with pytest.raises(RuntimeError, match="Training failed"):
            manager.train(mock_trainer)

        mock_clear_cache.assert_called_once()


class TestCreateCustomTrainerClass:
    """Test custom trainer class creation."""

    def test_create_custom_trainer_class(self, sample_config):
        """Test creating custom trainer class."""
        from transformers import Trainer

        custom_trainer_class = create_custom_trainer_class(sample_config)

        assert issubclass(custom_trainer_class, Trainer)  # CustomTrainer inherits from Trainer
        assert custom_trainer_class.__name__ == "CustomTrainer"

    @patch("src.training.clear_gpu_cache")
    def test_custom_trainer_compute_loss_success(self, mock_clear_cache, sample_config):
        """Test custom trainer compute_loss success case."""
        custom_trainer_class = create_custom_trainer_class(sample_config)

        # Create mock trainer instance
        with patch("transformers.Trainer.__init__", return_value=None):
            trainer = custom_trainer_class()

        mock_model = Mock()
        mock_inputs = {"input_ids": torch.tensor([[1, 2, 3]]), "labels": torch.tensor([[1, 2, 3]])}
        mock_loss = torch.tensor(0.5)

        with patch("transformers.Trainer.compute_loss", return_value=mock_loss):
            result = trainer.compute_loss(mock_model, mock_inputs)

        assert result == mock_loss
        mock_clear_cache.assert_not_called()

    @patch("src.training.clear_gpu_cache")
    def test_custom_trainer_compute_loss_oom_recovery(self, mock_clear_cache, sample_config):
        """Test custom trainer compute_loss OOM recovery."""
        custom_trainer_class = create_custom_trainer_class(sample_config)

        with patch("transformers.Trainer.__init__", return_value=None):
            trainer = custom_trainer_class()

        mock_model = Mock()
        mock_inputs = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),  # Batch size 2
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
            "labels": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        }
        mock_loss = torch.tensor(0.5)

        # First call raises OOM, second succeeds
        with patch("transformers.Trainer.compute_loss", side_effect=[torch.cuda.OutOfMemoryError(), mock_loss]):
            result = trainer.compute_loss(mock_model, mock_inputs)

        assert result == mock_loss
        mock_clear_cache.assert_called_once()

        # Verify batch size was reduced
        assert mock_inputs["input_ids"].shape[0] == 1
        assert mock_inputs["attention_mask"].shape[0] == 1
        assert mock_inputs["labels"].shape[0] == 1

    @patch("src.training.clear_gpu_cache")
    def test_custom_trainer_evaluation_loop(self, mock_clear_cache, sample_config):
        """Test custom trainer evaluation loop."""
        custom_trainer_class = create_custom_trainer_class(sample_config)

        with patch("transformers.Trainer.__init__", return_value=None):
            trainer = custom_trainer_class()

        mock_result = {"eval_loss": 0.3, "eval_f1": 0.85}

        with patch("transformers.Trainer.evaluation_loop", return_value=mock_result):
            result = trainer.evaluation_loop("arg1", "arg2", kwarg1="value1")

        assert result == mock_result
        assert mock_clear_cache.call_count == 2  # Before and after evaluation


class TestRunTrainingPipeline:
    """Test complete training pipeline."""

    @patch("src.training.TrainingManager")
    @patch("src.utils.setup_wandb_logging")
    def test_run_training_pipeline_with_wandb(self, mock_setup_wandb, mock_training_manager_class, sample_config):
        """Test running training pipeline with WandB enabled."""
        sample_config.logging.use_wandb = True

        # Setup mocks
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_train_dataset = Mock()
        mock_eval_dataset = Mock()
        mock_data_collator = Mock()

        mock_manager = Mock()
        mock_training_manager_class.return_value = mock_manager
        mock_trainer = Mock()
        mock_manager.create_trainer.return_value = mock_trainer
        mock_metrics = {"train_loss": 0.5, "eval_f1": 0.85}
        mock_manager.train.return_value = mock_metrics

        with patch.object(sample_config, "setup_wandb") as mock_setup_wandb_config:
            result = run_training_pipeline(
                model=mock_model,
                tokenizer=mock_tokenizer,
                train_dataset=mock_train_dataset,
                eval_dataset=mock_eval_dataset,
                data_collator=mock_data_collator,
                config=sample_config,
            )

        assert result == mock_metrics
        mock_setup_wandb_config.assert_called_once()
        mock_setup_wandb.assert_called_once_with(sample_config)
        mock_manager.create_trainer.assert_called_once()
        mock_manager.train.assert_called_once_with(mock_trainer)

    @patch("src.training.TrainingManager")
    def test_run_training_pipeline_without_wandb(self, mock_training_manager_class, sample_config):
        """Test running training pipeline without WandB."""
        sample_config.logging.use_wandb = False

        # Setup mocks
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_train_dataset = Mock()
        mock_eval_dataset = Mock()
        mock_data_collator = Mock()

        mock_manager = Mock()
        mock_training_manager_class.return_value = mock_manager
        mock_trainer = Mock()
        mock_manager.create_trainer.return_value = mock_trainer
        mock_metrics = {"train_loss": 0.5}
        mock_manager.train.return_value = mock_metrics

        result = run_training_pipeline(
            model=mock_model,
            tokenizer=mock_tokenizer,
            train_dataset=mock_train_dataset,
            eval_dataset=mock_eval_dataset,
            data_collator=mock_data_collator,
            config=sample_config,
        )

        assert result == mock_metrics
        mock_manager.create_trainer.assert_called_once()
        mock_manager.train.assert_called_once_with(mock_trainer)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
