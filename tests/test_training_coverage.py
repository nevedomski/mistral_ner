"""Focused tests for training.py to achieve 94%+ overall coverage."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
import torch

from src.training import (
    CustomWandbCallback,
    MemoryCallback,
    TrainingManager,
    create_custom_trainer_class,
    run_training_pipeline,
)


class TestMemoryCallback:
    """Test MemoryCallback class."""

    @patch("src.training.clear_gpu_cache")
    @patch("src.training.check_gpu_memory")
    @patch("src.training.logger")
    def test_memory_callback_on_step_end(
        self, mock_logger: Mock, mock_check_memory: Mock, mock_clear_cache: Mock
    ) -> None:
        """Test memory callback on step end."""
        # Arrange
        callback = MemoryCallback(clear_cache_steps=10)
        args = Mock()
        state = Mock()
        state.global_step = 10  # Should trigger cache clear
        control = Mock()
        mock_check_memory.return_value = {"gpu_0": {"allocated_gb": 2.5}}

        # Act
        callback.on_step_end(args, state, control)

        # Assert
        mock_clear_cache.assert_called_once()
        mock_check_memory.assert_called_once()
        mock_logger.debug.assert_called_once()

    @patch("src.training.clear_gpu_cache")
    def test_memory_callback_on_epoch_end(self, mock_clear_cache: Mock) -> None:
        """Test memory callback on epoch end."""
        # Arrange
        callback = MemoryCallback()
        args = Mock()
        state = Mock()
        control = Mock()

        # Act
        callback.on_epoch_end(args, state, control)

        # Assert
        mock_clear_cache.assert_called_once()

    @patch("src.training.clear_gpu_cache")
    @patch("src.training.check_gpu_memory")
    def test_memory_callback_step_not_divisible(self, mock_check_memory: Mock, mock_clear_cache: Mock) -> None:
        """Test memory callback when step is not divisible by clear_cache_steps."""
        # Arrange
        callback = MemoryCallback(clear_cache_steps=10)
        args = Mock()
        state = Mock()
        state.global_step = 5  # Should NOT trigger cache clear
        control = Mock()

        # Act
        callback.on_step_end(args, state, control)

        # Assert
        mock_clear_cache.assert_not_called()
        mock_check_memory.assert_not_called()


class TestCustomWandbCallback:
    """Test CustomWandbCallback class."""

    @patch("transformers.integrations.WandbCallback.__init__", return_value=None)
    @patch("src.training.check_gpu_memory")
    def test_custom_wandb_callback_on_log_with_gpu_stats(self, mock_check_memory: Mock, mock_wandb_init: Mock) -> None:
        """Test custom wandb callback logging with GPU stats."""
        # Arrange
        callback = CustomWandbCallback()
        callback._wandb = Mock()  # Simulate active wandb

        args = Mock()
        state = Mock()
        control = Mock()
        model = Mock()
        logs = {"train_loss": 0.5}

        mock_check_memory.return_value = {"gpu_0": {"allocated_gb": 2.5, "utilization_percent": 75.0}}

        # Mock the parent class method
        with patch("transformers.integrations.WandbCallback.on_log") as mock_super_on_log:
            # Act
            callback.on_log(args, state, control, model=model, logs=logs)

            # Assert
            assert "gpu/gpu_0/memory_used_gb" in logs
            assert "gpu/gpu_0/memory_util_percent" in logs
            assert logs["gpu/gpu_0/memory_used_gb"] == 2.5
            assert logs["gpu/gpu_0/memory_util_percent"] == 75.0
            mock_super_on_log.assert_called_once()

    @patch("transformers.integrations.WandbCallback.__init__", return_value=None)
    @patch("src.training.check_gpu_memory")
    def test_custom_wandb_callback_on_log_no_wandb(self, mock_check_memory: Mock, mock_wandb_init: Mock) -> None:
        """Test custom wandb callback when wandb is not active."""
        # Arrange
        callback = CustomWandbCallback()
        callback._wandb = None  # No active wandb

        args = Mock()
        state = Mock()
        control = Mock()
        logs = {"train_loss": 0.5}

        # Mock the parent class method
        with patch("transformers.integrations.WandbCallback.on_log") as mock_super_on_log:
            # Act
            callback.on_log(args, state, control, logs=logs)

            # Assert
            mock_check_memory.assert_not_called()
            mock_super_on_log.assert_called_once()

    @patch("transformers.integrations.WandbCallback.__init__", return_value=None)
    @patch("src.training.check_gpu_memory")
    def test_custom_wandb_callback_on_log_with_error(self, mock_check_memory: Mock, mock_wandb_init: Mock) -> None:
        """Test custom wandb callback when GPU memory check returns error."""
        # Arrange
        callback = CustomWandbCallback()
        callback._wandb = Mock()

        args = Mock()
        state = Mock()
        control = Mock()
        logs = {"train_loss": 0.5}

        mock_check_memory.return_value = {"error": "No GPU available"}

        # Mock the parent class method
        with patch("transformers.integrations.WandbCallback.on_log") as mock_super_on_log:
            # Act
            callback.on_log(args, state, control, logs=logs)

            # Assert - Should not add GPU stats when error
            assert "gpu/gpu_0/memory_used_gb" not in logs
            mock_super_on_log.assert_called_once()


class TestTrainingManager:
    """Test TrainingManager class."""

    def create_mock_config(self) -> Mock:
        """Create a mock config for testing."""
        config = Mock()

        # Training config
        config.training.fp16 = True
        config.training.bf16 = False
        config.training.tf32 = True
        config.training.output_dir = "/tmp/test_output"
        config.training.num_train_epochs = 3
        config.training.per_device_train_batch_size = 4
        config.training.per_device_eval_batch_size = 8
        config.training.gradient_accumulation_steps = 2
        config.training.gradient_checkpointing = True
        config.training.optim = "adamw_torch"
        config.training.learning_rate = 2e-5
        config.training.weight_decay = 0.01
        config.training.warmup_ratio = 0.1
        config.training.max_grad_norm = 1.0
        config.training.eval_strategy = "steps"
        config.training.save_strategy = "steps"
        config.training.logging_steps = 100
        config.training.save_total_limit = 3
        config.training.load_best_model_at_end = True
        config.training.metric_for_best_model = "f1"
        config.training.greater_is_better = True
        config.training.report_to = ["wandb"]
        config.training.seed = 42
        config.training.data_seed = 42
        config.training.local_rank = -1
        config.training.ddp_find_unused_parameters = False
        config.training.push_to_hub = False
        config.training.hub_model_id = None
        config.training.hub_strategy = "end"
        config.training.clear_cache_steps = 50
        config.training.early_stopping_patience = 3
        config.training.early_stopping_threshold = 0.01
        config.training.use_enhanced_evaluation = True
        config.training.compute_entity_level_metrics = True
        config.training.resume_from_checkpoint = None
        config.training.final_output_dir = "/tmp/test_final"

        # Logging config
        config.logging.use_wandb = True
        config.logging.disable_tqdm = False
        config.logging.log_level = "info"

        # Data config
        config.data.label_names = ["O", "B-PER", "I-PER"]

        return config

    @patch("src.training.detect_mixed_precision_support")
    def test_training_manager_init(self, mock_detect_mp: Mock) -> None:
        """Test TrainingManager initialization."""
        # Arrange
        config = self.create_mock_config()

        # Act
        manager = TrainingManager(config)

        # Assert
        assert manager.config == config

    @patch("src.training.detect_mixed_precision_support")
    @patch("src.training.Path")
    @patch("src.training.logger")
    def test_create_training_arguments_fp16_bf16_conflict(
        self, mock_logger: Mock, mock_path: Mock, mock_detect_mp: Mock
    ) -> None:
        """Test training arguments creation with fp16/bf16 conflict resolution."""
        # Arrange
        config = self.create_mock_config()
        config.training.fp16 = True
        config.training.bf16 = True  # Both enabled - should resolve to bf16 only

        mock_detect_mp.return_value = {"fp16": True, "bf16": True, "tf32": True}
        mock_path_instance = Mock()
        mock_path.return_value = mock_path_instance

        manager = TrainingManager(config)

        # Act
        with patch("src.training.TrainingArguments") as mock_training_args:
            mock_training_args.return_value = Mock()
            manager.create_training_arguments()

            # Assert - bf16 should be True, fp16 should be False
            call_kwargs = mock_training_args.call_args[1]
            assert call_kwargs["bf16"] is True
            assert call_kwargs["fp16"] is False
            mock_logger.info.assert_called()

    @patch("transformers.integrations.WandbCallback.__init__", return_value=None)
    @patch("src.evaluation.create_enhanced_compute_metrics")
    @patch("src.training.compute_metrics_factory")
    @patch("src.training.create_custom_trainer_class")
    def test_create_trainer_with_enhanced_evaluation(
        self,
        mock_create_trainer_class: Mock,
        mock_compute_factory: Mock,
        mock_enhanced_metrics: Mock,
        mock_wandb_init: Mock,
    ) -> None:
        """Test trainer creation with enhanced evaluation."""
        # Arrange
        config = self.create_mock_config()
        config.training.use_enhanced_evaluation = True

        manager = TrainingManager(config)
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_train_dataset = Mock()
        mock_eval_dataset = Mock()
        mock_data_collator = Mock()

        mock_enhanced_metrics.return_value = Mock()
        mock_trainer_class = Mock()
        mock_trainer_instance = Mock()
        mock_trainer_class.return_value = mock_trainer_instance
        mock_create_trainer_class.return_value = mock_trainer_class

        # Mock training arguments creation
        with patch.object(manager, "create_training_arguments") as mock_create_args:
            mock_create_args.return_value = Mock()

            # Act
            trainer = manager.create_trainer(
                model=mock_model,
                tokenizer=mock_tokenizer,
                train_dataset=mock_train_dataset,
                eval_dataset=mock_eval_dataset,
                data_collator=mock_data_collator,
            )

            # Assert
            mock_enhanced_metrics.assert_called_once()
            mock_compute_factory.assert_not_called()
            assert trainer == mock_trainer_instance

    @patch("src.training.logger")
    @patch("src.training.clear_gpu_cache")
    @patch("src.training.wandb")
    def test_train_with_keyboard_interrupt(self, mock_wandb: Mock, mock_clear_cache: Mock, mock_logger: Mock) -> None:
        """Test training with keyboard interrupt handling."""
        # Arrange
        config = self.create_mock_config()
        manager = TrainingManager(config)

        mock_trainer = Mock()
        mock_trainer.train.side_effect = KeyboardInterrupt("User interrupted")
        mock_trainer.save_model.return_value = None
        mock_trainer.save_state.return_value = None

        mock_wandb.run = Mock()

        # Act & Assert
        with pytest.raises(KeyboardInterrupt):
            manager.train(mock_trainer)

        # Assert cleanup happened
        mock_logger.info.assert_any_call("Training interrupted by user")
        mock_trainer.save_model.assert_called()
        mock_trainer.save_state.assert_called()
        mock_clear_cache.assert_called()
        mock_wandb.finish.assert_called_once()

    @patch("src.training.logger")
    @patch("src.training.clear_gpu_cache")
    @patch("src.training.wandb")
    def test_train_with_oom_error(self, mock_wandb: Mock, mock_clear_cache: Mock, mock_logger: Mock) -> None:
        """Test training with CUDA OOM error handling."""
        # Arrange
        config = self.create_mock_config()
        manager = TrainingManager(config)

        mock_trainer = Mock()
        oom_error = torch.cuda.OutOfMemoryError("CUDA out of memory")
        mock_trainer.train.side_effect = oom_error

        mock_wandb.run = Mock()

        # Act & Assert
        with pytest.raises(torch.cuda.OutOfMemoryError):
            manager.train(mock_trainer)

        # Assert error logging happened
        mock_logger.error.assert_any_call(f"CUDA out of memory during training: {oom_error}")
        mock_logger.error.assert_any_call("Try reducing batch size or sequence length")
        mock_clear_cache.assert_called()
        mock_wandb.finish.assert_called_once()

    @patch("src.training.logger")
    @patch("src.training.clear_gpu_cache")
    @patch("src.training.wandb")
    def test_train_success_with_final_output_dir(
        self, mock_wandb: Mock, mock_clear_cache: Mock, mock_logger: Mock
    ) -> None:
        """Test successful training with final output directory."""
        # Arrange
        config = self.create_mock_config()
        config.training.final_output_dir = "/tmp/final_model"
        manager = TrainingManager(config)

        mock_train_result = Mock()
        mock_train_result.metrics = {"train_loss": 0.5, "f1": 0.85}

        mock_trainer = Mock()
        mock_trainer.train.return_value = mock_train_result
        mock_trainer.save_model.return_value = None
        mock_trainer.save_state.return_value = None

        mock_wandb.run = Mock()

        # Act
        result = manager.train(mock_trainer)

        # Assert
        assert result == {"train_loss": 0.5, "f1": 0.85}
        mock_trainer.save_model.assert_called_with("/tmp/final_model")
        mock_trainer.save_state.assert_called_once()
        mock_logger.info.assert_any_call("Training completed. Metrics: {'train_loss': 0.5, 'f1': 0.85}")
        mock_clear_cache.assert_called()
        mock_wandb.finish.assert_called_once()


class TestCustomTrainerClass:
    """Test create_custom_trainer_class function and CustomTrainer behavior."""

    def create_mock_config(self) -> Mock:
        """Create a mock config for testing."""
        config = Mock()

        # Training config with custom loss
        config.training.loss_type = "focal"
        config.training.use_class_weights = True
        config.training.focal_alpha = None  # Auto-compute
        config.training.focal_gamma = 2.0
        config.training.manual_class_weights = None
        config.training.class_weight_type = "balanced"
        config.training.class_weight_smoothing = 0.0
        config.training.use_batch_balancing = False
        config.training.log_batch_composition = False
        config.training.lr_scheduler_type = "cosine"
        config.training.lr_scheduler_kwargs = {}

        # Model config
        config.model.num_labels = 3

        # Data config
        config.data.label_names = ["O", "B-PER", "I-PER"]
        config.training.label_smoothing = 0.1
        config.training.class_balanced_beta = 0.9999

        return config

    @patch("src.training.logger")
    @patch("src.training.compute_class_frequencies")
    @patch("src.training.create_loss_function")
    def test_create_custom_trainer_class_with_focal_loss(
        self, mock_create_loss: Mock, mock_compute_freq: Mock, mock_logger: Mock
    ) -> None:
        """Test creating custom trainer class with focal loss."""
        # Arrange
        config = self.create_mock_config()
        mock_train_dataset = Mock()

        mock_compute_freq.return_value = [100, 50, 25]  # Class frequencies
        mock_loss_fn = Mock()
        mock_create_loss.return_value = mock_loss_fn

        # Act
        CustomTrainerClass = create_custom_trainer_class(config, mock_train_dataset)

        # Assert
        assert issubclass(CustomTrainerClass, object)
        mock_compute_freq.assert_called_once_with(mock_train_dataset)
        mock_create_loss.assert_called_once()
        mock_logger.info.assert_called()

    @patch("src.training.logger")
    @patch("src.training.create_loss_function")
    def test_create_custom_trainer_class_cross_entropy_no_weights(
        self, mock_create_loss: Mock, mock_logger: Mock
    ) -> None:
        """Test creating custom trainer class with standard cross entropy."""
        # Arrange
        config = Mock()
        config.training.loss_type = "cross_entropy"
        config.training.use_class_weights = False
        config.training.focal_alpha = None
        config.training.focal_gamma = 2.0
        config.data.label_names = ["O", "B-PER", "I-PER"]
        config.training.label_smoothing = 0.0
        config.training.class_balanced_beta = 0.9999

        # Act
        CustomTrainerClass = create_custom_trainer_class(config, None)

        # Assert
        assert issubclass(CustomTrainerClass, object)
        mock_create_loss.assert_called_once()
        # Verify loss was created with correct parameters
        call_args = mock_create_loss.call_args
        assert call_args.kwargs["loss_type"] == "cross_entropy"
        assert call_args.kwargs["class_frequencies"] is None


class TestRunTrainingPipeline:
    """Test run_training_pipeline function."""

    @patch("src.utils.setup_wandb_logging")
    @patch("src.training.TrainingManager")
    def test_run_training_pipeline_with_wandb(self, mock_training_manager_class: Mock, mock_setup_wandb: Mock) -> None:
        """Test running training pipeline with wandb enabled."""
        # Arrange
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_train_dataset = Mock()
        mock_eval_dataset = Mock()
        mock_data_collator = Mock()

        config = Mock()
        config.logging.use_wandb = True
        config.setup_wandb.return_value = None

        mock_manager_instance = Mock()
        mock_trainer = Mock()
        mock_manager_instance.create_trainer.return_value = mock_trainer
        mock_manager_instance.train.return_value = {"f1": 0.85}
        mock_training_manager_class.return_value = mock_manager_instance

        # Act
        result = run_training_pipeline(
            model=mock_model,
            tokenizer=mock_tokenizer,
            train_dataset=mock_train_dataset,
            eval_dataset=mock_eval_dataset,
            data_collator=mock_data_collator,
            config=config,
        )

        # Assert
        assert result == {"f1": 0.85}
        config.setup_wandb.assert_called_once()
        mock_setup_wandb.assert_called_once_with(config)
        mock_manager_instance.create_trainer.assert_called_once()
        mock_manager_instance.train.assert_called_once_with(mock_trainer)

    @patch("src.training.TrainingManager")
    def test_run_training_pipeline_no_wandb(self, mock_training_manager_class: Mock) -> None:
        """Test running training pipeline without wandb."""
        # Arrange
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_train_dataset = Mock()
        mock_eval_dataset = Mock()
        mock_data_collator = Mock()

        config = Mock()
        config.logging.use_wandb = False

        mock_manager_instance = Mock()
        mock_trainer = Mock()
        mock_manager_instance.create_trainer.return_value = mock_trainer
        mock_manager_instance.train.return_value = {"f1": 0.85}
        mock_training_manager_class.return_value = mock_manager_instance

        # Act
        result = run_training_pipeline(
            model=mock_model,
            tokenizer=mock_tokenizer,
            train_dataset=mock_train_dataset,
            eval_dataset=mock_eval_dataset,
            data_collator=mock_data_collator,
            config=config,
        )

        # Assert
        assert result == {"f1": 0.85}
        # Should not call wandb setup
        assert not hasattr(config, "setup_wandb") or not config.setup_wandb.called
        mock_manager_instance.create_trainer.assert_called_once()
        mock_manager_instance.train.assert_called_once_with(mock_trainer)


# Edge cases and error conditions
class TestTrainingEdgeCases:
    """Test edge cases and error conditions."""

    @patch("src.training.clear_gpu_cache")
    @patch("src.training.check_gpu_memory")
    def test_memory_callback_large_clear_cache_steps(self, mock_check_memory: Mock, mock_clear_cache: Mock) -> None:
        """Test memory callback with large clear_cache_steps."""
        # Arrange
        callback = MemoryCallback(clear_cache_steps=100)
        args = Mock()
        state = Mock()
        state.global_step = 1  # Not divisible by 100
        control = Mock()

        # Act
        callback.on_step_end(args, state, control)

        # Assert - Should not clear cache when step is not divisible
        mock_clear_cache.assert_not_called()
        mock_check_memory.assert_not_called()

    def test_training_manager_train_generic_exception(self) -> None:
        """Test training manager handling generic exceptions."""
        # Arrange
        config = Mock()
        config.training.resume_from_checkpoint = None
        config.training.final_output_dir = None

        manager = TrainingManager(config)
        mock_trainer = Mock()
        mock_trainer.train.side_effect = ValueError("Something went wrong")

        # Act & Assert
        with (
            patch("src.training.logger") as mock_logger,
            patch("src.training.clear_gpu_cache"),
            patch("src.training.wandb") as mock_wandb,
        ):
            mock_wandb.run = None

            with pytest.raises(ValueError, match="Something went wrong"):
                manager.train(mock_trainer)

            mock_logger.error.assert_called_with("Training failed with error: Something went wrong")

    @patch("transformers.integrations.WandbCallback.__init__", return_value=None)
    @patch("src.training.check_gpu_memory")
    def test_custom_wandb_callback_non_dict_gpu_stats(self, mock_check_memory: Mock, mock_wandb_init: Mock) -> None:
        """Test custom wandb callback with non-dict GPU stats."""
        # Arrange
        callback = CustomWandbCallback()
        callback._wandb = Mock()

        args = Mock()
        state = Mock()
        control = Mock()
        logs = {"train_loss": 0.5}

        # Return non-dict stats for a GPU
        mock_check_memory.return_value = {
            "gpu_0": "invalid_stats_format"  # Not a dict
        }

        # Mock the parent class method
        with patch("transformers.integrations.WandbCallback.on_log") as mock_super_on_log:
            # Act
            callback.on_log(args, state, control, logs=logs)

            # Assert - Should not add GPU stats when format is invalid
            assert "gpu/gpu_0/memory_used_gb" not in logs
            mock_super_on_log.assert_called_once()
