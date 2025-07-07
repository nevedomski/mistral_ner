"""Tests for model module."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
from peft import LoraConfig, TaskType

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import Config
from src.model import (
    create_bnb_config,
    create_lora_config,
    load_base_model,
    load_model_from_checkpoint,
    load_tokenizer,
    merge_and_save_model,
    prepare_model_for_kbit_training,
    save_model_checkpoint,
    setup_model,
    setup_peft_model,
)


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return Config()


class TestCreateBnbConfig:
    """Test BitsAndBytes configuration creation."""

    @patch("src.model.BitsAndBytesConfig")
    def test_create_bnb_config_8bit(self, mock_bnb_config):
        """Test creating 8-bit quantization config."""
        mock_config = Mock()
        mock_bnb_config.return_value = mock_config

        config = create_bnb_config(load_in_8bit=True, load_in_4bit=False)

        assert config == mock_config
        mock_bnb_config.assert_called_once_with(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
            bnb_8bit_quant_type="nf8",
            bnb_8bit_use_double_quant=True,
        )

    @patch("src.model.BitsAndBytesConfig")
    def test_create_bnb_config_4bit(self, mock_bnb_config):
        """Test creating 4-bit quantization config."""
        mock_config = Mock()
        mock_bnb_config.return_value = mock_config

        config = create_bnb_config(load_in_8bit=False)

        assert config == mock_config
        mock_bnb_config.assert_called_once_with(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )


class TestCreateLoraConfig:
    """Test LoRA configuration creation."""

    def test_create_lora_config_default(self, sample_config):
        """Test creating LoRA config with default values."""
        lora_config = create_lora_config(sample_config)

        assert isinstance(lora_config, LoraConfig)
        assert lora_config.r == 32
        assert lora_config.lora_alpha == 64
        assert lora_config.lora_dropout == 0.1
        assert lora_config.bias == "none"
        assert lora_config.task_type == TaskType.TOKEN_CLS

    @patch("src.model.LoraConfig")
    def test_create_lora_config_custom(self, mock_lora_config, sample_config):
        """Test creating LoRA config with custom values."""
        sample_config.model.lora_r = 64
        sample_config.model.lora_alpha = 128
        sample_config.model.lora_dropout = 0.05
        sample_config.model.target_modules = ["q_proj", "v_proj"]

        mock_config = Mock()
        mock_lora_config.return_value = mock_config

        lora_config = create_lora_config(sample_config)

        assert lora_config == mock_config
        mock_lora_config.assert_called_once_with(
            r=64,
            lora_alpha=128,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.TOKEN_CLS,
        )


class TestLoadTokenizer:
    """Test tokenizer loading functionality."""

    @patch("src.model.AutoTokenizer.from_pretrained")
    def test_load_tokenizer_success(self, mock_from_pretrained):
        """Test successful tokenizer loading."""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "[PAD]"
        mock_tokenizer.eos_token = "[EOS]"
        mock_tokenizer.model_max_length = 512
        mock_from_pretrained.return_value = mock_tokenizer

        tokenizer = load_tokenizer("test-model")

        assert tokenizer == mock_tokenizer
        mock_from_pretrained.assert_called_once_with("test-model", trust_remote_code=True, add_prefix_space=True)

    @patch("src.model.AutoTokenizer.from_pretrained")
    def test_load_tokenizer_no_pad_token(self, mock_from_pretrained):
        """Test tokenizer loading when pad_token is None."""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "[EOS]"
        mock_tokenizer.model_max_length = 512
        mock_from_pretrained.return_value = mock_tokenizer

        tokenizer = load_tokenizer("test-model")

        assert tokenizer.pad_token == "[EOS]"

    @patch("src.model.AutoTokenizer.from_pretrained")
    def test_load_tokenizer_large_max_length(self, mock_from_pretrained):
        """Test tokenizer loading with large model_max_length."""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "[PAD]"
        mock_tokenizer.model_max_length = 1000000
        mock_from_pretrained.return_value = mock_tokenizer

        tokenizer = load_tokenizer("test-model")

        assert tokenizer.model_max_length == 2048

    @patch("src.model.AutoTokenizer.from_pretrained")
    def test_load_tokenizer_failure(self, mock_from_pretrained):
        """Test tokenizer loading failure."""
        mock_from_pretrained.side_effect = Exception("Loading failed")

        with pytest.raises(Exception, match="Loading failed"):
            load_tokenizer("test-model")


class TestLoadBaseModel:
    """Test base model loading functionality."""

    @patch("src.model.AutoModelForTokenClassification.from_pretrained")
    def test_load_base_model_success(self, mock_from_pretrained, sample_config):
        """Test successful base model loading."""
        mock_model = Mock()
        mock_model.gradient_checkpointing_enable.return_value = None
        # Remove the prepare_for_kbit_training method so it's not called
        if hasattr(mock_model, "prepare_for_kbit_training"):
            delattr(mock_model, "prepare_for_kbit_training")
        mock_from_pretrained.return_value = mock_model

        model = load_base_model("test-model", sample_config)

        assert model == mock_model
        mock_from_pretrained.assert_called_once()

    @patch("src.model.create_bnb_config")
    @patch("src.model.AutoModelForTokenClassification.from_pretrained")
    def test_load_base_model_with_quantization(self, mock_from_pretrained, mock_create_bnb, sample_config):
        """Test base model loading with quantization."""
        mock_model = Mock()
        # Remove the prepare_for_kbit_training method so it's not called
        if hasattr(mock_model, "prepare_for_kbit_training"):
            delattr(mock_model, "prepare_for_kbit_training")
        mock_from_pretrained.return_value = mock_model
        mock_bnb_config = Mock()

        model = load_base_model("test-model", sample_config, mock_bnb_config)

        assert model == mock_model
        # Verify quantization config was passed
        call_kwargs = mock_from_pretrained.call_args[1]
        assert "quantization_config" in call_kwargs

    @patch("src.model.AutoModelForTokenClassification.from_pretrained")
    @patch("src.model.gc.collect")
    @patch("torch.cuda.empty_cache")
    def test_load_base_model_oom_recovery(self, mock_empty_cache, mock_gc_collect, mock_from_pretrained, sample_config):
        """Test base model loading with OOM recovery."""
        mock_model = Mock()
        # Remove the prepare_for_kbit_training method so it's not called
        if hasattr(mock_model, "prepare_for_kbit_training"):
            delattr(mock_model, "prepare_for_kbit_training")

        # First call raises OOM, second succeeds
        mock_from_pretrained.side_effect = [torch.cuda.OutOfMemoryError(), mock_model]

        model = load_base_model("test-model", sample_config)

        assert model == mock_model
        assert mock_from_pretrained.call_count == 2
        mock_gc_collect.assert_called_once()
        mock_empty_cache.assert_called_once()

    @patch("src.model.AutoModelForTokenClassification.from_pretrained")
    def test_load_base_model_gradient_checkpointing(self, mock_from_pretrained, sample_config):
        """Test base model loading with gradient checkpointing."""
        mock_model = Mock()
        # Remove the prepare_for_kbit_training method so it's not called
        if hasattr(mock_model, "prepare_for_kbit_training"):
            delattr(mock_model, "prepare_for_kbit_training")
        mock_from_pretrained.return_value = mock_model
        sample_config.training.gradient_checkpointing = True

        load_base_model("test-model", sample_config)

        mock_model.gradient_checkpointing_enable.assert_called_once()

    @patch("src.model.AutoModelForTokenClassification.from_pretrained")
    @patch("src.model.prepare_model_for_kbit_training")
    def test_load_base_model_kbit_training(self, mock_prepare_kbit, mock_from_pretrained, sample_config):
        """Test base model loading with k-bit training preparation."""
        mock_model = Mock()
        mock_model.prepare_for_kbit_training = Mock()
        mock_from_pretrained.return_value = mock_model
        mock_prepare_kbit.return_value = mock_model

        load_base_model("test-model", sample_config)

        mock_prepare_kbit.assert_called_once_with(mock_model)


class TestPrepareModelForKbitTraining:
    """Test k-bit training preparation."""

    def test_prepare_model_for_kbit_training_with_enable_input_require_grads(self):
        """Test k-bit training preparation with enable_input_require_grads method."""
        mock_model = Mock()
        mock_model.gradient_checkpointing_enable.return_value = None
        mock_model.prepare_for_kbit_training.return_value = mock_model
        mock_model.enable_input_require_grads = Mock()

        # Mock parameters
        mock_param = Mock()
        mock_param.requires_grad = True
        mock_model.parameters.return_value = [mock_param]

        result = prepare_model_for_kbit_training(mock_model)

        assert result == mock_model
        mock_model.gradient_checkpointing_enable.assert_called_once()
        mock_model.prepare_for_kbit_training.assert_called_once()
        mock_model.enable_input_require_grads.assert_called_once()
        assert mock_param.requires_grad is False

    def test_prepare_model_for_kbit_training_without_enable_input_require_grads(self):
        """Test k-bit training preparation without enable_input_require_grads method."""
        mock_model = Mock()
        mock_model.gradient_checkpointing_enable.return_value = None
        mock_model.prepare_for_kbit_training.return_value = mock_model

        # Remove enable_input_require_grads method
        del mock_model.enable_input_require_grads

        # Mock parameters and input embeddings
        mock_param = Mock()
        mock_param.requires_grad = True
        mock_model.parameters.return_value = [mock_param]

        mock_embeddings = Mock()
        mock_model.get_input_embeddings.return_value = mock_embeddings

        result = prepare_model_for_kbit_training(mock_model)

        assert result == mock_model
        mock_embeddings.register_forward_hook.assert_called_once()


class TestSetupPeftModel:
    """Test PEFT model setup."""

    @patch("src.model.get_peft_model")
    def test_setup_peft_model_success(self, mock_get_peft_model, sample_config):
        """Test successful PEFT model setup."""
        mock_base_model = Mock()
        mock_peft_model = Mock()
        mock_get_peft_model.return_value = mock_peft_model

        # Mock named_parameters for trainable parameter calculation
        mock_param1 = Mock()
        mock_param1.numel.return_value = 1000
        mock_param1.requires_grad = True

        mock_param2 = Mock()
        mock_param2.numel.return_value = 2000
        mock_param2.requires_grad = False

        mock_peft_model.named_parameters.return_value = [("param1", mock_param1), ("param2", mock_param2)]

        lora_config = create_lora_config(sample_config)
        result = setup_peft_model(mock_base_model, lora_config)

        assert result == mock_peft_model
        mock_get_peft_model.assert_called_once_with(mock_base_model, lora_config)

    @patch("src.model.get_peft_model")
    def test_setup_peft_model_failure(self, mock_get_peft_model, sample_config):
        """Test PEFT model setup failure."""
        mock_get_peft_model.side_effect = Exception("PEFT setup failed")
        lora_config = create_lora_config(sample_config)

        with pytest.raises(Exception, match="PEFT setup failed"):
            setup_peft_model(Mock(), lora_config)


class TestSetupModel:
    """Test complete model setup."""

    @patch("src.model.setup_peft_model")
    @patch("src.model.create_lora_config")
    @patch("src.model.load_base_model")
    @patch("src.model.create_bnb_config")
    @patch("src.model.load_tokenizer")
    def test_setup_model_success(
        self, mock_load_tokenizer, mock_create_bnb, mock_load_base, mock_create_lora, mock_setup_peft, sample_config
    ):
        """Test successful complete model setup."""
        mock_tokenizer = Mock()
        mock_base_model = Mock()
        mock_peft_model = Mock()
        mock_bnb_config = Mock()
        mock_lora_config = Mock()

        mock_load_tokenizer.return_value = mock_tokenizer
        mock_create_bnb.return_value = mock_bnb_config
        mock_load_base.return_value = mock_base_model
        mock_create_lora.return_value = mock_lora_config
        mock_setup_peft.return_value = mock_peft_model

        model, tokenizer = setup_model("test-model", sample_config)

        assert model == mock_peft_model
        assert tokenizer == mock_tokenizer

        mock_load_tokenizer.assert_called_once_with("test-model")
        mock_create_bnb.assert_called_once_with(False, True)  # Default load_in_8bit=False, load_in_4bit=True
        mock_load_base.assert_called_once_with("test-model", sample_config, mock_bnb_config)
        mock_create_lora.assert_called_once_with(sample_config)
        mock_setup_peft.assert_called_once_with(mock_base_model, mock_lora_config)

    @patch("src.model.setup_peft_model")
    @patch("src.model.create_lora_config")
    @patch("src.model.load_base_model")
    @patch("src.model.create_bnb_config")
    @patch("src.model.load_tokenizer")
    def test_setup_model_without_quantization(
        self, mock_load_tokenizer, mock_create_bnb, mock_load_base, mock_create_lora, mock_setup_peft, sample_config
    ):
        """Test model setup without quantization."""
        sample_config.model.load_in_8bit = False
        sample_config.model.load_in_4bit = False

        mock_tokenizer = Mock()
        mock_base_model = Mock()
        mock_peft_model = Mock()
        mock_lora_config = Mock()

        mock_load_tokenizer.return_value = mock_tokenizer
        mock_load_base.return_value = mock_base_model
        mock_create_lora.return_value = mock_lora_config
        mock_setup_peft.return_value = mock_peft_model

        model, tokenizer = setup_model("test-model", sample_config)

        assert model == mock_peft_model
        assert tokenizer == mock_tokenizer

        # Verify quantization config creation was not called
        mock_create_bnb.assert_not_called()
        mock_load_base.assert_called_once_with("test-model", sample_config, None)

    @patch("src.model.setup_peft_model")
    @patch("src.model.create_lora_config")
    @patch("src.model.load_base_model")
    @patch("src.model.create_bnb_config")
    @patch("src.model.load_tokenizer")
    def test_setup_model_custom_device_map(
        self, mock_load_tokenizer, mock_create_bnb, mock_load_base, mock_create_lora, mock_setup_peft, sample_config
    ):
        """Test model setup with custom device map."""
        mock_tokenizer = Mock()
        mock_base_model = Mock()
        mock_peft_model = Mock()
        mock_bnb_config = Mock()
        mock_lora_config = Mock()

        mock_load_tokenizer.return_value = mock_tokenizer
        mock_create_bnb.return_value = mock_bnb_config
        mock_load_base.return_value = mock_base_model
        mock_create_lora.return_value = mock_lora_config
        mock_setup_peft.return_value = mock_peft_model

        model, tokenizer = setup_model("test-model", sample_config, device_map="balanced")

        assert sample_config.model.device_map == "balanced"


class TestSaveModelCheckpoint:
    """Test model checkpoint saving."""

    def test_save_model_checkpoint_success(self, sample_config):
        """Test successful model checkpoint saving."""
        mock_model = Mock()
        mock_tokenizer = Mock()

        with tempfile.TemporaryDirectory() as temp_dir:
            save_model_checkpoint(mock_model, mock_tokenizer, temp_dir, config=sample_config)

            mock_model.save_pretrained.assert_called_once_with(temp_dir)
            mock_tokenizer.save_pretrained.assert_called_once_with(temp_dir)

    def test_save_model_checkpoint_final(self, sample_config):
        """Test saving final model checkpoint."""
        mock_model = Mock()
        mock_tokenizer = Mock()

        with tempfile.TemporaryDirectory() as temp_dir:
            save_model_checkpoint(mock_model, mock_tokenizer, temp_dir, is_final=True, config=sample_config)

            mock_model.save_pretrained.assert_called_once_with(temp_dir)
            mock_tokenizer.save_pretrained.assert_called_once_with(temp_dir)

    @patch("src.model.Path")
    def test_save_model_checkpoint_failure(self, mock_path, sample_config):
        """Test model checkpoint saving failure."""
        mock_model = Mock()
        mock_model.save_pretrained.side_effect = Exception("Save failed")
        mock_tokenizer = Mock()

        # Mock Path to avoid directory creation errors
        mock_path_instance = Mock()
        mock_path.return_value = mock_path_instance
        mock_path_instance.mkdir.return_value = None

        with pytest.raises(Exception, match="Save failed"):
            save_model_checkpoint(mock_model, mock_tokenizer, "/invalid/path", config=sample_config)

    def test_save_model_checkpoint_without_config(self):
        """Test model checkpoint saving without config (backward compatibility)."""
        mock_model = Mock()
        mock_tokenizer = Mock()

        with tempfile.TemporaryDirectory() as temp_dir:
            save_model_checkpoint(mock_model, mock_tokenizer, temp_dir)

            mock_model.save_pretrained.assert_called_once_with(temp_dir)
            mock_tokenizer.save_pretrained.assert_called_once_with(temp_dir)
            # Verify config.yaml was not created
            assert not (Path(temp_dir) / "config.yaml").exists()

    def test_save_model_checkpoint_with_config_saves_yaml(self, sample_config):
        """Test that config is saved as YAML when provided."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        sample_config.to_yaml = Mock()

        with tempfile.TemporaryDirectory() as temp_dir:
            save_model_checkpoint(mock_model, mock_tokenizer, temp_dir, config=sample_config)

            # Verify config.to_yaml was called with the correct path
            sample_config.to_yaml.assert_called_once_with(Path(temp_dir) / "config.yaml")


class TestLoadModelFromCheckpoint:
    """Test model loading from checkpoint."""

    @patch("src.model.PeftModel.from_pretrained")
    @patch("src.model.load_base_model")
    @patch("src.model.create_bnb_config")
    @patch("src.model.AutoTokenizer.from_pretrained")
    def test_load_model_from_checkpoint_success(
        self, mock_tokenizer_load, mock_create_bnb, mock_load_base, mock_peft_load, sample_config
    ):
        """Test successful model loading from checkpoint."""
        mock_tokenizer = Mock()
        mock_base_model = Mock()
        mock_peft_model = Mock()
        mock_bnb_config = Mock()

        mock_tokenizer_load.return_value = mock_tokenizer
        mock_create_bnb.return_value = mock_bnb_config
        mock_load_base.return_value = mock_base_model
        mock_peft_load.return_value = mock_peft_model

        model, tokenizer = load_model_from_checkpoint("/path/to/checkpoint", sample_config)

        assert model == mock_peft_model
        assert tokenizer == mock_tokenizer

        mock_tokenizer_load.assert_called_once_with("/path/to/checkpoint")
        mock_peft_load.assert_called_once_with(mock_base_model, "/path/to/checkpoint")

    @patch("src.model.PeftModel.from_pretrained")
    @patch("src.model.load_base_model")
    @patch("src.model.create_bnb_config")
    @patch("src.model.AutoTokenizer.from_pretrained")
    def test_load_model_from_checkpoint_custom_base_model(
        self, mock_tokenizer_load, mock_create_bnb, mock_load_base, mock_peft_load, sample_config
    ):
        """Test model loading from checkpoint with custom base model."""
        mock_tokenizer = Mock()
        mock_base_model = Mock()
        mock_peft_model = Mock()
        mock_bnb_config = Mock()

        mock_tokenizer_load.return_value = mock_tokenizer
        mock_create_bnb.return_value = mock_bnb_config
        mock_load_base.return_value = mock_base_model
        mock_peft_load.return_value = mock_peft_model

        model, tokenizer = load_model_from_checkpoint("/path/to/checkpoint", sample_config, "custom-base-model")

        # Verify custom base model name was used
        mock_load_base.assert_called_once_with("custom-base-model", sample_config, mock_bnb_config)

    @patch("src.model.AutoTokenizer.from_pretrained")
    def test_load_model_from_checkpoint_failure(self, mock_tokenizer_load, sample_config):
        """Test model loading from checkpoint failure."""
        mock_tokenizer_load.side_effect = Exception("Loading failed")

        with pytest.raises(Exception, match="Loading failed"):
            load_model_from_checkpoint("/path/to/checkpoint", sample_config)


class TestMergeAndSaveModel:
    """Test model merging and saving."""

    def test_merge_and_save_model_success(self):
        """Test successful model merging and saving."""
        mock_peft_model = Mock()
        mock_merged_model = Mock()
        mock_peft_model.merge_and_unload.return_value = mock_merged_model
        mock_tokenizer = Mock()

        with tempfile.TemporaryDirectory() as temp_dir:
            merge_and_save_model(mock_peft_model, mock_tokenizer, temp_dir)

            mock_peft_model.merge_and_unload.assert_called_once()
            mock_merged_model.save_pretrained.assert_called_once_with(temp_dir, safe_serialization=True)
            mock_tokenizer.save_pretrained.assert_called_once_with(temp_dir)

    def test_merge_and_save_model_failure(self):
        """Test model merging and saving failure."""
        mock_peft_model = Mock()
        mock_peft_model.merge_and_unload.side_effect = Exception("Merge failed")
        mock_tokenizer = Mock()

        with pytest.raises(Exception, match="Merge failed"):
            merge_and_save_model(mock_peft_model, mock_tokenizer, "/invalid/path")


@pytest.mark.parametrize(
    ("load_in_8bit", "expected_8bit", "expected_4bit"),
    [
        (True, True, False),
        (False, False, True),
    ],
)
@patch("src.model.BitsAndBytesConfig")
def test_parametrized_bnb_config(mock_bnb_config, load_in_8bit, expected_8bit, expected_4bit):
    """Test BitsAndBytes config creation with different bit settings."""
    mock_config = Mock()
    mock_config.load_in_8bit = expected_8bit
    mock_config.load_in_4bit = expected_4bit
    mock_bnb_config.return_value = mock_config

    config = create_bnb_config(load_in_8bit=load_in_8bit)

    assert config.load_in_8bit == expected_8bit
    assert config.load_in_4bit == expected_4bit


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
