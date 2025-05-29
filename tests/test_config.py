"""Tests for configuration module."""

import sys
import tempfile
from pathlib import Path

import pytest

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import Config, DataConfig


def test_default_config():
    """Test default configuration creation."""
    config = Config()

    # Check model config
    assert config.model.model_name == "mistralai/Mistral-7B-v0.3"
    assert config.model.num_labels == 9
    assert config.model.load_in_8bit
    assert config.model.lora_r == 16

    # Check data config
    assert config.data.dataset_name == "conll2003"
    assert config.data.max_length == 256
    assert len(config.data.label_names) == 9
    assert config.data.id2label[0] == "O"
    assert config.data.label2id["O"] == 0

    # Check training config
    assert config.training.num_train_epochs == 5
    assert config.training.per_device_train_batch_size == 4
    assert config.training.learning_rate == 2e-4

    # Check logging config
    assert config.logging.log_level == "info"
    assert config.logging.use_wandb


def test_config_from_yaml():
    """Test loading configuration from YAML."""
    yaml_content = """
model:
  model_name: "test-model"
  num_labels: 5
  lora_r: 8

data:
  max_length: 128

training:
  num_train_epochs: 3
  learning_rate: 1e-4

logging:
  use_wandb: false
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()

        config = Config.from_yaml(f.name)

        assert config.model.model_name == "test-model"
        assert config.model.num_labels == 5
        assert config.model.lora_r == 8
        assert config.data.max_length == 128
        assert config.training.num_train_epochs == 3
        assert config.training.learning_rate == 1e-4
        assert not config.logging.use_wandb

        Path(f.name).unlink()


def test_config_to_yaml():
    """Test saving configuration to YAML."""
    config = Config()
    config.model.lora_r = 32
    config.training.num_train_epochs = 10

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        config.to_yaml(f.name)

        # Load it back
        loaded_config = Config.from_yaml(f.name)

        assert loaded_config.model.lora_r == 32
        assert loaded_config.training.num_train_epochs == 10

        Path(f.name).unlink()


def test_config_update_from_args():
    """Test updating configuration from arguments."""
    config = Config()

    # Mock args object
    class Args:
        model_name = "new-model"
        max_length = 512
        learning_rate = 3e-4
        use_wandb = False
        resume_from_checkpoint = "checkpoint-100"

    args = Args()
    config.update_from_args(args)

    assert config.model.model_name == "new-model"
    assert config.data.max_length == 512
    assert config.training.learning_rate == 3e-4
    assert not config.logging.use_wandb
    assert config.training.resume_from_checkpoint == "checkpoint-100"


def test_data_config_label_mappings():
    """Test label ID mappings are created correctly."""
    data_config = DataConfig()

    # Check mappings
    assert len(data_config.id2label) == 9
    assert len(data_config.label2id) == 9

    # Check specific mappings
    assert data_config.id2label[1] == "B-PER"
    assert data_config.label2id["B-PER"] == 1
    assert data_config.id2label[8] == "I-MISC"
    assert data_config.label2id["I-MISC"] == 8


def test_setup_wandb():
    """Test WandB setup."""
    import os

    config = Config()
    config.logging.use_wandb = True
    config.logging.wandb_project = "test-project"
    config.logging.wandb_mode = "offline"

    config.setup_wandb()

    assert os.environ.get("WANDB_PROJECT") == "test-project"
    assert os.environ.get("WANDB_MODE") == "offline"

    # Test disabling wandb
    config.logging.use_wandb = False
    config.setup_wandb()

    assert os.environ.get("WANDB_DISABLED") == "true"
    assert "wandb" not in config.training.report_to


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
