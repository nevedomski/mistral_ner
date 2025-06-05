"""Simple tests to boost coverage to 90%+."""

from unittest.mock import Mock, patch

import pytest
import torch
from transformers import TrainerControl, TrainerState, TrainingArguments

from datasets import Dataset, DatasetDict
from src.datasets.loaders.conll import CoNLLDataset
from src.datasets.loaders.fewnerd import FewNERDDataset
from src.datasets.mixers import DatasetMixer
from src.losses import FocalLoss, LabelSmoothingLoss

# from src.model import create_model_config
from src.schedulers import create_scheduler
from src.training import MemoryCallback, check_gpu_memory, clear_gpu_cache
from src.utils import setup_logging


class TestDatasetLoaders:
    """Simple tests for dataset loaders."""

    def test_conll_load(self):
        """Test CoNLL loader."""
        loader = CoNLLDataset()

        with patch("src.datasets.loaders.conll.load_dataset") as mock_load:
            mock_load.return_value = DatasetDict(
                {
                    "train": Dataset.from_dict({"tokens": [["test"]], "ner_tags": [[0]]}),
                    "validation": Dataset.from_dict({"tokens": [["val"]], "ner_tags": [[0]]}),
                    "test": Dataset.from_dict({"tokens": [["test"]], "ner_tags": [[0]]}),
                }
            )

            dataset = loader.load()
            assert dataset is not None

    def test_conll_mapping(self):
        """Test CoNLL label mapping."""
        loader = CoNLLDataset()
        mapping = loader.get_label_mapping()
        assert "O" in mapping
        assert "B-PER" in mapping

    def test_fewnerd_preprocess(self):
        """Test FewNERD preprocessing."""
        loader = FewNERDDataset()
        examples = {"tokens": [["test"]], "ner_tags": [[0]], "fine_ner_tags": [[1]]}
        result = loader.preprocess(examples)
        # FewNERD preprocess doesn't modify examples, it just returns them
        assert result == examples
        assert "tokens" in result
        assert "ner_tags" in result


class TestMixers:
    """Simple tests for dataset mixers."""

    def test_concat_mix(self):
        """Test concatenation mixing."""
        d1 = DatasetDict({"train": Dataset.from_dict({"text": ["a"], "label": [0]})})
        d2 = DatasetDict({"train": Dataset.from_dict({"text": ["b"], "label": [1]})})

        mixed = DatasetMixer.mix([d1, d2], strategy="concat")
        assert len(mixed["train"]) == 2

    def test_mix_error(self):
        """Test invalid strategy error."""
        d1 = DatasetDict({"train": Dataset.from_dict({"text": ["a"], "label": [0]})})

        with pytest.raises(ValueError, match="Unknown mixing strategy"):
            DatasetMixer.mix([d1], strategy="invalid")


class TestTrainingUtils:
    """Simple tests for training utilities."""

    def test_memory_callback(self):
        """Test memory callback."""
        callback = MemoryCallback(clear_cache_steps=1)

        args = TrainingArguments(output_dir="test")
        state = TrainerState()
        state.global_step = 1
        control = TrainerControl()

        with patch("src.training.clear_gpu_cache") as mock_clear:
            callback.on_step_end(args, state, control)
            mock_clear.assert_called_once()

    @patch("torch.cuda.is_available", return_value=False)
    def test_clear_gpu_cache(self, mock_available):
        """Test GPU cache clearing without CUDA."""
        # Should not raise error even without CUDA
        clear_gpu_cache()  # Just calls gc.collect()

    @patch("torch.cuda.is_available", return_value=False)
    def test_check_gpu_memory_no_cuda(self, mock_available):
        """Test GPU memory check without CUDA."""
        result = check_gpu_memory()
        assert result == {"error": "CUDA not available"}


class TestLosses:
    """Simple tests for loss functions."""

    def test_focal_loss(self):
        """Test focal loss initialization."""
        loss = FocalLoss(num_labels=5, gamma=2.0)
        assert loss.gamma == 2.0
        assert loss.num_labels == 5

    def test_label_smoothing_loss(self):
        """Test label smoothing loss."""
        loss = LabelSmoothingLoss(num_labels=5, smoothing=0.1)
        assert loss.smoothing == 0.1


# class TestModel:
#     """Simple tests for model utilities."""
#
#     def test_create_model_config(self):
#         """Test model config creation."""
#         config = Config()
#         config.model.load_in_4bit = True
#
#         with patch("src.model.BitsAndBytesConfig") as mock_bnb:
#             model_config = create_model_config(config)
#             mock_bnb.assert_called_once()


class TestSchedulers:
    """Simple tests for schedulers."""

    def test_create_scheduler_linear(self):
        """Test linear scheduler creation."""
        # Create a real optimizer instead of a mock
        model = torch.nn.Linear(10, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        args = TrainingArguments(output_dir="test", warmup_steps=100)

        scheduler = create_scheduler(optimizer, "linear", args, num_training_steps=1000)
        assert scheduler is not None


class TestUtils:
    """Simple tests for utilities."""

    @patch("logging.getLogger")
    def test_setup_logging(self, mock_get_logger):
        """Test logging setup."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        setup_logging("INFO")
        mock_logger.setLevel.assert_called()
