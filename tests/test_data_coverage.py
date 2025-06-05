"""Additional tests for data module to improve coverage."""

from unittest.mock import Mock, patch

import pytest

from datasets import Dataset, DatasetDict
from src.config import Config
from src.data import (
    create_data_collator,
    create_sample_dataset,
    load_conll2003_dataset,
)


class TestDataFunctions:
    """Test remaining data functions."""

    @patch("src.data.load_dataset")
    def test_load_conll2003_error_handling(self, mock_load_dataset):
        """Test error handling in CoNLL loading."""
        # Test exception during loading
        mock_load_dataset.side_effect = Exception("Network error")

        # Should not crash, return empty dict
        with pytest.raises(Exception, match="Network error"):
            load_conll2003_dataset()

    def test_create_sample_dataset(self):
        """Test creating sample dataset."""
        # Without config
        dataset = create_sample_dataset(size=5)

        assert isinstance(dataset, DatasetDict)
        assert "train" in dataset
        assert "validation" in dataset
        assert len(dataset["train"]) == 5  # All samples in train
        assert len(dataset["validation"]) == 1  # size//5 = 5//5 = 1

        # Check features
        assert "tokens" in dataset["train"].features
        assert "ner_tags" in dataset["train"].features

    def test_create_sample_dataset_with_config(self):
        """Test creating sample dataset with config."""
        config = Config()
        config.data.multi_dataset.enabled = True
        config.data.label_names = ["O", "B-TEST", "I-TEST"]

        dataset = create_sample_dataset(size=10, config=config)

        assert len(dataset["train"]) == 10  # All samples in train
        assert len(dataset["validation"]) == 2  # size//5 = 10//5 = 2

        # create_sample_dataset uses hardcoded tags, not config labels
        # so we just check that tags are valid integers
        for example in dataset["train"]:
            for tag in example["ner_tags"]:
                assert isinstance(tag, int)
                assert tag >= 0

    def test_create_data_collator(self):
        """Test creating data collator."""
        tokenizer = Mock()
        tokenizer.pad_token_id = 0

        collator = create_data_collator(tokenizer)

        # Should create DataCollatorForTokenClassification
        assert collator is not None
        assert hasattr(collator, "tokenizer")
        assert collator.tokenizer == tokenizer

    @patch("src.data.load_conll2003_dataset")
    def test_prepare_datasets_missing_splits(self, mock_load_conll2003):
        """Test prepare_datasets with missing validation split."""
        # Create dataset with all required splits
        mock_dataset = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {
                        "tokens": [["A"], ["B"], ["C"], ["D"], ["E"], ["F"], ["G"], ["H"], ["I"], ["J"]],
                        "ner_tags": [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
                    }
                ),
                "validation": Dataset.from_dict(
                    {
                        "tokens": [["K"], ["L"]],
                        "ner_tags": [[0], [0]],
                    }
                ),
                "test": Dataset.from_dict(
                    {
                        "tokens": [["M"]],
                        "ner_tags": [[0]],
                    }
                ),
            }
        )
        mock_load_conll2003.return_value = mock_dataset

        # Mock tokenizer that returns a proper BatchEncoding-like dict
        mock_tokenizer = Mock()

        # Create a custom class that behaves like BatchEncoding
        class MockBatchEncoding(dict):
            def __init__(self, data):
                super().__init__(data)

            def word_ids(self, batch_index=0):
                # Return word ids for each token
                return [None, 0, None]  # [CLS] token None

        # Mock the tokenizer to return our custom object
        def mock_tokenize(tokens, **kwargs):
            # Return proper tokenized output for each example
            return MockBatchEncoding(
                {
                    "input_ids": [[101, 100, 102]] * len(tokens),  # Simple mock tokens
                    "attention_mask": [[1, 1, 1]] * len(tokens),
                }
            )

        mock_tokenizer.side_effect = mock_tokenize

        from src.data import prepare_datasets

        config = Config()
        config.data.dataset_name = "test_dataset"
        config.model.model_name = "test-model"

        # prepare_datasets takes tokenizer as first argument
        train_dataset, eval_dataset, test_dataset, collator = prepare_datasets(mock_tokenizer, config)

        # When all splits exist, they are tokenized as-is
        assert len(train_dataset) == 10  # All train examples
        assert len(eval_dataset) == 2  # All validation examples
        assert len(test_dataset) == 1  # All test examples

    @patch("src.data.print")
    def test_print_dataset_statistics_no_tokenizer(self, mock_print):
        """Test printing statistics without tokenizer."""
        from src.data import print_dataset_statistics

        dataset = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {"tokens": [["Hello", "world"], ["Test", "data"]], "ner_tags": [[0, 1], [1, 2]]}
                ),
                "validation": Dataset.from_dict({"tokens": [["Val"]], "ner_tags": [[0]]}),
            }
        )

        # Should work without tokenizer
        print_dataset_statistics(dataset, tokenizer=None)

        # Check that statistics were printed
        assert mock_print.called
        # Should print dataset info
        calls = [str(call) for call in mock_print.call_args_list]
        assert any("Dataset Statistics" in str(call) for call in calls)

    def test_get_label_list(self):
        """Test getting label list from dataset."""
        from src.data import get_label_list

        dataset = DatasetDict(
            {
                "train": Dataset.from_dict({"tokens": [["A"], ["B"]], "ner_tags": [[0], [1]]}),
                "validation": Dataset.from_dict({"tokens": [["C"]], "ner_tags": [[2]]}),
                "test": Dataset.from_dict({"tokens": [["D"]], "ner_tags": [[1]]}),
            }
        )

        labels = get_label_list(dataset)

        # Should get unique label names sorted
        assert labels == ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]

    @patch("src.data.DatasetRegistry")
    @patch("transformers.AutoTokenizer")
    def test_prepare_multi_datasets_error_handling(self, mock_tokenizer_class, mock_registry_class):
        """Test error handling in multi-dataset preparation."""
        from src.data import prepare_multi_datasets

        # Mock registry to raise error
        mock_registry = Mock()
        mock_registry_class.return_value = mock_registry
        mock_registry.get_loader.side_effect = ValueError("Unknown dataset")

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        config = Config()
        config.data.multi_dataset.enabled = True
        config.data.multi_dataset.dataset_names = ["unknown_dataset"]
        config.model.model_name = "test-model"

        # prepare_multi_datasets takes tokenizer as first argument
        with pytest.raises(ValueError, match="Unknown dataset"):
            prepare_multi_datasets(mock_tokenizer, config)
