"""Tests for uncovered lines in data.py."""

from unittest.mock import Mock, patch

import pytest

from datasets import Dataset, DatasetDict
from src.config import Config


class TestDataUncoveredLines:
    """Test specific uncovered lines in data.py."""

    @patch("src.data.load_dataset")
    def test_load_conll2003_exception_line_15(self, mock_load_dataset):
        """Test line 15-17: Exception handling in load_conll2003_dataset."""
        from src.data import load_conll2003_dataset

        # Make load_dataset raise an exception
        mock_load_dataset.side_effect = Exception("Connection error")

        # This should trigger the exception handling
        with pytest.raises(Exception, match="Connection error"):
            load_conll2003_dataset()

    def test_validate_dataset_import_error_line_24(self):
        """Test lines 24-25: Import error handling."""
        # This is hard to test directly, but we can test the fallback behavior
        from src.data import validate_dataset

        # Create a valid dataset with all required splits
        dataset = DatasetDict({
            "train": Dataset.from_dict({"tokens": [["Hello"]], "ner_tags": [[0]]}),
            "validation": Dataset.from_dict({"tokens": [["World"]], "ner_tags": [[0]]}),
            "test": Dataset.from_dict({"tokens": [["Test"]], "ner_tags": [[0]]})
        })

        # Should work even if multi-dataset import failed
        validate_dataset(dataset, ["O"])

    def test_tokenize_align_labels_edge_cases(self):
        """Test lines 62-64, 106: Edge cases in tokenize_and_align_labels."""
        from src.data import tokenize_and_align_labels

        # Mock tokenizer that returns a proper BatchEncoding-like object
        tokenizer = Mock()

        # Test with empty examples (line 106)
        examples = {"tokens": [], "ner_tags": []}
        # Create a mock that behaves like a BatchEncoding
        class MockBatchEncoding(dict):
            def word_ids(self, batch_index=0):
                return []
        mock_result = MockBatchEncoding()
        tokenizer.return_value = mock_result

        result = tokenize_and_align_labels(examples, tokenizer)
        assert result["labels"] == []

        # Test with None word_ids (lines 62-64)
        examples = {"tokens": [["Test"]], "ner_tags": [[0]]}
        class MockBatchEncoding2(dict):
            def word_ids(self, batch_index=0):
                return [None, None, None]
        mock_result2 = MockBatchEncoding2({"input_ids": [[101, 2000, 102]]})
        tokenizer.return_value = mock_result2

        result = tokenize_and_align_labels(examples, tokenizer)
        # All should be -100 for None word_ids
        assert result["labels"] == [[-100, -100, -100]]

    def test_prepare_datasets_missing_ner_tags(self):
        """Test lines 141-143: Missing ner_tags column."""
        from src.data import prepare_datasets

        with patch("src.data.load_dataset") as mock_load_dataset:
            # Create dataset without ner_tags but with all splits
            mock_dataset = DatasetDict(
                {
                    "train": Dataset.from_dict(
                        {
                            "tokens": [["Hello"], ["World"]]
                            # Missing ner_tags
                        }
                    ),
                    "validation": Dataset.from_dict(
                        {
                            "tokens": [["Test"]]
                            # Missing ner_tags
                        }
                    ),
                    "test": Dataset.from_dict(
                        {
                            "tokens": [["Sample"]]
                            # Missing ner_tags
                        }
                    )
                }
            )
            mock_load_dataset.return_value = mock_dataset

            with patch("transformers.AutoTokenizer"), patch("src.data.load_conll2003_dataset") as mock_load_conll:
                # Return the dataset without ner_tags
                mock_load_conll.return_value = mock_dataset

                config = Config()
                config.data.dataset_name = "conll2003"

                # prepare_datasets takes tokenizer as first arg
                with pytest.raises(ValueError, match="missing 'ner_tags' feature"):
                    prepare_datasets(Mock(), config)

    def test_get_label_list_line_211(self):
        """Test line 211: get_label_list function."""
        from src.data import get_label_list

        # Create dataset with various label values
        dataset = DatasetDict(
            {
                "train": Dataset.from_dict({"tokens": [["A"], ["B"], ["C"]], "ner_tags": [[0], [2], [1]]}),
                "test": Dataset.from_dict({"tokens": [["D"]], "ner_tags": [[3]]}),
            }
        )

        labels = get_label_list(dataset)

        # Should return default label names (since features don't have names)
        assert labels == ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]

    @patch("builtins.print")
    def test_print_dataset_statistics_lines_248_251(self, mock_print):
        """Test lines 248-251: print_dataset_statistics."""
        from src.data import print_dataset_statistics

        # Create dataset
        dataset = DatasetDict(
            {"train": Dataset.from_dict({"tokens": [["Hello", "world"], ["Test"]], "ner_tags": [[0, 1], [2]]})}
        )

        # Test without tokenizer
        print_dataset_statistics(dataset, tokenizer=None)

        # Should print statistics
        assert mock_print.called

        # Test with tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.model_max_length = 512

        mock_print.reset_mock()
        print_dataset_statistics(dataset, tokenizer=mock_tokenizer)

        # Should print more detailed statistics with tokenizer
        assert mock_print.call_count > 2

    @patch("src.data.MULTI_DATASET_AVAILABLE", False)
    def test_prepare_multi_datasets_import_error(self):
        """Test lines 312-316: Import error in prepare_multi_datasets."""
        from src.data import prepare_datasets

        config = Config()
        config.data.multi_dataset.enabled = True

        # prepare_datasets checks MULTI_DATASET_AVAILABLE and raises ImportError
        with pytest.raises(ImportError, match="Multi-dataset components not available"):
            prepare_datasets(Mock(), config)
