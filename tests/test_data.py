"""Tests for data processing module."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from transformers import AutoTokenizer

from datasets import Dataset, DatasetDict
from src.config import Config
from src.data import (
    create_data_collator,
    create_sample_dataset,
    get_label_list,
    load_conll2003_dataset,
    prepare_datasets,
    print_dataset_statistics,
    tokenize_and_align_labels,
    validate_dataset,
)


def test_create_sample_dataset():
    """Test sample dataset creation."""
    dataset = create_sample_dataset(size=10)

    assert "train" in dataset
    assert "validation" in dataset
    assert "test" in dataset

    assert len(dataset["train"]) == 10
    assert len(dataset["validation"]) == 2
    assert len(dataset["test"]) == 2

    # Check features
    assert "tokens" in dataset["train"].features
    assert "ner_tags" in dataset["train"].features

    # Check sample
    sample = dataset["train"][0]
    assert isinstance(sample["tokens"], list)
    assert isinstance(sample["ner_tags"], list)
    assert len(sample["tokens"]) == len(sample["ner_tags"])


def test_tokenize_and_align_labels():
    """Test tokenization and label alignment."""
    # Create dummy tokenizer (you might want to use a real one for integration tests)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    examples = {
        "tokens": [["John", "lives", "in", "New", "York", "."]],
        "ner_tags": [[1, 0, 0, 5, 6, 0]],  # B-PER, O, O, B-LOC, I-LOC, O
    }

    result = tokenize_and_align_labels(examples=examples, tokenizer=tokenizer, max_length=128, label_all_tokens=False)

    assert "input_ids" in result
    assert "attention_mask" in result
    assert "labels" in result

    # Check that special tokens have -100 label
    labels = result["labels"][0]
    assert labels[0] == -100  # [CLS] token
    assert labels[-1] == -100  # [SEP] token


def test_validate_dataset():
    """Test dataset validation."""
    config = Config()
    dataset = create_sample_dataset(size=5, config=config)

    # Should not raise any exception
    validate_dataset(dataset, config.data.label_names)

    # Test with missing split
    incomplete_dataset = {
        "train": dataset["train"],
        "validation": dataset["validation"],
        # missing "test"
    }

    with pytest.raises(ValueError, match="missing required split"):
        validate_dataset(incomplete_dataset, config.data.label_names)


def test_get_label_list():
    """Test extracting label list from dataset."""
    dataset = create_sample_dataset(size=5)

    # The sample dataset doesn't have the label names in features,
    # so it should return the default labels
    labels = get_label_list(dataset)

    assert isinstance(labels, list)
    assert len(labels) == 9
    assert labels[0] == "O"
    assert "B-PER" in labels
    assert "I-LOC" in labels


@patch("src.data.load_dataset")
def test_load_conll2003_dataset_success(mock_load_dataset):
    """Test successful dataset loading."""
    # Mock successful dataset loading
    mock_dataset = {"train": Mock(), "validation": Mock(), "test": Mock()}
    mock_dataset["train"].__len__ = Mock(return_value=100)
    mock_dataset["validation"].__len__ = Mock(return_value=20)
    mock_dataset["test"].__len__ = Mock(return_value=20)
    mock_load_dataset.return_value = mock_dataset

    result = load_conll2003_dataset()

    mock_load_dataset.assert_called_once_with("conll2003")
    assert result == mock_dataset


@patch("src.data.load_dataset")
def test_load_conll2003_dataset_failure(mock_load_dataset):
    """Test dataset loading with exception."""
    # Mock dataset loading failure
    mock_load_dataset.side_effect = Exception("Network error")

    with pytest.raises(Exception, match="Network error"):
        load_conll2003_dataset()


def test_validate_dataset_missing_features():
    """Test dataset validation with missing features."""
    config = Config()

    # Create dataset missing 'tokens' feature
    incomplete_data = [{"id": "1", "ner_tags": [0, 1, 2]}]
    incomplete_dataset = DatasetDict(
        {
            "train": Dataset.from_list(incomplete_data),
            "validation": Dataset.from_list(incomplete_data),
            "test": Dataset.from_list(incomplete_data),
        }
    )

    with pytest.raises(ValueError, match="missing 'tokens' feature"):
        validate_dataset(incomplete_dataset, config.data.label_names)

    # Create dataset missing 'ner_tags' feature
    incomplete_data2 = [{"id": "1", "tokens": ["word1", "word2"]}]
    incomplete_dataset2 = DatasetDict(
        {
            "train": Dataset.from_list(incomplete_data2),
            "validation": Dataset.from_list(incomplete_data2),
            "test": Dataset.from_list(incomplete_data2),
        }
    )

    with pytest.raises(ValueError, match="missing 'ner_tags' feature"):
        validate_dataset(incomplete_dataset2, config.data.label_names)


def test_tokenize_and_align_labels_with_label_all_tokens():
    """Test tokenization with label_all_tokens=True."""
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    examples = {
        "tokens": [["John", "Smith", "lives", "in", "New", "York", "."]],
        "ner_tags": [[1, 2, 0, 0, 5, 6, 0]],  # B-PER, I-PER, O, O, B-LOC, I-LOC, O
    }

    result = tokenize_and_align_labels(examples=examples, tokenizer=tokenizer, max_length=128, label_all_tokens=True)

    assert "input_ids" in result
    assert "attention_mask" in result
    assert "labels" in result

    # Check that labels are assigned to all subword tokens
    labels = result["labels"][0]
    assert labels[0] == -100  # [CLS] token
    assert labels[-1] == -100  # [SEP] token


def test_create_data_collator():
    """Test data collator creation."""
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    data_collator = create_data_collator(tokenizer)

    # Check that it returns a DataCollatorForTokenClassification
    assert hasattr(data_collator, "tokenizer")
    assert data_collator.tokenizer == tokenizer
    assert data_collator.padding is True
    assert data_collator.label_pad_token_id == -100


@patch("src.data.load_conll2003_dataset")
def test_prepare_datasets(mock_load_dataset):
    """Test complete dataset preparation pipeline."""
    # Create mock dataset
    mock_dataset = create_sample_dataset(size=10)
    mock_load_dataset.return_value = mock_dataset

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    config = Config()

    train_dataset, eval_dataset, test_dataset, data_collator = prepare_datasets(tokenizer=tokenizer, config=config)

    # Check that datasets are returned
    assert train_dataset is not None
    assert eval_dataset is not None
    assert test_dataset is not None
    assert data_collator is not None

    # Check that load_conll2003_dataset was called
    mock_load_dataset.assert_called_once()


def test_prepare_datasets_with_provided_dataset():
    """Test dataset preparation with pre-loaded dataset."""
    dataset = create_sample_dataset(size=5)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    config = Config()

    train_dataset, eval_dataset, test_dataset, data_collator = prepare_datasets(
        tokenizer=tokenizer, config=config, dataset=dataset
    )

    # Check that datasets are returned
    assert train_dataset is not None
    assert eval_dataset is not None
    assert test_dataset is not None
    assert data_collator is not None


def test_get_label_list_without_feature_names():
    """Test get_label_list fallback when feature names are not available."""
    # Create a simple dataset without proper feature names
    data = [{"tokens": ["word1", "word2"], "ner_tags": [0, 1]}]
    simple_dataset = DatasetDict(
        {"train": Dataset.from_list(data), "validation": Dataset.from_list(data), "test": Dataset.from_list(data)}
    )

    labels = get_label_list(simple_dataset)

    # Should return default CoNLL-2003 labels
    expected_labels = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
    assert labels == expected_labels


def test_print_dataset_statistics(capsys):
    """Test dataset statistics printing."""
    dataset = create_sample_dataset(size=10)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    print_dataset_statistics(dataset, tokenizer)

    captured = capsys.readouterr()

    # Check that output contains expected sections
    assert "Dataset Statistics" in captured.out
    assert "TRAIN Split:" in captured.out
    assert "VALIDATION Split:" in captured.out
    assert "TEST Split:" in captured.out
    assert "Number of examples:" in captured.out
    assert "Average tokens per example:" in captured.out
    assert "Entity distribution:" in captured.out


def test_print_dataset_statistics_without_tokenizer(capsys):
    """Test dataset statistics printing without tokenizer."""
    dataset = create_sample_dataset(size=5)

    print_dataset_statistics(dataset)

    captured = capsys.readouterr()

    # Check that output contains expected sections
    assert "Dataset Statistics" in captured.out
    assert "Number of examples:" in captured.out


def test_validate_dataset_label_mismatch_warning(caplog):
    """Test dataset validation warns on label mismatch."""
    from datasets import ClassLabel, Features, Sequence, Value

    # Create dataset with different label names
    different_labels = ["O", "B-PERSON", "I-PERSON", "B-ORGANIZATION", "I-ORGANIZATION"]
    expected_labels = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]

    features = Features(
        {
            "tokens": Sequence(feature=Value(dtype="string")),
            "ner_tags": Sequence(feature=ClassLabel(names=different_labels)),
        }
    )

    data = [{"tokens": ["test"], "ner_tags": [0]}]
    dataset = DatasetDict(
        {
            "train": Dataset.from_list(data, features=features),
            "validation": Dataset.from_list(data, features=features),
            "test": Dataset.from_list(data, features=features),
        }
    )

    # Should log a warning about label mismatch
    validate_dataset(dataset, expected_labels)

    assert "Label mismatch" in caplog.text
    assert "Expected:" in caplog.text
    assert "Got:" in caplog.text


def test_tokenize_and_align_labels_edge_cases():
    """Test tokenization edge cases with label_all_tokens."""
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Test with label_all_tokens=True and subword tokens
    examples = {
        "tokens": [["playing"]],  # This will likely split into subwords
        "ner_tags": [[1]],  # B-PER
    }

    result = tokenize_and_align_labels(examples=examples, tokenizer=tokenizer, max_length=128, label_all_tokens=True)

    # The word "playing" should be split and all subwords should get the label
    labels = result["labels"][0]
    # Count non-special token labels
    valid_labels = [label for label in labels if label != -100]
    assert all(label == 1 for label in valid_labels)  # All subwords get the same label


def test_print_dataset_statistics_with_entity_labels(capsys):
    """Test dataset statistics printing with proper entity label names."""
    from datasets import ClassLabel, Features, Sequence, Value

    # Create dataset with proper label features
    label_names = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]

    data = [
        {
            "tokens": ["John", "Smith", "works", "at", "Microsoft", "in", "Seattle"],
            "ner_tags": [1, 2, 0, 0, 3, 0, 5],
        },  # B-PER, I-PER, O, O, B-ORG, O, B-LOC
        {
            "tokens": ["Apple", "Inc", "is", "based", "in", "Cupertino"],
            "ner_tags": [3, 4, 0, 0, 0, 5],
        },  # B-ORG, I-ORG, O, O, O, B-LOC
    ]

    features = Features(
        {
            "tokens": Sequence(feature=Value(dtype="string")),
            "ner_tags": Sequence(feature=ClassLabel(names=label_names)),
        }
    )

    dataset = DatasetDict(
        {
            "train": Dataset.from_list(data, features=features),
            "validation": Dataset.from_list(data[:1], features=features),
            "test": Dataset.from_list(data[1:], features=features),
        }
    )

    print_dataset_statistics(dataset)

    captured = capsys.readouterr()

    # Check that entity labels are printed with their names
    assert "B-PER: 1" in captured.out
    assert "I-PER: 1" in captured.out
    assert "B-ORG: 2" in captured.out
    assert "I-ORG: 1" in captured.out
    assert "B-LOC: 2" in captured.out
    assert "O:" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
