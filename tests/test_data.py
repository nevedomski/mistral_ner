"""Tests for data processing module."""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data import (
    tokenize_and_align_labels,
    create_sample_dataset,
    get_label_list,
    validate_dataset
)
from src.config import Config
from transformers import AutoTokenizer


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
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    examples = {
        "tokens": [["John", "lives", "in", "New", "York", "."]],
        "ner_tags": [[1, 0, 0, 5, 6, 0]]  # B-PER, O, O, B-LOC, I-LOC, O
    }
    
    result = tokenize_and_align_labels(
        examples=examples,
        tokenizer=tokenizer,
        max_length=128,
        label_all_tokens=False
    )
    
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
        "validation": dataset["validation"]
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])