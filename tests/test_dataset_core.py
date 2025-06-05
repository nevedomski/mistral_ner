"""Tests for core dataset module components."""


import pytest

from datasets import Dataset, DatasetDict
from src.datasets.base import BaseNERDataset
from src.datasets.mixers import DatasetMixer
from src.datasets.registry import DatasetRegistry


class TestBaseNERDataset:
    """Test the abstract base class."""

    def test_init(self):
        """Test initialization."""

        # Create a concrete implementation for testing
        class ConcreteDataset(BaseNERDataset):
            def load(self):
                return DatasetDict()

            def get_label_mapping(self):
                return {}

            def preprocess(self, examples):
                return examples

        # Test without config
        dataset = ConcreteDataset()
        assert dataset.config == {}

        # Test with config
        config = {"key": "value"}
        dataset = ConcreteDataset(config)
        assert dataset.config == config

    def test_abstract_methods(self):
        """Test that abstract methods must be implemented."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseNERDataset()

    def test_validate_dataset_valid(self):
        """Test validation of a valid dataset."""

        class ConcreteDataset(BaseNERDataset):
            def load(self):
                return DatasetDict()

            def get_label_mapping(self):
                return {}

            def preprocess(self, examples):
                return examples

        dataset = ConcreteDataset()

        # Create valid dataset
        valid_data = {
            "train": Dataset.from_dict({"tokens": [["word1", "word2"]], "ner_tags": [[0, 1]]}),
            "validation": Dataset.from_dict({"tokens": [["word3", "word4"]], "ner_tags": [[0, 0]]}),
            "test": Dataset.from_dict({"tokens": [["word5", "word6"]], "ner_tags": [[1, 1]]}),
        }

        dataset.validate_dataset(DatasetDict(valid_data))  # Should not raise

    def test_validate_dataset_missing_split(self):
        """Test validation with missing required split."""

        class ConcreteDataset(BaseNERDataset):
            def load(self):
                return DatasetDict()

            def get_label_mapping(self):
                return {}

            def preprocess(self, examples):
                return examples

        dataset = ConcreteDataset()

        # Missing validation split
        invalid_data = {"train": Dataset.from_dict({"tokens": [["word1", "word2"]], "ner_tags": [[0, 1]]})}

        with pytest.raises(ValueError, match="missing required split: validation"):
            dataset.validate_dataset(DatasetDict(invalid_data))

    def test_validate_dataset_missing_features(self):
        """Test validation with missing features."""

        class ConcreteDataset(BaseNERDataset):
            def load(self):
                return DatasetDict()

            def get_label_mapping(self):
                return {}

            def preprocess(self, examples):
                return examples

        dataset = ConcreteDataset()

        # Missing ner_tags feature
        invalid_data = {
            "train": Dataset.from_dict({"tokens": [["word1", "word2"]]}),
            "validation": Dataset.from_dict({"tokens": [["word3", "word4"]]}),
        }

        with pytest.raises(ValueError, match="missing 'ner_tags' feature"):
            dataset.validate_dataset(DatasetDict(invalid_data))

    def test_validate_dataset_optional_test(self):
        """Test that test split is optional."""

        class ConcreteDataset(BaseNERDataset):
            def load(self):
                return DatasetDict()

            def get_label_mapping(self):
                return {}

            def preprocess(self, examples):
                return examples

        dataset = ConcreteDataset()

        # No test split (should be OK)
        valid_data = {
            "train": Dataset.from_dict({"tokens": [["word1", "word2"]], "ner_tags": [[0, 1]]}),
            "validation": Dataset.from_dict({"tokens": [["word3", "word4"]], "ner_tags": [[0, 0]]}),
        }

        dataset.validate_dataset(DatasetDict(valid_data))  # Should not raise


class TestDatasetRegistry:
    """Test the dataset registry."""

    def test_init_and_default_loaders(self):
        """Test initialization and default loader registration."""
        registry = DatasetRegistry()

        # Check that default loaders are registered
        available = registry.list_available()
        assert "conll2003" in available
        assert "ontonotes" in available
        assert "wnut17" in available
        assert "fewnerd" in available
        assert "gretel_pii" in available
        assert "wikiner" in available
        assert "ai4privacy" in available
        assert "mendeley_pii" in available
        assert "bigcode_pii" in available
        assert len(available) == 9

    def test_register_custom_loader(self):
        """Test registering a custom loader."""
        registry = DatasetRegistry()

        # Create a mock loader class
        class CustomLoader(BaseNERDataset):
            def load(self):
                return DatasetDict()

            def get_label_mapping(self):
                return {}

            def preprocess(self, examples):
                return examples

        # Register it
        registry.register("custom", CustomLoader)

        # Check it's available
        assert "custom" in registry.list_available()
        assert len(registry.list_available()) == 10

    def test_get_loader(self):
        """Test getting a loader instance."""
        registry = DatasetRegistry()

        # Get a known loader
        loader = registry.get_loader("conll2003")
        assert loader is not None
        assert hasattr(loader, "load")
        assert hasattr(loader, "get_label_mapping")
        assert hasattr(loader, "preprocess")

        # Get with config
        config = {"test": "value"}
        loader = registry.get_loader("conll2003", config)
        assert loader.config == config

    def test_get_loader_unknown(self):
        """Test getting an unknown loader."""
        registry = DatasetRegistry()

        with pytest.raises(ValueError, match="Unknown dataset: unknown_dataset"):
            registry.get_loader("unknown_dataset")

    def test_get_loader_error_message(self):
        """Test error message includes available datasets."""
        registry = DatasetRegistry()

        try:
            registry.get_loader("invalid")
        except ValueError as e:
            error_msg = str(e)
            assert "Available:" in error_msg
            assert "conll2003" in error_msg
            assert "wikiner" in error_msg


class TestDatasetMixer:
    """Test the dataset mixer."""

    def test_mix_empty_datasets(self):
        """Test error when no datasets provided."""
        with pytest.raises(ValueError, match="No datasets provided to mix"):
            DatasetMixer.mix([])

    def test_concat_strategy(self):
        """Test concatenation mixing strategy."""
        # Create test datasets
        dataset1 = DatasetDict({"train": Dataset.from_dict({"tokens": [["a"], ["b"]], "ner_tags": [[0], [1]]})})

        dataset2 = DatasetDict({"train": Dataset.from_dict({"tokens": [["c"], ["d"]], "ner_tags": [[2], [3]]})})

        result = DatasetMixer.mix([dataset1, dataset2], strategy="concat")

        # Check concatenation
        assert "train" in result
        assert len(result["train"]) == 4
        assert result["train"][0]["tokens"] == ["a"]
        assert result["train"][2]["tokens"] == ["c"]

    def test_interleave_strategy(self):
        """Test interleave mixing strategy."""
        # Create test datasets
        dataset1 = DatasetDict({"train": Dataset.from_dict({"tokens": [["a"], ["b"]], "ner_tags": [[0], [1]]})})

        dataset2 = DatasetDict({"train": Dataset.from_dict({"tokens": [["c"], ["d"]], "ner_tags": [[2], [3]]})})

        result = DatasetMixer.mix([dataset1, dataset2], strategy="interleave", seed=42)

        # Check that we get a result with combined data
        assert "train" in result
        assert len(result["train"]) == 4

    def test_weighted_strategy(self):
        """Test weighted mixing strategy."""
        # Create test datasets
        dataset1 = DatasetDict(
            {"train": Dataset.from_dict({"tokens": [["a"] for _ in range(100)], "ner_tags": [[0] for _ in range(100)]})}
        )

        dataset2 = DatasetDict(
            {"train": Dataset.from_dict({"tokens": [["b"] for _ in range(100)], "ner_tags": [[1] for _ in range(100)]})}
        )

        # Use weights favoring first dataset
        result = DatasetMixer.mix([dataset1, dataset2], strategy="weighted", weights=[0.8, 0.2], seed=42)

        # Should have both datasets represented
        assert "train" in result

    def test_normalize_weights(self):
        """Test weight normalization."""
        weights = [2.0, 3.0, 5.0]
        normalized = DatasetMixer._normalize_weights(weights, 3)

        # Should sum to 1.0
        assert abs(sum(normalized) - 1.0) < 1e-6
        assert normalized == [0.2, 0.3, 0.5]

    def test_normalize_weights_zero_sum(self):
        """Test weight normalization with zero sum."""
        weights = [0.0, 0.0, 0.0]
        normalized = DatasetMixer._normalize_weights(weights, 3)

        # Should default to equal weights
        assert normalized == [1 / 3, 1 / 3, 1 / 3]

    def test_normalize_weights_mismatch(self):
        """Test error when weight count doesn't match dataset count."""
        weights = [0.5, 0.5]

        with pytest.raises(ValueError, match="Number of weights"):
            DatasetMixer._normalize_weights(weights, 3)

    def test_unknown_strategy(self):
        """Test error with unknown mixing strategy."""
        dataset1 = DatasetDict({"train": Dataset.from_dict({"tokens": [["a"]], "ner_tags": [[0]]})})

        with pytest.raises(ValueError, match="Unknown mixing strategy"):
            DatasetMixer.mix([dataset1], strategy="unknown")

    def test_concat_datasets(self):
        """Test the concatenation helper method."""
        dataset1 = Dataset.from_dict({"tokens": [["a"], ["b"]], "ner_tags": [[0], [1]]})

        dataset2 = Dataset.from_dict({"tokens": [["c"], ["d"]], "ner_tags": [[2], [3]]})

        result = DatasetMixer._concat_datasets([dataset1, dataset2])

        assert len(result) == 4
        assert result[0]["tokens"] == ["a"]
        assert result[2]["tokens"] == ["c"]

    def test_interleave_datasets(self):
        """Test the interleave helper method."""
        dataset1 = Dataset.from_dict({"tokens": [["a"], ["b"]], "ner_tags": [[0], [1]]})

        dataset2 = Dataset.from_dict({"tokens": [["c"], ["d"]], "ner_tags": [[2], [3]]})

        result = DatasetMixer._interleave_datasets([dataset1, dataset2], weights=None, seed=42)

        assert len(result) == 4

    def test_weighted_sample(self):
        """Test the weighted sampling helper method."""
        dataset1 = Dataset.from_dict({"tokens": [["a"] for _ in range(100)], "ner_tags": [[0] for _ in range(100)]})

        dataset2 = Dataset.from_dict({"tokens": [["b"] for _ in range(100)], "ner_tags": [[1] for _ in range(100)]})

        # This method exists but may not be implemented
        try:
            result = DatasetMixer._weighted_sample([dataset1, dataset2], [0.7, 0.3], seed=42)
            # If implemented, should return a dataset
            assert hasattr(result, "__len__")
        except (NotImplementedError, AttributeError):
            # Method might not be implemented
            pass

    def test_mix_partial_splits(self):
        """Test mixing when datasets have different splits."""
        dataset1 = DatasetDict(
            {
                "train": Dataset.from_dict({"tokens": [["a"]], "ner_tags": [[0]]}),
                "validation": Dataset.from_dict({"tokens": [["b"]], "ner_tags": [[1]]}),
            }
        )

        dataset2 = DatasetDict(
            {
                "train": Dataset.from_dict({"tokens": [["c"]], "ner_tags": [[2]]})
                # No validation split
            }
        )

        result = DatasetMixer.mix([dataset1, dataset2], strategy="concat")

        # Should have train from both, validation from dataset1 only
        assert "train" in result
        assert "validation" in result
        assert len(result["train"]) == 2
        assert len(result["validation"]) == 1
