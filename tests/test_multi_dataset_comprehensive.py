"""Comprehensive tests for multi_dataset module."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from datasets import Dataset
from src.datasets.multi_dataset import MultiDatasetLoader


@pytest.fixture
def mock_config():
    """Create a mock config for testing."""
    config = Mock()
    config.data.multi_dataset.dataset_names = ["conll2003", "wikiner"]
    config.data.multi_dataset.dataset_weights = [0.6, 0.4]
    config.data.multi_dataset.mixing_strategy = "weighted"
    config.data.multi_dataset.max_samples_per_dataset = 1000
    config.data.multi_dataset.label_mappings = None
    config.data.multi_dataset.label_mapping_profile = None
    config.data.label_names = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC"]
    config.data.id2label = None  # Explicitly set to None
    config.training.seed = 42
    return config


@pytest.fixture
def mock_dataset():
    """Create a mock dataset."""
    data = {
        "tokens": [["John", "lives", "in", "New", "York"], ["Mary", "works", "at", "Google"]],
        "ner_tags": [[1, 0, 0, 3, 4], [1, 0, 0, 0, 0]],
    }
    return Dataset.from_dict(data)


class TestMultiDatasetLoader:
    """Test MultiDatasetLoader functionality."""

    def test_init_with_valid_config(self, mock_config):
        """Test initialization with valid config."""
        loader = MultiDatasetLoader(mock_config)

        assert loader.dataset_names == ["conll2003", "wikiner"]
        assert loader.weights == [0.6, 0.4]
        assert loader.strategy == "weighted"
        assert len(loader.datasets) == 0

    def test_init_with_equal_weights(self, mock_config):
        """Test initialization with no weights specified."""
        mock_config.data.multi_dataset.dataset_weights = None
        loader = MultiDatasetLoader(mock_config)

        assert loader.weights == [0.5, 0.5]

    def test_validate_config_no_datasets(self, mock_config):
        """Test validation fails with no datasets."""
        mock_config.data.multi_dataset.dataset_names = []

        with pytest.raises(ValueError, match="No datasets specified"):
            MultiDatasetLoader(mock_config)

    def test_validate_config_invalid_strategy(self, mock_config):
        """Test validation fails with invalid strategy."""
        mock_config.data.multi_dataset.mixing_strategy = "invalid"

        with pytest.raises(ValueError, match="Invalid mixing strategy"):
            MultiDatasetLoader(mock_config)

    def test_validate_config_mismatched_weights(self, mock_config):
        """Test validation fails with mismatched weights."""
        mock_config.data.multi_dataset.dataset_weights = [0.5]  # Only 1 weight for 2 datasets

        with pytest.raises(ValueError, match="Number of weights.*must match"):
            MultiDatasetLoader(mock_config)

    def test_get_label_mapping_no_mapping(self, mock_config):
        """Test getting label mapping when none specified."""
        loader = MultiDatasetLoader(mock_config)
        mapping = loader._get_label_mapping("conll2003")

        assert mapping == {}

    def test_get_label_mapping_with_profile(self, mock_config):
        """Test getting label mapping with profile."""
        # Test that when a profile is set, it returns an empty dict when profile doesn't exist
        mock_config.data.multi_dataset.label_mapping_profile = "unknown_profile"

        with patch("src.datasets.multi_dataset.logger") as mock_logger:
            loader = MultiDatasetLoader(mock_config)
            mapping = loader._get_label_mapping("conll2003")

            assert mapping == {}
            # Verify warning was logged
            mock_logger.warning.assert_called()

    def test_get_label_mapping_with_custom_mapping(self, mock_config):
        """Test getting label mapping with custom mapping."""
        custom_mapping = {"B-MISC": "B-ORG", "I-MISC": "I-ORG"}
        mock_config.data.multi_dataset.label_mappings = {"conll2003": custom_mapping}

        loader = MultiDatasetLoader(mock_config)
        mapping = loader._get_label_mapping("conll2003")

        assert mapping == custom_mapping

    @patch("src.datasets.multi_dataset.DatasetRegistry")
    def test_load_and_prepare_datasets(self, mock_registry_class, mock_config, mock_dataset):
        """Test loading and preparing datasets."""
        # Setup mocks
        mock_registry = Mock()
        mock_registry_class.return_value = mock_registry

        mock_loader1 = Mock()
        mock_loader1.load.return_value = mock_dataset
        mock_loader2 = Mock()
        mock_loader2.load.return_value = mock_dataset

        mock_registry.get_loader.side_effect = [mock_loader1, mock_loader2]

        # Create loader and load datasets
        loader = MultiDatasetLoader(mock_config)
        datasets = loader.load_and_prepare_datasets(split="train", max_samples_per_dataset=500)

        assert len(datasets) == 2
        assert mock_registry.get_loader.call_count == 2

    @patch("src.datasets.multi_dataset.DatasetRegistry")
    def test_load_and_prepare_datasets_with_limit(self, mock_registry_class, mock_config, mock_dataset):
        """Test loading datasets with sample limit."""
        # Create a larger mock dataset
        large_data = {
            "tokens": [["token"] * 5] * 2000,
            "ner_tags": [[0] * 5] * 2000,
        }
        large_dataset = Dataset.from_dict(large_data)

        # Setup mocks
        mock_registry = Mock()
        mock_registry_class.return_value = mock_registry

        mock_loader = Mock()
        mock_loader.load.return_value = large_dataset
        mock_registry.get_loader.return_value = mock_loader

        # Create loader and load datasets with limit
        loader = MultiDatasetLoader(mock_config)
        loader.load_and_prepare_datasets(split="train", max_samples_per_dataset=100)

        # Verify dataset was limited
        assert all(len(d) <= 100 for d in loader.datasets)

    @patch("src.datasets.multi_dataset.DatasetRegistry")
    def test_load_and_prepare_datasets_error_handling(self, mock_registry_class, mock_config):
        """Test error handling when loading datasets."""
        mock_registry = Mock()
        mock_registry_class.return_value = mock_registry
        mock_registry.get_loader.side_effect = Exception("Dataset not found")

        loader = MultiDatasetLoader(mock_config)

        with pytest.raises(Exception, match="Dataset not found"):
            loader.load_and_prepare_datasets()

    @patch("src.datasets.multi_dataset.interleave_datasets")
    def test_create_interleaved_dataset_round_robin(self, mock_interleave, mock_config, mock_dataset):
        """Test creating interleaved dataset with round-robin strategy."""
        mock_config.data.multi_dataset.mixing_strategy = "interleave"
        mock_interleaved = Mock()
        mock_interleave.return_value = mock_interleaved

        loader = MultiDatasetLoader(mock_config)
        loader.datasets = [mock_dataset, mock_dataset]

        result = loader.create_interleaved_dataset()

        assert result == mock_interleaved
        mock_interleave.assert_called_once_with(
            loader.datasets, probabilities=None, seed=42, stopping_strategy="all_exhausted"
        )

    @patch("src.datasets.multi_dataset.interleave_datasets")
    def test_create_interleaved_dataset_weighted(self, mock_interleave, mock_config, mock_dataset):
        """Test creating interleaved dataset with weighted strategy."""
        mock_interleaved = Mock()
        mock_interleave.return_value = mock_interleaved

        loader = MultiDatasetLoader(mock_config)
        loader.datasets = [mock_dataset, mock_dataset]

        result = loader.create_interleaved_dataset(seed=123)

        assert result == mock_interleaved
        mock_interleave.assert_called_once_with(
            loader.datasets, probabilities=[0.6, 0.4], seed=123, stopping_strategy="first_exhausted"
        )

    def test_create_interleaved_dataset_no_datasets(self, mock_config):
        """Test error when creating interleaved dataset with no datasets."""
        loader = MultiDatasetLoader(mock_config)

        with pytest.raises(ValueError, match="No datasets loaded"):
            loader.create_interleaved_dataset()

    @patch("src.datasets.multi_dataset.DatasetRegistry")
    @patch("src.datasets.multi_dataset.interleave_datasets")
    def test_create_train_eval_datasets(self, mock_interleave, mock_registry_class, mock_config, mock_dataset):
        """Test creating train and eval datasets."""
        # Setup mocks
        mock_registry = Mock()
        mock_registry_class.return_value = mock_registry

        mock_loader = Mock()
        mock_loader.load.return_value = mock_dataset
        mock_registry.get_loader.return_value = mock_loader

        mock_train_interleaved = Mock()
        mock_eval_interleaved = Mock()
        mock_interleave.side_effect = [mock_train_interleaved, mock_eval_interleaved]

        # Create datasets
        loader = MultiDatasetLoader(mock_config)
        train_dataset, eval_dataset = loader.create_train_eval_datasets()

        assert train_dataset == mock_train_interleaved
        assert eval_dataset == mock_eval_interleaved
        assert mock_interleave.call_count == 2

    def test_get_dataset_statistics_no_datasets(self, mock_config):
        """Test getting statistics with no datasets."""
        loader = MultiDatasetLoader(mock_config)
        stats = loader.get_dataset_statistics()

        assert stats == {"error": "No datasets loaded"}

    @patch("src.datasets.multi_dataset.DatasetRegistry")
    def test_get_dataset_statistics_with_datasets(self, mock_registry_class, mock_config, mock_dataset):
        """Test getting statistics with loaded datasets."""
        # Setup
        mock_registry = Mock()
        mock_registry_class.return_value = mock_registry

        mock_loader = Mock()
        mock_loader.load.return_value = mock_dataset
        mock_registry.get_loader.return_value = mock_loader

        # Load datasets and get statistics
        loader = MultiDatasetLoader(mock_config)
        loader.load_and_prepare_datasets()
        stats = loader.get_dataset_statistics()

        assert stats["num_datasets"] == 2
        assert stats["total_samples"] == 4  # 2 samples per dataset
        assert stats["strategy"] == "weighted"
        assert stats["weights"] == [0.6, 0.4]
        assert "datasets" in stats

    def test_get_label_distribution(self, mock_config, mock_dataset):
        """Test getting label distribution."""
        loader = MultiDatasetLoader(mock_config)
        distribution = loader._get_label_distribution(mock_dataset)

        # Should count labels: {0: 6, 1: 2, 3: 1, 4: 1}
        assert distribution[0] == 6
        assert distribution[1] == 2
        assert distribution[3] == 1
        assert distribution[4] == 1

    def test_get_label_distribution_with_id2label(self, mock_config, mock_dataset):
        """Test getting label distribution with id2label mapping."""
        mock_config.data.id2label = {0: "O", 1: "B-PER", 3: "B-LOC", 4: "I-LOC"}

        loader = MultiDatasetLoader(mock_config)
        distribution = loader._get_label_distribution(mock_dataset)

        assert distribution["O"] == 6
        assert distribution["B-PER"] == 2
        assert distribution["B-LOC"] == 1
        assert distribution["I-LOC"] == 1

    def test_label_mapping_profile_reference(self, mock_config):
        """Test label mapping with profile reference in label_mappings."""
        # Test string reference to a profile
        mock_config.data.multi_dataset.label_mappings = {"conll2003": "unknown_profile"}

        with patch("src.datasets.multi_dataset.logger") as mock_logger:
            loader = MultiDatasetLoader(mock_config)
            mapping = loader._get_label_mapping("conll2003")

            assert mapping == {}
            # Should log warning about unknown profile
            mock_logger.warning.assert_called()
