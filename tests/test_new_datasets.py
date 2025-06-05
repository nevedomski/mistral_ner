"""Test new dataset loaders."""

from src.datasets import DatasetRegistry


class TestNewDatasets:
    """Test newly added dataset loaders."""

    def test_all_datasets_registered(self):
        """Test that all new datasets are registered."""
        registry = DatasetRegistry()
        available = registry.list_available()

        new_datasets = ["wikiner", "ai4privacy", "mendeley_pii", "bigcode_pii"]
        for dataset in new_datasets:
            assert dataset in available, f"Dataset {dataset} not registered"

    def test_wikiner_loader(self):
        """Test WikiNER loader initialization."""
        registry = DatasetRegistry()
        loader = registry.get_loader("wikiner", {"language": "en"})

        assert loader is not None
        assert loader.language == "en"

        # Test label mapping
        mapping = loader.get_label_mapping()
        assert "O" in mapping
        assert "B-PER" in mapping
        assert "I-PER" in mapping
        assert "B-ORG" in mapping
        assert "I-ORG" in mapping
        assert "B-LOC" in mapping
        assert "I-LOC" in mapping

    def test_ai4privacy_loader(self):
        """Test AI4Privacy loader initialization."""
        registry = DatasetRegistry()
        loader = registry.get_loader("ai4privacy")

        assert loader is not None

        # Test label mapping
        mapping = loader.get_label_mapping()
        assert "O" in mapping
        assert len(mapping) > 50  # Should have many PII entity types

    def test_mendeley_pii_loader(self):
        """Test Mendeley PII loader initialization."""
        registry = DatasetRegistry()
        loader = registry.get_loader("mendeley_pii", {"filter_english": True})

        assert loader is not None
        assert loader.filter_english is True

        # Test label mapping
        mapping = loader.get_label_mapping()
        assert "O" in mapping
        assert "B-CREDITCARDNUMBER" in mapping
        assert "B-SSN" in mapping

    def test_bigcode_pii_loader(self):
        """Test BigCode PII loader initialization."""
        registry = DatasetRegistry()
        loader = registry.get_loader("bigcode_pii")

        assert loader is not None
        assert loader.use_auth_token is None  # Default value

        # Test label mapping
        mapping = loader.get_label_mapping()
        assert "O" in mapping
        assert "B-KEY" in mapping
        assert "B-API_KEY" in mapping

    def test_complete_dataset_list(self):
        """Test that we have all 9 datasets registered."""
        registry = DatasetRegistry()
        available = registry.list_available()

        expected_datasets = [
            "conll2003",
            "ontonotes",
            "wnut17",
            "fewnerd",
            "gretel_pii",
            "wikiner",
            "ai4privacy",
            "mendeley_pii",
            "bigcode_pii",
        ]

        assert len(available) == len(expected_datasets)
        for dataset in expected_datasets:
            assert dataset in available
