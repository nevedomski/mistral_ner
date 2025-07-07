"""Tests for the label mapping system."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from src.datasets.base import BaseNERDataset
from src.datasets.mapping_profiles import MappingProfiles


class TestMappingProfiles:
    """Test the mapping profiles module."""

    def test_get_profile_bank_pii(self):
        """Test retrieving the bank_pii profile."""
        profile = MappingProfiles.get_profile("bank_pii")

        # Check it's a dictionary of dictionaries
        assert isinstance(profile, dict)
        assert "conll2003" in profile
        assert "ontonotes" in profile
        assert "gretel_pii" in profile
        assert "ai4privacy" in profile

        # Check some specific mappings
        assert profile["conll2003"]["B-LOC"] == "B-ADDR"
        assert profile["ontonotes"]["B-GPE"] == "B-ADDR"
        assert profile["gretel_pii"]["B-CREDIT_CARD"] == "B-CARD"
        assert profile["ai4privacy"]["B-FIRSTNAME"] == "B-PER"

    def test_get_profile_general(self):
        """Test retrieving the general profile."""
        profile = MappingProfiles.get_profile("general")

        assert isinstance(profile, dict)
        assert "conll2003" in profile

        # General profile preserves original labels
        assert profile["conll2003"]["B-LOC"] == "B-LOC"
        assert profile["conll2003"]["B-PER"] == "B-PER"

    def test_get_profile_case_insensitive(self):
        """Test profile names are case insensitive."""
        profile1 = MappingProfiles.get_profile("bank_pii")
        profile2 = MappingProfiles.get_profile("BANK_PII")
        profile3 = MappingProfiles.get_profile("Bank_PII")

        assert profile1 == profile2 == profile3

    def test_get_profile_invalid(self):
        """Test getting an invalid profile raises error."""
        with pytest.raises(ValueError, match="Unknown profile: invalid"):
            MappingProfiles.get_profile("invalid")

    def test_list_profiles(self):
        """Test listing available profiles."""
        profiles = MappingProfiles.list_profiles()

        assert isinstance(profiles, list)
        assert "bank_pii" in profiles
        assert "general" in profiles
        assert len(profiles) >= 2


class TestBaseDatasetLabelMapping:
    """Test label mapping functionality in base dataset."""

    class TestDataset(BaseNERDataset):
        """Concrete implementation for testing."""

        def load(self):
            return {}

        def get_default_label_mapping(self):
            return {
                "O": "O",
                "B-PER": "B-PERSON",
                "I-PER": "I-PERSON",
            }

        def preprocess(self, examples):
            return examples

    def test_default_mapping(self):
        """Test using default label mapping."""
        dataset = self.TestDataset()
        mapping = dataset.get_label_mapping()

        assert mapping["B-PER"] == "B-PERSON"
        assert mapping["I-PER"] == "I-PERSON"

    def test_direct_mapping_config(self):
        """Test providing mapping directly in config."""
        config = {
            "label_mapping": {
                "B-PER": "B-CUSTOM",
                "I-PER": "I-CUSTOM",
            }
        }
        dataset = self.TestDataset(config)
        mapping = dataset.get_label_mapping()

        assert mapping["B-PER"] == "B-CUSTOM"
        assert mapping["I-PER"] == "I-CUSTOM"

    def test_profile_mapping(self):
        """Test using a profile for mapping."""
        # Mock the dataset name to match a profile entry
        config = {"label_mapping": "profile:bank_pii", "dataset_name": "conll2003"}

        # Create a custom class that simulates CoNLLDataset
        class MockCoNLLDataset(self.TestDataset):
            pass

        # Change the class name to match what the profile loader expects
        MockCoNLLDataset.__name__ = "CoNLLDataset"

        dataset = MockCoNLLDataset(config)
        mapping = dataset.get_label_mapping()

        # Should use bank_pii profile for conll2003
        assert mapping["B-LOC"] == "B-ADDR"
        assert mapping["I-LOC"] == "I-ADDR"

    def test_file_mapping(self):
        """Test loading mapping from a file."""
        # Create a temporary mapping file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"B-PER": "B-FILE_PERSON", "I-PER": "I-FILE_PERSON", "O": "O"}, f)
            temp_path = f.name

        try:
            config = {"label_mapping": temp_path}
            dataset = self.TestDataset(config)
            mapping = dataset.get_label_mapping()

            assert mapping["B-PER"] == "B-FILE_PERSON"
            assert mapping["I-PER"] == "I-FILE_PERSON"
            assert mapping["O"] == "O"
        finally:
            Path(temp_path).unlink()

    def test_file_mapping_in_configs_dir(self):
        """Test loading mapping from configs/mappings directory."""
        # Create a mock file path that would be in configs/mappings
        config = {"label_mapping": "test_mapping.yaml"}

        # Mock the file loading
        mock_mapping = {
            "B-PER": "B-CONFIG_PERSON",
            "I-PER": "I-CONFIG_PERSON",
        }

        with (
            patch("builtins.open"),
            patch("yaml.safe_load", return_value=mock_mapping),
            patch("pathlib.Path.exists", return_value=True),
        ):
            dataset = self.TestDataset(config)
            mapping = dataset.get_label_mapping()

            assert mapping["B-PER"] == "B-CONFIG_PERSON"
            assert mapping["I-PER"] == "I-CONFIG_PERSON"

    def test_invalid_mapping_config_type(self):
        """Test invalid mapping config type raises error."""
        config = {"label_mapping": 123}  # Invalid type
        dataset = self.TestDataset(config)

        with pytest.raises(ValueError, match="Invalid mapping config type"):
            dataset.get_label_mapping()

    def test_missing_profile_mapping(self):
        """Test error when dataset not found in profile."""
        config = {"label_mapping": "profile:general", "dataset_name": "nonexistent"}

        # Create a class with a name that won't match any profile entries
        class NonexistentDataset(self.TestDataset):
            pass

        dataset = NonexistentDataset(config)
        with pytest.raises(ValueError, match="No mapping found for dataset in profile"):
            dataset.get_label_mapping()

    def test_mapping_file_not_found(self):
        """Test error when mapping file doesn't exist."""
        config = {"label_mapping": "/nonexistent/path.yaml"}
        dataset = self.TestDataset(config)

        with pytest.raises(FileNotFoundError, match="Mapping file not found"):
            dataset.get_label_mapping()

    def test_invalid_mapping_file_format(self):
        """Test error when mapping file has invalid format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: that: is: not: a: dict")
            temp_path = f.name

        try:
            config = {"label_mapping": temp_path}
            dataset = self.TestDataset(config)

            # Mock yaml.safe_load to return a non-dict
            with (
                patch("yaml.safe_load", return_value="not a dict"),
                pytest.raises(ValueError, match="Invalid mapping file format"),
            ):
                dataset.get_label_mapping()
        finally:
            Path(temp_path).unlink()

    def test_mapping_cached(self):
        """Test that mapping is cached after first load."""
        dataset = self.TestDataset()

        # Get mapping twice
        mapping1 = dataset.get_label_mapping()
        mapping2 = dataset.get_label_mapping()

        # Should be the same object (cached)
        assert mapping1 is mapping2
