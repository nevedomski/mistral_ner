"""Unit tests for multi-dataset NER functionality."""

from unittest.mock import Mock, patch

from datasets import Dataset, DatasetDict
from src.config import Config
from src.data import prepare_multi_datasets
from src.datasets import BaseNERDataset, DatasetMixer, DatasetRegistry, LabelMapper, UnifiedLabelSchema
from src.datasets.loaders import CoNLLDataset, GretelPIIDataset


class TestMultiDatasetConfig:
    """Test multi-dataset configuration."""

    def test_default_config_is_single_dataset(self):
        """Test that default config uses single dataset mode."""
        config = Config()
        assert config.data.multi_dataset.enabled is False
        assert len(config.data.label_names) == 9  # CoNLL labels
        assert config.data.label_names[0] == "O"

    def test_multi_dataset_config_enables_unified_labels(self):
        """Test that enabling multi-dataset uses unified labels."""
        config = Config()
        config.data.multi_dataset.enabled = True
        config.data.__post_init__()  # Re-initialize labels

        assert len(config.data.label_names) == 55  # Unified schema
        assert "B-CARD" in config.data.label_names  # PII label
        assert "B-SSN" in config.data.label_names

    def test_config_from_yaml_loads_multi_dataset(self, tmp_path):
        """Test loading multi-dataset config from YAML."""
        yaml_content = """
data:
  dataset_name: "conll2003"
  max_length: 256
  multi_dataset:
    enabled: true
    dataset_names: ["conll2003", "gretel_pii"]
    dataset_weights: [0.7, 0.3]
    mixing_strategy: "interleave"
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        config = Config.from_yaml(config_file)

        assert config.data.multi_dataset.enabled is True
        assert config.data.multi_dataset.dataset_names == ["conll2003", "gretel_pii"]
        assert config.data.multi_dataset.dataset_weights == [0.7, 0.3]
        assert config.data.multi_dataset.mixing_strategy == "interleave"


class TestUnifiedLabelSchema:
    """Test unified label schema."""

    def test_get_bio_labels(self):
        """Test BIO label generation."""
        labels = UnifiedLabelSchema.get_bio_labels()

        assert "O" in labels
        assert "B-PER" in labels
        assert "I-PER" in labels
        assert "B-CARD" in labels  # PII
        assert "I-CARD" in labels
        assert "B-SSN" in labels
        assert "I-SSN" in labels

    def test_all_entity_types_have_bio(self):
        """Test that all entity types have both B- and I- prefixes."""
        labels = UnifiedLabelSchema.get_bio_labels()

        # Extract entity types
        entity_types = set()
        for label in labels:
            if label != "O" and "-" in label:
                entity_types.add(label.split("-", 1)[1])

        # Check each entity type has both B- and I-
        for entity_type in entity_types:
            assert f"B-{entity_type}" in labels
            assert f"I-{entity_type}" in labels


class TestLabelMapper:
    """Test label mapping functionality."""

    def test_map_labels_with_strings(self):
        """Test mapping string labels."""
        mapper = LabelMapper(["O", "B-PER", "I-PER", "B-LOC", "I-LOC"])

        examples = {"ner_tags": [["O", "B-PER", "I-PER", "B-LOC"]]}
        mapping = {"O": "O", "B-PER": "B-PER", "I-PER": "I-PER", "B-LOC": "B-LOC"}

        result = mapper.map_labels(examples, mapping)

        assert result["ner_tags"][0] == [0, 1, 2, 3]  # Mapped to IDs

    def test_map_labels_with_unknown_labels(self):
        """Test mapping with labels not in unified schema."""
        mapper = LabelMapper(["O", "B-PER", "I-PER"])

        examples = {"ner_tags": [["O", "B-UNKNOWN", "I-UNKNOWN"]]}
        mapping = {"O": "O", "B-UNKNOWN": "B-UNKNOWN", "I-UNKNOWN": "I-UNKNOWN"}

        result = mapper.map_labels(examples, mapping)

        # Unknown labels should map to O (ID 0)
        assert result["ner_tags"][0] == [0, 0, 0]

    def test_get_label_distribution(self):
        """Test label distribution calculation."""
        mapper = LabelMapper(["O", "B-PER", "I-PER"])

        dataset = [{"ner_tags": [0, 1, 2, 0]}, {"ner_tags": [0, 0, 1, 2]}]

        distribution = mapper.get_label_distribution(dataset)

        assert distribution["O"] == 4
        assert distribution["B-PER"] == 2
        assert distribution["I-PER"] == 2


class TestDatasetRegistry:
    """Test dataset registry functionality."""

    def test_default_loaders_registered(self):
        """Test that default loaders are registered."""
        registry = DatasetRegistry()

        available = registry.list_available()
        assert "conll2003" in available
        assert "gretel_pii" in available
        assert "ontonotes" in available

    def test_get_loader(self):
        """Test getting a specific loader."""
        registry = DatasetRegistry()

        loader = registry.get_loader("conll2003")
        assert isinstance(loader, CoNLLDataset)

    def test_get_loader_with_config(self):
        """Test getting loader with configuration."""
        registry = DatasetRegistry()

        config = {"language": "English"}
        loader = registry.get_loader("gretel_pii", config)
        assert isinstance(loader, GretelPIIDataset)
        assert loader.language == "English"

    def test_register_custom_loader(self):
        """Test registering a custom loader."""
        registry = DatasetRegistry()

        class CustomDataset(BaseNERDataset):
            def load(self):
                return DatasetDict()

            def get_label_mapping(self):
                return {}

            def preprocess(self, examples):
                return examples

        registry.register("custom", CustomDataset)

        assert "custom" in registry.list_available()
        loader = registry.get_loader("custom")
        assert isinstance(loader, CustomDataset)


class TestDatasetMixer:
    """Test dataset mixing strategies."""

    def test_concat_strategy(self):
        """Test concatenation mixing."""
        ds1 = DatasetDict({"train": Dataset.from_dict({"tokens": [["A"], ["B"]]})})
        ds2 = DatasetDict({"train": Dataset.from_dict({"tokens": [["C"], ["D"]]})})

        result = DatasetMixer.mix([ds1, ds2], strategy="concat")

        assert len(result["train"]) == 4
        tokens = [ex["tokens"] for ex in result["train"]]
        assert tokens == [["A"], ["B"], ["C"], ["D"]]

    def test_interleave_strategy(self):
        """Test interleaving with weights."""
        ds1 = DatasetDict({"train": Dataset.from_dict({"tokens": [["A"], ["B"]]})})
        ds2 = DatasetDict({"train": Dataset.from_dict({"tokens": [["C"], ["D"]]})})

        result = DatasetMixer.mix([ds1, ds2], strategy="interleave", weights=[0.5, 0.5])

        # Interleaving should mix examples
        assert len(result["train"]) >= 4

    def test_weighted_strategy(self):
        """Test weighted sampling."""
        ds1 = DatasetDict({"train": Dataset.from_dict({"tokens": [["A"], ["B"], ["C"], ["D"]]})})
        ds2 = DatasetDict({"train": Dataset.from_dict({"tokens": [["E"], ["F"]]})})

        # Weight heavily towards ds1
        result = DatasetMixer.mix([ds1, ds2], strategy="weighted", weights=[0.8, 0.2], seed=42)

        # Should have samples from both datasets
        assert len(result["train"]) == 6  # Total samples

    def test_mix_with_missing_splits(self):
        """Test mixing when datasets have different splits."""
        ds1 = DatasetDict(
            {
                "train": Dataset.from_dict({"tokens": [["A"]]}),
                "validation": Dataset.from_dict({"tokens": [["B"]]}),
                "test": Dataset.from_dict({"tokens": [["C"]]}),
            }
        )
        ds2 = DatasetDict(
            {
                "train": Dataset.from_dict({"tokens": [["D"]]}),
                "validation": Dataset.from_dict({"tokens": [["E"]]}),
                # No test split
            }
        )

        result = DatasetMixer.mix([ds1, ds2], strategy="concat")

        assert "train" in result
        assert "validation" in result
        assert "test" in result  # Should include test from ds1
        assert len(result["test"]) == 1  # Only from ds1


class TestPrepareMultiDatasets:
    """Test the prepare_multi_datasets function."""

    @patch("src.data.DatasetRegistry")
    @patch("src.data.LabelMapper")
    @patch("src.data.DatasetMixer")
    def test_prepare_multi_datasets_flow(self, mock_mixer, mock_mapper, mock_registry):
        """Test the full multi-dataset preparation flow."""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_tokenizer.model_max_length = 512

        config = Config()
        config.data.multi_dataset.enabled = True
        config.data.multi_dataset.dataset_names = ["dataset1", "dataset2"]
        config.data.multi_dataset.dataset_weights = [0.6, 0.4]
        config.data.multi_dataset.mixing_strategy = "interleave"
        config.data.__post_init__()

        # Mock dataset loader
        mock_loader = Mock()
        mock_dataset = DatasetDict(
            {
                "train": Dataset.from_dict({"tokens": [["Hello", "world"]], "ner_tags": [[0, 0]]}),
                "validation": Dataset.from_dict({"tokens": [["Test"]], "ner_tags": [[0]]}),
                "test": Dataset.from_dict({"tokens": [["Test"]], "ner_tags": [[0]]}),
            }
        )
        mock_loader.load.return_value = mock_dataset
        mock_loader.get_label_mapping.return_value = {"O": "O"}
        mock_loader.preprocess = lambda x: x

        # Setup registry mock
        mock_registry_instance = mock_registry.return_value
        mock_registry_instance.get_loader.return_value = mock_loader

        # Setup mapper mock
        mock_mapper_instance = mock_mapper.return_value
        mock_mapper_instance.map_labels = lambda x, mapping: x

        # Setup mixer mock
        mock_mixer.mix.return_value = mock_dataset

        # Call function

        with patch("src.data.tokenize_and_align_labels") as mock_tokenize:
            mock_tokenize.side_effect = lambda x, **kwargs: {"input_ids": [[1, 2, 3]], "labels": [[0, 0, 0]]}

            train, val, test, collator = prepare_multi_datasets(mock_tokenizer, config)

        # Verify calls
        assert mock_registry_instance.get_loader.call_count == 2
        assert mock_mixer.mix.called
        mix_call = mock_mixer.mix.call_args
        assert mix_call[1]["strategy"] == "interleave"
        assert mix_call[1]["weights"] == [0.6, 0.4]


class TestDatasetLoaders:
    """Test individual dataset loaders."""

    def test_conll_loader_interface(self):
        """Test CoNLL loader implements required interface."""
        loader = CoNLLDataset()

        # Check required methods exist
        assert hasattr(loader, "load")
        assert hasattr(loader, "get_label_mapping")
        assert hasattr(loader, "preprocess")
        assert hasattr(loader, "validate_dataset")

        # Test label mapping
        mapping = loader.get_label_mapping()
        assert mapping["B-PER"] == "B-PER"  # Direct mapping
        assert mapping["B-ORG"] == "B-ORG"

    def test_gretel_loader_interface(self):
        """Test Gretel PII loader implements required interface."""
        loader = GretelPIIDataset({"language": "English"})

        # Check configuration
        assert loader.language == "English"

        # Test label mapping
        mapping = loader.get_label_mapping()
        assert mapping["B-CREDIT_CARD"] == "B-CARD"  # Maps to unified
        assert mapping["B-SSN"] == "B-SSN"
        assert mapping["B-PERSON"] == "B-PER"
