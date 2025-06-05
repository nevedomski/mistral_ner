"""Tests for NER label mapper."""

from typing import ClassVar

from src.datasets.label_mapper import LabelMapper, UnifiedLabelSchema


class TestUnifiedLabelSchema:
    """Test the unified label schema."""

    def test_get_bio_labels(self):
        """Test getting BIO labels."""
        labels = UnifiedLabelSchema.get_bio_labels()

        # Should include O tag
        assert "O" in labels

        # Should include core entities
        assert "B-PER" in labels
        assert "I-PER" in labels
        assert "B-ORG" in labels
        assert "I-ORG" in labels
        assert "B-LOC" in labels
        assert "I-LOC" in labels

        # Should include PII entities
        assert "B-CARD" in labels
        assert "I-CARD" in labels
        assert "B-SSN" in labels
        assert "I-SSN" in labels

        # Should be sorted
        assert labels == sorted(labels)


class TestLabelMapper:
    """Test the NER label mapper."""

    def test_init_default(self):
        """Test initialization with default labels."""
        mapper = LabelMapper()
        assert len(mapper.unified_labels) > 0
        assert "O" in mapper.unified_labels
        assert "B-PER" in mapper.unified_labels
        assert mapper.label2id["O"] == 0

    def test_init_custom(self):
        """Test initialization with custom labels."""
        custom_labels = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG"]
        mapper = LabelMapper(custom_labels)
        assert mapper.unified_labels == custom_labels
        assert mapper.label2id == {label: i for i, label in enumerate(custom_labels)}
        assert mapper.id2label == {i: label for i, label in enumerate(custom_labels)}

    def test_map_labels_batch(self):
        """Test mapping labels in a batch."""
        # Create mapper with simple schema
        custom_labels = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG"]
        mapper = LabelMapper(custom_labels)

        # Test batch with string labels
        examples = {"ner_tags": [["B-PER", "I-PER", "O"], ["O", "B-ORG", "I-ORG"]]}

        # Mapping from dataset labels to unified schema
        label_mapping = {"B-PER": "B-PER", "I-PER": "I-PER", "B-ORG": "B-ORG", "I-ORG": "I-ORG", "O": "O"}

        result = mapper.map_labels(examples, label_mapping)

        # Results should be mapped to IDs
        assert result["ner_tags"] == [
            [1, 2, 0],  # B-PER, I-PER, O
            [0, 3, 4],  # O, B-ORG, I-ORG
        ]

    def test_map_labels_with_integers(self):
        """Test mapping labels with integer inputs."""
        mapper = LabelMapper(["O", "B-PER", "I-PER"])

        # Mock examples with features
        examples = {"ner_tags": [[1, 2, 0], [0, 1, 2]]}

        # Mock the feature names
        class MockFeature:
            names: ClassVar[list[str]] = ["O", "B-PER", "I-PER"]

        class MockField:
            feature = MockFeature()

        examples["ner_tags"] = MockField()
        examples["ner_tags"] = [[1, 2, 0], [0, 1, 2]]  # Override with actual data

        # Create a custom examples object that has the feature attribute
        class MockExamples(dict):
            def __init__(self, data):
                super().__init__(data)
                self._ner_tags_field = MockField()

            def get(self, key):
                if key == "ner_tags":
                    return self._ner_tags_field
                return super().get(key)

        mock_examples = MockExamples({"ner_tags": [[1, 2, 0], [0, 1, 2]]})

        label_mapping = {"O": "O", "B-PER": "B-PER", "I-PER": "I-PER"}
        result = mapper.map_labels(mock_examples, label_mapping)

        # Should map integer IDs correctly
        assert result["ner_tags"] == [[1, 2, 0], [0, 1, 2]]

    def test_map_labels_missing_mapping(self):
        """Test mapping with missing label in mapping."""
        mapper = LabelMapper(["O", "B-PER", "I-PER"])

        examples = {"ner_tags": [["B-PER", "B-UNKNOWN", "O"]]}

        label_mapping = {
            "B-PER": "B-PER",
            "O": "O",
            # Missing B-UNKNOWN
        }

        result = mapper.map_labels(examples, label_mapping)

        # Unknown label should map to O (ID 0)
        assert result["ner_tags"] == [[1, 0, 0]]  # B-PER, O (fallback), O

    def test_get_label_distribution(self):
        """Test getting label distribution from dataset."""
        mapper = LabelMapper(["O", "B-PER", "I-PER"])

        # Create test dataset with ID labels
        dataset = [
            {"ner_tags": [1, 2, 0]},  # B-PER, I-PER, O
            {"ner_tags": [0, 0, 1]},  # O, O, B-PER
            {"ner_tags": [1, 0, 2]},  # B-PER, O, I-PER
        ]

        distribution = mapper.get_label_distribution(dataset)

        assert distribution["O"] == 4
        assert distribution["B-PER"] == 3
        assert distribution["I-PER"] == 2

    def test_label2id_mapping(self):
        """Test label to ID mapping."""
        custom_labels = ["O", "B-PER", "I-PER", "B-ORG"]
        mapper = LabelMapper(custom_labels)

        assert mapper.label2id["O"] == 0
        assert mapper.label2id["B-PER"] == 1
        assert mapper.label2id["I-PER"] == 2
        assert mapper.label2id["B-ORG"] == 3

        assert mapper.id2label[0] == "O"
        assert mapper.id2label[1] == "B-PER"
        assert mapper.id2label[2] == "I-PER"
        assert mapper.id2label[3] == "B-ORG"

    def test_map_labels_preserves_other_fields(self):
        """Test that mapping preserves other fields in examples."""
        mapper = LabelMapper(["O", "B-PER", "I-PER"])

        examples = {"tokens": [["John", "Smith"]], "ner_tags": [["B-PER", "I-PER"]], "id": [1]}

        label_mapping = {"B-PER": "B-PER", "I-PER": "I-PER"}
        result = mapper.map_labels(examples, label_mapping)

        # Other fields should be preserved
        assert result["tokens"] == [["John", "Smith"]]
        assert result["id"] == [1]
        # Labels should be mapped to IDs
        assert result["ner_tags"] == [[1, 2]]

    def test_empty_examples(self):
        """Test handling empty examples."""
        mapper = LabelMapper(["O", "B-PER"])

        examples = {"ner_tags": []}

        label_mapping = {"O": "O"}
        result = mapper.map_labels(examples, label_mapping)

        assert result["ner_tags"] == []

    def test_map_labels_with_custom_field(self):
        """Test mapping with custom label field name."""
        mapper = LabelMapper(["O", "B-PER", "I-PER"])

        examples = {"custom_tags": [["B-PER", "I-PER", "O"]]}

        label_mapping = {"B-PER": "B-PER", "I-PER": "I-PER", "O": "O"}
        result = mapper.map_labels(examples, label_mapping, label_field="custom_tags")

        assert result["custom_tags"] == [[1, 2, 0]]

    def test_label_distribution_unknown_id(self):
        """Test label distribution with unknown ID."""
        mapper = LabelMapper(["O", "B-PER"])

        # Dataset with unknown ID
        dataset = [
            {"ner_tags": [0, 1, 999]}  # 999 is unknown
        ]

        distribution = mapper.get_label_distribution(dataset)

        # Unknown ID should be counted as O
        assert distribution["O"] == 2  # Original O + fallback for unknown
        assert distribution["B-PER"] == 1
