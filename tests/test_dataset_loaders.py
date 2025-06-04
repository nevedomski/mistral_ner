"""Comprehensive tests for all dataset loaders."""

from unittest.mock import MagicMock, patch

import pytest

from datasets import Dataset, DatasetDict
from src.datasets.loaders.ai4privacy import AI4PrivacyDataset
from src.datasets.loaders.bigcode_pii import BigCodePIIDataset
from src.datasets.loaders.mendeley_pii import MendeleyPIIDataset
from src.datasets.loaders.wikiner import WikiNERDataset


class TestWikiNERDataset:
    """Test WikiNER dataset loader."""

    def test_init_default(self):
        """Test initialization with default config."""
        loader = WikiNERDataset()
        assert loader.language == "en"

    def test_init_custom_language(self):
        """Test initialization with custom language."""
        loader = WikiNERDataset({"language": "fr"})
        assert loader.language == "fr"

    @patch("src.datasets.loaders.wikiner.load_dataset")
    def test_load_success(self, mock_load_dataset):
        """Test successful dataset loading."""
        # Mock the dataset
        mock_dataset = MagicMock(spec=DatasetDict)
        mock_load_dataset.return_value = mock_dataset

        loader = WikiNERDataset({"language": "en"})
        result = loader.load()

        mock_load_dataset.assert_called_once_with("wikiann", "en")
        assert result == mock_dataset

    @patch("src.datasets.loaders.wikiner.load_dataset")
    def test_load_failure(self, mock_load_dataset):
        """Test dataset loading failure."""
        mock_load_dataset.side_effect = Exception("Loading failed")

        loader = WikiNERDataset()
        with pytest.raises(Exception, match="Loading failed"):
            loader.load()

    def test_get_label_mapping(self):
        """Test label mapping."""
        loader = WikiNERDataset()
        mapping = loader.get_label_mapping()

        assert mapping["O"] == "O"
        assert mapping["B-PER"] == "B-PER"
        assert mapping["I-PER"] == "I-PER"
        assert mapping["B-ORG"] == "B-ORG"
        assert mapping["I-ORG"] == "I-ORG"
        assert mapping["B-LOC"] == "B-LOC"
        assert mapping["I-LOC"] == "I-LOC"
        assert len(mapping) == 7

    def test_preprocess(self):
        """Test preprocessing (pass-through for WikiNER)."""
        loader = WikiNERDataset()
        examples = {"tokens": [["Hello", "world"]], "ner_tags": [[0, 0]]}
        result = loader.preprocess(examples)
        assert result == examples


class TestAI4PrivacyDataset:
    """Test AI4Privacy dataset loader."""

    def test_init(self):
        """Test initialization."""
        loader = AI4PrivacyDataset()
        assert loader.config == {}

        loader = AI4PrivacyDataset({"test": "value"})
        assert loader.config == {"test": "value"}

    @patch("src.datasets.loaders.ai4privacy.load_dataset")
    def test_load_with_split_creation(self, mock_load_dataset):
        """Test loading with automatic split creation."""
        # Mock dataset with only train split
        mock_train = MagicMock(spec=Dataset)
        mock_train.train_test_split.return_value = {"train": MagicMock(spec=Dataset), "test": MagicMock(spec=Dataset)}
        mock_train.train_test_split.return_value["test"].train_test_split.return_value = {
            "train": MagicMock(spec=Dataset),
            "test": MagicMock(spec=Dataset),
        }

        mock_dataset = {"train": mock_train}
        mock_load_dataset.return_value = mock_dataset

        loader = AI4PrivacyDataset()

        # Mock the preprocess method
        with patch.object(loader, "preprocess", return_value={"tokens": [], "ner_tags": []}):
            result = loader.load()

        mock_load_dataset.assert_called_once_with("ai4privacy/pii-masking-65k")
        assert "train" in result
        assert "validation" in result
        assert "test" in result

    @patch("src.datasets.loaders.ai4privacy.load_dataset")
    def test_load_with_existing_splits(self, mock_load_dataset):
        """Test loading when splits already exist."""
        # Mock dataset with all splits
        mock_dataset = MagicMock(spec=DatasetDict)
        mock_dataset.__contains__ = lambda self, key: key in ["train", "validation", "test"]
        mock_dataset.map.return_value = mock_dataset
        mock_load_dataset.return_value = mock_dataset

        loader = AI4PrivacyDataset()
        result = loader.load()

        assert result == mock_dataset

    @patch("src.datasets.loaders.ai4privacy.load_dataset")
    def test_load_failure(self, mock_load_dataset):
        """Test dataset loading failure."""
        mock_load_dataset.side_effect = Exception("API error")

        loader = AI4PrivacyDataset()
        with pytest.raises(Exception, match="API error"):
            loader.load()

    def test_get_label_mapping(self):
        """Test comprehensive label mapping."""
        loader = AI4PrivacyDataset()
        mapping = loader.get_label_mapping()

        # Check some key mappings
        assert mapping["O"] == "O"
        assert mapping["B-FIRSTNAME"] == "B-PER"
        assert mapping["I-FIRSTNAME"] == "I-PER"
        assert mapping["B-COMPANY_NAME"] == "B-ORG"
        assert mapping["B-CITY"] == "B-LOC"
        assert mapping["B-CREDITCARDNUMBER"] == "B-CARD"
        assert mapping["B-EMAIL"] == "B-EMAIL"
        assert mapping["B-SSN"] == "B-SSN"
        assert len(mapping) > 50  # Should have many mappings

    def test_preprocess_with_labels(self):
        """Test preprocessing with entity labels."""
        loader = AI4PrivacyDataset()
        examples = {
            "tokenised_unmasked_text": [["John", "Smith", "works"]],
            "token_entity_labels": [["B-FIRSTNAME", "B-LASTNAME", "O"]],
        }

        result = loader.preprocess(examples)

        assert result["tokens"] == [["John", "Smith", "works"]]
        assert result["ner_tags"] == [["B-FIRSTNAME", "B-LASTNAME", "O"]]

    def test_preprocess_without_labels(self):
        """Test preprocessing without entity labels."""
        loader = AI4PrivacyDataset()
        examples = {"tokenised_unmasked_text": [["Hello", "world"]], "token_entity_labels": [None]}

        result = loader.preprocess(examples)

        assert result["tokens"] == [["Hello", "world"]]
        assert result["ner_tags"] == [["O", "O"]]

    def test_preprocess_string_labels(self):
        """Test preprocessing with string representation of labels."""
        loader = AI4PrivacyDataset()
        examples = {"tokenised_unmasked_text": [["Test", "text"]], "token_entity_labels": ["['B-PER', 'O']"]}

        result = loader.preprocess(examples)

        assert result["tokens"] == [["Test", "text"]]
        assert result["ner_tags"] == [["B-PER", "O"]]

    def test_parse_entity_labels_invalid(self):
        """Test parsing invalid entity labels."""
        loader = AI4PrivacyDataset()
        result = loader._parse_entity_labels("invalid", 3)
        assert result == ["O", "O", "O"]


class TestMendeleyPIIDataset:
    """Test Mendeley PII dataset loader."""

    def test_init_default(self):
        """Test initialization with default config."""
        loader = MendeleyPIIDataset()
        assert loader.filter_english is True

    def test_init_no_filter(self):
        """Test initialization without English filter."""
        loader = MendeleyPIIDataset({"filter_english": False})
        assert loader.filter_english is False

    @patch("src.datasets.loaders.mendeley_pii.load_dataset")
    def test_load_with_english_filter(self, mock_load_dataset):
        """Test loading with English language filtering."""
        # Mock dataset
        mock_train = MagicMock(spec=Dataset)
        mock_train.features = {"language": "feature"}

        # Mock filter and split operations
        filtered_dataset = MagicMock()
        filtered_dataset.__getitem__ = lambda self, key: mock_train if key == "train" else None
        filtered_dataset.__contains__ = lambda self, key: key == "train"
        filtered_dataset.filter.return_value = filtered_dataset

        mock_load_dataset.return_value = filtered_dataset

        # Mock train_test_split
        mock_train.train_test_split.return_value = {"train": MagicMock(spec=Dataset), "test": MagicMock(spec=Dataset)}
        mock_train.train_test_split.return_value["test"].train_test_split.return_value = {
            "train": MagicMock(spec=Dataset),
            "test": MagicMock(spec=Dataset),
        }

        loader = MendeleyPIIDataset({"filter_english": True})

        # Mock the preprocess method
        with patch.object(loader, "preprocess", return_value={"tokens": [], "ner_tags": []}):
            loader.load()

        mock_load_dataset.assert_called_once_with("Isotonic/pii-masking-200k")
        filtered_dataset.filter.assert_called()

    @patch("src.datasets.loaders.mendeley_pii.load_dataset")
    def test_load_without_filter(self, mock_load_dataset):
        """Test loading without language filtering."""
        mock_train = MagicMock(spec=Dataset)
        mock_dataset = {"train": mock_train}
        mock_dataset["train"] = mock_train

        mock_load_dataset.return_value = mock_dataset

        # Mock train_test_split
        mock_train.train_test_split.return_value = {"train": MagicMock(spec=Dataset), "test": MagicMock(spec=Dataset)}
        mock_train.train_test_split.return_value["test"].train_test_split.return_value = {
            "train": MagicMock(spec=Dataset),
            "test": MagicMock(spec=Dataset),
        }

        loader = MendeleyPIIDataset({"filter_english": False})

        # Mock the preprocess method
        with patch.object(loader, "preprocess", return_value={"tokens": [], "ner_tags": []}):
            loader.load()

        mock_load_dataset.assert_called_once_with("Isotonic/pii-masking-200k")

    def test_get_label_mapping(self):
        """Test label mapping."""
        loader = MendeleyPIIDataset()
        mapping = loader.get_label_mapping()

        # Check key mappings
        assert mapping["O"] == "O"
        assert mapping["B-FIRSTNAME"] == "B-PER"
        assert mapping["B-CITY"] == "B-LOC"
        assert mapping["B-CREDITCARDNUMBER"] == "B-CARD"
        assert mapping["B-SSN"] == "B-SSN"
        assert mapping["B-EMAIL"] == "B-EMAIL"
        assert len(mapping) > 40

    def test_preprocess_with_bio_labels(self):
        """Test preprocessing with BIO labels."""
        loader = MendeleyPIIDataset()
        examples = {"tokenised_text": [["John", "lives", "in", "NYC"]], "bio_labels": [["B-PER", "O", "O", "B-LOC"]]}

        result = loader.preprocess(examples)

        assert result["tokens"] == [["John", "lives", "in", "NYC"]]
        assert result["ner_tags"] == [["B-PER", "O", "O", "B-LOC"]]

    def test_preprocess_without_bio_labels(self):
        """Test preprocessing without BIO labels."""
        loader = MendeleyPIIDataset()
        examples = {"tokenised_text": [["Hello", "world"]], "bio_labels": [None]}

        result = loader.preprocess(examples)

        assert result["tokens"] == [["Hello", "world"]]
        assert result["ner_tags"] == [["O", "O"]]

    def test_preprocess_label_mismatch(self):
        """Test preprocessing with label length mismatch."""
        loader = MendeleyPIIDataset()
        examples = {
            "tokenised_text": [["One", "Two", "Three"]],
            "bio_labels": [["B-PER", "O"]],  # Wrong length
        }

        with patch("src.datasets.loaders.mendeley_pii.logger") as mock_logger:
            result = loader.preprocess(examples)
            mock_logger.warning.assert_called()

        assert result["tokens"] == [["One", "Two", "Three"]]
        assert result["ner_tags"] == [["O", "O", "O"]]

    def test_parse_bio_labels_list(self):
        """Test parsing BIO labels from list string."""
        loader = MendeleyPIIDataset()
        result = loader._parse_bio_labels("['B-PER', 'O', 'B-LOC']", 3)
        assert result == ["B-PER", "O", "B-LOC"]

    def test_parse_bio_labels_space_separated(self):
        """Test parsing space-separated BIO labels."""
        loader = MendeleyPIIDataset()
        result = loader._parse_bio_labels("B-PER O B-LOC", 3)
        assert result == ["B-PER", "O", "B-LOC"]

    def test_parse_bio_labels_invalid(self):
        """Test parsing invalid BIO labels."""
        loader = MendeleyPIIDataset()
        result = loader._parse_bio_labels("invalid format", 3)
        assert result == ["O", "O", "O"]

    @patch("src.datasets.loaders.mendeley_pii.load_dataset")
    def test_load_exception_handling(self, mock_load_dataset):
        """Test exception handling during dataset loading."""
        mock_load_dataset.side_effect = Exception("Network error")

        loader = MendeleyPIIDataset()

        with patch("src.datasets.loaders.mendeley_pii.logger") as mock_logger:
            with pytest.raises(Exception, match="Network error"):
                loader.load()
            mock_logger.error.assert_called_once()
            assert "Failed to load Mendeley PII dataset" in mock_logger.error.call_args[0][0]

    def test_preprocess_string_bio_labels(self):
        """Test preprocessing with string representation of BIO labels."""
        loader = MendeleyPIIDataset()
        examples = {"tokenised_text": [["Test", "data"]], "bio_labels": ["['B-PER', 'O']"]}

        result = loader.preprocess(examples)

        assert result["tokens"] == [["Test", "data"]]
        assert result["ner_tags"] == [["B-PER", "O"]]


class TestBigCodePIIDataset:
    """Test BigCode PII dataset loader."""

    def test_init_default(self):
        """Test initialization with default config."""
        loader = BigCodePIIDataset()
        assert loader.dataset_name == "bigcode/the-stack-pii"
        assert loader.use_auth_token is None

    def test_init_custom(self):
        """Test initialization with custom config."""
        loader = BigCodePIIDataset({"dataset_name": "bigcode/custom-dataset", "use_auth_token": "token123"})
        assert loader.dataset_name == "bigcode/custom-dataset"
        assert loader.use_auth_token == "token123"

    @patch("src.datasets.loaders.bigcode_pii.load_dataset")
    def test_load_success_dict(self, mock_load_dataset):
        """Test successful loading with DatasetDict."""
        mock_dataset = MagicMock(spec=DatasetDict)
        mock_dataset.__contains__ = lambda self, key: key == "train"
        mock_dataset.map.return_value = mock_dataset

        mock_load_dataset.return_value = mock_dataset

        loader = BigCodePIIDataset()

        # Mock _create_splits
        with patch.object(loader, "_create_splits", return_value=mock_dataset):
            result = loader.load()

        mock_load_dataset.assert_called_once_with("bigcode/the-stack-pii", use_auth_token=None)
        assert result == mock_dataset

    @patch("src.datasets.loaders.bigcode_pii.load_dataset")
    def test_load_success_single_dataset(self, mock_load_dataset):
        """Test successful loading with single Dataset."""
        mock_dataset = MagicMock(spec=Dataset)
        mock_load_dataset.return_value = mock_dataset

        loader = BigCodePIIDataset()

        # Mock _create_splits and preprocess
        mock_splits = MagicMock(spec=DatasetDict)
        mock_splits.map.return_value = mock_splits

        with patch.object(loader, "_create_splits", return_value=mock_splits):
            result = loader.load()

        assert result == mock_splits

    @patch("src.datasets.loaders.bigcode_pii.load_dataset")
    def test_load_auth_failure(self, mock_load_dataset):
        """Test loading failure due to authentication."""
        mock_load_dataset.side_effect = Exception("401 Unauthorized")

        loader = BigCodePIIDataset()

        with patch("src.datasets.loaders.bigcode_pii.logger") as mock_logger:
            with pytest.raises(Exception, match="401 Unauthorized"):
                loader.load()

            # Check that auth error was logged
            mock_logger.error.assert_called()
            assert "Authentication failed" in mock_logger.error.call_args[0][0]

    @patch("src.datasets.loaders.bigcode_pii.load_dataset")
    def test_load_general_failure(self, mock_load_dataset):
        """Test loading failure with general error."""
        mock_load_dataset.side_effect = Exception("Network error")

        loader = BigCodePIIDataset()

        with pytest.raises(Exception, match="Network error"):
            loader.load()

    def test_create_splits(self):
        """Test split creation."""
        loader = BigCodePIIDataset()

        # Mock dataset
        mock_train = MagicMock(spec=Dataset)
        mock_train.train_test_split.return_value = {"train": MagicMock(spec=Dataset), "test": MagicMock(spec=Dataset)}
        mock_train.train_test_split.return_value["test"].train_test_split.return_value = {
            "train": MagicMock(spec=Dataset),
            "test": MagicMock(spec=Dataset),
        }

        dataset = {"train": mock_train}
        result = loader._create_splits(dataset)

        assert "train" in result
        assert "validation" in result
        assert "test" in result

    def test_get_label_mapping(self):
        """Test label mapping."""
        loader = BigCodePIIDataset()
        mapping = loader.get_label_mapping()

        # Check key mappings
        assert mapping["O"] == "O"
        assert mapping["B-NAME"] == "B-PER"
        assert mapping["B-EMAIL"] == "B-EMAIL"
        assert mapping["B-KEY"] == "B-MISC"
        assert mapping["B-API_KEY"] == "B-MISC"
        assert mapping["B-URL"] == "B-MISC"
        assert len(mapping) > 20

    def test_preprocess_with_tokens(self):
        """Test preprocessing with existing tokens."""
        loader = BigCodePIIDataset()
        examples = {"tokens": [["def", "main", "():"]], "labels": [["O", "O", "O"]]}

        result = loader.preprocess(examples)

        assert result["tokens"] == [["def", "main", "():"]]
        assert result["ner_tags"] == [["O", "O", "O"]]

    def test_preprocess_with_text(self):
        """Test preprocessing with text field."""
        loader = BigCodePIIDataset()
        examples = {"text": ["def main():"]}

        result = loader.preprocess(examples)

        assert len(result["tokens"]) == 1
        assert len(result["tokens"][0]) == 2  # Split by whitespace
        assert result["ner_tags"] == [["O", "O"]]

    def test_preprocess_with_code(self):
        """Test preprocessing with code field."""
        loader = BigCodePIIDataset()
        examples = {"code": ["print('hello world')"]}

        result = loader.preprocess(examples)

        assert len(result["tokens"]) == 1
        assert len(result["ner_tags"][0]) == len(result["tokens"][0])
        assert all(tag == "O" for tag in result["ner_tags"][0])

    def test_preprocess_no_valid_field(self):
        """Test preprocessing with no valid field."""
        loader = BigCodePIIDataset()
        examples = {"invalid": ["data"]}

        with pytest.raises(ValueError, match="Could not find tokens or text field"):
            loader.preprocess(examples)

    def test_preprocess_with_alternative_token_fields(self):
        """Test preprocessing with alternative token field names."""
        loader = BigCodePIIDataset()
        examples = {"input_tokens": [["test", "tokens"]], "entity_labels": [["O", "B-KEY"]]}

        result = loader.preprocess(examples)

        assert result["tokens"] == [["test", "tokens"]]
        assert result["ner_tags"] == [["O", "B-KEY"]]

    def test_preprocess_tokens_without_labels(self):
        """Test preprocessing tokens without labels."""
        loader = BigCodePIIDataset()
        examples = {"text_tokens": [["hello", "world"]]}

        result = loader.preprocess(examples)

        assert result["tokens"] == [["hello", "world"]]
        assert result["ner_tags"] == [["O", "O"]]
