"""Tests for scripts/inference.py module."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.inference import (
    extract_entities,
    format_output,
    load_model_for_inference,
    main,
    parse_args,
    predict_entities,
)


class TestParseArgs:
    """Test argument parsing functionality."""

    def test_parse_args_required(self):
        """Test argument parsing with required arguments."""
        with patch("sys.argv", ["inference.py", "--model-path", "/test/model"]):
            args = parse_args()

        assert args.model_path == "/test/model"
        assert args.base_model == "mistralai/Mistral-7B-v0.3"
        assert args.text is None
        assert args.file is None
        assert args.output is None
        assert args.batch_size == 1
        assert args.config == "configs/default.yaml"

    def test_parse_args_all_options(self):
        """Test argument parsing with all options."""
        test_argv = [
            "inference.py",
            "--model-path",
            "/test/model",
            "--base-model",
            "custom-base",
            "--text",
            "Test text",
            "--file",
            "/test/input.txt",
            "--output",
            "/test/output.txt",
            "--batch-size",
            "4",
            "--device",
            "cpu",
            "--config",
            "custom.yaml",
        ]
        with patch("sys.argv", test_argv):
            args = parse_args()

        assert args.model_path == "/test/model"
        assert args.base_model == "custom-base"
        assert args.text == "Test text"
        assert args.file == "/test/input.txt"
        assert args.output == "/test/output.txt"
        assert args.batch_size == 4
        assert args.device == "cpu"
        assert args.config == "custom.yaml"

    @patch("torch.cuda.is_available")
    def test_parse_args_device_default_cuda(self, mock_cuda_available):
        """Test device defaults to cuda when available."""
        mock_cuda_available.return_value = True

        with patch("sys.argv", ["inference.py", "--model-path", "/test/model"]):
            args = parse_args()

        assert args.device == "cuda"

    @patch("torch.cuda.is_available")
    def test_parse_args_device_default_cpu(self, mock_cuda_available):
        """Test device defaults to cpu when cuda not available."""
        mock_cuda_available.return_value = False

        with patch("sys.argv", ["inference.py", "--model-path", "/test/model"]):
            args = parse_args()

        assert args.device == "cpu"


class TestLoadModelForInference:
    """Test model loading for inference."""

    @patch("scripts.inference.load_model_from_checkpoint")
    @patch("scripts.inference.setup_logging")
    def test_load_model_lora_checkpoint(self, mock_setup_logging, mock_load_checkpoint):
        """Test loading LoRA checkpoint."""
        mock_logger = Mock()
        mock_setup_logging.return_value = mock_logger

        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load_checkpoint.return_value = (mock_model, mock_tokenizer)

        mock_config = Mock()

        # Mock LoRA checkpoint (adapter_config.json exists)
        with patch("pathlib.Path.exists", return_value=True):
            model, tokenizer = load_model_for_inference("/test/lora", "base-model", mock_config)

        assert model == mock_model
        assert tokenizer == mock_tokenizer
        mock_load_checkpoint.assert_called_once_with(
            checkpoint_path="/test/lora", config=mock_config, base_model_name="base-model"
        )

    @patch("scripts.inference.setup_model")
    @patch("scripts.inference.AutoTokenizer.from_pretrained")
    @patch("scripts.inference.setup_logging")
    def test_load_model_full_checkpoint(self, mock_setup_logging, mock_tokenizer_load, mock_setup_model):
        """Test loading full model checkpoint."""
        mock_logger = Mock()
        mock_setup_logging.return_value = mock_logger

        mock_tokenizer = Mock()
        mock_tokenizer_load.return_value = mock_tokenizer

        mock_model = Mock()
        mock_setup_model.return_value = (mock_model, mock_tokenizer)

        mock_config = Mock()

        # Mock full model checkpoint (adapter_config.json doesn't exist)
        with patch("pathlib.Path.exists", return_value=False):
            model, tokenizer = load_model_for_inference("/test/full", "base-model", mock_config)

        assert model == mock_model
        assert tokenizer == mock_tokenizer
        mock_tokenizer_load.assert_called_once_with("/test/full")
        mock_setup_model.assert_called_once_with(model_name="/test/full", config=mock_config)


class TestPredictEntities:
    """Test entity prediction functionality."""

    def test_predict_entities_basic(self):
        """Test basic entity prediction."""
        # Mock model
        mock_model = Mock()
        mock_model.eval.return_value = None
        mock_model.to.return_value = mock_model

        # Mock outputs
        mock_outputs = Mock()
        # Predictions for: "John works" -> [1, 0] (B-PER, O)
        mock_outputs.logits = torch.tensor([[[0.1, 0.9, 0.05], [0.8, 0.1, 0.1]]])  # batch=1, seq=2, labels=3
        mock_model.return_value = mock_outputs

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])}

        # Mock encoding for word alignment
        mock_encoding = Mock()
        mock_encoding.word_ids.return_value = [0, 1]  # Two words
        mock_tokenizer.side_effect = [
            {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])},
            mock_encoding,
        ]

        texts = ["John works"]
        label_names = ["O", "B-PER", "I-PER"]

        with patch("torch.no_grad"):
            predictions = predict_entities(mock_model, mock_tokenizer, texts, label_names, device="cpu")

        assert len(predictions) == 1
        pred = predictions[0]
        assert pred["text"] == "John works"
        assert "words" in pred
        assert "labels" in pred
        assert "entities" in pred

    def test_predict_entities_batch_processing(self):
        """Test batch processing of multiple texts."""
        mock_model = Mock()
        mock_model.eval.return_value = None
        mock_model.to.return_value = mock_model

        # Mock outputs for batch of 2
        mock_outputs = Mock()
        mock_outputs.logits = torch.tensor(
            [
                [[0.1, 0.9], [0.8, 0.2]],  # First text
                [[0.9, 0.1], [0.1, 0.9]],  # Second text
            ]
        )
        mock_model.return_value = mock_outputs

        mock_tokenizer = Mock()

        def mock_tokenizer_call(*args, **kwargs):
            if isinstance(args[0], list):  # Batch call
                return {"input_ids": torch.tensor([[1, 2], [3, 4]]), "attention_mask": torch.tensor([[1, 1], [1, 1]])}
            else:  # Single call for word alignment
                mock_encoding = Mock()
                mock_encoding.word_ids.return_value = [0, 1]
                return mock_encoding

        mock_tokenizer.side_effect = mock_tokenizer_call

        texts = ["Text one", "Text two"]
        label_names = ["O", "B-PER"]

        with patch("torch.no_grad"):
            predictions = predict_entities(mock_model, mock_tokenizer, texts, label_names, device="cpu", batch_size=2)

        assert len(predictions) == 2

    def test_predict_entities_label_alignment_fallback(self):
        """Test label alignment fallback when lengths don't match."""
        mock_model = Mock()
        mock_model.eval.return_value = None
        mock_model.to.return_value = mock_model

        mock_outputs = Mock()
        mock_outputs.logits = torch.tensor([[[0.9, 0.1], [0.1, 0.9], [0.5, 0.5]]])  # 3 predictions
        mock_model.return_value = mock_outputs

        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        # Mock encoding that produces only 1 word prediction but text has 2 words
        mock_encoding = Mock()
        mock_encoding.word_ids.return_value = [0, 0, 0]  # All tokens map to first word
        mock_tokenizer.side_effect = [
            {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])},
            mock_encoding,
        ]

        texts = ["Two words"]  # 2 words but only 1 prediction
        label_names = ["O", "B-PER"]

        with patch("torch.no_grad"):
            predictions = predict_entities(mock_model, mock_tokenizer, texts, label_names, device="cpu")

        pred = predictions[0]
        # Should pad with 'O' to match word count
        assert len(pred["labels"]) == len(pred["words"])


class TestExtractEntities:
    """Test entity extraction from BIO labels."""

    def test_extract_entities_basic(self):
        """Test basic entity extraction."""
        words = ["John", "Smith", "works", "at", "Microsoft"]
        labels = ["B-PER", "I-PER", "O", "O", "B-ORG"]

        entities = extract_entities(words, labels)

        assert len(entities) == 2

        person_entity = entities[0]
        assert person_entity["text"] == "John Smith"
        assert person_entity["type"] == "PER"
        assert person_entity["start"] == 0
        assert person_entity["end"] == 2

        org_entity = entities[1]
        assert org_entity["text"] == "Microsoft"
        assert org_entity["type"] == "ORG"
        assert org_entity["start"] == 4
        assert org_entity["end"] == 5

    def test_extract_entities_no_entities(self):
        """Test extraction when no entities present."""
        words = ["This", "is", "a", "test"]
        labels = ["O", "O", "O", "O"]

        entities = extract_entities(words, labels)

        assert len(entities) == 0

    def test_extract_entities_incomplete_entity(self):
        """Test extraction with incomplete entity at end."""
        words = ["Apple", "Inc"]
        labels = ["B-ORG", "I-ORG"]

        entities = extract_entities(words, labels)

        assert len(entities) == 1
        assert entities[0]["text"] == "Apple Inc"
        assert entities[0]["type"] == "ORG"

    def test_extract_entities_broken_sequence(self):
        """Test extraction with broken I- without B-."""
        words = ["Apple", "Inc", "is", "great"]
        labels = ["I-ORG", "I-ORG", "O", "O"]  # I- without B-

        entities = extract_entities(words, labels)

        # Should not create entity for broken sequence
        assert len(entities) == 0

    def test_extract_entities_multiple_same_type(self):
        """Test extraction with multiple entities of same type."""
        words = ["John", "and", "Mary", "work", "at", "Google"]
        labels = ["B-PER", "O", "B-PER", "O", "O", "B-ORG"]

        entities = extract_entities(words, labels)

        assert len(entities) == 3
        assert entities[0]["text"] == "John"
        assert entities[1]["text"] == "Mary"
        assert entities[2]["text"] == "Google"


class TestFormatOutput:
    """Test output formatting functionality."""

    def test_format_output_inline(self):
        """Test inline format output."""
        predictions = [
            {
                "text": "John works at Microsoft",
                "words": ["John", "works", "at", "Microsoft"],
                "labels": ["B-PER", "O", "O", "B-ORG"],
                "entities": [],
            }
        ]

        output = format_output(predictions, format_type="inline")

        assert "John/B-PER works at Microsoft/B-ORG" in output

    def test_format_output_conll(self):
        """Test CoNLL format output."""
        predictions = [{"text": "John works", "words": ["John", "works"], "labels": ["B-PER", "O"], "entities": []}]

        output = format_output(predictions, format_type="conll")
        lines = output.strip().split("\n")

        assert "John\tB-PER" in lines
        assert "works\tO" in lines

    def test_format_output_json(self):
        """Test JSON format output."""
        predictions = [
            {
                "text": "John works",
                "words": ["John", "works"],
                "labels": ["B-PER", "O"],
                "entities": [{"text": "John", "type": "PER", "start": 0, "end": 1}],
            }
        ]

        output = format_output(predictions, format_type="json")

        # Should contain JSON structure
        assert '"text": "John works"' in output
        assert '"entities"' in output

    def test_format_output_multiple_predictions(self):
        """Test formatting multiple predictions."""
        predictions = [
            {"text": "First text", "words": ["First", "text"], "labels": ["O", "O"], "entities": []},
            {"text": "Second text", "words": ["Second", "text"], "labels": ["O", "O"], "entities": []},
        ]

        output = format_output(predictions, format_type="inline")
        lines = output.strip().split("\n")

        assert len(lines) == 2
        assert "First text" in lines[0]
        assert "Second text" in lines[1]


class TestMain:
    """Test main inference function."""

    @patch("scripts.inference.format_output")
    @patch("scripts.inference.predict_entities")
    @patch("scripts.inference.load_model_for_inference")
    @patch("scripts.inference.Config.from_yaml")
    @patch("scripts.inference.setup_logging")
    @patch("scripts.inference.parse_args")
    def test_main_text_input(
        self, mock_parse_args, mock_setup_logging, mock_config_from_yaml, mock_load_model, mock_predict, mock_format
    ):
        """Test main function with text input."""
        # Setup mocks
        mock_args = Mock()
        mock_args.model_path = "/test/model"
        mock_args.base_model = "base-model"
        mock_args.config = "config.yaml"
        mock_args.text = "Test text"
        mock_args.file = None
        mock_args.output = None
        mock_args.device = "cpu"
        mock_args.batch_size = 1
        mock_parse_args.return_value = mock_args

        mock_logger = Mock()
        mock_setup_logging.return_value = mock_logger

        mock_config = Mock()
        mock_config.data.label_names = ["O", "B-PER"]
        mock_config_from_yaml.return_value = mock_config

        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)

        mock_predictions = [{"text": "Test text", "entities": []}]
        mock_predict.return_value = mock_predictions

        mock_format.return_value = "Formatted output"

        with patch("builtins.print"):
            main()

        mock_load_model.assert_called_once_with(model_path="/test/model", base_model="base-model", config=mock_config)
        mock_predict.assert_called_once_with(
            model=mock_model,
            tokenizer=mock_tokenizer,
            texts=["Test text"],
            label_names=["O", "B-PER"],
            device="cpu",
            batch_size=1,
        )

    @patch("scripts.inference.format_output")
    @patch("scripts.inference.predict_entities")
    @patch("scripts.inference.load_model_for_inference")
    @patch("scripts.inference.Config.from_yaml")
    @patch("scripts.inference.setup_logging")
    @patch("scripts.inference.parse_args")
    def test_main_file_input(
        self, mock_parse_args, mock_setup_logging, mock_config_from_yaml, mock_load_model, mock_predict, mock_format
    ):
        """Test main function with file input."""
        mock_args = Mock()
        mock_args.model_path = "/test/model"
        mock_args.base_model = "base-model"
        mock_args.config = "config.yaml"
        mock_args.text = None
        mock_args.file = "/test/input.txt"
        mock_args.output = "/test/output.txt"
        mock_args.device = "cpu"
        mock_args.batch_size = 1
        mock_parse_args.return_value = mock_args

        mock_logger = Mock()
        mock_setup_logging.return_value = mock_logger

        mock_config = Mock()
        mock_config.data.label_names = ["O", "B-PER"]
        mock_config_from_yaml.return_value = mock_config

        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)

        mock_predictions = [{"text": "Line 1", "entities": []}, {"text": "Line 2", "entities": []}]
        mock_predict.return_value = mock_predictions

        mock_format.return_value = "Formatted output"

        # Mock file reading
        mock_file_content = "Line 1\nLine 2\n"
        with patch("builtins.open", mock_file_content) as mock_open:
            mock_open.return_value.__enter__.return_value = mock_file_content.split("\n")
            main()

        # Verify output file was written
        mock_open.assert_called()

    @patch("scripts.inference.setup_logging")
    @patch("scripts.inference.parse_args")
    def test_main_interactive_mode(self, mock_parse_args, mock_setup_logging):
        """Test main function in interactive mode."""
        mock_args = Mock()
        mock_args.text = None
        mock_args.file = None
        mock_parse_args.return_value = mock_args

        mock_logger = Mock()
        mock_setup_logging.return_value = mock_logger

        # Mock config and model loading
        with (
            patch("scripts.inference.Config.from_yaml"),
            patch("scripts.inference.load_model_for_inference"),
            patch("builtins.input", side_effect=["Test input", EOFError()]),
            patch("scripts.inference.predict_entities"),
            patch("scripts.inference.format_output"),
            patch("builtins.print"),
        ):
            main()

        # Should handle EOFError gracefully

    @patch("scripts.inference.setup_logging")
    @patch("scripts.inference.parse_args")
    def test_main_no_input(self, mock_parse_args, mock_setup_logging):
        """Test main function with no input provided."""
        mock_args = Mock()
        mock_args.text = None
        mock_args.file = None
        mock_parse_args.return_value = mock_args

        mock_logger = Mock()
        mock_setup_logging.return_value = mock_logger

        # Mock config and model loading
        with (
            patch("scripts.inference.Config.from_yaml"),
            patch("scripts.inference.load_model_for_inference"),
            patch("builtins.input", side_effect=EOFError()),
            pytest.raises(SystemExit) as exc_info,
        ):  # No input provided
            main()

        assert exc_info.value.code == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
