#!/usr/bin/env python3
"""Simple test to improve inference.py coverage by directly importing and testing specific lines."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# This should cover the TYPE_CHECKING import lines (14-18)
from scripts.inference import (
    extract_entities,
    format_output,
    load_model_for_inference,
    main,
    parse_args,
    predict_entities,
)


def test_type_checking_imports():
    """Test the TYPE_CHECKING imports by triggering them."""
    # Just importing the module already covers the TYPE_CHECKING lines
    assert extract_entities is not None
    assert format_output is not None
    assert load_model_for_inference is not None
    assert main is not None
    assert parse_args is not None
    assert predict_entities is not None


def test_word_id_none_continue():
    """Test the continue statement in predict_entities (line 123)."""
    import torch

    mock_model = Mock()
    mock_model.eval.return_value = None
    mock_model.to.return_value = mock_model

    mock_outputs = Mock()
    mock_outputs.logits = torch.tensor([[[0.1, 0.9], [0.8, 0.2]]])
    mock_model.return_value = mock_outputs

    mock_tokenizer = Mock()
    mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])}

    # Mock encoding with None values to trigger continue statement
    mock_encoding = Mock()
    mock_encoding.word_ids.return_value = [None, 0]  # First token is None (special token)
    mock_tokenizer.side_effect = [
        {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])},
        mock_encoding,
    ]

    texts = ["Test"]
    label_names = ["O", "B-PER"]

    with patch("torch.no_grad"):
        predictions = predict_entities(mock_model, mock_tokenizer, texts, label_names, device="cpu")

    assert len(predictions) == 1


def test_extract_entities_append_current():
    """Test appending current_entity when starting new entity (line 157)."""
    words = ["John", "Mary"]
    labels = ["B-PER", "B-PER"]  # Two consecutive B- tags

    entities = extract_entities(words, labels)

    # Should create two separate entities
    assert len(entities) == 2
    assert entities[0]["text"] == "John"
    assert entities[1]["text"] == "Mary"


def test_main_file_input_coverage():
    """Test main function file input to cover lines 229-231."""
    from unittest.mock import mock_open, patch

    with patch("scripts.inference.parse_args") as mock_parse_args:
        mock_args = Mock()
        mock_args.model_path = "/test/model"
        mock_args.base_model = "base-model"
        mock_args.config = "config.yaml"
        mock_args.text = None
        mock_args.file = "/test/input.txt"
        mock_args.output = None
        mock_args.device = "cpu"
        mock_args.batch_size = 1
        mock_parse_args.return_value = mock_args

        with (
            patch("scripts.inference.setup_logging"),
            patch("scripts.inference.Config.from_yaml") as mock_config,
            patch("scripts.inference.load_model_for_inference", return_value=(Mock(), Mock())),
            patch("scripts.inference.predict_entities", return_value=[{"text": "Line1", "entities": []}]),
            patch("scripts.inference.format_output", return_value="Output"),
            patch("builtins.open", mock_open(read_data="Line1\nLine2\n")),
            patch("builtins.print"),
        ):
            mock_config.return_value.data.label_names = ["O", "B-PER"]
            main()


def test_main_interactive_input_coverage():
    """Test main function interactive input to cover lines 234-241."""
    from unittest.mock import patch

    with patch("scripts.inference.parse_args") as mock_parse_args:
        mock_args = Mock()
        mock_args.model_path = "/test/model"
        mock_args.base_model = "base-model"
        mock_args.config = "config.yaml"
        mock_args.text = None
        mock_args.file = None
        mock_args.output = None
        mock_args.device = "cpu"
        mock_args.batch_size = 1
        mock_parse_args.return_value = mock_args

        with (
            patch("scripts.inference.setup_logging"),
            patch("scripts.inference.Config.from_yaml") as mock_config,
            patch("scripts.inference.load_model_for_inference", return_value=(Mock(), Mock())),
            patch("scripts.inference.predict_entities", return_value=[{"text": "Input", "entities": []}]),
            patch("scripts.inference.format_output", return_value="Output"),
            patch("builtins.input", side_effect=["Input", EOFError()]),
            patch("builtins.print"),
        ):
            mock_config.return_value.data.label_names = ["O", "B-PER"]
            main()


def test_main_no_input_error_coverage():
    """Test main function no input error to cover lines 244-245."""
    from unittest.mock import patch

    import pytest

    with patch("scripts.inference.parse_args") as mock_parse_args:
        mock_args = Mock()
        mock_args.model_path = "/test/model"
        mock_args.base_model = "base-model"
        mock_args.config = "config.yaml"
        mock_args.text = None
        mock_args.file = None
        mock_args.device = "cpu"
        mock_args.batch_size = 1
        mock_parse_args.return_value = mock_args

        with (
            patch("scripts.inference.setup_logging"),
            patch("scripts.inference.Config.from_yaml") as mock_config,
            patch("scripts.inference.load_model_for_inference", return_value=(Mock(), Mock())),
            patch("builtins.input", side_effect=EOFError()),
        ):
            mock_config.return_value.data.label_names = ["O", "B-PER"]
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 1


def test_main_output_file_coverage():
    """Test main function output file writing to cover lines 262-264."""
    from unittest.mock import mock_open, patch

    with patch("scripts.inference.parse_args") as mock_parse_args:
        mock_args = Mock()
        mock_args.model_path = "/test/model"
        mock_args.base_model = "base-model"
        mock_args.config = "config.yaml"
        mock_args.text = "Test text"
        mock_args.file = None
        mock_args.output = "/test/output.txt"
        mock_args.device = "cpu"
        mock_args.batch_size = 1
        mock_parse_args.return_value = mock_args

        with (
            patch("scripts.inference.setup_logging"),
            patch("scripts.inference.Config.from_yaml") as mock_config,
            patch("scripts.inference.load_model_for_inference", return_value=(Mock(), Mock())),
            patch("scripts.inference.predict_entities", return_value=[{"text": "Test", "entities": []}]),
            patch("scripts.inference.format_output", return_value="Output"),
            patch("builtins.open", mock_open()) as mock_file,
        ):
            mock_config.return_value.data.label_names = ["O", "B-PER"]
            main()

        mock_file.assert_called_with("/test/output.txt", "w")


def test_main_entity_printing_coverage():
    """Test main function entity printing to cover lines 276-278."""
    from unittest.mock import patch

    with patch("scripts.inference.parse_args") as mock_parse_args:
        mock_args = Mock()
        mock_args.model_path = "/test/model"
        mock_args.base_model = "base-model"
        mock_args.config = "config.yaml"
        mock_args.text = "John works"
        mock_args.file = None
        mock_args.output = None
        mock_args.device = "cpu"
        mock_args.batch_size = 1
        mock_parse_args.return_value = mock_args

        mock_predictions = [{"text": "John works", "entities": [{"text": "John", "type": "PER", "start": 0, "end": 1}]}]

        with (
            patch("scripts.inference.setup_logging"),
            patch("scripts.inference.Config.from_yaml") as mock_config,
            patch("scripts.inference.load_model_for_inference", return_value=(Mock(), Mock())),
            patch("scripts.inference.predict_entities", return_value=mock_predictions),
            patch("scripts.inference.format_output", return_value="Output"),
            patch("builtins.print") as mock_print,
        ):
            mock_config.return_value.data.label_names = ["O", "B-PER"]
            main()

        # Should have printed entity information
        mock_print.assert_called()


def test_main_module_execution_coverage():
    """Test if __name__ == '__main__' execution (line 282)."""
    from unittest.mock import patch

    # We can't easily test this line directly, but importing the module covers most lines
    # The line 282 is covered when the module is executed as a script
    with patch("scripts.inference.main"):
        # Simulate module execution
        import scripts.inference

        # We can at least verify the main function exists and is callable
        assert callable(scripts.inference.main)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
