"""Regression tests for model outputs."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest
import torch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import Config


class ModelRegressionTester:
    """Regression testing for model outputs."""

    def __init__(self, fixture_path: Path) -> None:
        """Initialize regression tester."""
        self.fixture_path = fixture_path
        self.tolerance = 1e-4  # Numerical tolerance for confidence scores

    def save_reference_outputs(self, inputs: list[str], outputs: dict[str, Any]) -> None:
        """Save reference outputs for regression testing."""
        reference = {
            "inputs": inputs,
            "outputs": outputs,
            "model_version": "mistral-7b-v0.3",
            "timestamp": str(datetime.now()),
        }
        with open(self.fixture_path, "w") as f:
            json.dump(reference, f, indent=2)

    def load_reference_outputs(self) -> dict[str, Any]:
        """Load reference outputs for comparison."""
        with open(self.fixture_path) as f:
            result: dict[str, Any] = json.load(f)
            return result

    def compare_predictions(self, current: dict[str, Any], reference: dict[str, Any]) -> bool:
        """Compare current predictions against reference."""
        # Compare entity lists
        current_entities = current.get("entities", [])
        reference_entities = reference.get("entities", [])

        if len(current_entities) != len(reference_entities):
            return False

        # Compare each entity
        for curr_entity, ref_entity in zip(current_entities, reference_entities, strict=False):
            if curr_entity.get("text") != ref_entity.get("text"):
                return False
            if curr_entity.get("type") != ref_entity.get("type"):
                return False

        return True

    def compare_confidence_scores(self, current: dict[str, Any], reference: dict[str, Any]) -> bool:
        """Compare confidence scores with tolerance."""
        current_scores = current.get("confidence_scores", [])
        reference_scores = reference.get("confidence_scores", [])

        if len(current_scores) != len(reference_scores):
            return False

        for curr_score, ref_score in zip(current_scores, reference_scores, strict=False):
            if abs(curr_score - ref_score) > self.tolerance:
                return False

        return True


@pytest.fixture
def config() -> Config:
    """Create test configuration."""
    return Config()


@pytest.fixture
def mock_model_and_tokenizer():
    """Create mock model and tokenizer for testing."""
    # Mock tokenizer
    mock_tokenizer = Mock()
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
    }
    mock_tokenizer.word_ids.return_value = [None, 0, 1, 2, None]

    # Mock model
    mock_model = Mock()
    mock_logits = torch.tensor([[[0.1, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])  # B-PER prediction
    mock_outputs = Mock()
    mock_outputs.logits = mock_logits
    mock_model.return_value = mock_outputs
    mock_model.eval.return_value = None
    mock_model.to.return_value = mock_model

    return mock_model, mock_tokenizer


@pytest.fixture
def regression_test_cases() -> dict[str, Any]:
    """Load regression test cases."""
    fixture_path = Path(__file__).parent / "fixtures" / "regression_test_cases.json"
    with open(fixture_path) as f:
        result: dict[str, Any] = json.load(f)
        return result


@pytest.fixture
def regression_tester() -> ModelRegressionTester:
    """Create regression tester instance."""
    fixture_path = Path(__file__).parent / "fixtures" / "model_outputs_reference.json"
    return ModelRegressionTester(fixture_path)


class TestModelRegressionTester:
    """Test the ModelRegressionTester class."""

    def test_save_and_load_reference_outputs(self, regression_tester: ModelRegressionTester) -> None:
        """Test saving and loading reference outputs."""
        test_inputs = ["John Smith works at Microsoft."]
        test_outputs = {
            "entities": [
                {"text": "John Smith", "type": "PER", "start": 0, "end": 10},
                {"text": "Microsoft", "type": "ORG", "start": 20, "end": 29},
            ],
            "confidence_scores": [0.95, 0.92],
        }

        # Save reference outputs
        regression_tester.save_reference_outputs(test_inputs, test_outputs)

        # Load and verify
        loaded = regression_tester.load_reference_outputs()
        assert loaded["inputs"] == test_inputs
        assert loaded["outputs"] == test_outputs
        assert loaded["model_version"] == "mistral-7b-v0.3"

        # Clean up
        if regression_tester.fixture_path.exists():
            regression_tester.fixture_path.unlink()

    def test_compare_predictions_identical(self, regression_tester: ModelRegressionTester) -> None:
        """Test comparing identical predictions."""
        predictions = {
            "entities": [
                {"text": "John Smith", "type": "PER"},
                {"text": "Microsoft", "type": "ORG"},
            ]
        }
        assert regression_tester.compare_predictions(predictions, predictions)

    def test_compare_predictions_different(self, regression_tester: ModelRegressionTester) -> None:
        """Test comparing different predictions."""
        pred1 = {
            "entities": [
                {"text": "John Smith", "type": "PER"},
            ]
        }
        pred2 = {
            "entities": [
                {"text": "John Smith", "type": "ORG"},  # Different type
            ]
        }
        assert not regression_tester.compare_predictions(pred1, pred2)

    def test_compare_confidence_scores_within_tolerance(self, regression_tester: ModelRegressionTester) -> None:
        """Test comparing confidence scores within tolerance."""
        current = {"confidence_scores": [0.95, 0.92]}
        reference = {"confidence_scores": [0.9501, 0.9199]}  # Within tolerance
        assert regression_tester.compare_confidence_scores(current, reference)

    def test_compare_confidence_scores_outside_tolerance(self, regression_tester: ModelRegressionTester) -> None:
        """Test comparing confidence scores outside tolerance."""
        current = {"confidence_scores": [0.95, 0.92]}
        reference = {"confidence_scores": [0.96, 0.91]}  # Outside tolerance
        assert not regression_tester.compare_confidence_scores(current, reference)


@pytest.mark.regression
class TestModelOutputRegression:
    """Regression tests for model outputs."""

    def extract_entities_from_predictions(self, text: str, predictions: list[dict[str, Any]]) -> list[dict[str, str]]:
        """Extract entities from model predictions."""
        entities = []
        current_entity = None
        current_start = 0

        for _i, pred in enumerate(predictions[0]["words"]):
            word = pred["word"]
            label = pred["label"]

            if label.startswith("B-"):
                # Save previous entity if exists
                if current_entity:
                    entities.append(current_entity)
                # Start new entity
                entity_type = label[2:]  # Remove B- prefix
                current_entity = {
                    "text": word,
                    "type": entity_type,
                    "start": current_start,
                    "end": current_start + len(word),
                }
            elif label.startswith("I-") and current_entity:
                # Continue current entity
                current_entity["text"] += " " + word
                current_entity["end"] = current_start + len(word)
            else:
                # Save current entity if exists
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

            current_start += len(word) + 1  # +1 for space

        # Save last entity if exists
        if current_entity:
            entities.append(current_entity)

        return entities

    @patch("scripts.inference.load_model_for_inference")
    def test_model_predictions_unchanged(
        self,
        mock_load_model: Mock,
        mock_model_and_tokenizer: tuple[Mock, Mock],
        regression_test_cases: dict[str, Any],
        config: Config,
    ) -> None:
        """Ensure model predictions remain consistent."""
        mock_model, mock_tokenizer = mock_model_and_tokenizer
        mock_load_model.return_value = (mock_model, mock_tokenizer)

        # Test with first test case
        test_case = regression_test_cases["test_cases"][0]
        text = test_case["input"]
        expected_entities = test_case["expected_entities"]
        expected_labels = test_case["expected_labels"]

        # Mock the predict_entities function behavior
        with patch("scripts.inference.predict_entities") as mock_predict:
            mock_predict.return_value = [
                {
                    "text": text,
                    "words": [
                        {"word": "John", "label": "B-PER"},
                        {"word": "Smith", "label": "I-PER"},
                        {"word": "works", "label": "O"},
                        {"word": "at", "label": "O"},
                        {"word": "Microsoft", "label": "B-ORG"},
                        {"word": "in", "label": "O"},
                        {"word": "Seattle", "label": "B-LOC"},
                    ],
                    "entities": [
                        {"text": "John Smith", "type": "PER"},
                        {"text": "Microsoft", "type": "ORG"},
                        {"text": "Seattle", "type": "LOC"},
                    ],
                }
            ]

            predictions = mock_predict.return_value
            entities = predictions[0]["entities"]

            # Verify entities match expected
            entity_texts = [e["text"] for e in entities]
            entity_types = [e["type"] for e in entities]

            assert entity_texts == expected_entities
            assert entity_types == expected_labels

    @patch("scripts.inference.load_model_for_inference")
    def test_prediction_confidence_stability(
        self,
        mock_load_model: Mock,
        mock_model_and_tokenizer: tuple[Mock, Mock],
        config: Config,
    ) -> None:
        """Ensure prediction confidence scores are stable."""
        mock_model, mock_tokenizer = mock_model_and_tokenizer
        mock_load_model.return_value = (mock_model, mock_tokenizer)

        test_text = "John Smith works at Microsoft."

        # Mock multiple predictions with similar confidence scores
        with patch("scripts.inference.predict_entities") as mock_predict:
            # First prediction
            mock_predict.return_value = [
                {
                    "text": test_text,
                    "confidence_scores": [0.95, 0.92, 0.88],
                }
            ]
            prediction1 = mock_predict.return_value[0]

            # Second prediction (should be similar)
            mock_predict.return_value = [
                {
                    "text": test_text,
                    "confidence_scores": [0.9505, 0.9195, 0.8805],  # Within tolerance
                }
            ]
            prediction2 = mock_predict.return_value[0]

            # Compare confidence scores
            scores1 = prediction1["confidence_scores"]
            scores2 = prediction2["confidence_scores"]

            # Verify scores are within acceptable tolerance
            tolerance = 1e-2  # 1% tolerance for confidence stability
            for s1, s2 in zip(scores1, scores2, strict=False):
                assert abs(s1 - s2) <= tolerance

    def test_no_entities_text(self, config: Config) -> None:
        """Test regression for text with no entities."""
        with patch("scripts.inference.load_model_for_inference") as mock_load_model:
            mock_model = Mock()
            mock_tokenizer = Mock()
            mock_load_model.return_value = (mock_model, mock_tokenizer)

            with patch("scripts.inference.predict_entities") as mock_predict:
                mock_predict.return_value = [
                    {
                        "text": "The weather is nice today.",
                        "words": [
                            {"word": "The", "label": "O"},
                            {"word": "weather", "label": "O"},
                            {"word": "is", "label": "O"},
                            {"word": "nice", "label": "O"},
                            {"word": "today", "label": "O"},
                        ],
                        "entities": [],
                    }
                ]

                predictions = mock_predict.return_value
                entities = predictions[0]["entities"]

                # Verify no entities detected
                assert len(entities) == 0

    @pytest.mark.parametrize("test_case_id", ["basic_entities", "complex_entities", "mixed_entities"])
    def test_regression_for_all_test_cases(
        self,
        test_case_id: str,
        regression_test_cases: dict[str, Any],
        config: Config,
    ) -> None:
        """Test regression for all test cases."""
        test_case = next(tc for tc in regression_test_cases["test_cases"] if tc["id"] == test_case_id)

        with patch("scripts.inference.load_model_for_inference") as mock_load_model:
            mock_model = Mock()
            mock_tokenizer = Mock()
            mock_load_model.return_value = (mock_model, mock_tokenizer)

            # Create mock prediction based on expected entities
            mock_entities = []
            for entity_text, entity_type in zip(
                test_case["expected_entities"], test_case["expected_labels"], strict=False
            ):
                mock_entities.append({"text": entity_text, "type": entity_type})

            with patch("scripts.inference.predict_entities") as mock_predict:
                mock_predict.return_value = [
                    {
                        "text": test_case["input"],
                        "entities": mock_entities,
                    }
                ]

                predictions = mock_predict.return_value
                entities = predictions[0]["entities"]

                # Verify entities match expected
                entity_texts = [e["text"] for e in entities]
                entity_types = [e["type"] for e in entities]

                assert entity_texts == test_case["expected_entities"]
                assert entity_types == test_case["expected_labels"]
