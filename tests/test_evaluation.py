"""Tests for evaluation module."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from transformers import EvalPrediction

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation import (
    align_predictions,
    compute_confusion_matrix,
    compute_metrics_factory,
    evaluate_model,
    load_seqeval_metric,
    log_confusion_matrix_to_wandb,
    print_evaluation_report,
    save_predictions,
)


@pytest.fixture
def sample_labels():
    """Sample label names for testing."""
    return ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


@pytest.fixture
def sample_predictions():
    """Sample predictions array."""
    # Shape: (batch_size=2, seq_len=4, num_labels=7)
    return np.array(
        [
            [
                [0.9, 0.05, 0.02, 0.01, 0.01, 0.005, 0.005],  # O
                [0.1, 0.8, 0.05, 0.02, 0.02, 0.005, 0.005],  # B-PER
                [0.1, 0.05, 0.8, 0.02, 0.02, 0.005, 0.005],  # I-PER
                [0.9, 0.05, 0.02, 0.01, 0.01, 0.005, 0.005],
            ],  # O
            [
                [0.1, 0.05, 0.02, 0.8, 0.02, 0.005, 0.005],  # B-ORG
                [0.1, 0.05, 0.02, 0.02, 0.8, 0.005, 0.005],  # I-ORG
                [0.9, 0.05, 0.02, 0.01, 0.01, 0.005, 0.005],  # O
                [0.9, 0.05, 0.02, 0.01, 0.01, 0.005, 0.005],
            ],  # O (padding)
        ]
    )


@pytest.fixture
def sample_label_ids():
    """Sample label IDs array."""
    return np.array(
        [
            [0, 1, 2, 0],  # O, B-PER, I-PER, O
            [3, 4, 0, -100],  # B-ORG, I-ORG, O, padding
        ]
    )


class TestLoadSeqevalMetric:
    """Test seqeval metric loading."""

    @patch("src.evaluation.evaluate.load")
    def test_load_seqeval_metric_success(self, mock_load):
        """Test successful loading of seqeval metric."""
        mock_metric = Mock()
        mock_load.return_value = mock_metric

        result = load_seqeval_metric()

        mock_load.assert_called_once_with("seqeval")
        assert result == mock_metric

    @patch("src.evaluation.evaluate.load")
    def test_load_seqeval_metric_failure(self, mock_load):
        """Test fallback when seqeval loading fails."""
        mock_load.side_effect = Exception("Loading failed")

        result = load_seqeval_metric()

        assert result is None


class TestAlignPredictions:
    """Test prediction alignment functionality."""

    def test_align_predictions_basic(self, sample_predictions, sample_label_ids, sample_labels):
        """Test basic prediction alignment."""
        true_labels, pred_labels = align_predictions(sample_predictions, sample_label_ids, sample_labels)

        assert len(true_labels) == 2
        assert len(pred_labels) == 2

        # First sequence: O, B-PER, I-PER, O
        assert true_labels[0] == ["O", "B-PER", "I-PER", "O"]
        assert pred_labels[0] == ["O", "B-PER", "I-PER", "O"]

        # Second sequence: B-ORG, I-ORG, O (no padding)
        assert true_labels[1] == ["B-ORG", "I-ORG", "O"]
        assert pred_labels[1] == ["B-ORG", "I-ORG", "O"]

    def test_align_predictions_with_padding(self, sample_labels):
        """Test alignment with padding tokens."""
        predictions = np.array([[[0.9, 0.1], [0.1, 0.9]]])  # Shape: (1, 2, 2)
        label_ids = np.array([[0, -100]])  # Second token is padding

        true_labels, pred_labels = align_predictions(predictions, label_ids, ["O", "B-PER"])

        assert len(true_labels) == 1
        assert true_labels[0] == ["O"]  # Only non-padding token
        assert pred_labels[0] == ["O"]

    def test_align_predictions_empty_sequence(self, sample_labels):
        """Test alignment with completely padded sequence."""
        predictions = np.array([[[0.9, 0.1], [0.1, 0.9]]])
        label_ids = np.array([[-100, -100]])  # All padding

        true_labels, pred_labels = align_predictions(predictions, label_ids, ["O", "B-PER"])

        assert len(true_labels) == 0
        assert len(pred_labels) == 0


class TestComputeMetricsFactory:
    """Test metrics computation factory."""

    def test_compute_metrics_with_seqeval_metric(self, sample_labels):
        """Test metrics computation with seqeval metric."""
        mock_seqeval = Mock()
        mock_seqeval.compute.return_value = {
            "overall_precision": 0.85,
            "overall_recall": 0.80,
            "overall_f1": 0.82,
            "overall_accuracy": 0.90,
        }

        compute_metrics = compute_metrics_factory(sample_labels, mock_seqeval)

        # Create test data
        predictions = np.array([[[0.9, 0.1, 0, 0, 0, 0, 0]]])
        labels = np.array([[0]])
        eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)

        with patch("src.evaluation.classification_report") as mock_report:
            mock_report.return_value = {"O": {"f1-score": 0.95, "precision": 0.90, "recall": 0.85}}

            result = compute_metrics(eval_pred)

            assert result["precision"] == 0.85
            assert result["recall"] == 0.80
            assert result["f1"] == 0.82
            assert result["accuracy"] == 0.90

    @patch("src.evaluation.wandb")
    def test_compute_metrics_without_seqeval_metric(self, mock_wandb, sample_labels):
        """Test metrics computation without seqeval metric."""
        mock_wandb.run = None

        compute_metrics = compute_metrics_factory(sample_labels, None)

        predictions = np.array([[[0.9, 0.1, 0, 0, 0, 0, 0]]])
        labels = np.array([[0]])
        eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)

        with (
            patch("src.evaluation.precision_score", return_value=0.85),
            patch("src.evaluation.recall_score", return_value=0.80),
            patch("src.evaluation.f1_score", return_value=0.82),
            patch("src.evaluation.accuracy_score", return_value=0.90),
            patch("src.evaluation.classification_report") as mock_report,
        ):
            mock_report.return_value = {"O": {"f1-score": 0.95, "precision": 0.90, "recall": 0.85}}

            result = compute_metrics(eval_pred)

            assert result["precision"] == 0.85
            assert result["recall"] == 0.80
            assert result["f1"] == 0.82
            assert result["accuracy"] == 0.90

    @patch("src.evaluation.wandb")
    def test_compute_metrics_with_wandb_logging(self, mock_wandb, sample_labels):
        """Test metrics computation with WandB logging."""
        mock_wandb.run = Mock()  # Simulate active run

        compute_metrics = compute_metrics_factory(sample_labels, None)

        predictions = np.array([[[0.9, 0.1, 0, 0, 0, 0, 0]]])
        labels = np.array([[0]])
        eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)

        with (
            patch("src.evaluation.precision_score", return_value=0.85),
            patch("src.evaluation.recall_score", return_value=0.80),
            patch("src.evaluation.f1_score", return_value=0.82),
            patch("src.evaluation.accuracy_score", return_value=0.90),
            patch("src.evaluation.classification_report") as mock_report,
        ):
            mock_report.return_value = {}

            compute_metrics(eval_pred)

            # Verify WandB logging was called
            mock_wandb.log.assert_called_once()
            log_args = mock_wandb.log.call_args[0][0]
            assert "eval/precision" in log_args
            assert "eval/recall" in log_args


class TestEvaluateModel:
    """Test model evaluation functionality."""

    @patch("torch.utils.data.DataLoader")
    @patch("src.evaluation.logger")
    def test_evaluate_model(self, mock_logger, mock_dataloader, sample_labels):
        """Test model evaluation."""
        # Mock model
        mock_model = Mock()
        mock_model.eval.return_value = None
        mock_param = torch.tensor([1.0], device="cpu")
        mock_model.parameters.return_value = iter([mock_param])  # For device detection

        # Mock outputs
        mock_outputs = Mock()
        mock_outputs.logits = torch.tensor([[[0.9, 0.1, 0, 0, 0, 0, 0]]])
        mock_model.return_value = mock_outputs

        # Mock dataset and other inputs
        mock_dataset = Mock()
        mock_data_collator = Mock()
        mock_tokenizer = Mock()

        # Mock dataloader
        mock_batch = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "labels": torch.tensor([[0]]),
        }
        mock_dataloader_instance = [mock_batch]  # Simple list for iteration
        mock_dataloader.return_value = mock_dataloader_instance

        with patch("src.evaluation.compute_metrics_factory") as mock_factory:
            mock_compute_metrics = Mock(return_value={"f1": 0.85})
            mock_factory.return_value = mock_compute_metrics

            result = evaluate_model(
                mock_model, mock_dataset, mock_data_collator, mock_tokenizer, sample_labels, batch_size=8
            )

            assert result == {"f1": 0.85}
            mock_model.eval.assert_called_once()


class TestPrintEvaluationReport:
    """Test evaluation report printing."""

    @patch("builtins.print")
    def test_print_evaluation_report(self, mock_print, sample_labels):
        """Test printing evaluation report."""
        predictions = [["O", "B-PER"], ["B-ORG", "O"]]
        labels = [["O", "B-PER"], ["B-ORG", "I-ORG"]]

        with (
            patch("src.evaluation.classification_report", return_value="Mock Report"),
            patch("src.evaluation.precision_score", return_value=0.85),
            patch("src.evaluation.recall_score", return_value=0.80),
            patch("src.evaluation.f1_score", return_value=0.82),
            patch("src.evaluation.accuracy_score", return_value=0.90),
        ):
            print_evaluation_report(predictions, labels, sample_labels)

            # Verify print was called multiple times
            assert mock_print.call_count > 5


class TestSavePredictions:
    """Test prediction saving functionality."""

    def test_save_predictions(self):
        """Test saving predictions to file."""
        predictions = [["O", "B-PER"], ["B-ORG"]]
        labels = [["O", "I-PER"], ["B-ORG"]]
        tokens = [["John", "Smith"], ["Microsoft"]]

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            output_file = f.name

        try:
            save_predictions(predictions, labels, tokens, output_file)

            # Read and verify file contents
            with open(output_file) as f:
                content = f.read()

            lines = content.strip().split("\n")
            assert "John\tO\tO" in lines
            assert "Smith\tI-PER\tB-PER" in lines
            assert "Microsoft\tB-ORG\tB-ORG" in lines

        finally:
            Path(output_file).unlink()


class TestComputeConfusionMatrix:
    """Test confusion matrix computation."""

    def test_compute_confusion_matrix(self, sample_labels):
        """Test confusion matrix computation."""
        predictions = [["O", "B-PER"], ["B-ORG", "O"]]
        labels = [["O", "B-PER"], ["B-ORG", "I-ORG"]]

        with patch("sklearn.metrics.confusion_matrix") as mock_cm:
            mock_cm.return_value = np.array([[1, 0], [0, 1]])

            result = compute_confusion_matrix(predictions, labels, sample_labels)

            mock_cm.assert_called_once()
            assert result.shape == (2, 2)


class TestLogConfusionMatrixToWandB:
    """Test WandB confusion matrix logging."""

    @patch("src.evaluation.wandb")
    def test_log_confusion_matrix_to_wandb_with_active_run(self, mock_wandb, sample_labels):
        """Test logging confusion matrix when WandB run is active."""
        mock_wandb.run = Mock()  # Active run
        mock_wandb.plot.confusion_matrix.return_value = "mock_plot"

        confusion_matrix = np.array([[10, 1], [2, 8]])

        log_confusion_matrix_to_wandb(confusion_matrix, sample_labels[:2])

        mock_wandb.plot.confusion_matrix.assert_called_once()
        mock_wandb.log.assert_called_once()

    @patch("src.evaluation.wandb")
    def test_log_confusion_matrix_to_wandb_without_active_run(self, mock_wandb, sample_labels):
        """Test logging confusion matrix when WandB run is not active."""
        mock_wandb.run = None  # No active run

        confusion_matrix = np.array([[10, 1], [2, 8]])

        log_confusion_matrix_to_wandb(confusion_matrix, sample_labels[:2])

        # Should not call any WandB functions
        mock_wandb.plot.confusion_matrix.assert_not_called()
        mock_wandb.log.assert_not_called()


@pytest.mark.parametrize(
    ("metric_name", "expected_calls"),
    [
        ("precision", 1),
        ("recall", 1),
        ("f1", 1),
        ("accuracy", 1),
    ],
)
def test_parametrized_metric_calls(metric_name, expected_calls, sample_labels):
    """Test that different metrics are called correctly."""
    compute_metrics = compute_metrics_factory(sample_labels, None)

    predictions = np.array([[[0.9, 0.1, 0, 0, 0, 0, 0]]])
    labels = np.array([[0]])
    eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)

    with (
        patch(f"src.evaluation.{metric_name}_score", return_value=0.85) as mock_metric,
        patch("src.evaluation.classification_report", return_value={}),
    ):
        result = compute_metrics(eval_pred)

        assert mock_metric.call_count == expected_calls
        assert metric_name in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
