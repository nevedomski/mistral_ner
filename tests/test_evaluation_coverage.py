"""Comprehensive tests for evaluation.py to achieve 85%+ coverage."""

from __future__ import annotations

import tempfile
from unittest.mock import Mock, patch

import numpy as np
import torch
from transformers import EvalPrediction

from src.evaluation import (
    align_predictions,
    compute_confusion_matrix,
    compute_detailed_metrics,
    compute_metrics_factory,
    create_enhanced_compute_metrics,
    evaluate_model,
    log_confusion_matrix_to_wandb,
    print_evaluation_report,
    save_predictions,
)


class TestAlignPredictions:
    """Test align_predictions function."""

    def test_align_predictions_basic(self) -> None:
        """Test basic prediction alignment."""
        # Arrange
        # Shape: (batch_size=2, seq_len=4, num_classes=3)
        predictions = np.array(
            [
                [[0.1, 0.8, 0.1], [0.7, 0.2, 0.1], [0.1, 0.1, 0.8], [0.9, 0.05, 0.05]],
                [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.9, 0.05, 0.05]],
            ]
        )
        label_ids = np.array(
            [
                [1, 0, 2, -100],  # Last token is padding
                [0, 1, 2, -100],  # Last token is padding
            ]
        )
        label_names = ["O", "B-PER", "I-PER"]

        # Act
        true_labels, pred_labels = align_predictions(predictions, label_ids, label_names)

        # Assert
        assert len(true_labels) == 2
        assert len(pred_labels) == 2
        assert true_labels[0] == ["B-PER", "O", "I-PER"]
        assert pred_labels[0] == ["B-PER", "O", "I-PER"]
        assert true_labels[1] == ["O", "B-PER", "I-PER"]
        assert pred_labels[1] == ["O", "B-PER", "I-PER"]

    def test_align_predictions_empty_sequences(self) -> None:
        """Test alignment with all-padding sequences."""
        # Arrange
        predictions = np.array([[[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]], [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]]])
        label_ids = np.array(
            [
                [-100, -100],  # All padding
                [-100, -100],  # All padding
            ]
        )
        label_names = ["O", "B-PER", "I-PER"]

        # Act
        true_labels, pred_labels = align_predictions(predictions, label_ids, label_names)

        # Assert
        assert len(true_labels) == 0
        assert len(pred_labels) == 0

    def test_align_predictions_partial_padding(self) -> None:
        """Test alignment with some padding tokens."""
        # Arrange
        predictions = np.array(
            [
                [[0.1, 0.8, 0.1], [0.7, 0.2, 0.1], [0.1, 0.1, 0.8]],
            ]
        )
        label_ids = np.array(
            [
                [1, -100, 2],  # Middle token is padding
            ]
        )
        label_names = ["O", "B-PER", "I-PER"]

        # Act
        true_labels, pred_labels = align_predictions(predictions, label_ids, label_names)

        # Assert
        assert len(true_labels) == 1
        assert true_labels[0] == ["B-PER", "I-PER"]
        assert pred_labels[0] == ["B-PER", "I-PER"]


class TestComputeMetricsFactory:
    """Test compute_metrics_factory function."""

    @patch("src.evaluation.wandb")
    def test_compute_metrics_factory_with_seqeval(self, mock_wandb: Mock) -> None:
        """Test compute_metrics with seqeval metric."""
        # Arrange
        label_names = ["O", "B-PER", "I-PER"]
        mock_seqeval = Mock()
        mock_seqeval.compute.return_value = {
            "overall_precision": 0.85,
            "overall_recall": 0.80,
            "overall_f1": 0.82,
            "overall_accuracy": 0.88,
        }
        mock_wandb.run = None

        # Create predictions and labels
        predictions = np.array([[[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]]])
        labels = np.array([[1, 0]])
        eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)

        # Act
        compute_metrics = compute_metrics_factory(label_names)

        # Mock all seqeval functions
        with (
            patch("src.evaluation.precision_score", return_value=0.85),
            patch("src.evaluation.recall_score", return_value=0.80),
            patch("src.evaluation.f1_score", return_value=0.82),
            patch("src.evaluation.accuracy_score", return_value=0.88),
            patch("src.evaluation.classification_report") as mock_report,
        ):
            mock_report.return_value = {
                "B-PER": {"precision": 0.9, "recall": 0.8, "f1-score": 0.85},
                "O": {"precision": 0.95, "recall": 0.9, "f1-score": 0.92},
            }

            result = compute_metrics(eval_pred)

        # Assert
        assert result["precision"] == 0.85
        assert result["recall"] == 0.80
        assert result["f1"] == 0.82
        assert result["accuracy"] == 0.88
        assert "B-PER_f1" in result
        assert "O_f1" in result

    @patch("src.evaluation.wandb")
    @patch("src.evaluation.precision_score")
    @patch("src.evaluation.recall_score")
    @patch("src.evaluation.f1_score")
    @patch("src.evaluation.accuracy_score")
    @patch("src.evaluation.classification_report")
    def test_compute_metrics_factory_without_seqeval(
        self,
        mock_report: Mock,
        mock_accuracy: Mock,
        mock_f1: Mock,
        mock_recall: Mock,
        mock_precision: Mock,
        mock_wandb: Mock,
    ) -> None:
        """Test compute_metrics without seqeval metric."""
        # Arrange
        label_names = ["O", "B-PER"]
        mock_wandb.run = None

        # Mock direct seqeval calls
        mock_precision.return_value = 0.85
        mock_recall.return_value = 0.80
        mock_f1.return_value = 0.82
        mock_accuracy.return_value = 0.88

        mock_report.return_value = {
            "B-PER": {"precision": 0.9, "recall": 0.8, "f1-score": 0.85},
        }

        predictions = np.array([[[0.1, 0.8], [0.7, 0.3]]])
        labels = np.array([[1, 0]])
        eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)

        # Act
        compute_metrics = compute_metrics_factory(label_names)
        result = compute_metrics(eval_pred)

        # Assert
        assert result["precision"] == 0.85
        assert result["recall"] == 0.80
        assert result["f1"] == 0.82
        assert result["accuracy"] == 0.88

    @patch("src.evaluation.wandb")
    def test_compute_metrics_factory_with_wandb_logging(self, mock_wandb: Mock) -> None:
        """Test compute_metrics with wandb logging."""
        # Arrange
        label_names = ["O", "B-PER"]
        mock_wandb.run = Mock()
        mock_wandb.log = Mock()

        predictions = np.array([[[0.1, 0.8], [0.7, 0.3]]])
        labels = np.array([[1, 0]])
        eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)

        # Act
        compute_metrics = compute_metrics_factory(label_names)

        with (
            patch("src.evaluation.precision_score", return_value=0.85),
            patch("src.evaluation.recall_score", return_value=0.80),
            patch("src.evaluation.f1_score", return_value=0.82),
            patch("src.evaluation.accuracy_score", return_value=0.88),
            patch("src.evaluation.classification_report", return_value={}),
        ):
            compute_metrics(eval_pred)

        # Assert
        mock_wandb.log.assert_called_once()


class TestEvaluateModel:
    """Test evaluate_model function."""

    @patch("src.evaluation.logger")
    @patch("torch.utils.data.DataLoader")
    def test_evaluate_model_basic(self, mock_dataloader_class: Mock, mock_logger: Mock) -> None:
        """Test basic model evaluation."""
        # Arrange
        mock_model = Mock()
        mock_model.eval.return_value = None

        # Mock parameters() to return an iterator
        mock_param = Mock()
        mock_param.device = torch.device("cpu")
        mock_model.parameters.return_value = iter([mock_param])

        # Mock model outputs
        mock_outputs = Mock()
        mock_outputs.logits = torch.tensor([[[0.1, 0.8, 0.1]]])
        mock_model.return_value = mock_outputs

        mock_dataset = Mock()
        mock_collator = Mock()
        mock_tokenizer = Mock()
        label_names = ["O", "B-PER", "I-PER"]

        # Mock dataloader
        mock_batch = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "labels": torch.tensor([[1, 0, 2]]),
        }
        mock_dataloader = Mock()
        mock_dataloader.__iter__ = Mock(return_value=iter([mock_batch]))
        mock_dataloader_class.return_value = mock_dataloader

        # Act
        with patch("src.evaluation.compute_metrics_factory") as mock_factory:
            mock_compute_metrics = Mock(return_value={"f1": 0.85})
            mock_factory.return_value = mock_compute_metrics

            result = evaluate_model(
                model=mock_model,
                eval_dataset=mock_dataset,
                data_collator=mock_collator,
                tokenizer=mock_tokenizer,
                label_names=label_names,
                batch_size=8,
            )

        # Assert
        assert result == {"f1": 0.85}
        mock_model.eval.assert_called_once()
        mock_logger.info.assert_called()


class TestComputeDetailedMetrics:
    """Test compute_detailed_metrics function."""

    def test_compute_detailed_metrics_basic(self) -> None:
        """Test basic detailed metrics computation."""
        # Arrange
        true_labels = [["O", "B-PER", "I-PER", "O"], ["B-ORG", "I-ORG", "O", "B-PER"]]
        pred_labels = [
            ["O", "B-PER", "I-PER", "O"],  # Perfect match
            ["B-ORG", "I-ORG", "B-PER", "B-PER"],  # Some errors
        ]
        label_names = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG"]

        # Act
        with patch("src.evaluation.classification_report") as mock_report:
            mock_report.return_value = {
                "macro avg": {"precision": 0.85, "recall": 0.80, "f1-score": 0.82, "support": 100},
                "B-PER": {"precision": 0.9, "recall": 0.8, "f1-score": 0.85, "support": 10},
                "I-PER": {"precision": 0.8, "recall": 0.9, "f1-score": 0.85, "support": 5},
                "B-ORG": {"precision": 0.95, "recall": 0.85, "f1-score": 0.90, "support": 8},
                "I-ORG": {"precision": 0.85, "recall": 0.95, "f1-score": 0.90, "support": 3},
            }

            result = compute_detailed_metrics(true_labels, pred_labels, label_names)

        # Assert
        assert "overall" in result
        assert "per_entity_type" in result
        assert "confusion_matrix" in result
        assert "error_analysis" in result

        # Check entity types
        assert "PER" in result["per_entity_type"]
        assert "ORG" in result["per_entity_type"]

        # Check error analysis structure
        assert "false_positives" in result["error_analysis"]
        assert "false_negatives" in result["error_analysis"]
        assert "boundary_errors" in result["error_analysis"]
        assert "type_confusion" in result["error_analysis"]

    def test_compute_detailed_metrics_with_errors(self) -> None:
        """Test detailed metrics with various error types."""
        # Arrange
        true_labels = [["O", "B-PER", "I-PER"]]
        pred_labels = [["B-ORG", "O", "B-PER"]]  # FP, FN, type confusion
        label_names = ["O", "B-PER", "I-PER", "B-ORG"]

        # Act
        with patch("src.evaluation.classification_report") as mock_report:
            mock_report.return_value = {
                "macro avg": {"precision": 0.5, "recall": 0.5, "f1-score": 0.5, "support": 10},
            }

            result = compute_detailed_metrics(true_labels, pred_labels, label_names)

        # Assert
        # Check confusion matrix
        assert result["confusion_matrix"]["O"]["B-ORG"] == 1  # False positive
        assert result["confusion_matrix"]["B-PER"]["O"] == 1  # False negative
        assert result["confusion_matrix"]["I-PER"]["B-PER"] == 1  # Type/boundary error


class TestCreateEnhancedComputeMetrics:
    """Test create_enhanced_compute_metrics function."""

    @patch("src.evaluation.compute_metrics_factory")
    def test_create_enhanced_compute_metrics_basic(self, mock_factory: Mock) -> None:
        """Test enhanced compute metrics creation."""
        # Arrange
        label_names = ["O", "B-PER", "I-PER"]
        mock_base_metrics = {"precision": 0.85, "recall": 0.80}
        mock_base_compute = Mock(return_value=mock_base_metrics)
        mock_factory.return_value = mock_base_compute

        predictions = np.array([[[0.1, 0.8, 0.1]]])
        labels = np.array([[1]])
        eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)

        # Act
        enhanced_compute_metrics = create_enhanced_compute_metrics(label_names, compute_detailed=True)

        with patch("src.evaluation.compute_detailed_metrics") as mock_detailed:
            mock_detailed.return_value = {
                "per_entity_type": {
                    "PER": {
                        "precision": 0.9,
                        "recall": 0.85,
                        "f1": 0.87,
                        "support": 10,
                        "b_tag_accuracy": 0.95,
                        "i_tag_accuracy": 0.90,
                    }
                },
                "confusion_matrix": {"B-PER": {"O": 2}},
                "error_analysis": {
                    "type_confusion": {"PER": {"ORG": 1}},
                    "boundary_errors": {"PER": 1},
                },
            }

            result = enhanced_compute_metrics(eval_pred)

        # Assert
        assert result["precision"] == 0.85  # Base metric
        assert result["PER_precision"] == 0.9  # Enhanced metric
        assert result["PER_f1"] == 0.87
        assert result["PER_b_accuracy"] == 0.95
        assert "error_rate" in result
        assert "type_confusion_rate" in result

    @patch("src.evaluation.compute_metrics_factory")
    def test_create_enhanced_compute_metrics_no_detailed(self, mock_factory: Mock) -> None:
        """Test enhanced compute metrics without detailed analysis."""
        # Arrange
        label_names = ["O", "B-PER"]
        mock_base_metrics = {"precision": 0.85}
        mock_base_compute = Mock(return_value=mock_base_metrics)
        mock_factory.return_value = mock_base_compute

        predictions = np.array([[[0.1, 0.8]]])
        labels = np.array([[1]])
        eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)

        # Act
        enhanced_compute_metrics = create_enhanced_compute_metrics(label_names, compute_detailed=False)
        result = enhanced_compute_metrics(eval_pred)

        # Assert
        assert result == {"precision": 0.85}  # Only base metrics


class TestPrintEvaluationReport:
    """Test print_evaluation_report function."""

    @patch("builtins.print")
    @patch("src.evaluation.classification_report")
    @patch("src.evaluation.precision_score")
    @patch("src.evaluation.recall_score")
    @patch("src.evaluation.f1_score")
    @patch("src.evaluation.accuracy_score")
    def test_print_evaluation_report(
        self,
        mock_accuracy: Mock,
        mock_f1: Mock,
        mock_recall: Mock,
        mock_precision: Mock,
        mock_report: Mock,
        mock_print: Mock,
    ) -> None:
        """Test evaluation report printing."""
        # Arrange
        predictions = [["B-PER", "O"]]
        labels = [["B-PER", "O"]]
        label_names = ["O", "B-PER"]

        mock_report.return_value = "Classification Report"
        mock_precision.return_value = 0.85
        mock_recall.return_value = 0.80
        mock_f1.return_value = 0.82
        mock_accuracy.return_value = 0.88

        # Act
        print_evaluation_report(predictions, labels, label_names)

        # Assert
        assert mock_print.call_count > 0  # Should print multiple lines
        mock_report.assert_called_once_with(labels, predictions, digits=4, zero_division=0)


class TestSavePredictions:
    """Test save_predictions function."""

    @patch("src.evaluation.logger")
    def test_save_predictions(self, mock_logger: Mock) -> None:
        """Test saving predictions to file."""
        # Arrange
        predictions = [["B-PER", "O"], ["O", "B-ORG"]]
        labels = [["B-PER", "I-PER"], ["O", "B-ORG"]]
        tokens = [["John", "works"], ["at", "Microsoft"]]

        # Act
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            output_file = f.name

        save_predictions(predictions, labels, tokens, output_file)

        # Assert
        with open(output_file) as f:
            content = f.read()

        lines = content.strip().split("\n")
        assert "John\tB-PER\tB-PER" in lines
        assert "works\tI-PER\tO" in lines
        assert "at\tO\tO" in lines
        assert "Microsoft\tB-ORG\tB-ORG" in lines
        mock_logger.info.assert_called_once()


class TestComputeConfusionMatrix:
    """Test compute_confusion_matrix function."""

    @patch("sklearn.metrics.confusion_matrix")
    def test_compute_confusion_matrix(self, mock_sklearn_cm: Mock) -> None:
        """Test confusion matrix computation."""
        # Arrange
        predictions = [["B-PER", "O"], ["O", "B-ORG"]]
        labels = [["B-PER", "I-PER"], ["O", "B-ORG"]]
        label_names = ["O", "B-PER", "I-PER", "B-ORG"]

        mock_cm_result = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        mock_sklearn_cm.return_value = mock_cm_result

        # Act
        result = compute_confusion_matrix(predictions, labels, label_names)

        # Assert
        mock_sklearn_cm.assert_called_once_with(
            ["B-PER", "I-PER", "O", "B-ORG"],  # Flattened labels
            ["B-PER", "O", "O", "B-ORG"],  # Flattened predictions
            labels=label_names,
        )
        np.testing.assert_array_equal(result, mock_cm_result)


class TestLogConfusionMatrixToWandb:
    """Test log_confusion_matrix_to_wandb function."""

    @patch("src.evaluation.wandb")
    def test_log_confusion_matrix_to_wandb_with_run(self, mock_wandb: Mock) -> None:
        """Test logging confusion matrix to wandb when run is active."""
        # Arrange
        confusion_matrix = np.array([[10, 2], [1, 15]])
        label_names = ["O", "B-PER"]

        mock_wandb.run = Mock()
        mock_wandb.log = Mock()
        mock_wandb.plot.confusion_matrix = Mock()

        # Act
        log_confusion_matrix_to_wandb(confusion_matrix, label_names)

        # Assert
        mock_wandb.log.assert_called_once()
        mock_wandb.plot.confusion_matrix.assert_called_once_with(
            probs=None, y_true=None, preds=None, class_names=label_names, matrix=confusion_matrix.tolist()
        )

    @patch("src.evaluation.wandb")
    def test_log_confusion_matrix_to_wandb_no_run(self, mock_wandb: Mock) -> None:
        """Test logging confusion matrix when no wandb run is active."""
        # Arrange
        confusion_matrix = np.array([[10, 2], [1, 15]])
        label_names = ["O", "B-PER"]

        mock_wandb.run = None

        # Act
        log_confusion_matrix_to_wandb(confusion_matrix, label_names)

        # Assert - Should not log anything
        assert not hasattr(mock_wandb, "log") or not mock_wandb.log.called


# Edge cases and error conditions
class TestEvaluationEdgeCases:
    """Test edge cases and error conditions."""

    def test_align_predictions_single_class(self) -> None:
        """Test alignment with single class predictions."""
        predictions = np.array([[[1.0, 0.0]]])
        label_ids = np.array([[0]])
        label_names = ["O", "B-PER"]

        true_labels, pred_labels = align_predictions(predictions, label_ids, label_names)

        assert len(true_labels) == 1
        assert true_labels[0] == ["O"]
        assert pred_labels[0] == ["O"]

    def test_compute_detailed_metrics_no_entities(self) -> None:
        """Test detailed metrics with only O labels."""
        true_labels = [["O", "O", "O"]]
        pred_labels = [["O", "O", "O"]]
        label_names = ["O"]

        with patch("src.evaluation.classification_report") as mock_report:
            mock_report.return_value = {
                "macro avg": {"precision": 1.0, "recall": 1.0, "f1-score": 1.0, "support": 3},
            }

            result = compute_detailed_metrics(true_labels, pred_labels, label_names)

        assert result["per_entity_type"] == {}  # No entity types found

    def test_enhanced_compute_metrics_no_errors(self) -> None:
        """Test enhanced metrics with perfect predictions (no errors)."""
        with patch("src.evaluation.compute_metrics_factory") as mock_factory:
            mock_base_compute = Mock(return_value={"precision": 1.0})
            mock_factory.return_value = mock_base_compute

            enhanced_compute = create_enhanced_compute_metrics(["O", "B-PER"])

            with patch("src.evaluation.compute_detailed_metrics") as mock_detailed:
                mock_detailed.return_value = {
                    "per_entity_type": {},
                    "confusion_matrix": {},  # No errors
                    "error_analysis": {
                        "type_confusion": {},
                        "boundary_errors": {},
                    },
                }

                predictions = np.array([[[1.0, 0.0]]])
                labels = np.array([[0]])
                eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)

                result = enhanced_compute(eval_pred)

        assert result["precision"] == 1.0
        # Should not have error rates when no errors
        assert "error_rate" not in result or result.get("error_rate", 0) == 0
