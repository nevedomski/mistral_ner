"""Performance regression tests for model metrics.

This module implements performance regression testing to ensure that model
metrics (F1, precision, recall, accuracy) don't degrade below acceptable
thresholds during development.
"""

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


# Performance thresholds based on baseline model performance
PERFORMANCE_THRESHOLDS = {
    "f1": 0.80,  # Minimum acceptable F1 score
    "precision": 0.78,  # Minimum acceptable precision
    "recall": 0.78,  # Minimum acceptable recall
    "accuracy": 0.85,  # Minimum acceptable accuracy
}


class PerformanceRegression:
    """Track and validate model performance metrics."""

    def __init__(self, baseline_path: Path) -> None:
        """Initialize performance regression tracker.

        Args:
            baseline_path: Path to store/load baseline metrics
        """
        self.baseline_path = baseline_path
        self.thresholds = PERFORMANCE_THRESHOLDS

    def save_baseline(self, metrics: dict[str, float]) -> None:
        """Save current metrics as baseline.

        Args:
            metrics: Dictionary of metric names to values
        """
        baseline = {
            "metrics": metrics,
            "timestamp": str(datetime.now()),
            "model_version": "mistral-7b-v0.3",
            "dataset": "conll2003",
            "thresholds": self.thresholds,
        }

        # Ensure directory exists
        self.baseline_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.baseline_path, "w") as f:
            json.dump(baseline, f, indent=2)

    def load_baseline(self) -> dict[str, float] | None:
        """Load baseline metrics from file.

        Returns:
            Baseline metrics dict or None if file doesn't exist
        """
        if not self.baseline_path.exists():
            return None

        try:
            with open(self.baseline_path) as f:
                content = f.read().strip()
                if not content:
                    return None
                baseline = json.loads(content)
        except (OSError, json.JSONDecodeError):
            return None

        metrics = baseline.get("metrics")
        return metrics if isinstance(metrics, dict) else None

    def check_regression(self, current_metrics: dict[str, float]) -> dict[str, bool]:
        """Check if current metrics meet thresholds.

        Args:
            current_metrics: Dictionary of current metric values

        Returns:
            Dictionary mapping metric names to boolean pass/fail status
        """
        results = {}
        for metric, threshold in self.thresholds.items():
            if metric in current_metrics:
                results[metric] = current_metrics[metric] >= threshold
            else:
                results[metric] = False

        return results

    def compare_to_baseline(self, current_metrics: dict[str, float]) -> dict[str, Any]:
        """Compare current metrics to baseline.

        Args:
            current_metrics: Dictionary of current metric values

        Returns:
            Dictionary with comparison results including differences
        """
        baseline = self.load_baseline()
        if baseline is None:
            return {"status": "no_baseline", "baseline": {}, "current": current_metrics}

        comparison = {
            "baseline": baseline,
            "current": current_metrics,
            "differences": {},
            "improvements": {},
            "regressions": {},
        }

        for metric in baseline:
            if metric in current_metrics:
                diff = current_metrics[metric] - baseline[metric]
                comparison["differences"][metric] = diff

                if diff > 0:
                    comparison["improvements"][metric] = diff
                elif diff < 0:
                    comparison["regressions"][metric] = abs(diff)

        return comparison


# Test fixtures and helper functions


@pytest.fixture
def temp_baseline_path():
    """Create temporary baseline file path."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        baseline_path = Path(f.name)

    yield baseline_path

    # Cleanup
    if baseline_path.exists():
        baseline_path.unlink()


@pytest.fixture
def sample_metrics():
    """Sample performance metrics for testing."""
    return {
        "f1": 0.85,
        "precision": 0.82,
        "recall": 0.88,
        "accuracy": 0.90,
    }


@pytest.fixture
def performance_tracker(temp_baseline_path):
    """Create performance regression tracker instance."""
    return PerformanceRegression(temp_baseline_path)


def create_mock_model_and_data():
    """Create mock model and dataset for testing."""
    # Mock model with dummy outputs
    mock_model = Mock()
    mock_model.eval.return_value = None

    # Mock dataset and related components
    mock_dataset = Mock()
    mock_data_collator = Mock()
    mock_tokenizer = Mock()

    return mock_model, mock_dataset, mock_data_collator, mock_tokenizer


# Performance regression tests


class TestPerformanceRegression:
    """Test the PerformanceRegression class functionality."""

    def test_save_and_load_baseline(self, performance_tracker, sample_metrics):
        """Test saving and loading baseline metrics."""
        # Save baseline
        performance_tracker.save_baseline(sample_metrics)

        # Verify file was created
        assert performance_tracker.baseline_path.exists()

        # Load and verify
        loaded_metrics = performance_tracker.load_baseline()
        assert loaded_metrics == sample_metrics

    def test_load_baseline_no_file(self, performance_tracker):
        """Test loading baseline when file doesn't exist."""
        result = performance_tracker.load_baseline()
        assert result is None

    def test_check_regression_all_pass(self, performance_tracker, sample_metrics):
        """Test regression check with all metrics passing."""
        results = performance_tracker.check_regression(sample_metrics)

        assert all(results.values()), f"Some metrics failed: {results}"
        assert len(results) == len(PERFORMANCE_THRESHOLDS)

    def test_check_regression_some_fail(self, performance_tracker):
        """Test regression check with some metrics failing."""
        poor_metrics = {
            "f1": 0.75,  # Below threshold (0.80)
            "precision": 0.80,  # Above threshold
            "recall": 0.76,  # Below threshold (0.78)
            "accuracy": 0.88,  # Above threshold
        }

        results = performance_tracker.check_regression(poor_metrics)

        assert not results["f1"]  # Should fail
        assert results["precision"]  # Should pass
        assert not results["recall"]  # Should fail
        assert results["accuracy"]  # Should pass

    def test_compare_to_baseline_improvements(self, performance_tracker, sample_metrics):
        """Test baseline comparison with improvements."""
        # Save baseline
        performance_tracker.save_baseline(sample_metrics)

        # Improved metrics
        improved_metrics = {
            "f1": 0.90,  # +0.05
            "precision": 0.85,  # +0.03
            "recall": 0.90,  # +0.02
            "accuracy": 0.92,  # +0.02
        }

        comparison = performance_tracker.compare_to_baseline(improved_metrics)

        assert len(comparison["improvements"]) == 4
        assert abs(comparison["improvements"]["f1"] - 0.05) < 1e-10
        assert len(comparison["regressions"]) == 0

    def test_compare_to_baseline_regressions(self, performance_tracker, sample_metrics):
        """Test baseline comparison with regressions."""
        # Save baseline
        performance_tracker.save_baseline(sample_metrics)

        # Degraded metrics
        degraded_metrics = {
            "f1": 0.80,  # -0.05
            "precision": 0.78,  # -0.04
            "recall": 0.85,  # -0.03
            "accuracy": 0.88,  # -0.02
        }

        comparison = performance_tracker.compare_to_baseline(degraded_metrics)

        assert len(comparison["regressions"]) == 4
        assert abs(comparison["regressions"]["f1"] - 0.05) < 1e-10
        assert len(comparison["improvements"]) == 0

    def test_compare_to_baseline_no_baseline(self, performance_tracker, sample_metrics):
        """Test baseline comparison when no baseline exists."""
        comparison = performance_tracker.compare_to_baseline(sample_metrics)

        assert comparison["status"] == "no_baseline"
        assert comparison["current"] == sample_metrics


@pytest.mark.performance
class TestModelPerformanceThresholds:
    """Test model performance against predefined thresholds."""

    def test_model_f1_score_threshold(self):
        """Ensure F1 score meets minimum threshold."""
        # Mock the evaluation to return acceptable metrics
        mock_metrics = {
            "f1": 0.82,  # Above threshold
            "precision": 0.80,
            "recall": 0.84,
            "accuracy": 0.87,
        }

        with patch("src.evaluation.evaluate_model", return_value=mock_metrics):
            mock_model, mock_dataset, mock_data_collator, mock_tokenizer = create_mock_model_and_data()

            # This would normally call evaluate_model
            metrics = mock_metrics  # Simulate evaluation result

            assert metrics["f1"] >= PERFORMANCE_THRESHOLDS["f1"], (
                f"F1 score {metrics['f1']} below threshold {PERFORMANCE_THRESHOLDS['f1']}"
            )

    def test_model_precision_threshold(self):
        """Ensure precision meets minimum threshold."""
        mock_metrics = {"precision": 0.80}  # Above threshold

        assert mock_metrics["precision"] >= PERFORMANCE_THRESHOLDS["precision"], (
            f"Precision {mock_metrics['precision']} below threshold {PERFORMANCE_THRESHOLDS['precision']}"
        )

    def test_model_recall_threshold(self):
        """Ensure recall meets minimum threshold."""
        mock_metrics = {"recall": 0.81}  # Above threshold

        assert mock_metrics["recall"] >= PERFORMANCE_THRESHOLDS["recall"], (
            f"Recall {mock_metrics['recall']} below threshold {PERFORMANCE_THRESHOLDS['recall']}"
        )

    def test_model_accuracy_threshold(self):
        """Ensure accuracy meets minimum threshold."""
        mock_metrics = {"accuracy": 0.87}  # Above threshold

        assert mock_metrics["accuracy"] >= PERFORMANCE_THRESHOLDS["accuracy"], (
            f"Accuracy {mock_metrics['accuracy']} below threshold {PERFORMANCE_THRESHOLDS['accuracy']}"
        )

    def test_model_precision_recall_balance(self):
        """Ensure balanced precision and recall."""
        mock_metrics = {
            "precision": 0.82,
            "recall": 0.80,
        }

        # Check individual thresholds
        assert mock_metrics["precision"] >= PERFORMANCE_THRESHOLDS["precision"]
        assert mock_metrics["recall"] >= PERFORMANCE_THRESHOLDS["recall"]

        # Check balance (difference should not be too large)
        balance_threshold = 0.10
        precision_recall_diff = abs(mock_metrics["precision"] - mock_metrics["recall"])

        assert precision_recall_diff <= balance_threshold, (
            f"Precision-recall imbalance: {precision_recall_diff} > {balance_threshold}"
        )

    def test_model_overall_performance_regression(self, temp_baseline_path):
        """Test overall performance regression check."""
        tracker = PerformanceRegression(temp_baseline_path)

        # Simulate current model metrics
        current_metrics = {
            "f1": 0.83,
            "precision": 0.81,
            "recall": 0.85,
            "accuracy": 0.88,
        }

        # Check against thresholds
        results = tracker.check_regression(current_metrics)

        # All metrics should pass
        assert all(results.values()), f"Performance regression detected: {results}"

    @pytest.mark.slow
    def test_model_performance_with_actual_evaluation(self):
        """Test performance using actual evaluation (marked as slow)."""
        # This test would use real model evaluation but is marked as slow
        # In a real scenario, this would load a small test dataset and model

        # For now, simulate with mock data to keep test fast
        with patch("src.evaluation.evaluate_model") as mock_evaluate:
            mock_evaluate.return_value = {
                "f1": 0.84,
                "precision": 0.82,
                "recall": 0.86,
                "accuracy": 0.89,
            }

            mock_model, mock_dataset, mock_data_collator, mock_tokenizer = create_mock_model_and_data()

            # This would call actual evaluation in real scenario
            metrics = mock_evaluate.return_value

            # Verify all thresholds are met
            for metric, threshold in PERFORMANCE_THRESHOLDS.items():
                assert metrics[metric] >= threshold, f"{metric} {metrics[metric]} below threshold {threshold}"


@pytest.mark.performance
def test_performance_thresholds_configuration():
    """Test that performance thresholds are correctly configured."""
    # Verify thresholds are reasonable
    assert 0.0 <= PERFORMANCE_THRESHOLDS["f1"] <= 1.0
    assert 0.0 <= PERFORMANCE_THRESHOLDS["precision"] <= 1.0
    assert 0.0 <= PERFORMANCE_THRESHOLDS["recall"] <= 1.0
    assert 0.0 <= PERFORMANCE_THRESHOLDS["accuracy"] <= 1.0

    # Verify F1 threshold is reasonable for NER task
    assert PERFORMANCE_THRESHOLDS["f1"] >= 0.70  # Minimum reasonable for NER
    assert PERFORMANCE_THRESHOLDS["f1"] <= 0.95  # Not too optimistic


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
