"""Comprehensive tests for visualization.py to achieve 85%+ coverage."""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock, patch

from src.visualization import (
    create_performance_summary_plot,
    plot_confusion_matrix,
    plot_error_distribution,
    plot_per_entity_metrics,
)


class TestPlotConfusionMatrix:
    """Test plot_confusion_matrix function."""

    @patch("src.visualization.wandb")
    @patch("src.visualization.plt")
    @patch("src.visualization.sns")
    def test_plot_confusion_matrix_basic(self, mock_sns: Mock, mock_plt: Mock, mock_wandb: Mock) -> None:
        """Test basic confusion matrix plotting."""
        # Arrange
        confusion_matrix = {
            "PER": {"PER": 100, "ORG": 5, "O": 2},
            "ORG": {"ORG": 80, "PER": 3, "O": 1},
            "O": {"O": 500, "PER": 1, "ORG": 2},
        }
        label_names = ["PER", "ORG", "O"]
        save_path = "/tmp/confusion_matrix.png"

        # Mock pyplot methods
        mock_figure = Mock()
        mock_plt.figure.return_value = mock_figure
        mock_plt.savefig = Mock()
        mock_plt.close = Mock()
        mock_plt.title = Mock()
        mock_plt.xlabel = Mock()
        mock_plt.ylabel = Mock()
        mock_plt.tight_layout = Mock()

        # Mock seaborn
        mock_sns.heatmap = Mock()

        # Mock wandb
        mock_wandb.run = None

        # Act
        plot_confusion_matrix(
            confusion_matrix=confusion_matrix,
            label_names=label_names,
            save_path=save_path,
            log_to_wandb=False,
        )

        # Assert
        mock_plt.figure.assert_called_once_with(figsize=(12, 10))
        mock_sns.heatmap.assert_called_once()
        mock_plt.savefig.assert_called_once_with(save_path, dpi=300, bbox_inches="tight")
        mock_plt.close.assert_called_once()

    @patch("src.visualization.wandb")
    @patch("src.visualization.plt")
    @patch("src.visualization.sns")
    def test_plot_confusion_matrix_with_wandb(self, mock_sns: Mock, mock_plt: Mock, mock_wandb: Mock) -> None:
        """Test confusion matrix plotting with wandb logging."""
        # Arrange
        confusion_matrix = {"PER": {"PER": 50}}
        label_names = ["PER"]

        # Mock wandb
        mock_wandb.run = Mock()
        mock_wandb.Image = Mock()
        mock_wandb.log = Mock()

        # Mock pyplot
        mock_plt.figure = Mock()
        mock_plt.close = Mock()
        mock_plt.title = Mock()
        mock_plt.xlabel = Mock()
        mock_plt.ylabel = Mock()
        mock_plt.tight_layout = Mock()

        # Act
        plot_confusion_matrix(
            confusion_matrix=confusion_matrix,
            label_names=label_names,
            save_path=None,
            log_to_wandb=True,
        )

        # Assert
        mock_wandb.log.assert_called_once()

    @patch("src.visualization.plt")
    @patch("src.visualization.sns")
    def test_plot_confusion_matrix_missing_labels(self, mock_sns: Mock, mock_plt: Mock) -> None:
        """Test confusion matrix with missing labels in data."""
        # Arrange
        confusion_matrix = {
            "PER": {"UNKNOWN": 5},  # UNKNOWN not in label_names
            "MISSING": {"PER": 3},  # MISSING not in label_names
        }
        label_names = ["PER", "ORG"]

        # Mock pyplot
        mock_plt.figure = Mock()
        mock_plt.close = Mock()
        mock_plt.title = Mock()
        mock_plt.xlabel = Mock()
        mock_plt.ylabel = Mock()
        mock_plt.tight_layout = Mock()

        # Act
        plot_confusion_matrix(
            confusion_matrix=confusion_matrix,
            label_names=label_names,
            save_path=None,
            log_to_wandb=False,
        )

        # Assert - Should handle missing labels gracefully
        mock_sns.heatmap.assert_called_once()
        mock_plt.close.assert_called_once()


class TestPlotPerEntityMetrics:
    """Test plot_per_entity_metrics function."""

    @patch("src.visualization.wandb")
    @patch("src.visualization.plt")
    def test_plot_per_entity_metrics_basic(self, mock_plt: Mock, mock_wandb: Mock) -> None:
        """Test basic per-entity metrics plotting."""
        # Arrange
        metrics = {
            "PER_precision": 0.85,
            "PER_recall": 0.80,
            "PER_f1": 0.82,
            "ORG_precision": 0.90,
            "ORG_recall": 0.85,
            "ORG_f1": 0.87,
            "LOC_precision": 0.75,
            "LOC_recall": 0.70,
            "LOC_f1": 0.72,
        }
        save_path = "/tmp/per_entity_metrics.png"

        # Mock pyplot components
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_plt.tight_layout = Mock()
        mock_plt.savefig = Mock()
        mock_plt.close = Mock()

        # Mock wandb
        mock_wandb.run = None

        # Act
        plot_per_entity_metrics(metrics=metrics, save_path=save_path, log_to_wandb=False)

        # Assert
        mock_plt.subplots.assert_called_once_with(figsize=(10, 6))
        mock_plt.savefig.assert_called_once_with(save_path, dpi=300, bbox_inches="tight")
        mock_plt.close.assert_called_once()

    @patch("src.visualization.wandb")
    @patch("src.visualization.plt")
    @patch("src.visualization.logger")
    def test_plot_per_entity_metrics_no_metrics(self, mock_logger: Mock, mock_plt: Mock, mock_wandb: Mock) -> None:
        """Test per-entity metrics plotting with no valid metrics."""
        # Arrange
        metrics = {
            "eval_loss": 0.5,
            "accuracy": 0.95,
            # No per-entity metrics
        }

        # Act
        plot_per_entity_metrics(metrics=metrics, save_path=None, log_to_wandb=False)

        # Assert
        mock_logger.warning.assert_called_once_with("No per-entity metrics found to plot")
        # Should not create any plots
        mock_plt.subplots.assert_not_called()

    @patch("src.visualization.wandb")
    @patch("src.visualization.plt")
    def test_plot_per_entity_metrics_with_wandb(self, mock_plt: Mock, mock_wandb: Mock) -> None:
        """Test per-entity metrics plotting with wandb logging."""
        # Arrange
        metrics = {"PER_f1": 0.85}

        # Mock pyplot
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_plt.tight_layout = Mock()
        mock_plt.close = Mock()

        # Mock wandb
        mock_wandb.run = Mock()
        mock_wandb.Image = Mock()
        mock_wandb.log = Mock()

        # Act
        plot_per_entity_metrics(metrics=metrics, save_path=None, log_to_wandb=True)

        # Assert
        mock_wandb.log.assert_called_once()

    @patch("src.visualization.plt")
    def test_plot_per_entity_metrics_complex_entity_names(self, mock_plt: Mock) -> None:
        """Test per-entity metrics with complex entity names."""
        # Arrange
        metrics = {
            "PERSON_NAME_precision": 0.85,
            "PERSON_NAME_recall": 0.80,
            "PERSON_NAME_f1": 0.82,
            "ORGANIZATION_precision": 0.90,
        }

        # Mock pyplot
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_plt.tight_layout = Mock()
        mock_plt.close = Mock()

        # Act
        plot_per_entity_metrics(metrics=metrics, save_path=None, log_to_wandb=False)

        # Assert
        mock_plt.subplots.assert_called_once()
        mock_plt.close.assert_called_once()


class TestPlotErrorDistribution:
    """Test plot_error_distribution function."""

    @patch("src.visualization.wandb")
    @patch("src.visualization.plt")
    @patch("src.visualization.sns")
    def test_plot_error_distribution_basic(self, mock_sns: Mock, mock_plt: Mock, mock_wandb: Mock) -> None:
        """Test basic error distribution plotting."""
        # Arrange
        error_analysis = {
            "false_positives": {"PER": ["example1", "example2"], "ORG": ["example3"]},
            "false_negatives": {"PER": ["example4"], "LOC": ["example5", "example6"]},
            "type_confusion": {"PER": {"ORG": 2}, "ORG": {"PER": 1}},
            "boundary_errors": {"PER": 3, "ORG": 1},
        }
        save_path = "/tmp/error_distribution.png"

        # Mock pyplot
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
        mock_plt.tight_layout = Mock()
        mock_plt.savefig = Mock()
        mock_plt.close = Mock()

        # Mock wandb
        mock_wandb.run = None

        # Act
        plot_error_distribution(error_analysis=error_analysis, save_path=save_path, log_to_wandb=False)

        # Assert
        mock_plt.subplots.assert_called_once_with(1, 2, figsize=(12, 5))
        mock_plt.savefig.assert_called_once_with(save_path, dpi=300, bbox_inches="tight")
        mock_plt.close.assert_called_once()

    @patch("src.visualization.wandb")
    @patch("src.visualization.plt")
    @patch("src.visualization.sns")
    def test_plot_error_distribution_no_errors(self, mock_sns: Mock, mock_plt: Mock, mock_wandb: Mock) -> None:
        """Test error distribution plotting with no errors."""
        # Arrange
        error_analysis = {
            "false_positives": {},
            "false_negatives": {},
            "type_confusion": {},
            "boundary_errors": {},
        }

        # Mock pyplot
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
        mock_plt.tight_layout = Mock()
        mock_plt.close = Mock()

        # Act
        plot_error_distribution(error_analysis=error_analysis, save_path=None, log_to_wandb=False)

        # Assert
        mock_ax1.text.assert_called_with(0.5, 0.5, "No errors found", ha="center", va="center")
        mock_ax2.text.assert_called_with(0.5, 0.5, "No type confusion", ha="center", va="center")
        mock_plt.close.assert_called_once()

    @patch("src.visualization.wandb")
    @patch("src.visualization.plt")
    @patch("src.visualization.sns")
    def test_plot_error_distribution_with_wandb(self, mock_sns: Mock, mock_plt: Mock, mock_wandb: Mock) -> None:
        """Test error distribution plotting with wandb logging."""
        # Arrange
        error_analysis = {"false_positives": {"PER": ["example1"]}}

        # Mock pyplot
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
        mock_plt.tight_layout = Mock()
        mock_plt.close = Mock()

        # Mock wandb
        mock_wandb.run = Mock()
        mock_wandb.Image = Mock()
        mock_wandb.log = Mock()

        # Act
        plot_error_distribution(error_analysis=error_analysis, save_path=None, log_to_wandb=True)

        # Assert
        mock_wandb.log.assert_called_once()

    @patch("src.visualization.plt")
    @patch("src.visualization.sns")
    def test_plot_error_distribution_partial_data(self, mock_sns: Mock, mock_plt: Mock) -> None:
        """Test error distribution with partial error data."""
        # Arrange
        error_analysis = {
            "false_positives": {"PER": ["example1"]},
            # Missing other error types
        }

        # Mock pyplot
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
        mock_plt.tight_layout = Mock()
        mock_plt.close = Mock()

        # Act
        plot_error_distribution(error_analysis=error_analysis, save_path=None, log_to_wandb=False)

        # Assert - Should handle missing error types gracefully
        mock_plt.subplots.assert_called_once()
        mock_plt.close.assert_called_once()


class TestCreatePerformanceSummaryPlot:
    """Test create_performance_summary_plot function."""

    @patch("src.visualization.wandb")
    @patch("src.visualization.plt")
    def test_create_performance_summary_plot_basic(self, mock_plt: Mock, mock_wandb: Mock) -> None:
        """Test basic performance summary plot creation."""
        # Arrange
        metrics = {
            "precision": 0.85,
            "recall": 0.80,
            "f1": 0.82,
            "accuracy": 0.88,
            "PER_f1": 0.85,
            "ORG_f1": 0.90,
            "LOC_f1": 0.75,
            "PER_support": 100,
            "ORG_support": 80,
            "LOC_support": 60,
            "error_rate": 0.12,
            "type_confusion_rate": 0.05,
            "boundary_error_rate": 0.03,
        }
        save_path = "/tmp/performance_summary.png"

        # Mock pyplot components
        mock_figure = Mock()
        mock_plt.figure.return_value = mock_figure

        # Mock subplot with proper bar mock
        mock_subplot = Mock()
        mock_plt.subplot.return_value = mock_subplot

        # Mock bar chart with proper bar objects
        mock_bar = Mock()
        mock_bar.get_height.return_value = 0.85
        mock_bar.get_x.return_value = 0.0
        mock_bar.get_width.return_value = 1.0
        mock_subplot.bar.return_value = [mock_bar, mock_bar, mock_bar, mock_bar]
        mock_subplot.pie.return_value = (Mock(), Mock())

        mock_plt.suptitle = Mock()
        mock_plt.tight_layout = Mock()
        mock_plt.savefig = Mock()
        mock_plt.close = Mock()
        mock_plt.xticks = Mock()

        # Mock wandb
        mock_wandb.run = None

        # Act
        create_performance_summary_plot(metrics=metrics, save_path=save_path, log_to_wandb=False)

        # Assert
        mock_plt.figure.assert_called_once_with(figsize=(15, 10))
        assert mock_plt.subplot.call_count == 4  # 4 subplots
        mock_plt.savefig.assert_called_once_with(save_path, dpi=300, bbox_inches="tight")
        mock_plt.close.assert_called_once()

    @patch("src.visualization.wandb")
    @patch("src.visualization.plt")
    def test_create_performance_summary_plot_minimal_metrics(self, mock_plt: Mock, mock_wandb: Mock) -> None:
        """Test performance summary plot with minimal metrics."""
        # Arrange
        metrics = {
            "precision": 0.85,
            "recall": 0.80,
            # Missing many optional metrics
        }

        # Mock pyplot
        mock_figure = Mock()
        mock_plt.figure.return_value = mock_figure
        mock_subplot = Mock()
        mock_plt.subplot.return_value = mock_subplot

        # Mock bar chart
        mock_bar = Mock()
        mock_bar.get_height.return_value = 0.85
        mock_bar.get_x.return_value = 0.0
        mock_bar.get_width.return_value = 1.0
        mock_subplot.bar.return_value = [mock_bar, mock_bar, mock_bar, mock_bar]
        mock_subplot.text.return_value = Mock()

        mock_plt.suptitle = Mock()
        mock_plt.tight_layout = Mock()
        mock_plt.close = Mock()

        # Act
        create_performance_summary_plot(metrics=metrics, save_path=None, log_to_wandb=False)

        # Assert
        mock_plt.figure.assert_called_once()
        mock_plt.close.assert_called_once()

    @patch("src.visualization.wandb")
    @patch("src.visualization.plt")
    def test_create_performance_summary_plot_with_wandb(self, mock_plt: Mock, mock_wandb: Mock) -> None:
        """Test performance summary plot with wandb logging."""
        # Arrange
        metrics = {"precision": 0.85}

        # Mock pyplot
        mock_figure = Mock()
        mock_plt.figure.return_value = mock_figure
        mock_subplot = Mock()
        mock_plt.subplot.return_value = mock_subplot

        # Mock bar chart
        mock_bar = Mock()
        mock_bar.get_height.return_value = 0.85
        mock_bar.get_x.return_value = 0.0
        mock_bar.get_width.return_value = 1.0
        mock_subplot.bar.return_value = [mock_bar, mock_bar, mock_bar, mock_bar]
        mock_subplot.text.return_value = Mock()

        mock_plt.suptitle = Mock()
        mock_plt.tight_layout = Mock()
        mock_plt.close = Mock()

        # Mock wandb
        mock_wandb.run = Mock()
        mock_wandb.Image = Mock()
        mock_wandb.log = Mock()

        # Act
        create_performance_summary_plot(metrics=metrics, save_path=None, log_to_wandb=True)

        # Assert
        mock_wandb.log.assert_called_once()

    @patch("src.visualization.plt")
    def test_create_performance_summary_plot_no_entity_metrics(self, mock_plt: Mock) -> None:
        """Test performance summary plot with no entity-specific metrics."""
        # Arrange
        metrics = {
            "precision": 0.85,
            "recall": 0.80,
            "f1": 0.82,
            "accuracy": 0.88,
            # No entity-specific metrics
        }

        # Mock pyplot
        mock_figure = Mock()
        mock_plt.figure.return_value = mock_figure
        mock_subplot = Mock()
        mock_plt.subplot.return_value = mock_subplot

        # Mock bar chart
        mock_bar = Mock()
        mock_bar.get_height.return_value = 0.85
        mock_bar.get_x.return_value = 0.0
        mock_bar.get_width.return_value = 1.0
        mock_subplot.bar.return_value = [mock_bar, mock_bar, mock_bar, mock_bar]
        mock_subplot.text.return_value = Mock()

        mock_plt.suptitle = Mock()
        mock_plt.tight_layout = Mock()
        mock_plt.close = Mock()

        # Act
        create_performance_summary_plot(metrics=metrics, save_path=None, log_to_wandb=False)

        # Assert - Should handle missing entity metrics gracefully
        mock_plt.figure.assert_called_once()
        mock_plt.close.assert_called_once()

    @patch("src.visualization.plt")
    def test_create_performance_summary_plot_no_error_metrics(self, mock_plt: Mock) -> None:
        """Test performance summary plot with no error metrics."""
        # Arrange
        metrics = {
            "precision": 0.85,
            "PER_f1": 0.85,
            "PER_support": 100,
            # No error metrics
        }

        # Mock pyplot
        mock_figure = Mock()
        mock_plt.figure.return_value = mock_figure
        mock_subplot = Mock()
        mock_plt.subplot.return_value = mock_subplot

        # Mock bar chart and pie chart
        mock_bar = Mock()
        mock_bar.get_height.return_value = 0.85
        mock_bar.get_x.return_value = 0.0
        mock_bar.get_width.return_value = 1.0
        mock_subplot.bar.return_value = [mock_bar]
        mock_subplot.text.return_value = Mock()
        mock_subplot.pie.return_value = (Mock(), Mock())

        mock_plt.suptitle = Mock()
        mock_plt.tight_layout = Mock()
        mock_plt.close = Mock()
        mock_plt.xticks = Mock()

        # Act
        create_performance_summary_plot(metrics=metrics, save_path=None, log_to_wandb=False)

        # Assert
        mock_plt.figure.assert_called_once()
        mock_plt.close.assert_called_once()

    @patch("src.visualization.plt")
    def test_create_performance_summary_plot_no_support_data(self, mock_plt: Mock) -> None:
        """Test performance summary plot with no support data."""
        # Arrange
        metrics = {
            "precision": 0.85,
            "PER_f1": 0.85,
            "error_rate": 0.12,
            # No support metrics
        }

        # Mock pyplot
        mock_figure = Mock()
        mock_plt.figure.return_value = mock_figure
        mock_subplot = Mock()
        mock_plt.subplot.return_value = mock_subplot

        # Mock bar chart
        mock_bar = Mock()
        mock_bar.get_height.return_value = 0.85
        mock_bar.get_x.return_value = 0.0
        mock_bar.get_width.return_value = 1.0
        mock_subplot.bar.return_value = [mock_bar, mock_bar]
        mock_subplot.text.return_value = Mock()

        mock_plt.suptitle = Mock()
        mock_plt.tight_layout = Mock()
        mock_plt.close = Mock()
        mock_plt.xticks = Mock()

        # Act
        create_performance_summary_plot(metrics=metrics, save_path=None, log_to_wandb=False)

        # Assert
        mock_plt.figure.assert_called_once()
        mock_plt.close.assert_called_once()


# Integration tests for edge cases
class TestVisualizationEdgeCases:
    """Test edge cases and error conditions."""

    @patch("src.visualization.plt")
    @patch("src.visualization.sns")
    def test_empty_confusion_matrix(self, mock_sns: Mock, mock_plt: Mock) -> None:
        """Test plotting with empty confusion matrix."""
        # Arrange
        confusion_matrix: dict[str, dict[str, int]] = {}
        label_names: list[str] = []

        # Mock pyplot
        mock_plt.figure = Mock()
        mock_plt.close = Mock()
        mock_plt.title = Mock()
        mock_plt.xlabel = Mock()
        mock_plt.ylabel = Mock()
        mock_plt.tight_layout = Mock()

        # Act & Assert - Should not raise an error
        plot_confusion_matrix(
            confusion_matrix=confusion_matrix,
            label_names=label_names,
            save_path=None,
            log_to_wandb=False,
        )

        mock_plt.close.assert_called_once()

    @patch("src.visualization.plt")
    def test_empty_metrics_dict(self, mock_plt: Mock) -> None:
        """Test plotting with empty metrics dictionary."""
        # Arrange
        metrics: dict[str, float] = {}

        # Mock pyplot
        mock_figure = Mock()
        mock_plt.figure.return_value = mock_figure
        mock_subplot = Mock()
        mock_plt.subplot.return_value = mock_subplot

        # Mock bar chart
        mock_bar = Mock()
        mock_bar.get_height.return_value = 0.0
        mock_bar.get_x.return_value = 0.0
        mock_bar.get_width.return_value = 1.0
        mock_subplot.bar.return_value = [mock_bar, mock_bar, mock_bar, mock_bar]
        mock_subplot.text.return_value = Mock()

        mock_plt.suptitle = Mock()
        mock_plt.tight_layout = Mock()
        mock_plt.close = Mock()

        # Act & Assert - Should not raise an error
        create_performance_summary_plot(metrics=metrics, save_path=None, log_to_wandb=False)

        mock_plt.close.assert_called_once()

    @patch("src.visualization.plt")
    @patch("src.visualization.sns")
    def test_empty_error_analysis(self, mock_sns: Mock, mock_plt: Mock) -> None:
        """Test plotting with empty error analysis."""
        # Arrange
        error_analysis: dict[str, Any] = {}

        # Mock pyplot
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
        mock_plt.tight_layout = Mock()
        mock_plt.close = Mock()

        # Act & Assert - Should not raise an error
        plot_error_distribution(error_analysis=error_analysis, save_path=None, log_to_wandb=False)

        mock_plt.close.assert_called_once()
