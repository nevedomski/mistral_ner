"""Visualization utilities for NER performance analysis."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt  # type: ignore[import-not-found]
import numpy as np
import seaborn as sns  # type: ignore[import-untyped]
import wandb

if TYPE_CHECKING:
    pass

logger = logging.getLogger("mistral_ner")


def plot_confusion_matrix(
    confusion_matrix: dict[str, dict[str, int]],
    label_names: list[str],
    save_path: str | None = None,
    log_to_wandb: bool = False,
) -> None:
    """
    Plot confusion matrix heatmap.

    Args:
        confusion_matrix: Confusion matrix as nested dict
        label_names: List of label names
        save_path: Path to save the plot
        log_to_wandb: Whether to log to wandb
    """
    # Convert to numpy array
    n_labels = len(label_names)
    matrix = np.zeros((n_labels, n_labels))

    label_to_idx = {label: i for i, label in enumerate(label_names)}

    for true_label, predictions in confusion_matrix.items():
        if true_label in label_to_idx:
            true_idx = label_to_idx[true_label]
            for pred_label, count in predictions.items():
                if pred_label in label_to_idx:
                    pred_idx = label_to_idx[pred_label]
                    matrix[true_idx, pred_idx] = count

    # Create plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        cbar_kws={"label": "Count"},
    )
    plt.title("NER Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if log_to_wandb and wandb.run is not None:
        wandb.log({"confusion_matrix": wandb.Image(plt)})

    plt.close()


def plot_per_entity_metrics(
    metrics: dict[str, float],
    save_path: str | None = None,
    log_to_wandb: bool = False,
) -> None:
    """
    Plot per-entity type metrics comparison.

    Args:
        metrics: Dictionary of metrics
        save_path: Path to save the plot
        log_to_wandb: Whether to log to wandb
    """
    # Extract entity types and their metrics
    entity_metrics: dict[str, dict[str, float]] = {}
    for key, value in metrics.items():
        if "_" in key and not key.startswith("eval_"):
            parts = key.split("_")
            if len(parts) >= 2 and parts[-1] in ["precision", "recall", "f1"]:
                entity_type = "_".join(parts[:-1])
                metric_type = parts[-1]

                if entity_type not in entity_metrics:
                    entity_metrics[entity_type] = {}
                entity_metrics[entity_type][metric_type] = value

    if not entity_metrics:
        logger.warning("No per-entity metrics found to plot")
        return

    # Prepare data for plotting
    entity_types = sorted(entity_metrics.keys())
    metric_types = ["precision", "recall", "f1"]

    x = np.arange(len(entity_types))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot bars for each metric type
    for i, metric in enumerate(metric_types):
        values = [entity_metrics[entity].get(metric, 0) for entity in entity_types]
        ax.bar(x + i * width, values, width, label=metric.capitalize())

    # Customize plot
    ax.set_xlabel("Entity Type")
    ax.set_ylabel("Score")
    ax.set_title("Per-Entity Type Performance Metrics")
    ax.set_xticks(x + width)
    ax.set_xticklabels(entity_types)
    ax.legend()
    ax.set_ylim(0, 1.0)

    # Add value labels on bars
    for i, metric in enumerate(metric_types):
        values = [entity_metrics[entity].get(metric, 0) for entity in entity_types]
        for j, v in enumerate(values):
            ax.text(j + i * width, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if log_to_wandb and wandb.run is not None:
        wandb.log({"per_entity_metrics": wandb.Image(plt)})

    plt.close()


def plot_error_distribution(
    error_analysis: dict[str, Any],
    save_path: str | None = None,
    log_to_wandb: bool = False,
) -> None:
    """
    Plot error distribution analysis.

    Args:
        error_analysis: Error analysis dictionary
        save_path: Path to save the plot
        log_to_wandb: Whether to log to wandb
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Error type distribution
    error_types = {
        "False Positives": sum(len(v) for v in error_analysis.get("false_positives", {}).values()),
        "False Negatives": sum(len(v) for v in error_analysis.get("false_negatives", {}).values()),
        "Type Confusion": sum(sum(v.values()) for v in error_analysis.get("type_confusion", {}).values()),
        "Boundary Errors": sum(error_analysis.get("boundary_errors", {}).values()),
    }

    if sum(error_types.values()) > 0:
        ax1.pie(
            error_types.values(),
            labels=error_types.keys(),
            autopct="%1.1f%%",
            startangle=90,
        )
        ax1.set_title("Error Type Distribution")
    else:
        ax1.text(0.5, 0.5, "No errors found", ha="center", va="center")
        ax1.set_title("Error Type Distribution")

    # Plot 2: Type confusion heatmap
    type_confusion = error_analysis.get("type_confusion", {})
    if type_confusion:
        # Get unique entity types
        entity_types = sorted(
            set(list(type_confusion.keys()) + [pred for true in type_confusion.values() for pred in true])
        )

        # Build confusion matrix
        n_types = len(entity_types)
        confusion = np.zeros((n_types, n_types))

        type_to_idx = {t: i for i, t in enumerate(entity_types)}

        for true_type, predictions in type_confusion.items():
            if true_type in type_to_idx:
                true_idx = type_to_idx[true_type]
                for pred_type, count in predictions.items():
                    if pred_type in type_to_idx:
                        pred_idx = type_to_idx[pred_type]
                        confusion[true_idx, pred_idx] = count

        # Plot heatmap
        sns.heatmap(
            confusion,
            annot=True,
            fmt=".0f",
            cmap="Reds",
            xticklabels=entity_types,
            yticklabels=entity_types,
            ax=ax2,
            cbar_kws={"label": "Count"},
        )
        ax2.set_title("Entity Type Confusion")
        ax2.set_xlabel("Predicted Type")
        ax2.set_ylabel("True Type")
    else:
        ax2.text(0.5, 0.5, "No type confusion", ha="center", va="center")
        ax2.set_title("Entity Type Confusion")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if log_to_wandb and wandb.run is not None:
        wandb.log({"error_distribution": wandb.Image(plt)})

    plt.close()


def create_performance_summary_plot(
    metrics: dict[str, float],
    save_path: str | None = None,
    log_to_wandb: bool = False,
) -> None:
    """
    Create a comprehensive performance summary plot.

    Args:
        metrics: Dictionary of all metrics
        save_path: Path to save the plot
        log_to_wandb: Whether to log to wandb
    """
    plt.figure(figsize=(15, 10))

    # Overall metrics subplot
    ax1 = plt.subplot(2, 2, 1)
    overall_metrics = {
        "Precision": metrics.get("precision", 0),
        "Recall": metrics.get("recall", 0),
        "F1": metrics.get("f1", 0),
        "Accuracy": metrics.get("accuracy", 0),
    }

    bars = ax1.bar(overall_metrics.keys(), overall_metrics.values())
    ax1.set_ylim(0, 1.0)
    ax1.set_title("Overall Performance Metrics")
    ax1.set_ylabel("Score")

    # Add value labels on bars
    for bar, (_, value) in zip(bars, overall_metrics.items(), strict=False):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, height + 0.01, f"{value:.3f}", ha="center", va="bottom")

    # Entity-specific F1 scores subplot
    ax2 = plt.subplot(2, 2, 2)
    entity_f1s = {}
    for key, value in metrics.items():
        if key.endswith("_f1") and not key.startswith("eval_") and key != "f1":
            entity_type = key.replace("_f1", "")
            entity_f1s[entity_type] = value

    if entity_f1s:
        sorted_entities = sorted(entity_f1s.items(), key=lambda x: x[1], reverse=True)
        entities, f1_scores = zip(*sorted_entities, strict=False)

        bars = ax2.bar(entities, f1_scores)
        ax2.set_ylim(0, 1.0)
        ax2.set_title("F1 Scores by Entity Type")
        ax2.set_ylabel("F1 Score")
        ax2.set_xlabel("Entity Type")
        plt.xticks(rotation=45)

        # Add value labels
        for bar, score in zip(bars, f1_scores, strict=False):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2.0, height + 0.01, f"{score:.3f}", ha="center", va="bottom")

    # Error analysis subplot
    ax3 = plt.subplot(2, 2, 3)
    error_metrics = {
        "Error Rate": metrics.get("error_rate", 0),
        "Type Confusion": metrics.get("type_confusion_rate", 0),
        "Boundary Errors": metrics.get("boundary_error_rate", 0),
    }

    if any(v > 0 for v in error_metrics.values()):
        bars = ax3.bar(error_metrics.keys(), error_metrics.values())
        ax3.set_title("Error Analysis")
        ax3.set_ylabel("Rate")

        # Add value labels
        for bar, (_, value) in zip(bars, error_metrics.items(), strict=False):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2.0, height + 0.001, f"{value:.3f}", ha="center", va="bottom")
    else:
        ax3.text(0.5, 0.5, "No error metrics available", ha="center", va="center")
        ax3.set_title("Error Analysis")

    # Support distribution subplot
    ax4 = plt.subplot(2, 2, 4)
    entity_supports = {}
    for key, value in metrics.items():
        if key.endswith("_support") and not key.startswith("eval_"):
            entity_type = key.replace("_support", "")
            entity_supports[entity_type] = value

    if entity_supports:
        sorted_supports = sorted(entity_supports.items(), key=lambda x: x[1], reverse=True)
        entities, supports = zip(*sorted_supports, strict=False)

        ax4.pie(supports, labels=entities, autopct="%1.1f%%", startangle=90)
        ax4.set_title("Entity Type Distribution in Dataset")
    else:
        ax4.text(0.5, 0.5, "No support data available", ha="center", va="center")
        ax4.set_title("Entity Type Distribution")

    plt.suptitle("NER Model Performance Summary", fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if log_to_wandb and wandb.run is not None:
        wandb.log({"performance_summary": wandb.Image(plt)})

    plt.close()
