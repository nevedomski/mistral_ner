"""Evaluation metrics and functions for NER."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import wandb
from seqeval.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from transformers import EvalPrediction, PreTrainedModel, PreTrainedTokenizerBase

from datasets import Dataset

if TYPE_CHECKING:
    from transformers import DataCollatorForTokenClassification

logger = logging.getLogger("mistral_ner")


def align_predictions(
    predictions: np.ndarray[Any, Any], label_ids: np.ndarray[Any, Any], label_names: list[str]
) -> tuple[list[list[str]], list[list[str]]]:
    """
    Align predictions with labels, removing padding and special tokens.

    Args:
        predictions: Predicted label indices
        label_ids: True label indices
        label_names: List of label names

    Returns:
        Tuple of (true_labels, predicted_labels) as lists of label strings
    """
    preds = np.argmax(predictions, axis=2)

    batch_size, seq_len = preds.shape
    true_labels = []
    pred_labels = []

    for i in range(batch_size):
        true_label_seq = []
        pred_label_seq = []

        for j in range(seq_len):
            if label_ids[i, j] != -100:  # Skip padding and special tokens
                true_label_seq.append(label_names[label_ids[i, j]])
                pred_label_seq.append(label_names[preds[i, j]])

        if true_label_seq:  # Only add non-empty sequences
            true_labels.append(true_label_seq)
            pred_labels.append(pred_label_seq)

    return true_labels, pred_labels


def compute_metrics_factory(label_names: list[str]) -> Callable[[EvalPrediction], dict[str, float]]:
    """
    Factory function to create compute_metrics function with label names.

    Args:
        label_names: List of label names

    Returns:
        compute_metrics function for trainer
    """

    def compute_metrics(p: EvalPrediction) -> dict[str, float]:
        """Compute NER metrics."""
        predictions, labels = p

        # Align predictions and labels
        true_labels, pred_labels = align_predictions(predictions, labels, label_names)

        # Calculate metrics using seqeval directly
        results = {
            "precision": precision_score(true_labels, pred_labels),
            "recall": recall_score(true_labels, pred_labels),
            "f1": f1_score(true_labels, pred_labels),
            "accuracy": accuracy_score(true_labels, pred_labels),
        }

        # Add per-entity metrics
        report = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)

        # Add entity-specific F1 scores
        for entity, metrics in report.items():
            if isinstance(metrics, dict) and entity not in ["accuracy", "macro avg", "weighted avg"]:
                results[f"{entity}_f1"] = metrics.get("f1-score", 0.0)
                results[f"{entity}_precision"] = metrics.get("precision", 0.0)
                results[f"{entity}_recall"] = metrics.get("recall", 0.0)

        # Log to wandb if available
        if wandb.run is not None:
            wandb.log({f"eval/{k}": v for k, v in results.items()})

        return results

    return compute_metrics


def evaluate_model(
    model: PreTrainedModel,
    eval_dataset: Dataset,
    data_collator: DataCollatorForTokenClassification,
    tokenizer: PreTrainedTokenizerBase,
    label_names: list[str],
    batch_size: int = 8,
) -> dict[str, float]:
    """
    Evaluate model on a dataset.

    Args:
        model: The model to evaluate
        eval_dataset: Evaluation dataset
        data_collator: Data collator
        tokenizer: Tokenizer
        label_names: List of label names
        batch_size: Batch size for evaluation

    Returns:
        Dictionary of metrics
    """
    from torch.utils.data import DataLoader

    logger.info("Starting model evaluation...")

    # Create dataloader
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=data_collator, shuffle=False)

    model.eval()
    all_predictions = []
    all_labels = []

    device = next(model.parameters()).device

    with torch.no_grad():
        for batch_raw in eval_dataloader:
            # Move batch to device
            batch: dict[str, Any] = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_raw.items()
            }

            # Forward pass
            outputs = model(**batch)
            predictions = outputs.logits.detach().cpu().numpy()
            labels = batch["labels"].detach().cpu().numpy()

            all_predictions.append(predictions)
            all_labels.append(labels)

    # Concatenate all batches
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Compute metrics
    eval_prediction = EvalPrediction(predictions=all_predictions, label_ids=all_labels)
    compute_metrics = compute_metrics_factory(label_names)
    metrics = compute_metrics(eval_prediction)

    logger.info(f"Evaluation metrics: {metrics}")

    return metrics


def compute_detailed_metrics(
    true_labels: list[list[str]],
    pred_labels: list[list[str]],
    label_names: list[str],
) -> dict[str, Any]:
    """
    Compute detailed per-entity metrics and error analysis.

    Args:
        true_labels: True label sequences
        pred_labels: Predicted label sequences
        label_names: List of all possible labels

    Returns:
        Dictionary with detailed metrics and analysis
    """
    from collections import defaultdict

    from seqeval.metrics import classification_report

    # Get standard classification report
    report = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)

    # Initialize detailed metrics
    detailed_metrics = {
        "overall": {
            "precision": report.get("macro avg", {}).get("precision", 0),
            "recall": report.get("macro avg", {}).get("recall", 0),
            "f1": report.get("macro avg", {}).get("f1-score", 0),
            "support": report.get("macro avg", {}).get("support", 0),
        },
        "per_entity_type": {},
        "confusion_matrix": defaultdict(lambda: defaultdict(int)),
        "error_analysis": {
            "false_positives": defaultdict(list),
            "false_negatives": defaultdict(list),
            "boundary_errors": defaultdict(int),
            "type_confusion": defaultdict(lambda: defaultdict(int)),
        },
    }

    # Group entities by type (PER, ORG, LOC, MISC, etc.)
    entity_types = defaultdict(list)
    for label in label_names:
        if label != "O" and "-" in label:
            entity_type = label.split("-")[1]
            entity_types[entity_type].append(label)

    # Compute per-entity-type metrics
    for entity_type, type_labels in entity_types.items():
        type_metrics = {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "support": 0,
            "b_tag_accuracy": 0.0,
            "i_tag_accuracy": 0.0,
        }

        # Aggregate metrics for this entity type
        total_support = 0
        weighted_precision = 0.0
        weighted_recall = 0.0
        weighted_f1 = 0.0

        for label in type_labels:
            if label in report:
                label_metrics = report[label]
                support = label_metrics.get("support", 0)
                total_support += support

                weighted_precision += label_metrics.get("precision", 0) * support
                weighted_recall += label_metrics.get("recall", 0) * support
                weighted_f1 += label_metrics.get("f1-score", 0) * support

                # Track B vs I tag accuracy
                if label.startswith("B-"):
                    type_metrics["b_tag_accuracy"] = label_metrics.get("f1-score", 0)
                elif label.startswith("I-"):
                    type_metrics["i_tag_accuracy"] = label_metrics.get("f1-score", 0)

        if total_support > 0:
            type_metrics["precision"] = weighted_precision / total_support
            type_metrics["recall"] = weighted_recall / total_support
            type_metrics["f1"] = weighted_f1 / total_support
            type_metrics["support"] = total_support

        detailed_metrics["per_entity_type"][entity_type] = type_metrics

    # Error analysis
    for true_seq, pred_seq in zip(true_labels, pred_labels, strict=False):
        for i, (true_label, pred_label) in enumerate(zip(true_seq, pred_seq, strict=False)):
            if true_label != pred_label:
                # Update confusion matrix
                detailed_metrics["confusion_matrix"][true_label][pred_label] += 1

                # Analyze error types
                if true_label == "O" and pred_label != "O":
                    # False positive
                    detailed_metrics["error_analysis"]["false_positives"][pred_label].append(i)
                elif true_label != "O" and pred_label == "O":
                    # False negative
                    detailed_metrics["error_analysis"]["false_negatives"][true_label].append(i)
                elif true_label != "O" and pred_label != "O":
                    # Type confusion or boundary error
                    true_bio, true_type = true_label.split("-") if "-" in true_label else ("O", "O")
                    pred_bio, pred_type = pred_label.split("-") if "-" in pred_label else ("O", "O")

                    if true_type != pred_type:
                        # Type confusion (e.g., PER predicted as ORG)
                        detailed_metrics["error_analysis"]["type_confusion"][true_type][pred_type] += 1
                    else:
                        # Boundary error (B/I mismatch)
                        detailed_metrics["error_analysis"]["boundary_errors"][true_type] += 1

    return detailed_metrics


def create_enhanced_compute_metrics(
    label_names: list[str],
    compute_detailed: bool = True,
) -> Callable[[EvalPrediction], dict[str, float]]:
    """
    Create enhanced compute_metrics function with detailed per-entity analysis.

    Args:
        label_names: List of label names
        compute_detailed: Whether to compute detailed metrics

    Returns:
        Enhanced compute_metrics function
    """
    base_compute_metrics = compute_metrics_factory(label_names)

    def enhanced_compute_metrics(p: EvalPrediction) -> dict[str, float]:
        """Compute enhanced NER metrics with detailed analysis."""
        # Get base metrics
        metrics = base_compute_metrics(p)

        if compute_detailed:
            # Get aligned labels for detailed analysis
            predictions, labels = p
            true_labels, pred_labels = align_predictions(predictions, labels, label_names)

            # Compute detailed metrics
            detailed = compute_detailed_metrics(true_labels, pred_labels, label_names)

            # Add per-entity-type metrics to results
            for entity_type, type_metrics in detailed["per_entity_type"].items():
                metrics[f"{entity_type}_precision"] = type_metrics["precision"]
                metrics[f"{entity_type}_recall"] = type_metrics["recall"]
                metrics[f"{entity_type}_f1"] = type_metrics["f1"]
                metrics[f"{entity_type}_support"] = type_metrics["support"]

                # Add B/I tag accuracy if available
                if type_metrics["b_tag_accuracy"] > 0:
                    metrics[f"{entity_type}_b_accuracy"] = type_metrics["b_tag_accuracy"]
                if type_metrics["i_tag_accuracy"] > 0:
                    metrics[f"{entity_type}_i_accuracy"] = type_metrics["i_tag_accuracy"]

            # Add error analysis summary
            total_errors = sum(sum(conf.values()) for conf in detailed["confusion_matrix"].values())
            if total_errors > 0:
                metrics["error_rate"] = total_errors / sum(len(seq) for seq in true_labels)

                # Add type confusion rate
                total_type_confusion = sum(
                    sum(confusion.values()) for confusion in detailed["error_analysis"]["type_confusion"].values()
                )
                metrics["type_confusion_rate"] = total_type_confusion / total_errors

                # Add boundary error rate
                total_boundary_errors = sum(detailed["error_analysis"]["boundary_errors"].values())
                metrics["boundary_error_rate"] = total_boundary_errors / total_errors

        return metrics

    return enhanced_compute_metrics


def print_evaluation_report(predictions: list[list[str]], labels: list[list[str]], label_names: list[str]) -> None:
    """Print detailed evaluation report."""
    report = classification_report(labels, predictions, digits=4, zero_division=0)

    print("\n" + "=" * 80)
    print("EVALUATION REPORT")
    print("=" * 80)
    print(report)
    print("=" * 80)

    # Print overall metrics
    print("\nOverall Metrics:")
    print(f"  Precision: {precision_score(labels, predictions):.4f}")
    print(f"  Recall: {recall_score(labels, predictions):.4f}")
    print(f"  F1-Score: {f1_score(labels, predictions):.4f}")
    print(f"  Accuracy: {accuracy_score(labels, predictions):.4f}")
    print("=" * 80 + "\n")


def save_predictions(
    predictions: list[list[str]], labels: list[list[str]], tokens: list[list[str]], output_file: str
) -> None:
    """
    Save predictions to a file in CoNLL format.

    Args:
        predictions: Predicted labels
        labels: True labels
        tokens: Original tokens
        output_file: Path to output file
    """
    with open(output_file, "w") as f:
        for tok_seq, true_seq, pred_seq in zip(tokens, labels, predictions, strict=False):
            for token, true_label, pred_label in zip(tok_seq, true_seq, pred_seq, strict=False):
                f.write(f"{token}\t{true_label}\t{pred_label}\n")
            f.write("\n")  # Empty line between sentences

    logger.info(f"Predictions saved to {output_file}")


def compute_confusion_matrix(
    predictions: list[list[str]], labels: list[list[str]], label_names: list[str]
) -> np.ndarray[Any, Any]:
    """Compute confusion matrix for NER predictions."""
    from sklearn.metrics import confusion_matrix

    # Flatten predictions and labels
    flat_predictions = [label for seq in predictions for label in seq]
    flat_labels = [label for seq in labels for label in seq]

    # Compute confusion matrix
    cm = confusion_matrix(flat_labels, flat_predictions, labels=label_names)

    return cm  # type: ignore[no-any-return]


def log_confusion_matrix_to_wandb(confusion_matrix: np.ndarray[Any, Any], label_names: list[str]) -> None:
    """Log confusion matrix to Weights & Biases."""
    if wandb.run is not None:
        wandb.log(
            {
                "confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None, y_true=None, preds=None, class_names=label_names, matrix=confusion_matrix.tolist()
                )
            }
        )
