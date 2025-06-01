"""Evaluation metrics and functions for NER."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import evaluate
import numpy as np
import torch
from datasets import Dataset
from seqeval.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from transformers import EvalPrediction, PreTrainedModel, PreTrainedTokenizerBase

import wandb

if TYPE_CHECKING:
    from evaluate import EvaluationModule
    from transformers import DataCollatorForTokenClassification

logger = logging.getLogger("mistral_ner")


def load_seqeval_metric() -> EvaluationModule | None:
    """Load the seqeval metric for NER evaluation."""
    try:
        return evaluate.load("seqeval")
    except Exception as e:
        logger.warning(f"Failed to load seqeval from evaluate library: {e}")
        # Fallback to direct seqeval usage
        return None


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


def compute_metrics_factory(
    label_names: list[str], seqeval_metric: EvaluationModule | None = None
) -> Callable[[EvalPrediction], dict[str, float]]:
    """
    Factory function to create compute_metrics function with label names.

    Args:
        label_names: List of label names
        seqeval_metric: Pre-loaded seqeval metric (optional)

    Returns:
        compute_metrics function for trainer
    """

    def compute_metrics(p: EvalPrediction) -> dict[str, float]:
        """Compute NER metrics."""
        predictions, labels = p

        # Align predictions and labels
        true_labels, pred_labels = align_predictions(predictions, labels, label_names)

        # Calculate metrics using seqeval
        results = {}

        if seqeval_metric is not None:
            # Use evaluate library's seqeval
            metric_results = seqeval_metric.compute(predictions=pred_labels, references=true_labels)
            results.update(
                {
                    "precision": metric_results["overall_precision"],
                    "recall": metric_results["overall_recall"],
                    "f1": metric_results["overall_f1"],
                    "accuracy": metric_results["overall_accuracy"],
                }
            )
        else:
            # Use seqeval directly
            results.update(
                {
                    "precision": precision_score(true_labels, pred_labels),
                    "recall": recall_score(true_labels, pred_labels),
                    "f1": f1_score(true_labels, pred_labels),
                    "accuracy": accuracy_score(true_labels, pred_labels),
                }
            )

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
