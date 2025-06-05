"""Custom loss functions for NER fine-tuning."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as functional

if TYPE_CHECKING:
    pass

logger = logging.getLogger("mistral_ner")


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in token classification.

    Paper: "Focal Loss for Dense Object Detection" (https://arxiv.org/abs/1708.02002)

    The focal loss applies a modulating term to the cross entropy loss in order to
    focus learning on hard negative examples. It is defined as:

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where:
    - alpha_t is a weighting factor for class t
    - gamma is the focusing parameter (gamma=0 reduces to standard cross-entropy)
    - p_t is the predicted probability for class t
    """

    def __init__(
        self,
        num_labels: int,
        alpha: float | list[float] | None = None,
        gamma: float = 2.0,
        ignore_index: int = -100,
        reduction: str = "mean",
    ) -> None:
        """
        Initialize Focal Loss.

        Args:
            num_labels: Number of classes
            alpha: Weighting factor for rare class (default: None for balanced)
            gamma: Focusing parameter, higher gamma puts more focus on hard examples
            ignore_index: Index to ignore in loss computation
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.num_labels = num_labels
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        # Set up alpha (class weighting)
        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, int | float):
            # Single alpha value for binary-like case
            self.alpha = torch.ones(num_labels) * alpha
        else:
            # List of alpha values per class
            self.alpha = torch.tensor(alpha, dtype=torch.float32)

        logger.info(f"Initialized Focal Loss with gamma={gamma}, alpha={alpha}, num_labels={num_labels}")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of focal loss.

        Args:
            inputs: Logits tensor of shape [batch_size, seq_len, num_labels] or [N, num_labels]
            targets: Target tensor of shape [batch_size, seq_len] or [N]

        Returns:
            Focal loss value
        """
        # Flatten inputs and targets for easier computation
        if inputs.dim() > 2:
            inputs = inputs.view(-1, self.num_labels)
            targets = targets.view(-1)

        # Create mask for valid targets (not ignore_index)
        valid_mask = targets != self.ignore_index

        if not valid_mask.any():
            # No valid targets
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)

        # Filter out ignored indices
        inputs_valid = inputs[valid_mask]
        targets_valid = targets[valid_mask]

        # Compute cross entropy loss
        ce_loss = functional.cross_entropy(inputs_valid, targets_valid, reduction="none")

        # Compute p_t (probability of correct class)
        pt = torch.exp(-ce_loss)

        # Apply alpha weighting if specified
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets_valid]
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedFocalLoss(FocalLoss):
    """
    Weighted Focal Loss that combines class weighting with focal loss.
    This is specifically designed for severe class imbalance in NER.
    """

    def __init__(
        self,
        num_labels: int,
        class_weights: list[float] | None = None,
        gamma: float = 2.0,
        ignore_index: int = -100,
        reduction: str = "mean",
    ) -> None:
        """
        Initialize Weighted Focal Loss.

        Args:
            num_labels: Number of classes
            class_weights: Class weights for handling imbalance
            gamma: Focusing parameter
            ignore_index: Index to ignore in loss computation
            reduction: Reduction method
        """
        # Convert class weights to alpha for focal loss
        super().__init__(
            num_labels=num_labels,
            alpha=class_weights,
            gamma=gamma,
            ignore_index=ignore_index,
            reduction=reduction,
        )
        logger.info("Initialized Weighted Focal Loss for severe class imbalance handling")


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss for token classification.

    Note: Based on 2024 research, label smoothing can degrade OOD detection.
    Use with caution and consider alternatives like focal loss for NER tasks.
    """

    def __init__(
        self,
        num_labels: int,
        smoothing: float = 0.1,
        ignore_index: int = -100,
    ) -> None:
        """
        Initialize Label Smoothing Loss.

        Args:
            num_labels: Number of classes
            smoothing: Smoothing factor (0.0 = no smoothing, standard cross-entropy)
            ignore_index: Index to ignore in loss computation
        """
        super().__init__()
        self.num_labels = num_labels
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing

        logger.info(f"Initialized Label Smoothing Loss with smoothing={smoothing}, num_labels={num_labels}")
        if smoothing > 0:
            logger.warning(
                "Label smoothing may degrade OOD detection according to 2024 research. "
                "Consider using focal loss instead for NER tasks."
            )

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of label smoothing loss.

        Args:
            inputs: Logits tensor of shape [batch_size, seq_len, num_labels] or [N, num_labels]
            targets: Target tensor of shape [batch_size, seq_len] or [N]

        Returns:
            Label smoothing loss value
        """
        # Flatten for easier computation
        if inputs.dim() > 2:
            inputs = inputs.view(-1, self.num_labels)
            targets = targets.view(-1)

        # Create mask for valid targets
        valid_mask = targets != self.ignore_index

        if not valid_mask.any():
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)

        # Filter valid inputs/targets
        inputs_valid = inputs[valid_mask]
        targets_valid = targets[valid_mask]

        # Create smooth labels
        smooth_labels = torch.full_like(inputs_valid, self.smoothing / (self.num_labels - 1))
        smooth_labels.scatter_(1, targets_valid.unsqueeze(1), self.confidence)

        # Compute loss
        log_probs = functional.log_softmax(inputs_valid, dim=-1)
        loss = -(smooth_labels * log_probs).sum(dim=-1)

        return loss.mean()


class BatchBalancedFocalLoss(nn.Module):
    """
    Batch-Balanced Focal Loss for extreme class imbalance.

    This loss dynamically adjusts class weights based on the batch composition,
    ensuring better gradient flow for rare classes even in imbalanced batches.
    """

    def __init__(
        self,
        num_labels: int,
        gamma: float = 2.0,
        base_alpha: float | list[float] | None = None,
        batch_balance_beta: float = 0.999,
        ignore_index: int = -100,
        reduction: str = "mean",
    ) -> None:
        """
        Initialize Batch-Balanced Focal Loss.

        Args:
            num_labels: Number of classes
            gamma: Focal loss focusing parameter
            base_alpha: Base class weights (can be updated per batch)
            batch_balance_beta: Re-weighting factor for batch balancing
            ignore_index: Index to ignore in loss
            reduction: Reduction method
        """
        super().__init__()
        self.num_labels = num_labels
        self.gamma = gamma
        self.batch_balance_beta = batch_balance_beta
        self.ignore_index = ignore_index
        self.reduction = reduction

        # Initialize base focal loss
        self.focal_loss = FocalLoss(
            num_labels=num_labels,
            alpha=base_alpha,
            gamma=gamma,
            ignore_index=ignore_index,
            reduction="none",  # We'll handle reduction after re-weighting
        )

        logger.info(f"Initialized Batch-Balanced Focal Loss with gamma={gamma}, beta={batch_balance_beta}")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dynamic batch-based re-weighting.

        Args:
            inputs: Logits [batch_size, seq_len, num_labels]
            targets: Target labels [batch_size, seq_len]

        Returns:
            Batch-balanced focal loss value
        """
        # Get per-sample focal loss
        focal_losses = self.focal_loss(inputs, targets)

        # Compute batch statistics for re-weighting
        valid_mask = targets != self.ignore_index

        if not valid_mask.any():
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)

        # Flatten for easier computation
        targets_valid = targets[valid_mask]

        # Handle case where focal_losses is already reduced
        if focal_losses.dim() == 0:
            # If already reduced to scalar, re-compute without reduction
            focal_losses = self.focal_loss.forward(inputs, targets)

        # Flatten focal losses
        if focal_losses.dim() > 1:
            focal_losses = focal_losses.view(-1)

        focal_losses_valid = focal_losses[valid_mask.view(-1)]

        # Compute batch class frequencies
        batch_class_counts = torch.zeros(self.num_labels, device=targets.device)
        for i in range(self.num_labels):
            batch_class_counts[i] = (targets_valid == i).sum().float()

        # Compute effective numbers for batch
        batch_effective_num = 1.0 - torch.pow(self.batch_balance_beta, batch_class_counts)
        batch_weights = (1.0 - self.batch_balance_beta) / (batch_effective_num + 1e-8)

        # Normalize weights
        batch_weights = batch_weights / batch_weights.sum() * self.num_labels

        # Apply batch-based re-weighting
        if focal_losses_valid.dim() == 0:
            # Single element
            weight = batch_weights[targets_valid.item()]
            balanced_loss = focal_losses_valid * weight
        else:
            # Multiple elements
            sample_weights = batch_weights[targets_valid]
            balanced_loss = focal_losses_valid * sample_weights

        # Apply reduction
        if self.reduction == "mean":
            return balanced_loss.mean()
        elif self.reduction == "sum":
            return balanced_loss.sum()
        else:
            return balanced_loss


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss based on effective number of samples.

    Paper: "Class-Balanced Loss Based on Effective Number of Samples"
    Addresses class imbalance by re-weighting loss based on effective sample numbers.
    """

    def __init__(
        self,
        num_labels: int,
        class_frequencies: list[int],
        beta: float = 0.9999,
        ignore_index: int = -100,
        loss_type: str = "focal",  # 'focal', 'cross_entropy', 'sigmoid'
        **loss_kwargs: Any,
    ) -> None:
        """
        Initialize Class-Balanced Loss.

        Args:
            num_labels: Number of classes
            class_frequencies: List of sample counts per class
            beta: Re-weighting hyperparameter (0 = no re-weighting, 0.9999 = strong re-weighting)
            ignore_index: Index to ignore in loss computation
            loss_type: Type of base loss ('focal', 'cross_entropy')
            **loss_kwargs: Additional arguments for base loss
        """
        super().__init__()
        self.num_labels = num_labels
        self.beta = beta
        self.ignore_index = ignore_index

        # Calculate effective numbers and weights
        effective_num = 1.0 - torch.pow(beta, torch.tensor(class_frequencies, dtype=torch.float32))
        weights = (1.0 - beta) / effective_num
        self.class_weights = weights / weights.sum() * num_labels  # Normalize

        # Initialize base loss
        if loss_type == "focal":
            self.base_loss = FocalLoss(
                num_labels=num_labels,
                alpha=self.class_weights.tolist(),
                ignore_index=ignore_index,
                **loss_kwargs,
            )
        elif loss_type == "cross_entropy":
            self.base_loss = nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=ignore_index, **loss_kwargs)
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

        logger.info(f"Initialized Class-Balanced Loss with beta={beta}, loss_type={loss_type}")
        logger.info(f"Class weights: {self.class_weights.tolist()}")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass using the configured base loss."""
        return self.base_loss(inputs, targets)


def calculate_class_weights(
    class_frequencies: list[int],
    weight_type: str = "inverse",
    smoothing: float = 1.0,
) -> list[float]:
    """
    Calculate class weights from frequencies to handle imbalance.

    Args:
        class_frequencies: List of sample counts per class
        weight_type: Type of weighting ('inverse', 'inverse_sqrt', 'effective')
        smoothing: Smoothing factor to prevent extreme weights

    Returns:
        List of weights per class
    """
    import numpy as np

    frequencies = np.array(class_frequencies, dtype=np.float32)

    # Add smoothing to prevent division by zero
    frequencies = frequencies + smoothing

    if weight_type == "inverse":
        # Inverse frequency weighting
        weights = 1.0 / frequencies
    elif weight_type == "inverse_sqrt":
        # Inverse square root (less aggressive)
        weights = 1.0 / np.sqrt(frequencies)
    elif weight_type == "effective":
        # Effective number based weighting (similar to class-balanced loss)
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, frequencies)
        weights = (1.0 - beta) / effective_num
    else:
        raise ValueError(f"Unknown weight_type: {weight_type}")

    # Normalize weights to sum to number of classes
    weights = weights / weights.sum() * len(weights)

    weights_list: list[float] = weights.tolist()
    logger.info(f"Calculated class weights ({weight_type}): {weights_list}")
    return weights_list


def compute_class_frequencies(dataset: Any, label_column: str = "labels") -> list[int]:
    """
    Compute class frequencies from a dataset.

    Args:
        dataset: Dataset with labels
        label_column: Name of the label column

    Returns:
        List of frequencies per class
    """
    from collections import Counter

    # Flatten all labels and count frequencies
    all_labels = []
    for example in dataset:
        labels = example[label_column]
        if isinstance(labels, torch.Tensor):
            labels = labels.tolist()
        # Filter out ignore_index (-100)
        valid_labels = [label for label in labels if label != -100]
        all_labels.extend(valid_labels)

    # Count frequencies
    label_counts = Counter(all_labels)

    # Create frequency list (ensure all classes are represented)
    max_label = max(label_counts.keys()) if label_counts else 0
    frequencies = []
    for i in range(max_label + 1):
        frequencies.append(label_counts.get(i, 1))  # Default to 1 to avoid division by zero

    logger.info(f"Computed class frequencies: {frequencies}")
    return frequencies


def create_loss_function(
    loss_type: str,
    num_labels: int,
    class_frequencies: list[int] | None = None,
    class_weights: list[float] | None = None,
    auto_weight: bool = False,
    weight_type: str = "inverse",
    smoothing: float = 1.0,
    **kwargs: Any,
) -> nn.Module:
    """
    Factory function to create loss functions.

    Args:
        loss_type: Type of loss ('focal', 'label_smoothing', 'class_balanced', 'cross_entropy', 'weighted_cross_entropy')
        num_labels: Number of classes
        class_frequencies: Class frequencies for auto-weight calculation
        class_weights: Manual class weights (overrides auto_weight)
        auto_weight: Automatically calculate class weights from frequencies
        **kwargs: Additional arguments for specific loss functions

    Returns:
        Configured loss function
    """
    # Calculate class weights if requested
    if auto_weight and class_weights is None and class_frequencies is not None:
        class_weights = calculate_class_weights(class_frequencies, weight_type=weight_type, smoothing=smoothing)
        logger.info(f"Auto-calculated class weights: {class_weights}")

    if loss_type == "focal":
        # For focal loss, weights are passed as alpha parameter
        if class_weights is not None and "alpha" not in kwargs:
            kwargs["alpha"] = class_weights
        return FocalLoss(num_labels=num_labels, **kwargs)
    elif loss_type == "label_smoothing":
        return LabelSmoothingLoss(num_labels=num_labels, **kwargs)
    elif loss_type == "class_balanced":
        if class_frequencies is None:
            raise ValueError("class_frequencies required for class_balanced loss")
        return ClassBalancedLoss(num_labels=num_labels, class_frequencies=class_frequencies, **kwargs)
    elif loss_type == "cross_entropy" or loss_type == "weighted_cross_entropy":
        if class_weights is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
            return nn.CrossEntropyLoss(weight=weight_tensor, **kwargs)
        return nn.CrossEntropyLoss(**kwargs)
    elif loss_type == "batch_balanced_focal":
        # For batch-balanced focal loss
        return BatchBalancedFocalLoss(num_labels=num_labels, base_alpha=class_weights, **kwargs)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")
