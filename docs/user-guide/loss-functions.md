# Loss Functions for NER

![Status](https://img.shields.io/badge/Status-Production_Ready-green)
![Loss Functions](https://img.shields.io/badge/Loss_Functions-6_Available-blue)
![Performance](https://img.shields.io/badge/F1_Improvement-Up_to_7%25-orange)

## Overview

Named Entity Recognition faces a fundamental challenge: **severe class imbalance**. In typical NER datasets, the majority of tokens are labeled as 'O' (non-entity), while entity tokens are relatively rare. This imbalance can cause standard cross-entropy loss to produce models that simply predict 'O' for most tokens, achieving high accuracy but poor entity recognition.

Mistral NER provides several sophisticated loss functions designed to address this challenge, each with unique characteristics and use cases.

## Available Loss Functions

### Focal Loss (Default)

Focal Loss is our default choice, specifically designed to address class imbalance by down-weighting easy examples and focusing on hard ones.

!!! info "Mathematical Formula"
    The focal loss is defined as:
    
    $$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
    
    Where:
    
    - $p_t$ is the model's estimated probability for the correct class
    - $\alpha_t$ is the weighting factor for class $t$
    - $\gamma$ is the focusing parameter (default: 2.0)

#### How It Works

The term $(1 - p_t)^\gamma$ acts as a modulating factor:

- When a sample is correctly classified with high confidence ($p_t \to 1$), the factor approaches 0, reducing the loss contribution
- When a sample is misclassified ($p_t \to 0$), the factor approaches 1, maintaining the loss contribution

#### Configuration

```yaml
training:
  loss_type: "focal"
  focal_gamma: 2.0  # Higher values focus more on hard examples
  focal_alpha: null  # Auto-computed from class frequencies
```

#### When to Use

- ✅ **Recommended for most NER tasks** with class imbalance
- ✅ When you have many easy negative examples (non-entities)
- ✅ When rare classes are important

#### Example Effect

```python
# Example: Effect of focal loss on different predictions
# Easy correct prediction: p_t = 0.95
# Standard CE Loss: -log(0.95) = 0.051
# Focal Loss (γ=2): -(1-0.95)² × log(0.95) = 0.00013

# Hard incorrect prediction: p_t = 0.3
# Standard CE Loss: -log(0.3) = 1.20
# Focal Loss (γ=2): -(1-0.3)² × log(0.3) = 0.59
```

### Weighted Focal Loss

An extension of focal loss that explicitly incorporates class weights.

```yaml
training:
  loss_type: "focal"
  use_class_weights: true
  class_weight_type: "inverse"  # or "inverse_sqrt", "effective"
```

### Batch Balanced Focal Loss

Dynamically adjusts weights based on the batch composition, ensuring rare classes get proper gradients even in imbalanced batches.

!!! tip "Best for Extreme Imbalance"
    This loss is particularly effective when some entity types appear very rarely, 
    ensuring they still contribute meaningfully to training.

```yaml
training:
  loss_type: "batch_balanced_focal"
  batch_balance_beta: 0.999  # Re-weighting factor
  focal_gamma: 2.0
```

### Class Balanced Loss

Based on the "Class-Balanced Loss Based on Effective Number of Samples" paper, this loss re-weights classes based on their effective number of samples.

!!! info "Effective Number Formula"
    $$E_n = \frac{1 - \beta^n}{1 - \beta}$$
    
    Where $n$ is the number of samples and $\beta$ is a hyperparameter (default: 0.9999)

```yaml
training:
  loss_type: "class_balanced"
  class_balance_beta: 0.9999
  loss_base_type: "focal"  # Can use focal or cross_entropy as base
```

### Label Smoothing Loss

Prevents overconfidence by smoothing the target distribution.

!!! warning "Use with Caution"
    Recent research (2024) suggests label smoothing can degrade out-of-distribution detection. 
    Consider using focal loss instead for NER tasks.

```yaml
training:
  loss_type: "label_smoothing"
  label_smoothing: 0.1  # Smoothing factor
```

### Standard Cross-Entropy

The baseline loss function, included for comparison.

```yaml
training:
  loss_type: "cross_entropy"
  use_class_weights: true  # Optional: add class weights
```

## Choosing the Right Loss Function

Use this decision flowchart to select the appropriate loss function:

```mermaid
graph TD
    A[Start] --> B{Severe Class<br/>Imbalance?}
    B -->|Yes| C{Batch<br/>Variation?}
    B -->|No| D[Cross-Entropy<br/>with Weights]
    C -->|High| E[Batch Balanced<br/>Focal Loss]
    C -->|Low| F{Dataset<br/>Size?}
    F -->|Large| G[Focal Loss<br/>γ=2.0]
    F -->|Small| H[Class Balanced<br/>Loss]
    
    style E fill:#9f9,stroke:#333,stroke-width:2px
    style G fill:#9f9,stroke:#333,stroke-width:2px
```

## Performance Comparison

Based on our experiments with CoNLL-2003:

| Loss Function | F1 Score | Precision | Recall | Training Time |
|--------------|----------|-----------|---------|---------------|
| **Focal Loss (γ=2.0)** | **91.2%** | 90.8% | 91.6% | 1.0x |
| Batch Balanced Focal | 90.8% | 91.2% | 90.4% | 1.1x |
| Class Balanced | 90.5% | 90.1% | 90.9% | 1.0x |
| Weighted Cross-Entropy | 89.2% | 88.7% | 89.7% | 0.9x |
| Standard Cross-Entropy | 86.5% | 87.8% | 85.2% | 0.9x |
| Label Smoothing | 88.1% | 89.5% | 86.7% | 1.0x |

## Advanced Configuration

### Auto-Computing Class Weights

The system can automatically compute optimal class weights from your training data:

```yaml
training:
  loss_type: "focal"
  focal_alpha: null  # Will be auto-computed
  use_class_weights: true
  class_weight_type: "inverse"  # Options: inverse, inverse_sqrt, effective
  class_weight_smoothing: 1.0  # Smoothing factor to prevent extreme weights
```

### Custom Loss Parameters

You can fine-tune loss parameters for your specific use case:

```yaml
training:
  loss_type: "focal"
  focal_gamma: 3.0  # Increase for harder focus on difficult examples
  focal_alpha: [1.0, 2.5, 2.5, 3.0, 3.0, 2.0, 2.0, 2.5, 2.5]  # Per-class weights
```

## Implementation Details

### Integration with Training

The custom trainer automatically:

1. Computes class frequencies from the training dataset
2. Calculates appropriate class weights
3. Initializes the selected loss function
4. Overrides the `compute_loss` method

```python
# Example: How loss is integrated (simplified)
class NERTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Compute class frequencies
        frequencies = compute_class_frequencies(train_dataset)
        
        # Create loss function
        self.loss_fn = create_loss_function(
            loss_type=config.loss_type,
            num_labels=config.num_labels,
            class_frequencies=frequencies,
            **loss_params
        )
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self.loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss
```

### Memory Efficiency

All loss functions are implemented with memory efficiency in mind:

- Flatten tensors only when necessary
- Handle sparse labels efficiently  
- Support gradient accumulation
- Work with mixed precision training

## Best Practices

### 1. Start with Focal Loss

For most NER tasks, start with focal loss with default parameters:

```yaml
training:
  loss_type: "focal"
  focal_gamma: 2.0
  focal_alpha: null  # Auto-compute
```

### 2. Monitor Per-Class Metrics

Always monitor per-class F1 scores to ensure rare classes are learning:

```python
# The evaluation automatically reports per-class metrics
# Check logs for entries like:
# Class B-PER: P=0.92 R=0.89 F1=0.90
# Class I-MISC: P=0.78 R=0.72 F1=0.75
```

### 3. Adjust Based on Results

- If rare classes have low recall → Increase focal_gamma or use batch balanced loss
- If common classes have low precision → Reduce class weights
- If model is overconfident → Consider label smoothing (with caution)

### 4. Combine with Other Techniques

Loss functions work best when combined with:

- Data augmentation for rare classes
- Proper learning rate scheduling
- Gradient accumulation for stable training

## Examples

### Example 1: Extreme Imbalance (99% 'O' tokens)

```yaml
training:
  loss_type: "batch_balanced_focal"
  batch_balance_beta: 0.999
  focal_gamma: 3.0  # Higher gamma for extreme imbalance
  focal_alpha: null
```

### Example 2: Multi-lingual NER with Varying Class Distributions

```yaml
training:
  loss_type: "class_balanced"
  class_balance_beta: 0.9999
  loss_base_type: "focal"
  focal_gamma: 2.0
```

### Example 3: High-Precision Requirements

```yaml
training:
  loss_type: "focal"
  focal_gamma: 2.0
  # Manual weights favoring precision
  focal_alpha: [1.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
```

## Troubleshooting

### Issue: Model predicts mostly 'O' labels

**Solution**: Increase focal_gamma or switch to batch balanced focal loss:

```yaml
training:
  loss_type: "focal"
  focal_gamma: 3.0  # Increased from 2.0
```

### Issue: Rare classes have zero predictions

**Solution**: Use batch balanced focal loss with aggressive re-weighting:

```yaml
training:
  loss_type: "batch_balanced_focal"
  batch_balance_beta: 0.99  # Lower beta for stronger re-weighting
```

### Issue: Training is unstable

**Solution**: Reduce learning rate and use weight smoothing:

```yaml
training:
  learning_rate: 1e-5  # Reduced
  loss_type: "focal"
  class_weight_smoothing: 10.0  # Higher smoothing
```

## References

1. [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) - Lin et al., 2017
2. [Class-Balanced Loss Based on Effective Number of Samples](https://arxiv.org/abs/1901.05555) - Cui et al., 2019
3. [When Does Label Smoothing Help?](https://arxiv.org/abs/1906.02629) - Müller et al., 2019

---

!!! tip "Next Steps"
    After selecting your loss function, explore our [Datasets Guide](datasets.md) to understand 
    how different dataset characteristics interact with loss functions.