import evaluate
import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    BitsAndBytesConfig,  # Add import for BitsAndBytesConfig
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

# 1. Load dataset
dataset = load_dataset("conll2003")

# 2. Load tokenizer and model
model_checkpoint = "mistralai/Mistral-7B-v0.3"  # Replace with your checkpoint path if local
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

# Set up quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,  # Changed from 4-bit to 8-bit quantization
    llm_int8_threshold=6.0,  # Threshold for outlier features in 8-bit quantization
    llm_int8_has_fp16_weight=False,  # Keep weights in pure int8
)

# Load base model with quantization for more memory efficiency
base_model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=dataset["train"].features["ner_tags"].feature.num_classes,
    quantization_config=quantization_config,
    device_map="auto",
)
base_model.gradient_checkpointing_enable()

# Configure LoRA for parameter-efficient fine-tuning
peft_config = LoraConfig(
    task_type=TaskType.TOKEN_CLS,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

# Apply LoRA to the model
model = get_peft_model(base_model, peft_config)
print(f"Trainable parameters: {model.print_trainable_parameters()}")

label_list = dataset["train"].features["ner_tags"].feature.names


# 3. Tokenize and align labels
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, max_length=256)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have word_idx = None
            if word_idx is None:
                label_ids.append(-100)
            # First token of a word
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # Subword tokens: assign -100 to ignore in loss calculation
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


tokenized_datasets = dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=dataset["train"].column_names,
    num_proc=4,  # Speed up processing with multiple cores
)

# 4. Data collator
data_collator = DataCollatorForTokenClassification(tokenizer)

# 5. Metrics
metric = evaluate.load("seqeval")


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
    true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


# 6. Training arguments
has_cuda = torch.cuda.is_available()
use_bf16 = has_cuda and torch.cuda.is_bf16_supported()

training_args = TrainingArguments(
    output_dir="./mistral-ner-finetuned",
    eval_strategy="steps",  # Changed to eval_strategy for compatibility with your version
    eval_steps=500,  # Reduced frequency of evaluation to speed up training
    save_strategy="steps",
    save_steps=500,
    learning_rate=5e-5,
    per_device_train_batch_size=2,  # Reduced batch size
    per_device_eval_batch_size=2,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    report_to="none",
    fp16=has_cuda and not use_bf16,  # FP16 only works on CUDA GPUs, not on CPU
    bf16=use_bf16,  # BF16 only works with specific GPU support
    # On CPU, both will be False and training will use FP32
    push_to_hub=False,
    gradient_accumulation_steps=8,  # Effectively increases batch size to 16
    warmup_ratio=0.1,  # Warm up learning rate for first 10% of steps
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
)


# Add memory optimization for training
class GradientCheckpointingCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        # Clear CUDA cache periodically to prevent OOM
        if state.global_step % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()


# 7. Trainer with additional callbacks
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3), GradientCheckpointingCallback()],
)

# 8. Train and evaluate
trainer.train()
metrics = trainer.evaluate()
print(metrics)

# 9. Save the final model
model.save_pretrained("./mistral-ner-finetuned-final")
tokenizer.save_pretrained("./mistral-ner-finetuned-final")
