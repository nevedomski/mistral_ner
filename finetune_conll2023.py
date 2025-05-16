import evaluate
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, Trainer, TrainingArguments

# 1. Load dataset
dataset = load_dataset("conll2003")

# 2. Load tokenizer and model
model_checkpoint = "mistralai/Mistral-7B-v0.3"  # Replace with your checkpoint path if local
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token for Mistral
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint, num_labels=dataset["train"].features["ner_tags"].feature.num_classes
)

label_list = dataset["train"].features["ner_tags"].feature.names


# 3. Tokenize and align labels
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if True else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True, remove_columns=dataset["train"].column_names)

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
training_args = TrainingArguments(
    output_dir="./mistral-ner-finetuned",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,  # Adjust as needed for multi-GPU
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    report_to="none",
    fp16=True,
    bf16=False,
    push_to_hub=False,
    # Multi-GPU is handled automatically when launched with accelerate/torchrun
)

# 7. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 8. Train and evaluate
trainer.train()
metrics = trainer.evaluate()
print(metrics)
