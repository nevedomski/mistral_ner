#!/usr/bin/env python3
"""Inference script for Mistral NER model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from argparse import Namespace
    from transformers import PreTrainedModel, PreTrainedTokenizerBase
    from src.config import Config

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from transformers import AutoTokenizer

from src.config import Config
from src.model import load_model_from_checkpoint, setup_model
from src.utils import setup_logging


def parse_args() -> Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned Mistral NER model")

    parser.add_argument("--model-path", type=str, required=True, help="Path to fine-tuned model checkpoint")
    parser.add_argument(
        "--base-model",
        type=str,
        default="mistralai/Mistral-7B-v0.3",
        help="Base model name (if loading LoRA checkpoint)",
    )
    parser.add_argument("--text", type=str, help="Text to run inference on")
    parser.add_argument("--file", type=str, help="File containing text to run inference on")
    parser.add_argument("--output", type=str, help="Output file for predictions")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on"
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to configuration file")

    return parser.parse_args()


def load_model_for_inference(
    model_path: str, base_model: str, config: Config
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load model for inference."""
    logger = setup_logging()

    # Check if it's a full model or LoRA checkpoint
    model_path_obj = Path(model_path)
    is_lora = (model_path_obj / "adapter_config.json").exists()

    if is_lora:
        logger.info(f"Loading LoRA model from {model_path}")
        model, tokenizer = load_model_from_checkpoint(
            checkpoint_path=model_path, config=config, base_model_name=base_model
        )
    else:
        logger.info(f"Loading full model from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model, _ = setup_model(model_name=model_path, config=config)

    return model, tokenizer


def predict_entities(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    label_names: list[str],
    device: str = "cuda",
    batch_size: int = 1,
) -> list[dict[str, Any]]:
    """
    Predict entities in texts.

    Returns:
        List of dictionaries with tokens and predicted labels
    """
    model.eval()
    model.to(device)

    all_predictions = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        # Tokenize
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=256, return_tensors="pt")

        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)

        # Process predictions
        for j, text in enumerate(batch_texts):
            token_predictions = predictions[j].cpu().tolist()

            # Get word ids to align predictions
            encoding = tokenizer(text, truncation=True, max_length=256)
            word_ids = encoding.word_ids()

            # Align predictions with words
            word_labels = []
            previous_word_idx = None

            for idx, word_idx in enumerate(word_ids):
                if word_idx is None:
                    continue

                if word_idx != previous_word_idx and idx < len(token_predictions):
                    label_id = token_predictions[idx]
                    if label_id < len(label_names):
                        word_labels.append(label_names[label_id])

                previous_word_idx = word_idx

            # Get original words
            words = text.split()

            # Ensure alignment
            if len(words) != len(word_labels):
                # Simple fallback: pad with 'O'
                word_labels = word_labels[: len(words)]
                word_labels.extend(["O"] * (len(words) - len(word_labels)))

            all_predictions.append(
                {"text": text, "words": words, "labels": word_labels, "entities": extract_entities(words, word_labels)}
            )

    return all_predictions


def extract_entities(words: list[str], labels: list[str]) -> list[dict[str, Any]]:
    """Extract entities from BIO labels."""
    entities = []
    current_entity = None

    for i, (word, label) in enumerate(zip(words, labels, strict=False)):
        if label.startswith("B-"):
            # Start new entity
            if current_entity:
                entities.append(current_entity)

            entity_type = label[2:]
            current_entity = {"text": word, "type": entity_type, "start": i, "end": i + 1}

        elif label.startswith("I-") and current_entity:
            # Continue entity
            current_entity["text"] += " " + word
            current_entity["end"] = i + 1

        else:
            # End entity
            if current_entity:
                entities.append(current_entity)
                current_entity = None

    # Don't forget last entity
    if current_entity:
        entities.append(current_entity)

    return entities


def format_output(predictions: list[dict[str, Any]], format_type: str = "inline") -> str:
    """Format predictions for output."""
    output_lines = []

    for pred in predictions:
        if format_type == "inline":
            # Inline format: word/LABEL
            tokens_with_labels = []
            for word, label in zip(pred["words"], pred["labels"], strict=False):
                if label != "O":
                    tokens_with_labels.append(f"{word}/{label}")
                else:
                    tokens_with_labels.append(word)

            output_lines.append(" ".join(tokens_with_labels))

        elif format_type == "conll":
            # CoNLL format
            for word, label in zip(pred["words"], pred["labels"], strict=False):
                output_lines.append(f"{word}\t{label}")
            output_lines.append("")  # Empty line between sentences

        elif format_type == "json":
            # JSON format with entities
            import json

            output_lines.append(json.dumps({"text": pred["text"], "entities": pred["entities"]}, indent=2))

    return "\n".join(output_lines)


def main() -> None:
    """Main inference function."""
    args = parse_args()

    # Setup logging
    logger = setup_logging()

    # Load config
    config = Config.from_yaml(args.config)

    # Load model
    logger.info("Loading model...")
    model, tokenizer = load_model_for_inference(model_path=args.model_path, base_model=args.base_model, config=config)

    # Get input texts
    texts = []
    if args.text:
        texts.append(args.text)
    elif args.file:
        with open(args.file) as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        # Interactive mode
        logger.info("Enter text for NER (Ctrl+D to finish):")
        try:
            while True:
                text = input("> ")
                if text.strip():
                    texts.append(text.strip())
        except EOFError:
            pass

    if not texts:
        logger.error("No input text provided")
        sys.exit(1)

    # Run inference
    logger.info(f"Running inference on {len(texts)} texts...")
    predictions = predict_entities(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        label_names=config.data.label_names,
        device=args.device,
        batch_size=args.batch_size,
    )

    # Format and output results
    output = format_output(predictions, format_type="inline")

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        logger.info(f"Results saved to {args.output}")
    else:
        print("\n" + "=" * 50)
        print("PREDICTIONS")
        print("=" * 50)
        print(output)
        print("=" * 50)

        # Also print entities
        print("\nEXTRACTED ENTITIES:")
        for pred in predictions:
            if pred["entities"]:
                print(f"\nText: {pred['text']}")
                for entity in pred["entities"]:
                    print(f"  - {entity['text']} [{entity['type']}]")


if __name__ == "__main__":
    main()
