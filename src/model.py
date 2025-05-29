"""Model setup and configuration for Mistral NER fine-tuning."""

import gc
import logging
from typing import Any

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

logger = logging.getLogger("mistral_ner")


def create_bnb_config(load_in_8bit: bool = True) -> BitsAndBytesConfig:
    """Create BitsAndBytes configuration for model quantization."""
    if load_in_8bit:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
            bnb_8bit_quant_type="nf8",
            bnb_8bit_use_double_quant=True,
        )
        logger.info("Created 8-bit quantization config")
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        logger.info("Created 4-bit quantization config")

    return bnb_config


def create_lora_config(config: Any) -> LoraConfig:
    """Create LoRA configuration."""
    lora_config = LoraConfig(
        r=config.model.lora_r,
        lora_alpha=config.model.lora_alpha,
        target_modules=config.model.target_modules,
        lora_dropout=config.model.lora_dropout,
        bias=config.model.lora_bias,
        task_type=getattr(TaskType, config.model.task_type),
    )

    logger.info(
        f"Created LoRA config: r={config.model.lora_r}, "
        f"alpha={config.model.lora_alpha}, "
        f"target_modules={config.model.target_modules}"
    )

    return lora_config


def load_tokenizer(model_name: str) -> PreTrainedTokenizer:
    """Load and configure tokenizer."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, add_prefix_space=True)

        # Set padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Set pad_token to eos_token: {tokenizer.pad_token}")

        # Set model max length if not set
        if tokenizer.model_max_length > 100000:
            tokenizer.model_max_length = 2048
            logger.info("Set model_max_length to 2048")

        return tokenizer

    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise


def load_base_model(model_name: str, config: Any, bnb_config: BitsAndBytesConfig | None = None) -> PreTrainedModel:
    """Load the base model with optional quantization."""
    try:
        logger.info(f"Loading base model: {model_name}")

        # Model loading arguments
        model_kwargs = {
            "num_labels": config.model.num_labels,
            "id2label": config.data.id2label,
            "label2id": config.data.label2id,
            "trust_remote_code": config.model.trust_remote_code,
            "use_cache": config.model.use_cache,
        }

        # Add quantization config if provided
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["device_map"] = config.model.device_map

        # Handle potential out of memory errors
        try:
            model = AutoModelForTokenClassification.from_pretrained(model_name, **model_kwargs)
        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA out of memory. Trying to free cache...")
            gc.collect()
            torch.cuda.empty_cache()

            # Try again with CPU offloading
            model_kwargs["device_map"] = {"": "cpu", "lm_head": 0, "score": 0}
            model = AutoModelForTokenClassification.from_pretrained(model_name, **model_kwargs)
            logger.warning("Model loaded with CPU offloading due to memory constraints")

        # Enable gradient checkpointing if specified
        if config.training.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

        # Prepare model for k-bit training
        if hasattr(model, "prepare_for_kbit_training"):
            model = prepare_model_for_kbit_training(model)

        return model

    except Exception as e:
        logger.error(f"Failed to load base model: {e}")
        raise


def prepare_model_for_kbit_training(model: PreTrainedModel) -> PreTrainedModel:
    """Prepare model for k-bit training."""
    model.gradient_checkpointing_enable()
    model = model.prepare_for_kbit_training()

    # Fix for potential issues with some model architectures
    for param in model.parameters():
        param.requires_grad = False

    # Enable input require grads
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:

        def make_inputs_require_grad(module, inputs, output):
            if hasattr(inputs, "requires_grad_"):
                inputs.requires_grad_(True)
            elif isinstance(inputs, list | tuple):
                for inp in inputs:
                    if hasattr(inp, "requires_grad_"):
                        inp.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    return model


def setup_peft_model(model: PreTrainedModel, lora_config: LoraConfig) -> PeftModel:
    """Setup PEFT model with LoRA."""
    try:
        logger.info("Setting up PEFT model with LoRA...")

        # Get PEFT model
        model = get_peft_model(model, lora_config)

        # Print trainable parameters
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        trainable_percent = 100 * trainable_params / all_param
        logger.info(
            f"Trainable params: {trainable_params:,} || "
            f"All params: {all_param:,} || "
            f"Trainable%: {trainable_percent:.4f}"
        )

        return model

    except Exception as e:
        logger.error(f"Failed to setup PEFT model: {e}")
        raise


def setup_model(
    model_name: str, config: Any, device_map: str | None = None
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Complete model setup including tokenizer, base model, and LoRA.

    Args:
        model_name: Name or path of the model
        config: Configuration object
        device_map: Device mapping for model placement

    Returns:
        Tuple of (model, tokenizer)
    """
    # Override device map if provided
    if device_map is not None:
        config.model.device_map = device_map

    # Load tokenizer
    tokenizer = load_tokenizer(model_name)

    # Create quantization config
    bnb_config = create_bnb_config(config.model.load_in_8bit) if config.model.load_in_8bit else None

    # Load base model
    model = load_base_model(model_name, config, bnb_config)

    # Create LoRA config and setup PEFT
    lora_config = create_lora_config(config)
    model = setup_peft_model(model, lora_config)

    return model, tokenizer


def save_model_checkpoint(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, output_dir: str, is_final: bool = False
) -> None:
    """Save model checkpoint."""
    try:
        logger.info(f"Saving {'final' if is_final else 'checkpoint'} to {output_dir}")

        # Save model
        model.save_pretrained(output_dir)

        # Save tokenizer
        tokenizer.save_pretrained(output_dir)

        logger.info("Model and tokenizer saved successfully")

    except Exception as e:
        logger.error(f"Failed to save model checkpoint: {e}")
        raise


def load_model_from_checkpoint(
    checkpoint_path: str, config: Any, base_model_name: str | None = None
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load model from a checkpoint."""
    try:
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

        # If base model name not provided, try to get it from config
        if base_model_name is None:
            base_model_name = config.model.model_name

        # Create quantization config
        bnb_config = create_bnb_config(config.model.load_in_8bit) if config.model.load_in_8bit else None

        # Load base model
        base_model = load_base_model(base_model_name, config, bnb_config)

        # Load PEFT model
        model = PeftModel.from_pretrained(base_model, checkpoint_path)

        logger.info("Model loaded successfully from checkpoint")

        return model, tokenizer

    except Exception as e:
        logger.error(f"Failed to load model from checkpoint: {e}")
        raise


def merge_and_save_model(model: PeftModel, tokenizer: PreTrainedTokenizer, output_dir: str) -> None:
    """Merge LoRA weights with base model and save."""
    try:
        logger.info("Merging LoRA weights with base model...")

        # Merge LoRA weights
        merged_model = model.merge_and_unload()

        # Save merged model
        merged_model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)

        logger.info(f"Merged model saved to {output_dir}")

    except Exception as e:
        logger.error(f"Failed to merge and save model: {e}")
        raise
