"""Configuration management for Mistral NER fine-tuning."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ModelConfig:
    """Model configuration."""

    model_name: str = "mistralai/Mistral-7B-v0.3"
    num_labels: int = 9
    load_in_8bit: bool = True
    device_map: str = "auto"
    trust_remote_code: bool = True
    use_cache: bool = False

    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_bias: str = "none"
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    task_type: str = "TOKEN_CLS"


@dataclass
class DataConfig:
    """Data configuration."""

    dataset_name: str = "conll2003"
    max_length: int = 256
    label_all_tokens: bool = False
    return_entity_level_metrics: bool = True

    # Label configuration
    label_names: list[str] = field(
        default_factory=lambda: ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
    )
    id2label: dict[int, str] = field(init=False)
    label2id: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.id2label = {i: label for i, label in enumerate(self.label_names)}
        self.label2id = {label: i for i, label in enumerate(self.label_names)}


@dataclass
class TrainingConfig:
    """Training configuration."""

    output_dir: str = "./mistral-ner-finetuned"
    final_output_dir: str = "./mistral-ner-finetuned-final"
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_32bit"
    learning_rate: float = 2e-4
    weight_decay: float = 0.001
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0

    # Training strategy
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    logging_steps: int = 10
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_f1"
    greater_is_better: bool = True
    report_to: list[str] = field(default_factory=lambda: ["wandb"])

    # Mixed precision
    fp16: bool = False
    bf16: bool = False
    tf32: bool = True

    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.01

    # Memory management
    clear_cache_steps: int = 50

    # Resume from checkpoint
    resume_from_checkpoint: str | None = None

    # Seed
    seed: int = 42
    data_seed: int = 42

    # Distributed training
    local_rank: int = -1
    ddp_find_unused_parameters: bool = False

    # Hub
    push_to_hub: bool = False
    hub_model_id: str | None = None
    hub_strategy: str = "every_save"


@dataclass
class LoggingConfig:
    """Logging configuration."""

    log_level: str = "info"
    log_dir: str = "./logs"
    disable_tqdm: bool = False

    # WandB configuration
    use_wandb: bool = True
    wandb_project: str = "mistral-ner"
    wandb_entity: str | None = None
    wandb_name: str | None = None
    wandb_tags: list[str] = field(default_factory=list)
    wandb_notes: str | None = None
    wandb_mode: str = "online"  # online, offline, disabled
    wandb_dir: str = "./wandb"  # Directory for offline runs
    wandb_resume: str = "allow"  # allow, must, never, auto
    wandb_run_id: str | None = None  # For resuming specific runs
    wandb_api_key: str | None = None  # API key (can be set via env var)


@dataclass
class Config:
    """Main configuration combining all configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> Config:
        """Load configuration from YAML file."""
        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)

        # Create config objects from dictionaries
        config = cls()

        if "model" in config_dict:
            config.model = ModelConfig(**config_dict["model"])
        if "data" in config_dict:
            config.data = DataConfig(**config_dict["data"])
        if "training" in config_dict:
            config.training = TrainingConfig(**config_dict["training"])
        if "logging" in config_dict:
            config.logging = LoggingConfig(**config_dict["logging"])

        return config

    def to_yaml(self, yaml_path: str | Path) -> None:
        """Save configuration to YAML file."""
        # Create clean dictionaries without computed fields
        data_dict = {k: v for k, v in self.data.__dict__.items() if k not in ["id2label", "label2id"]}

        config_dict: dict[str, dict[str, Any]] = {
            "model": self.model.__dict__,
            "data": data_dict,
            "training": self.training.__dict__,
            "logging": self.logging.__dict__,
        }

        with open(yaml_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def update_from_args(self, args: Any) -> None:
        """Update configuration from command line arguments."""
        # Update model config
        if hasattr(args, "model_name") and args.model_name:
            self.model.model_name = args.model_name
        if hasattr(args, "load_in_8bit"):
            self.model.load_in_8bit = args.load_in_8bit

        # Update data config
        if hasattr(args, "max_length") and args.max_length:
            self.data.max_length = args.max_length

        # Update training config
        if hasattr(args, "output_dir") and args.output_dir:
            self.training.output_dir = args.output_dir
        if hasattr(args, "num_train_epochs") and args.num_train_epochs:
            self.training.num_train_epochs = args.num_train_epochs
        if hasattr(args, "per_device_train_batch_size") and args.per_device_train_batch_size:
            self.training.per_device_train_batch_size = args.per_device_train_batch_size
        if hasattr(args, "learning_rate") and args.learning_rate:
            self.training.learning_rate = args.learning_rate
        if hasattr(args, "resume_from_checkpoint") and args.resume_from_checkpoint:
            self.training.resume_from_checkpoint = args.resume_from_checkpoint

        # Update logging config
        if hasattr(args, "use_wandb") and args.use_wandb is not None:
            self.logging.use_wandb = args.use_wandb
        if hasattr(args, "wandb_project") and args.wandb_project:
            self.logging.wandb_project = args.wandb_project
        if hasattr(args, "wandb_entity") and args.wandb_entity:
            self.logging.wandb_entity = args.wandb_entity
        if hasattr(args, "wandb_name") and args.wandb_name:
            self.logging.wandb_name = args.wandb_name
        if hasattr(args, "wandb_mode") and args.wandb_mode:
            self.logging.wandb_mode = args.wandb_mode
        if hasattr(args, "wandb_dir") and args.wandb_dir:
            self.logging.wandb_dir = args.wandb_dir
        if hasattr(args, "wandb_resume") and args.wandb_resume:
            self.logging.wandb_resume = args.wandb_resume
        if hasattr(args, "wandb_run_id") and args.wandb_run_id:
            self.logging.wandb_run_id = args.wandb_run_id
        if hasattr(args, "wandb_tags") and args.wandb_tags:
            self.logging.wandb_tags = args.wandb_tags
        if hasattr(args, "wandb_notes") and args.wandb_notes:
            self.logging.wandb_notes = args.wandb_notes

    def setup_wandb(self) -> None:
        """Setup WandB based on configuration."""
        from .utils import validate_wandb_config
        
        # Validate configuration first
        validate_wandb_config(self.logging)
        
        if self.logging.use_wandb and self.logging.wandb_mode != "disabled":
            # Ensure wandb directory exists
            os.makedirs(self.logging.wandb_dir, exist_ok=True)
            
            # Set core environment variables
            os.environ["WANDB_PROJECT"] = self.logging.wandb_project
            os.environ["WANDB_MODE"] = self.logging.wandb_mode
            os.environ["WANDB_DIR"] = self.logging.wandb_dir
            
            # Set optional configuration
            if self.logging.wandb_entity:
                os.environ["WANDB_ENTITY"] = self.logging.wandb_entity
            if self.logging.wandb_name:
                os.environ["WANDB_NAME"] = self.logging.wandb_name
            if self.logging.wandb_notes:
                os.environ["WANDB_NOTES"] = self.logging.wandb_notes
            if self.logging.wandb_run_id:
                os.environ["WANDB_RUN_ID"] = self.logging.wandb_run_id
            if self.logging.wandb_api_key:
                os.environ["WANDB_API_KEY"] = self.logging.wandb_api_key
                
            # Set resume strategy
            os.environ["WANDB_RESUME"] = self.logging.wandb_resume
            
            # Set tags if provided
            if self.logging.wandb_tags:
                os.environ["WANDB_TAGS"] = ",".join(self.logging.wandb_tags)
        else:
            os.environ["WANDB_DISABLED"] = "true"
            os.environ["WANDB_MODE"] = "disabled"
            # Remove wandb from report_to if disabled
            if "wandb" in self.training.report_to:
                self.training.report_to.remove("wandb")
