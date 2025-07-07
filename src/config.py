"""Configuration management for Mistral NER fine-tuning."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from .hyperopt.config import HyperoptConfig

load_dotenv()


@dataclass
class ModelConfig:
    """Model configuration."""

    model_name: str = "mistralai/Mistral-7B-v0.3"
    num_labels: int = 9
    load_in_8bit: bool = False  # Updated default for QLoRA
    load_in_4bit: bool = True  # Added 4-bit quantization support
    device_map: str = "auto"
    trust_remote_code: bool = True
    use_cache: bool = False

    # LoRA configuration - Enhanced for better capacity
    lora_r: int = 32  # Increased from 16 for better capacity
    lora_alpha: int = 64  # Increased proportionally (2x rank)
    lora_dropout: float = 0.1
    lora_bias: str = "none"
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",  # Attention projections
            "gate_proj",
            "up_proj",
            "down_proj",  # MLP projections
        ]
    )
    task_type: str = "TOKEN_CLS"


@dataclass
class MultiDatasetConfig:
    """Configuration for multi-dataset training."""

    enabled: bool = False  # Enable multi-dataset mode
    dataset_names: list[str] = field(default_factory=lambda: ["conll2003"])
    dataset_weights: list[float] | None = None  # None = equal weights
    mixing_strategy: str = "interleave"  # "concat", "interleave", "weighted"
    filter_english: bool = True  # For multilingual datasets
    max_samples_per_dataset: int | None = None  # For memory management

    # Unified label schema - comprehensive set covering all datasets
    unified_labels: list[str] = field(
        default_factory=lambda: [
            "O",
            # Core entities
            "B-PER",
            "I-PER",  # Person
            "B-ORG",
            "I-ORG",  # Organization
            "B-LOC",
            "I-LOC",  # Location
            "B-MISC",
            "I-MISC",  # Miscellaneous
            # Extended entities
            "B-DATE",
            "I-DATE",  # Date
            "B-TIME",
            "I-TIME",  # Time
            "B-MONEY",
            "I-MONEY",  # Money
            "B-PERCENT",
            "I-PERCENT",  # Percentage
            "B-FAC",
            "I-FAC",  # Facility
            "B-GPE",
            "I-GPE",  # Geopolitical entity
            "B-PROD",
            "I-PROD",  # Product
            "B-EVENT",
            "I-EVENT",  # Event
            "B-ART",
            "I-ART",  # Work of art
            "B-LANG",
            "I-LANG",  # Language
            "B-LAW",
            "I-LAW",  # Law
            # PII entities
            "B-CARD",
            "I-CARD",  # Credit card
            "B-SSN",
            "I-SSN",  # Social Security Number
            "B-PHONE",
            "I-PHONE",  # Phone number
            "B-EMAIL",
            "I-EMAIL",  # Email
            "B-ADDR",
            "I-ADDR",  # Address
            "B-BANK",
            "I-BANK",  # Bank account
            "B-PASSPORT",
            "I-PASSPORT",  # Passport
            "B-LICENSE",
            "I-LICENSE",  # Driver's license
            # Numeric entities
            "B-QUANT",
            "I-QUANT",  # Quantity
            "B-ORD",
            "I-ORD",  # Ordinal
            "B-CARD_NUM",
            "I-CARD_NUM",  # Cardinal number
            "B-NORP",
            "I-NORP",  # Nationality, religious, political group
        ]
    )


@dataclass
class DataConfig:
    """Data configuration."""

    dataset_name: str = "conll2003"  # For single dataset mode
    max_length: int = 256
    label_all_tokens: bool = False
    return_entity_level_metrics: bool = True

    # Multi-dataset configuration
    multi_dataset: MultiDatasetConfig = field(default_factory=MultiDatasetConfig)

    # Label configuration - will be set based on mode
    _label_names: list[str] | None = field(init=False, default=None)
    _id2label: dict[int, str] | None = field(init=False, default=None)
    _label2id: dict[str, int] | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        """Initialize label mappings based on dataset mode."""
        self._initialize_labels()

    def _initialize_labels(self) -> None:
        """Initialize label mappings if not already set."""
        if self._label_names is None:
            # Set labels based on mode
            if self.multi_dataset.enabled:
                self._label_names = self.multi_dataset.unified_labels
            else:
                # Default CoNLL-2003 labels for backward compatibility
                self._label_names = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]

            self._id2label = {i: label for i, label in enumerate(self._label_names)}
            self._label2id = {label: i for i, label in enumerate(self._label_names)}

    @property
    def label_names(self) -> list[str]:
        """Get label names, initializing if necessary."""
        if self._label_names is None:
            self._initialize_labels()
        assert self._label_names is not None  # Type narrowing for mypy
        return self._label_names

    @property
    def id2label(self) -> dict[int, str]:
        """Get id to label mapping, initializing if necessary."""
        if self._id2label is None:
            self._initialize_labels()
        assert self._id2label is not None  # Type narrowing for mypy
        return self._id2label

    @property
    def label2id(self) -> dict[str, int]:
        """Get label to id mapping, initializing if necessary."""
        if self._label2id is None:
            self._initialize_labels()
        assert self._label2id is not None  # Type narrowing for mypy
        return self._label2id


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

    # Advanced optimizer configuration
    adam_beta1: float = 0.9  # Beta1 for Adam optimizer
    adam_beta2: float = 0.999  # Beta2 for Adam optimizer (try 0.95 for better convergence)
    adam_epsilon: float = 1e-8  # Epsilon for numerical stability
    gradient_clipping_value: float = 1.0  # Gradient clipping value
    use_gradient_clipping: bool = True  # Enable gradient clipping

    # Loss function configuration
    loss_type: str = "focal"  # 'focal', 'cross_entropy', 'label_smoothing', 'class_balanced', 'weighted_cross_entropy'
    focal_alpha: float | None = None  # Auto-computed from class frequencies if None
    focal_gamma: float = 2.0  # Focusing parameter for focal loss
    label_smoothing: float = 0.1  # Only used if loss_type='label_smoothing'
    class_balanced_beta: float = 0.9999  # Only used if loss_type='class_balanced'

    # Class weighting configuration
    use_class_weights: bool = False  # Enable automatic class weight calculation
    class_weight_type: str = "inverse"  # 'inverse', 'inverse_sqrt', 'effective'
    class_weight_smoothing: float = 1.0  # Smoothing factor for weight calculation
    manual_class_weights: list[float] | None = None  # Manual weights override auto-calculation

    # Batch balancing configuration
    use_batch_balancing: bool = False  # Enable balanced batch sampling
    batch_balance_type: str = "balanced"  # 'balanced', 'entity_aware'
    min_positive_ratio: float = 0.3  # Minimum ratio of positive samples per batch
    batch_balance_beta: float = 0.999  # Beta for batch-balanced focal loss
    log_batch_composition: bool = False  # Enable batch composition logging
    log_batch_every_n: int = 100  # Log batch stats every N batches

    # Learning rate scheduling
    lr_scheduler_type: str = "cosine"  # 'linear', 'cosine', 'cosine_with_restarts', 'polynomial'
    lr_scheduler_kwargs: dict[str, Any] = field(default_factory=dict)  # Additional scheduler params

    # Training strategy
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    logging_steps: int = 10
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_f1"
    greater_is_better: bool = True
    report_to: list[str] = field(default_factory=lambda: ["wandb"])

    # Enhanced evaluation configuration
    use_enhanced_evaluation: bool = False  # Enable detailed per-entity metrics
    compute_entity_level_metrics: bool = True  # Compute metrics per entity type
    log_confusion_matrix: bool = False  # Log confusion matrix to wandb
    analyze_errors: bool = False  # Enable error analysis

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
    wandb_dir: str = "./wandb_logs"  # Directory for offline runs
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
    hyperopt: HyperoptConfig = field(default_factory=HyperoptConfig)

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
            data_dict = config_dict["data"].copy()
            # Remove computed fields that might be in the YAML
            computed_fields = ["label_names", "id2label", "label2id", "_label_names", "_id2label", "_label2id"]
            for field in computed_fields:
                data_dict.pop(field, None)

            # Handle nested multi_dataset config
            if "multi_dataset" in data_dict:
                multi_dataset_dict = data_dict.pop("multi_dataset")
                config.data = DataConfig(**data_dict)
                config.data.multi_dataset = MultiDatasetConfig(**multi_dataset_dict)
                # Re-initialize labels based on multi-dataset mode
                config.data._initialize_labels()
            else:
                config.data = DataConfig(**data_dict)
        if "training" in config_dict:
            config.training = TrainingConfig(**config_dict["training"])
        if "logging" in config_dict:
            config.logging = LoggingConfig(**config_dict["logging"])
        if "hyperopt" in config_dict:
            config.hyperopt = HyperoptConfig(**config_dict["hyperopt"])

        return config

    def to_yaml(self, yaml_path: str | Path) -> None:
        """Save configuration to YAML file."""
        # Create clean dictionaries without computed fields
        data_dict = {
            k: v
            for k, v in self.data.__dict__.items()
            if k not in ["_id2label", "_label2id", "_label_names", "id2label", "label2id", "label_names"]
        }

        # Convert multi_dataset to dict if it exists
        if "multi_dataset" in data_dict and data_dict["multi_dataset"] is not None:
            data_dict["multi_dataset"] = data_dict["multi_dataset"].__dict__

        config_dict: dict[str, dict[str, Any]] = {
            "model": self.model.__dict__,
            "data": data_dict,
            "training": self.training.__dict__,
            "logging": self.logging.__dict__,
            "hyperopt": self.hyperopt.__dict__,
        }

        with open(yaml_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def update_from_args(self, args: Any) -> None:
        """Update configuration from command line arguments."""
        # Update model config
        if hasattr(args, "model_name") and args.model_name:
            self.model.model_name = args.model_name
        if hasattr(args, "load_in_8bit") and args.load_in_8bit is not None:
            self.model.load_in_8bit = args.load_in_8bit
        if hasattr(args, "load_in_4bit") and args.load_in_4bit is not None:
            self.model.load_in_4bit = args.load_in_4bit

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
