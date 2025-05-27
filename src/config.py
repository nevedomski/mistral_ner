"""Configuration management for Mistral NER fine-tuning."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import yaml
from pathlib import Path
import os
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
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    task_type: str = "TOKEN_CLS"


@dataclass
class DataConfig:
    """Data configuration."""
    dataset_name: str = "conll2003"
    max_length: int = 256
    label_all_tokens: bool = False
    return_entity_level_metrics: bool = True
    
    # Label configuration
    label_names: List[str] = field(default_factory=lambda: [
        "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"
    ])
    id2label: Dict[int, str] = field(init=False)
    label2id: Dict[str, int] = field(init=False)
    
    def __post_init__(self):
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
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    logging_steps: int = 10
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_f1"
    greater_is_better: bool = True
    report_to: List[str] = field(default_factory=lambda: ["wandb"])
    
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
    resume_from_checkpoint: Optional[str] = None
    
    # Seed
    seed: int = 42
    data_seed: int = 42
    
    # Distributed training
    local_rank: int = -1
    ddp_find_unused_parameters: bool = False
    
    # Hub
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_strategy: str = "every_save"


@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_level: str = "info"
    log_dir: str = "./logs"
    disable_tqdm: bool = False
    
    # WandB configuration
    wandb_project: str = "mistral-ner"
    wandb_entity: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    wandb_notes: Optional[str] = None
    wandb_mode: str = "online"  # online, offline, disabled
    use_wandb: bool = True


@dataclass
class Config:
    """Main configuration combining all configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Create config objects from dictionaries
        config = cls()
        
        if 'model' in config_dict:
            config.model = ModelConfig(**config_dict['model'])
        if 'data' in config_dict:
            config.data = DataConfig(**config_dict['data'])
        if 'training' in config_dict:
            config.training = TrainingConfig(**config_dict['training'])
        if 'logging' in config_dict:
            config.logging = LoggingConfig(**config_dict['logging'])
        
        return config
    
    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            'model': self.model.__dict__,
            'data': self.data.__dict__,
            'training': self.training.__dict__,
            'logging': self.logging.__dict__
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    def update_from_args(self, args: Any) -> None:
        """Update configuration from command line arguments."""
        # Update model config
        if hasattr(args, 'model_name') and args.model_name:
            self.model.model_name = args.model_name
        if hasattr(args, 'load_in_8bit'):
            self.model.load_in_8bit = args.load_in_8bit
            
        # Update data config
        if hasattr(args, 'max_length') and args.max_length:
            self.data.max_length = args.max_length
            
        # Update training config
        if hasattr(args, 'output_dir') and args.output_dir:
            self.training.output_dir = args.output_dir
        if hasattr(args, 'num_train_epochs') and args.num_train_epochs:
            self.training.num_train_epochs = args.num_train_epochs
        if hasattr(args, 'per_device_train_batch_size') and args.per_device_train_batch_size:
            self.training.per_device_train_batch_size = args.per_device_train_batch_size
        if hasattr(args, 'learning_rate') and args.learning_rate:
            self.training.learning_rate = args.learning_rate
        if hasattr(args, 'resume_from_checkpoint') and args.resume_from_checkpoint:
            self.training.resume_from_checkpoint = args.resume_from_checkpoint
            
        # Update logging config
        if hasattr(args, 'use_wandb'):
            self.logging.use_wandb = args.use_wandb
        if hasattr(args, 'wandb_project') and args.wandb_project:
            self.logging.wandb_project = args.wandb_project
            
    def setup_wandb(self) -> None:
        """Setup WandB based on configuration."""
        if self.logging.use_wandb and self.logging.wandb_mode != "disabled":
            os.environ["WANDB_PROJECT"] = self.logging.wandb_project
            os.environ["WANDB_MODE"] = self.logging.wandb_mode
            
            if self.logging.wandb_entity:
                os.environ["WANDB_ENTITY"] = self.logging.wandb_entity
        else:
            os.environ["WANDB_DISABLED"] = "true"
            # Remove wandb from report_to if disabled
            if "wandb" in self.training.report_to:
                self.training.report_to.remove("wandb")