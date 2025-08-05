"""Logging configuration for Mistral NER training and evaluation."""

from __future__ import annotations

import logging
import os
from typing import Any, Literal

from pydantic import Field, model_validator
from pydantic.dataclasses import dataclass


@dataclass
class LoggingConfig:
    """Configuration for logging and experiment tracking in Mistral NER.

    This class manages logging settings for both local file logging and
    Weights & Biases (WandB) experiment tracking. Supports online, offline,
    and disabled modes for WandB integration.

    Attributes:
        log_level (str): Logging level - 'debug', 'info', 'warning', 'error', or 'critical'.
        log_dir (str): Directory for storing log files (default: './logs').
        disable_tqdm (bool): Whether to disable tqdm progress bars.

        use_wandb (bool): Enable Weights & Biases experiment tracking.
        wandb_project (str): WandB project name (default: 'mistral-ner').
        wandb_entity (str | None): WandB entity/team name (can be set via WANDB_ENTITY env var).
        wandb_name (str | None): Run name in WandB (auto-generated if not specified).
        wandb_tags (list[str]): Tags for categorizing runs (e.g., ['experiment', 'baseline']).
        wandb_notes (str | None): Additional notes or description for the run.
        wandb_mode (str): WandB mode - 'online', 'offline', or 'disabled'.
        wandb_dir (str): Directory for storing offline WandB runs (default: './wandb_logs').
        wandb_resume (str): Resume behavior - 'allow', 'must', 'never', or 'auto'.
        wandb_run_id (str | None): Specific run ID for resuming previous runs.
        wandb_api_key (str | None): WandB API key (can be set via WANDB_API_KEY env var).

    Note:
        For production use, it's recommended to set WANDB_ENTITY and WANDB_API_KEY
        as environment variables rather than in configuration files.
    """

    log_level: Literal["debug", "info", "warning", "error", "critical"] = "info"
    log_dir: str = "./logs"
    disable_tqdm: bool = False

    # WandB configuration
    use_wandb: bool = False
    wandb_project: str = "mistral-ner"
    wandb_entity: str | None = None  # Set via WANDB_ENTITY env var or specify here
    wandb_name: str | None = None  # Auto-generated if not specified
    wandb_tags: list[str] = Field(default_factory=list)  # Add tags like ["experiment", "baseline"]
    wandb_notes: str | None = None  # Add experiment notes
    wandb_mode: Literal["online", "offline", "disabled"] = "disabled"
    wandb_dir: str = "./wandb_logs"  # Directory for offline runs
    wandb_resume: Literal["allow", "must", "never", "auto"] = "allow"
    wandb_run_id: str | None = None  # For resuming specific runs
    wandb_api_key: str | None = None  # Set via WANDB_API_KEY env var or specify here

    @model_validator(mode="before")
    @classmethod
    def wandb_api_key_validate(cls, values: Any) -> Any:
        """Validate WandB API key and adjust mode if necessary."""
        # Access dict keys instead of attributes
        if values.get("wandb_mode") == "online":
            api_key = values.get("wandb_api_key") or os.getenv("WANDB_API_KEY")
            if not api_key:
                logging.warning("WANDB_API_KEY not found...")
                values["wandb_mode"] = "offline"
        return values

    @model_validator(mode="after")
    def setup_wandb(self) -> LoggingConfig:
        """Setup WandB based on configuration."""

        if self.use_wandb and self.wandb_mode != "disabled":
            # Ensure wandb directory exists
            os.makedirs(self.wandb_dir, exist_ok=True)

            # Set core environment variables
            os.environ["WANDB_PROJECT"] = self.wandb_project
            os.environ["WANDB_MODE"] = self.wandb_mode
            os.environ["WANDB_DIR"] = self.wandb_dir

            # Set optional configuration
            if self.wandb_entity:
                os.environ["WANDB_ENTITY"] = self.wandb_entity
            if self.wandb_name:
                os.environ["WANDB_NAME"] = self.wandb_name
            if self.wandb_notes:
                os.environ["WANDB_NOTES"] = self.wandb_notes
            if self.wandb_run_id:
                os.environ["WANDB_RUN_ID"] = self.wandb_run_id
            if self.wandb_api_key:
                os.environ["WANDB_API_KEY"] = self.wandb_api_key

            # Set resume strategy
            os.environ["WANDB_RESUME"] = self.wandb_resume

            # Set tags if provided
            if self.wandb_tags:
                os.environ["WANDB_TAGS"] = ",".join(self.wandb_tags)
        else:
            os.environ["WANDB_DISABLED"] = "true"
            os.environ["WANDB_MODE"] = "disabled"
        return self
