"""Configuration loading and validation."""

import os
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

from ..exceptions import ConfigurationError
from .models import Config

# Load environment variables
load_dotenv()


def load_config(config_path: Optional[Path] = None) -> Config:
    """
    Load and validate configuration from YAML file.

    Args:
        config_path: Path to configuration file. If None, uses "config.yaml" in current directory.

    Returns:
        Validated Config object.

    Raises:
        ConfigurationError: If configuration file is not found or invalid.
    """
    if config_path is None:
        config_path = Path("config.yaml")

    if not config_path.exists():
        raise ConfigurationError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
    except Exception as e:
        raise ConfigurationError(f"Failed to load configuration file: {e}") from e

    if config_dict is None:
        raise ConfigurationError("Configuration file is empty")

    try:
        return Config(**config_dict)
    except Exception as e:
        raise ConfigurationError(f"Invalid configuration: {e}") from e


def get_hf_token() -> str:
    """
    Get HuggingFace token from environment.

    Returns:
        HuggingFace token.

    Raises:
        ConfigurationError: If HF_TOKEN is not set.
    """
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ConfigurationError(
            "HF_TOKEN environment variable is not set. "
            "Please set it in your .env file or environment."
        )
    return token

