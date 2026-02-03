"""
Pipeline Configuration

All configuration loaded from environment variables with sensible defaults.
Secrets should be provided via Kubernetes secrets, not hardcoded.
"""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class RunPodConfig:
    """RunPod API configuration."""
    api_key: str = os.getenv("RUNPOD_API_KEY", "")
    endpoint: str = os.getenv("RUNPOD_ENDPOINT", "")
    api_base: str = os.getenv("RUNPOD_API_BASE", "https://api.runpod.ai/v2")


@dataclass
class AWSConfig:
    """AWS credentials and settings."""
    access_key_id: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    secret_access_key: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    session_token: str = os.getenv("AWS_SESSION_TOKEN", "")
    region: str = os.getenv("AWS_REGION", "us-east-1")
    s3_bucket: str = os.getenv("S3_BUCKET", "")
    s3_prefix: str = os.getenv("S3_PREFIX", "models")


@dataclass
class ModelConfig:
    """Model training defaults."""
    base_model: str = os.getenv("BASE_MODEL", "emilyalsentzer/Bio_ClinicalBERT")
    max_samples: int = int(os.getenv("MAX_SAMPLES", "10000"))
    epochs: int = int(os.getenv("EPOCHS", "3"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "16"))
    eval_split: float = float(os.getenv("EVAL_SPLIT", "0.1"))
    learning_rate: float = float(os.getenv("LEARNING_RATE", "2e-5"))


@dataclass
class PipelineConfig:
    """Combined pipeline configuration."""
    runpod: RunPodConfig
    aws: AWSConfig
    model: ModelConfig
    
    # Pipeline settings
    poll_interval: int = int(os.getenv("POLL_INTERVAL_SECONDS", "10"))
    timeout: int = int(os.getenv("TRAINING_TIMEOUT_SECONDS", "3600"))
    
    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Load configuration from environment variables."""
        return cls(
            runpod=RunPodConfig(),
            aws=AWSConfig(),
            model=ModelConfig(),
        )


# Task-specific defaults (can override base config)
TASK_DEFAULTS = {
    "ddi": {
        "max_samples": 10000,
        "batch_size": 16,
        "s3_prefix": "ddi-models",
    },
    "ade": {
        "max_samples": 10000,
        "batch_size": 16,
        "s3_prefix": "ade-models",
    },
    "triage": {
        "max_samples": 5000,
        "batch_size": 8,
        "s3_prefix": "triage-models",
    },
    "symptom_disease": {
        "max_samples": 5000,
        "batch_size": 16,
        "s3_prefix": "symptom-disease-models",
    },
}


def get_task_config(task: str, overrides: Optional[dict] = None) -> dict:
    """Get task-specific configuration with optional overrides."""
    config = TASK_DEFAULTS.get(task, {}).copy()
    if overrides:
        config.update(overrides)
    return config
