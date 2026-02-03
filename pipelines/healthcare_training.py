"""
Healthcare ML Training Pipelines

Multi-task training pipelines for:
- Adverse Drug Event (ADE) Classification
- Medical Triage Classification
- Symptom-to-Disease Prediction
- Drug-Drug Interaction (DDI) Classification

All use RunPod serverless GPU infrastructure.
Configuration via environment variables - see config.py for details.

Environment Variables:
    RUNPOD_API_KEY      - RunPod API key (required)
    RUNPOD_ENDPOINT     - RunPod serverless endpoint ID (required)
    AWS_ACCESS_KEY_ID   - AWS credentials for S3 upload
    AWS_SECRET_ACCESS_KEY
    AWS_SESSION_TOKEN   - Optional session token for assumed roles
    AWS_REGION          - Default: us-east-1
    S3_BUCKET           - Bucket for model artifacts (required)
    BASE_MODEL          - HuggingFace model ID (default: Bio_ClinicalBERT)
    MAX_SAMPLES         - Training samples (default: 10000)
    EPOCHS              - Training epochs (default: 3)
    BATCH_SIZE          - Batch size (default: 16)
"""
import os
from kfp import dsl
from kfp import compiler
from typing import Optional


# =============================================================================
# Reusable Training Component
# =============================================================================
@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["requests"]
)
def train_healthcare_model(
    task: str,
    runpod_api_key: str,
    runpod_endpoint: str,
    model_name: str,
    max_samples: int,
    epochs: int,
    batch_size: int,
    eval_split: float,
    s3_bucket: str,
    s3_prefix: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    aws_session_token: str,
    aws_region: str,
    poll_interval: int,
    timeout: int,
) -> dict:
    """
    Generic healthcare model training component.
    
    Submits training job to RunPod serverless GPU and polls for completion.
    Trained model is uploaded to S3 by the RunPod handler.
    
    Args:
        task: Training task (ddi, ade, triage, symptom_disease)
        runpod_api_key: RunPod API key
        runpod_endpoint: RunPod serverless endpoint ID
        model_name: HuggingFace model ID
        max_samples: Maximum training samples
        epochs: Training epochs
        batch_size: Training batch size
        eval_split: Validation split ratio
        s3_bucket: S3 bucket for model output
        s3_prefix: S3 key prefix for this task
        aws_*: AWS credentials for S3 access
        poll_interval: Seconds between status checks
        timeout: Maximum training time in seconds
        
    Returns:
        Training output including metrics and S3 URI
    """
    import requests
    import time
    
    api_base = os.getenv("RUNPOD_API_BASE", "https://api.runpod.ai/v2")
    
    # Submit training job
    response = requests.post(
        f"{api_base}/{runpod_endpoint}/run",
        headers={"Authorization": f"Bearer {runpod_api_key}"},
        json={
            "input": {
                "task": task,
                "model_name": model_name,
                "max_samples": max_samples,
                "epochs": epochs,
                "batch_size": batch_size,
                "eval_split": eval_split,
                "s3_bucket": s3_bucket,
                "s3_prefix": s3_prefix,
                "aws_access_key_id": aws_access_key_id,
                "aws_secret_access_key": aws_secret_access_key,
                "aws_session_token": aws_session_token,
                "aws_region": aws_region,
            }
        },
        timeout=30,
    )
    response.raise_for_status()
    
    job_id = response.json()["id"]
    print(f"[{task}] RunPod job submitted: {job_id}")
    
    # Poll for completion
    start_time = time.time()
    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            raise TimeoutError(f"Training exceeded timeout of {timeout}s")
        
        status_resp = requests.get(
            f"{api_base}/{runpod_endpoint}/status/{job_id}",
            headers={"Authorization": f"Bearer {runpod_api_key}"},
            timeout=30,
        )
        status_resp.raise_for_status()
        status = status_resp.json()
        
        if status["status"] == "COMPLETED":
            print(f"[{task}] Training completed in {elapsed:.0f}s")
            return status.get("output", {})
        elif status["status"] == "FAILED":
            error = status.get("error", "Unknown error")
            raise RuntimeError(f"[{task}] Training failed: {error}")
        elif status["status"] in ["IN_QUEUE", "IN_PROGRESS"]:
            print(f"[{task}] Status: {status['status']} ({elapsed:.0f}s elapsed)")
        
        time.sleep(poll_interval)


# =============================================================================
# Pipeline Definitions
# =============================================================================
def _get_env(name: str, default: str = "") -> str:
    """Helper to get env var with default."""
    return os.getenv(name, default)


def _get_env_int(name: str, default: int) -> int:
    """Helper to get int env var with default."""
    return int(os.getenv(name, str(default)))


def _get_env_float(name: str, default: float) -> float:
    """Helper to get float env var with default."""
    return float(os.getenv(name, str(default)))


@dsl.pipeline(name="ade-classification-pipeline")
def ade_classification_pipeline(
    # RunPod config - from env or override
    runpod_api_key: str = "",
    runpod_endpoint: str = "",
    # Model config
    model_name: str = "",
    max_samples: int = 10000,
    epochs: int = 3,
    batch_size: int = 16,
    eval_split: float = 0.1,
    # AWS config
    s3_bucket: str = "",
    aws_access_key_id: str = "",
    aws_secret_access_key: str = "",
    aws_session_token: str = "",
    aws_region: str = "us-east-1",
    # Runtime config
    poll_interval: int = 10,
    timeout: int = 3600,
):
    """
    Adverse Drug Event Classification Pipeline
    
    Trains Bio_ClinicalBERT on ADE Corpus V2 (30K samples).
    Binary classification: ADE present / No ADE.
    
    All parameters can be provided via environment variables:
    - RUNPOD_API_KEY, RUNPOD_ENDPOINT
    - BASE_MODEL, MAX_SAMPLES, EPOCHS, BATCH_SIZE
    - S3_BUCKET, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, etc.
    """
    train_healthcare_model(
        task="ade",
        runpod_api_key=runpod_api_key or _get_env("RUNPOD_API_KEY"),
        runpod_endpoint=runpod_endpoint or _get_env("RUNPOD_ENDPOINT"),
        model_name=model_name or _get_env("BASE_MODEL", "emilyalsentzer/Bio_ClinicalBERT"),
        max_samples=max_samples,
        epochs=epochs,
        batch_size=batch_size,
        eval_split=eval_split,
        s3_bucket=s3_bucket or _get_env("S3_BUCKET"),
        s3_prefix="ade-models/bert",
        aws_access_key_id=aws_access_key_id or _get_env("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=aws_secret_access_key or _get_env("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=aws_session_token or _get_env("AWS_SESSION_TOKEN"),
        aws_region=aws_region,
        poll_interval=poll_interval,
        timeout=timeout,
    )


@dsl.pipeline(name="triage-classification-pipeline")
def triage_classification_pipeline(
    runpod_api_key: str = "",
    runpod_endpoint: str = "",
    model_name: str = "",
    max_samples: int = 5000,
    epochs: int = 3,
    batch_size: int = 8,
    eval_split: float = 0.1,
    s3_bucket: str = "",
    aws_access_key_id: str = "",
    aws_secret_access_key: str = "",
    aws_session_token: str = "",
    aws_region: str = "us-east-1",
    poll_interval: int = 10,
    timeout: int = 3600,
):
    """
    Medical Triage Classification Pipeline
    
    Trains classifier for ER triage urgency levels.
    Multi-class: Emergency, Urgent, Standard, Non-urgent.
    """
    train_healthcare_model(
        task="triage",
        runpod_api_key=runpod_api_key or _get_env("RUNPOD_API_KEY"),
        runpod_endpoint=runpod_endpoint or _get_env("RUNPOD_ENDPOINT"),
        model_name=model_name or _get_env("BASE_MODEL", "emilyalsentzer/Bio_ClinicalBERT"),
        max_samples=max_samples,
        epochs=epochs,
        batch_size=batch_size,
        eval_split=eval_split,
        s3_bucket=s3_bucket or _get_env("S3_BUCKET"),
        s3_prefix="triage-models/bert",
        aws_access_key_id=aws_access_key_id or _get_env("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=aws_secret_access_key or _get_env("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=aws_session_token or _get_env("AWS_SESSION_TOKEN"),
        aws_region=aws_region,
        poll_interval=poll_interval,
        timeout=timeout,
    )


@dsl.pipeline(name="symptom-disease-classification-pipeline")
def symptom_disease_pipeline(
    runpod_api_key: str = "",
    runpod_endpoint: str = "",
    model_name: str = "",
    max_samples: int = 5000,
    epochs: int = 3,
    batch_size: int = 16,
    eval_split: float = 0.1,
    s3_bucket: str = "",
    aws_access_key_id: str = "",
    aws_secret_access_key: str = "",
    aws_session_token: str = "",
    aws_region: str = "us-east-1",
    poll_interval: int = 10,
    timeout: int = 3600,
):
    """
    Symptom-to-Disease Classification Pipeline
    
    Predicts disease from symptom descriptions.
    Multi-class: 41 disease categories.
    """
    train_healthcare_model(
        task="symptom_disease",
        runpod_api_key=runpod_api_key or _get_env("RUNPOD_API_KEY"),
        runpod_endpoint=runpod_endpoint or _get_env("RUNPOD_ENDPOINT"),
        model_name=model_name or _get_env("BASE_MODEL", "emilyalsentzer/Bio_ClinicalBERT"),
        max_samples=max_samples,
        epochs=epochs,
        batch_size=batch_size,
        eval_split=eval_split,
        s3_bucket=s3_bucket or _get_env("S3_BUCKET"),
        s3_prefix="symptom-disease-models/bert",
        aws_access_key_id=aws_access_key_id or _get_env("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=aws_secret_access_key or _get_env("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=aws_session_token or _get_env("AWS_SESSION_TOKEN"),
        aws_region=aws_region,
        poll_interval=poll_interval,
        timeout=timeout,
    )


@dsl.pipeline(name="ddi-classification-pipeline")
def ddi_classification_pipeline(
    runpod_api_key: str = "",
    runpod_endpoint: str = "",
    model_name: str = "",
    max_samples: int = 10000,
    epochs: int = 3,
    batch_size: int = 16,
    eval_split: float = 0.1,
    s3_bucket: str = "",
    aws_access_key_id: str = "",
    aws_secret_access_key: str = "",
    aws_session_token: str = "",
    aws_region: str = "us-east-1",
    poll_interval: int = 10,
    timeout: int = 3600,
):
    """
    Drug-Drug Interaction Classification Pipeline
    
    Trains on 176K DrugBank DDI samples.
    Multi-class severity: Minor, Moderate, Major, Contraindicated.
    """
    train_healthcare_model(
        task="ddi",
        runpod_api_key=runpod_api_key or _get_env("RUNPOD_API_KEY"),
        runpod_endpoint=runpod_endpoint or _get_env("RUNPOD_ENDPOINT"),
        model_name=model_name or _get_env("BASE_MODEL", "emilyalsentzer/Bio_ClinicalBERT"),
        max_samples=max_samples,
        epochs=epochs,
        batch_size=batch_size,
        eval_split=eval_split,
        s3_bucket=s3_bucket or _get_env("S3_BUCKET"),
        s3_prefix="ddi-models/bert",
        aws_access_key_id=aws_access_key_id or _get_env("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=aws_secret_access_key or _get_env("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=aws_session_token or _get_env("AWS_SESSION_TOKEN"),
        aws_region=aws_region,
        poll_interval=poll_interval,
        timeout=timeout,
    )


@dsl.pipeline(name="healthcare-multi-task-pipeline")
def healthcare_multi_task_pipeline(
    runpod_api_key: str = "",
    runpod_endpoint: str = "",
    model_name: str = "",
    s3_bucket: str = "",
    aws_access_key_id: str = "",
    aws_secret_access_key: str = "",
    aws_session_token: str = "",
    aws_region: str = "us-east-1",
    poll_interval: int = 10,
    timeout: int = 3600,
):
    """
    Train all healthcare models in parallel.
    
    Outputs:
    - DDI classifier (s3://bucket/ddi-models/...)
    - ADE classifier (s3://bucket/ade-models/...)
    - Triage classifier (s3://bucket/triage-models/...)
    - Symptom-Disease classifier (s3://bucket/symptom-disease-models/...)
    """
    # Resolve env vars once
    _runpod_key = runpod_api_key or _get_env("RUNPOD_API_KEY")
    _runpod_endpoint = runpod_endpoint or _get_env("RUNPOD_ENDPOINT")
    _model = model_name or _get_env("BASE_MODEL", "emilyalsentzer/Bio_ClinicalBERT")
    _bucket = s3_bucket or _get_env("S3_BUCKET")
    _aws_key = aws_access_key_id or _get_env("AWS_ACCESS_KEY_ID")
    _aws_secret = aws_secret_access_key or _get_env("AWS_SECRET_ACCESS_KEY")
    _aws_token = aws_session_token or _get_env("AWS_SESSION_TOKEN")
    
    # Run all training tasks in parallel (no dependencies between them)
    ddi_task = train_healthcare_model(
        task="ddi",
        runpod_api_key=_runpod_key,
        runpod_endpoint=_runpod_endpoint,
        model_name=_model,
        max_samples=10000,
        epochs=3,
        batch_size=16,
        eval_split=0.1,
        s3_bucket=_bucket,
        s3_prefix="ddi-models/bert",
        aws_access_key_id=_aws_key,
        aws_secret_access_key=_aws_secret,
        aws_session_token=_aws_token,
        aws_region=aws_region,
        poll_interval=poll_interval,
        timeout=timeout,
    )
    
    ade_task = train_healthcare_model(
        task="ade",
        runpod_api_key=_runpod_key,
        runpod_endpoint=_runpod_endpoint,
        model_name=_model,
        max_samples=10000,
        epochs=3,
        batch_size=16,
        eval_split=0.1,
        s3_bucket=_bucket,
        s3_prefix="ade-models/bert",
        aws_access_key_id=_aws_key,
        aws_secret_access_key=_aws_secret,
        aws_session_token=_aws_token,
        aws_region=aws_region,
        poll_interval=poll_interval,
        timeout=timeout,
    )
    
    triage_task = train_healthcare_model(
        task="triage",
        runpod_api_key=_runpod_key,
        runpod_endpoint=_runpod_endpoint,
        model_name=_model,
        max_samples=5000,
        epochs=3,
        batch_size=8,
        eval_split=0.1,
        s3_bucket=_bucket,
        s3_prefix="triage-models/bert",
        aws_access_key_id=_aws_key,
        aws_secret_access_key=_aws_secret,
        aws_session_token=_aws_token,
        aws_region=aws_region,
        poll_interval=poll_interval,
        timeout=timeout,
    )
    
    symptom_task = train_healthcare_model(
        task="symptom_disease",
        runpod_api_key=_runpod_key,
        runpod_endpoint=_runpod_endpoint,
        model_name=_model,
        max_samples=5000,
        epochs=3,
        batch_size=16,
        eval_split=0.1,
        s3_bucket=_bucket,
        s3_prefix="symptom-disease-models/bert",
        aws_access_key_id=_aws_key,
        aws_secret_access_key=_aws_secret,
        aws_session_token=_aws_token,
        aws_region=aws_region,
        poll_interval=poll_interval,
        timeout=timeout,
    )


# =============================================================================
# Compile Pipelines
# =============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compile Kubeflow pipelines")
    parser.add_argument("--output-dir", default=".", help="Output directory for compiled YAML")
    args = parser.parse_args()
    
    pipelines = [
        (ade_classification_pipeline, "ade_classification_pipeline.yaml"),
        (triage_classification_pipeline, "triage_classification_pipeline.yaml"),
        (symptom_disease_pipeline, "symptom_disease_pipeline.yaml"),
        (ddi_classification_pipeline, "ddi_classification_pipeline.yaml"),
        (healthcare_multi_task_pipeline, "healthcare_multi_task_pipeline.yaml"),
    ]
    
    for pipeline_func, filename in pipelines:
        output_path = os.path.join(args.output_dir, filename)
        compiler.Compiler().compile(pipeline_func, output_path)
        print(f"Compiled: {output_path}")
    
    print(f"\n✓ All {len(pipelines)} pipelines compiled!")
