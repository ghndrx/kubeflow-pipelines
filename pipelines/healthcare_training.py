"""
Healthcare ML Training Pipelines

Multi-task training pipelines for:
- Adverse Drug Event (ADE) Classification
- Medical Triage Classification
- Symptom-to-Disease Prediction
- Drug-Drug Interaction (DDI) Classification

All use RunPod serverless GPU infrastructure.
"""
from kfp import dsl
from kfp import compiler


# ============================================================================
# ADE (Adverse Drug Event) Classification Pipeline
# ============================================================================
@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["requests"]
)
def train_ade_model(
    runpod_api_key: str,
    runpod_endpoint: str,
    model_name: str,
    max_samples: int,
    epochs: int,
    batch_size: int,
    s3_bucket: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    aws_session_token: str,
) -> dict:
    """Train ADE classifier on RunPod serverless GPU."""
    import requests
    import time
    
    response = requests.post(
        f"https://api.runpod.ai/v2/{runpod_endpoint}/run",
        headers={"Authorization": f"Bearer {runpod_api_key}"},
        json={
            "input": {
                "task": "ade",
                "model_name": model_name,
                "max_samples": max_samples,
                "epochs": epochs,
                "batch_size": batch_size,
                "eval_split": 0.1,
                "s3_bucket": s3_bucket,
                "s3_prefix": "ade-models/bert",
                "aws_access_key_id": aws_access_key_id,
                "aws_secret_access_key": aws_secret_access_key,
                "aws_session_token": aws_session_token,
            }
        }
    )
    
    job_id = response.json()["id"]
    print(f"RunPod job submitted: {job_id}")
    
    # Poll for completion
    while True:
        status = requests.get(
            f"https://api.runpod.ai/v2/{runpod_endpoint}/status/{job_id}",
            headers={"Authorization": f"Bearer {runpod_api_key}"}
        ).json()
        
        if status["status"] == "COMPLETED":
            return status["output"]
        elif status["status"] == "FAILED":
            raise Exception(f"Training failed: {status}")
        
        time.sleep(10)


@dsl.pipeline(name="ade-classification-pipeline")
def ade_classification_pipeline(
    runpod_api_key: str,
    runpod_endpoint: str = "k57do7afav01es",
    model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
    max_samples: int = 10000,
    epochs: int = 3,
    batch_size: int = 16,
    s3_bucket: str = "",
    aws_access_key_id: str = "",
    aws_secret_access_key: str = "",
    aws_session_token: str = "",
):
    """
    Adverse Drug Event Classification Pipeline
    
    Trains Bio_ClinicalBERT on ADE Corpus V2 (30K samples)
    Binary classification: ADE present / No ADE
    """
    train_task = train_ade_model(
        runpod_api_key=runpod_api_key,
        runpod_endpoint=runpod_endpoint,
        model_name=model_name,
        max_samples=max_samples,
        epochs=epochs,
        batch_size=batch_size,
        s3_bucket=s3_bucket,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
    )


# ============================================================================
# Medical Triage Classification Pipeline
# ============================================================================
@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["requests"]
)
def train_triage_model(
    runpod_api_key: str,
    runpod_endpoint: str,
    model_name: str,
    max_samples: int,
    epochs: int,
    batch_size: int,
    s3_bucket: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    aws_session_token: str,
) -> dict:
    """Train Medical Triage classifier on RunPod."""
    import requests
    import time
    
    response = requests.post(
        f"https://api.runpod.ai/v2/{runpod_endpoint}/run",
        headers={"Authorization": f"Bearer {runpod_api_key}"},
        json={
            "input": {
                "task": "triage",
                "model_name": model_name,
                "max_samples": max_samples,
                "epochs": epochs,
                "batch_size": batch_size,
                "eval_split": 0.1,
                "s3_bucket": s3_bucket,
                "s3_prefix": "triage-models/bert",
                "aws_access_key_id": aws_access_key_id,
                "aws_secret_access_key": aws_secret_access_key,
                "aws_session_token": aws_session_token,
            }
        }
    )
    
    job_id = response.json()["id"]
    print(f"RunPod job submitted: {job_id}")
    
    while True:
        status = requests.get(
            f"https://api.runpod.ai/v2/{runpod_endpoint}/status/{job_id}",
            headers={"Authorization": f"Bearer {runpod_api_key}"}
        ).json()
        
        if status["status"] == "COMPLETED":
            return status["output"]
        elif status["status"] == "FAILED":
            raise Exception(f"Training failed: {status}")
        
        time.sleep(10)


@dsl.pipeline(name="triage-classification-pipeline")
def triage_classification_pipeline(
    runpod_api_key: str,
    runpod_endpoint: str = "k57do7afav01es",
    model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
    max_samples: int = 5000,
    epochs: int = 3,
    batch_size: int = 8,
    s3_bucket: str = "",
    aws_access_key_id: str = "",
    aws_secret_access_key: str = "",
    aws_session_token: str = "",
):
    """
    Medical Triage Classification Pipeline
    
    Trains classifier for ER triage urgency levels.
    Multi-class: Emergency, Urgent, Standard, etc.
    """
    train_task = train_triage_model(
        runpod_api_key=runpod_api_key,
        runpod_endpoint=runpod_endpoint,
        model_name=model_name,
        max_samples=max_samples,
        epochs=epochs,
        batch_size=batch_size,
        s3_bucket=s3_bucket,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
    )


# ============================================================================
# Symptom-to-Disease Classification Pipeline
# ============================================================================
@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["requests"]
)
def train_symptom_disease_model(
    runpod_api_key: str,
    runpod_endpoint: str,
    model_name: str,
    max_samples: int,
    epochs: int,
    batch_size: int,
    s3_bucket: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    aws_session_token: str,
) -> dict:
    """Train Symptom-to-Disease classifier on RunPod."""
    import requests
    import time
    
    response = requests.post(
        f"https://api.runpod.ai/v2/{runpod_endpoint}/run",
        headers={"Authorization": f"Bearer {runpod_api_key}"},
        json={
            "input": {
                "task": "symptom_disease",
                "model_name": model_name,
                "max_samples": max_samples,
                "epochs": epochs,
                "batch_size": batch_size,
                "eval_split": 0.1,
                "s3_bucket": s3_bucket,
                "s3_prefix": "symptom-disease-models/bert",
                "aws_access_key_id": aws_access_key_id,
                "aws_secret_access_key": aws_secret_access_key,
                "aws_session_token": aws_session_token,
            }
        }
    )
    
    job_id = response.json()["id"]
    print(f"RunPod job submitted: {job_id}")
    
    while True:
        status = requests.get(
            f"https://api.runpod.ai/v2/{runpod_endpoint}/status/{job_id}",
            headers={"Authorization": f"Bearer {runpod_api_key}"}
        ).json()
        
        if status["status"] == "COMPLETED":
            return status["output"]
        elif status["status"] == "FAILED":
            raise Exception(f"Training failed: {status}")
        
        time.sleep(10)


@dsl.pipeline(name="symptom-disease-classification-pipeline")
def symptom_disease_pipeline(
    runpod_api_key: str,
    runpod_endpoint: str = "k57do7afav01es",
    model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
    max_samples: int = 5000,
    epochs: int = 3,
    batch_size: int = 16,
    s3_bucket: str = "",
    aws_access_key_id: str = "",
    aws_secret_access_key: str = "",
    aws_session_token: str = "",
):
    """
    Symptom-to-Disease Classification Pipeline
    
    Predicts disease from symptom descriptions.
    Multi-class: 40+ disease categories
    """
    train_task = train_symptom_disease_model(
        runpod_api_key=runpod_api_key,
        runpod_endpoint=runpod_endpoint,
        model_name=model_name,
        max_samples=max_samples,
        epochs=epochs,
        batch_size=batch_size,
        s3_bucket=s3_bucket,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
    )


# ============================================================================
# Full Healthcare Training Pipeline (All Tasks)
# ============================================================================
@dsl.pipeline(name="healthcare-multi-task-pipeline")
def healthcare_multi_task_pipeline(
    runpod_api_key: str,
    runpod_endpoint: str = "k57do7afav01es",
    model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
    s3_bucket: str = "",
    aws_access_key_id: str = "",
    aws_secret_access_key: str = "",
    aws_session_token: str = "",
):
    """
    Train all healthcare models in parallel.
    
    Outputs:
    - ADE classifier (s3://bucket/ade-models/...)
    - Triage classifier (s3://bucket/triage-models/...)
    - Symptom-Disease classifier (s3://bucket/symptom-disease-models/...)
    """
    # Run all training tasks in parallel
    ade_task = train_ade_model(
        runpod_api_key=runpod_api_key,
        runpod_endpoint=runpod_endpoint,
        model_name=model_name,
        max_samples=10000,
        epochs=3,
        batch_size=16,
        s3_bucket=s3_bucket,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
    )
    
    triage_task = train_triage_model(
        runpod_api_key=runpod_api_key,
        runpod_endpoint=runpod_endpoint,
        model_name=model_name,
        max_samples=5000,
        epochs=3,
        batch_size=8,
        s3_bucket=s3_bucket,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
    )
    
    symptom_task = train_symptom_disease_model(
        runpod_api_key=runpod_api_key,
        runpod_endpoint=runpod_endpoint,
        model_name=model_name,
        max_samples=5000,
        epochs=3,
        batch_size=16,
        s3_bucket=s3_bucket,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
    )


if __name__ == "__main__":
    # Compile pipelines
    compiler.Compiler().compile(
        ade_classification_pipeline,
        "ade_classification_pipeline.yaml"
    )
    compiler.Compiler().compile(
        triage_classification_pipeline,
        "triage_classification_pipeline.yaml"
    )
    compiler.Compiler().compile(
        symptom_disease_pipeline,
        "symptom_disease_pipeline.yaml"
    )
    compiler.Compiler().compile(
        healthcare_multi_task_pipeline,
        "healthcare_multi_task_pipeline.yaml"
    )
    print("All pipelines compiled!")
