"""
DDI Training Pipeline with RunPod GPU

Fully automated pipeline that:
1. Preprocesses CCDA/FHIR clinical data
2. Uploads to MinIO
3. Triggers RunPod serverless GPU training
4. Evaluates and registers the model
"""
import os
from kfp import dsl
from kfp import compiler


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["boto3", "requests"]
)
def create_sample_dataset(
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    output_path: str = "ddi_train.json"
) -> str:
    """Create a sample DDI training dataset for testing."""
    import json
    import boto3
    
    # Sample DDI training data (drug pairs with interaction labels)
    # Labels: 0=none, 1=minor, 2=moderate, 3=major, 4=contraindicated
    sample_data = [
        {"text": "Patient taking warfarin and aspirin together", "label": 3},
        {"text": "Metformin administered with lisinopril", "label": 0},
        {"text": "Concurrent use of simvastatin and amiodarone", "label": 3},
        {"text": "Patient prescribed omeprazole with clopidogrel", "label": 2},
        {"text": "Fluoxetine and tramadol co-administration", "label": 4},
        {"text": "Atorvastatin given with diltiazem", "label": 2},
        {"text": "Methotrexate and NSAIDs used together", "label": 3},
        {"text": "Levothyroxine taken with calcium supplements", "label": 1},
        {"text": "Ciprofloxacin and theophylline interaction", "label": 3},
        {"text": "ACE inhibitor with potassium supplement", "label": 2},
        # Add more samples for better training
        {"text": "Digoxin and amiodarone combination therapy", "label": 3},
        {"text": "SSRIs with MAO inhibitors", "label": 4},
        {"text": "Lithium and ACE inhibitors together", "label": 3},
        {"text": "Benzodiazepines with opioids", "label": 4},
        {"text": "Metronidazole and alcohol consumption", "label": 4},
    ]
    
    # Upload to MinIO
    s3 = boto3.client(
        's3',
        endpoint_url=minio_endpoint,
        aws_access_key_id=minio_access_key,
        aws_secret_access_key=minio_secret_key,
        region_name='us-east-1'
    )
    
    data_json = json.dumps(sample_data)
    s3.put_object(
        Bucket='datasets',
        Key=output_path,
        Body=data_json.encode('utf-8'),
        ContentType='application/json'
    )
    
    print(f"Uploaded sample dataset to datasets/{output_path}")
    return output_path


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["requests"]
)
def trigger_runpod_training(
    runpod_api_key: str,
    runpod_endpoint_id: str,
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    dataset_path: str,
    model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
    epochs: int = 3,
    learning_rate: float = 2e-5,
    output_model_path: str = "ddi_model_v1"
) -> str:
    """Trigger RunPod serverless training job."""
    import requests
    import json
    import time
    
    # RunPod API endpoint
    url = f"https://api.runpod.ai/v2/{runpod_endpoint_id}/runsync"
    
    headers = {
        "Authorization": f"Bearer {runpod_api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "input": {
            "model_name": model_name,
            "dataset_path": dataset_path,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": 16,
            "output_path": output_model_path,
            # MinIO credentials for the worker
            "minio_endpoint": minio_endpoint,
            "minio_access_key": minio_access_key,
            "minio_secret_key": minio_secret_key
        }
    }
    
    print(f"Triggering RunPod training job...")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_path}")
    print(f"Epochs: {epochs}")
    
    response = requests.post(url, headers=headers, json=payload, timeout=3600)
    result = response.json()
    
    if response.status_code != 200:
        raise Exception(f"RunPod API error: {result}")
    
    if result.get('status') == 'FAILED':
        raise Exception(f"Training failed: {result.get('error')}")
    
    output = result.get('output', {})
    print(f"Training complete!")
    print(f"Model path: {output.get('model_path')}")
    print(f"Metrics: {output.get('metrics')}")
    
    return output.get('model_path', f"s3://models/{output_model_path}")


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["boto3"]
)
def register_model(
    model_path: str,
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    model_name: str = "ddi-detector",
    version: str = "v1"
) -> str:
    """Register the trained model in the model registry."""
    import boto3
    import json
    from datetime import datetime
    
    s3 = boto3.client(
        's3',
        endpoint_url=minio_endpoint,
        aws_access_key_id=minio_access_key,
        aws_secret_access_key=minio_secret_key,
        region_name='us-east-1'
    )
    
    # Create model registry entry
    registry_entry = {
        "name": model_name,
        "version": version,
        "path": model_path,
        "created_at": datetime.utcnow().isoformat(),
        "framework": "transformers",
        "task": "sequence-classification",
        "labels": ["none", "minor", "moderate", "major", "contraindicated"]
    }
    
    registry_key = f"registry/{model_name}/{version}/metadata.json"
    s3.put_object(
        Bucket='models',
        Key=registry_key,
        Body=json.dumps(registry_entry).encode('utf-8'),
        ContentType='application/json'
    )
    
    print(f"Model registered: {model_name} v{version}")
    print(f"Registry path: models/{registry_key}")
    
    return f"models/{registry_key}"


@dsl.pipeline(
    name="ddi-training-runpod",
    description="Train DDI detection model using RunPod serverless GPU"
)
def ddi_training_pipeline(
    # RunPod settings
    runpod_endpoint_id: str = "YOUR_ENDPOINT_ID",
    
    # Model settings
    model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
    epochs: int = 3,
    learning_rate: float = 2e-5,
    model_version: str = "v1",
    
    # MinIO settings (these will be injected from secrets)
    minio_endpoint: str = "https://minio.walleye-frog.ts.net",
):
    """
    Full DDI training pipeline:
    1. Create/upload sample dataset
    2. Trigger RunPod GPU training
    3. Register trained model
    """
    import os
    
    # These would come from k8s secrets in production
    minio_access_key = "minioadmin"
    minio_secret_key = "minioadmin123!"
    runpod_api_key = os.environ.get("RUNPOD_API_KEY", "")
    
    # Step 1: Create sample dataset
    dataset_task = create_sample_dataset(
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        output_path=f"ddi_train_{model_version}.json"
    )
    
    # Step 2: Trigger RunPod training
    training_task = trigger_runpod_training(
        runpod_api_key=runpod_api_key,
        runpod_endpoint_id=runpod_endpoint_id,
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        dataset_path=dataset_task.output,
        model_name=model_name,
        epochs=epochs,
        learning_rate=learning_rate,
        output_model_path=f"ddi_model_{model_version}"
    )
    
    # Step 3: Register model
    register_task = register_model(
        model_path=training_task.output,
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        model_name="ddi-detector",
        version=model_version
    )


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=ddi_training_pipeline,
        package_path="ddi_training_runpod.yaml"
    )
    print("Pipeline compiled to ddi_training_runpod.yaml")
