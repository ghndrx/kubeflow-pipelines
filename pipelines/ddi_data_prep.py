"""
DDI Data Preparation Pipeline

Prepares training data for DDI detection model.
Training can be triggered manually on RunPod or any GPU environment.
"""
from kfp import dsl
from kfp import compiler


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["boto3", "botocore", "requests"]
)
def create_ddi_dataset(
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    output_path: str = "ddi_train.json"
) -> str:
    """Create DDI training dataset and upload to MinIO."""
    import json
    import boto3
    
    # DDI training data (drug pairs with interaction severity)
    # Labels: 0=none, 1=minor, 2=moderate, 3=major, 4=contraindicated
    training_data = [
        # Major interactions
        {"text": "Patient taking warfarin and aspirin together", "label": 3},
        {"text": "Concurrent use of simvastatin and amiodarone", "label": 3},
        {"text": "Methotrexate and NSAIDs used together", "label": 3},
        {"text": "Ciprofloxacin and theophylline interaction", "label": 3},
        {"text": "Digoxin and amiodarone combination therapy", "label": 3},
        {"text": "Lithium and ACE inhibitors together", "label": 3},
        
        # Contraindicated
        {"text": "Fluoxetine and tramadol co-administration", "label": 4},
        {"text": "SSRIs with MAO inhibitors", "label": 4},
        {"text": "Benzodiazepines with opioids", "label": 4},
        {"text": "Metronidazole and alcohol consumption", "label": 4},
        {"text": "Linezolid with serotonergic drugs", "label": 4},
        
        # Moderate
        {"text": "Patient prescribed omeprazole with clopidogrel", "label": 2},
        {"text": "Atorvastatin given with diltiazem", "label": 2},
        {"text": "ACE inhibitor with potassium supplement", "label": 2},
        {"text": "Metformin with contrast dye procedures", "label": 2},
        
        # Minor
        {"text": "Levothyroxine taken with calcium supplements", "label": 1},
        {"text": "Antacids with oral antibiotics timing", "label": 1},
        {"text": "Iron supplements with dairy products", "label": 1},
        
        # No interaction
        {"text": "Metformin administered with lisinopril", "label": 0},
        {"text": "Amlodipine with metoprolol combination", "label": 0},
        {"text": "Omeprazole and acetaminophen together", "label": 0},
        {"text": "Vitamin D with calcium supplements", "label": 0},
    ]
    
    # Upload to MinIO with proper config for Tailscale endpoints
    from botocore.config import Config
    
    s3_config = Config(
        connect_timeout=30,
        read_timeout=60,
        retries={'max_attempts': 3},
        s3={'addressing_style': 'path'}
    )
    
    s3 = boto3.client(
        's3',
        endpoint_url=minio_endpoint,
        aws_access_key_id=minio_access_key,
        aws_secret_access_key=minio_secret_key,
        region_name='us-east-1',
        config=s3_config,
        verify=True
    )
    
    data_json = json.dumps(training_data, indent=2)
    s3.put_object(
        Bucket='datasets',
        Key=output_path,
        Body=data_json.encode('utf-8'),
        ContentType='application/json'
    )
    
    print(f"✅ Uploaded {len(training_data)} samples to datasets/{output_path}")
    print(f"   - Contraindicated: {sum(1 for d in training_data if d['label'] == 4)}")
    print(f"   - Major: {sum(1 for d in training_data if d['label'] == 3)}")
    print(f"   - Moderate: {sum(1 for d in training_data if d['label'] == 2)}")
    print(f"   - Minor: {sum(1 for d in training_data if d['label'] == 1)}")
    print(f"   - None: {sum(1 for d in training_data if d['label'] == 0)}")
    
    return f"s3://datasets/{output_path}"


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["boto3"]
)
def create_training_config(
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    dataset_path: str,
    model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
    epochs: int = 3,
    learning_rate: float = 2e-5,
    batch_size: int = 16
) -> str:
    """Create training configuration file."""
    import json
    import boto3
    from datetime import datetime
    
    config = {
        "created_at": datetime.utcnow().isoformat(),
        "dataset": {
            "path": dataset_path,
            "format": "json",
            "text_field": "text",
            "label_field": "label"
        },
        "model": {
            "base_model": model_name,
            "num_labels": 5,
            "label_names": ["none", "minor", "moderate", "major", "contraindicated"]
        },
        "training": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "warmup_steps": 100,
            "weight_decay": 0.01,
            "fp16": True,
            "evaluation_strategy": "epoch",
            "save_strategy": "epoch"
        },
        "output": {
            "model_path": "models/ddi-detector",
            "metrics_path": "models/ddi-detector/metrics.json"
        }
    }
    
    s3 = boto3.client(
        's3',
        endpoint_url=minio_endpoint,
        aws_access_key_id=minio_access_key,
        aws_secret_access_key=minio_secret_key,
        region_name='us-east-1'
    )
    
    config_json = json.dumps(config, indent=2)
    config_path = "configs/ddi_training_config.json"
    
    s3.put_object(
        Bucket='training-data',
        Key=config_path,
        Body=config_json.encode('utf-8'),
        ContentType='application/json'
    )
    
    print(f"✅ Training config saved to training-data/{config_path}")
    print(f"   Model: {model_name}")
    print(f"   Epochs: {epochs}")
    print(f"   Learning rate: {learning_rate}")
    
    return f"s3://training-data/{config_path}"


@dsl.pipeline(
    name="ddi-data-preparation",
    description="Prepare DDI training data and configuration"
)
def ddi_data_prep_pipeline(
    model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
    epochs: int = 3,
    learning_rate: float = 2e-5,
    minio_endpoint: str = "http://minio.minio.svc.cluster.local:9000",
):
    """
    Data preparation pipeline:
    1. Create DDI training dataset
    2. Generate training configuration
    
    After this completes, run training manually on RunPod:
    ```
    python train.py --config s3://training-data/configs/ddi_training_config.json
    ```
    """
    minio_access_key = "minioadmin"
    minio_secret_key = "minioadmin123!"
    
    # Create dataset
    dataset_task = create_ddi_dataset(
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        output_path="ddi_train.json"
    )
    
    # Create config
    config_task = create_training_config(
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        dataset_path=dataset_task.output,
        model_name=model_name,
        epochs=epochs,
        learning_rate=learning_rate
    )


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=ddi_data_prep_pipeline,
        package_path="ddi_data_prep.yaml"
    )
    print("Pipeline compiled to ddi_data_prep.yaml")
