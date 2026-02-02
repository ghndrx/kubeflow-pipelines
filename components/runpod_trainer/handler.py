"""
RunPod Serverless Handler for DDI Model Training

This runs on RunPod GPU instances and trains the Bio_ClinicalBERT model
for drug-drug interaction detection.
"""
import os
import json
import runpod
from typing import Dict, Any


def download_from_minio(bucket: str, key: str, local_path: str):
    """Download file from MinIO."""
    import boto3
    
    s3 = boto3.client(
        's3',
        endpoint_url=os.environ['MINIO_ENDPOINT'],
        aws_access_key_id=os.environ['MINIO_ACCESS_KEY'],
        aws_secret_access_key=os.environ['MINIO_SECRET_KEY']
    )
    s3.download_file(bucket, key, local_path)


def upload_to_minio(local_path: str, bucket: str, key: str):
    """Upload file to MinIO."""
    import boto3
    
    s3 = boto3.client(
        's3',
        endpoint_url=os.environ['MINIO_ENDPOINT'],
        aws_access_key_id=os.environ['MINIO_ACCESS_KEY'],
        aws_secret_access_key=os.environ['MINIO_SECRET_KEY']
    )
    s3.upload_file(local_path, bucket, key)


def train_ddi_model(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train DDI detection model.
    
    Expected input:
    {
        "model_name": "emilyalsentzer/Bio_ClinicalBERT",
        "dataset_path": "datasets/ddi_train.json",
        "epochs": 3,
        "learning_rate": 2e-5,
        "batch_size": 16,
        "output_path": "models/ddi_model_v1"
    }
    """
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer
    )
    from datasets import Dataset
    import tempfile
    import shutil
    
    # Extract parameters
    model_name = job_input.get('model_name', 'emilyalsentzer/Bio_ClinicalBERT')
    dataset_path = job_input.get('dataset_path', 'datasets/ddi_train.json')
    epochs = job_input.get('epochs', 3)
    learning_rate = job_input.get('learning_rate', 2e-5)
    batch_size = job_input.get('batch_size', 16)
    output_path = job_input.get('output_path', 'models/ddi_model')
    
    # Create temp directory
    work_dir = tempfile.mkdtemp()
    data_file = os.path.join(work_dir, 'train.json')
    model_dir = os.path.join(work_dir, 'model')
    
    try:
        # Download training data from MinIO
        print(f"Downloading dataset from {dataset_path}...")
        download_from_minio('datasets', dataset_path, data_file)
        
        # Load dataset
        with open(data_file, 'r') as f:
            train_data = json.load(f)
        
        dataset = Dataset.from_list(train_data)
        
        # Load model and tokenizer
        print(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=5  # DDI severity levels: none, minor, moderate, major, contraindicated
        )
        
        # Tokenize dataset
        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=512
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=model_dir,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=os.path.join(work_dir, 'logs'),
            logging_steps=10,
            save_strategy='epoch',
            evaluation_strategy='epoch' if 'validation' in train_data else 'no',
            load_best_model_at_end=True if 'validation' in train_data else False,
            fp16=torch.cuda.is_available(),
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )
        
        # Train
        print("Starting training...")
        train_result = trainer.train()
        
        # Save model
        print("Saving model...")
        trainer.save_model(model_dir)
        tokenizer.save_pretrained(model_dir)
        
        # Save training metrics
        metrics = {
            'train_loss': train_result.training_loss,
            'epochs': epochs,
            'model_name': model_name,
            'samples': len(dataset)
        }
        
        with open(os.path.join(model_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f)
        
        # Upload model to MinIO
        print(f"Uploading model to {output_path}...")
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                local_file = os.path.join(root, file)
                relative_path = os.path.relpath(local_file, model_dir)
                minio_key = f"{output_path}/{relative_path}"
                upload_to_minio(local_file, 'models', minio_key)
        
        return {
            'status': 'success',
            'model_path': f"s3://models/{output_path}",
            'metrics': metrics
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }
    finally:
        # Cleanup
        shutil.rmtree(work_dir, ignore_errors=True)


def handler(job):
    """RunPod serverless handler."""
    job_input = job['input']
    return train_ddi_model(job_input)


# RunPod serverless entrypoint
runpod.serverless.start({'handler': handler})
