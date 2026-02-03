"""
RunPod Serverless Handler for DDI Model Training

This runs on RunPod GPU instances and trains the Bio_ClinicalBERT model
for drug-drug interaction detection.
"""
import os
import json
import runpod
from typing import Dict, Any, List, Optional


def train_ddi_model(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train DDI detection model.
    
    Expected input:
    {
        "model_name": "emilyalsentzer/Bio_ClinicalBERT",
        "training_data": [{"text": "...", "label": 0}, ...],  # Inline data
        "epochs": 3,
        "learning_rate": 2e-5,
        "batch_size": 16
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
    training_data = job_input.get('training_data', None)
    epochs = job_input.get('epochs', 3)
    learning_rate = job_input.get('learning_rate', 2e-5)
    batch_size = job_input.get('batch_size', 16)
    
    # Use sample data if none provided
    if not training_data:
        print("No training data provided, using sample DDI dataset...")
        training_data = [
            {"text": "warfarin and aspirin interaction causes bleeding risk", "label": 3},
            {"text": "metformin with lisinopril is safe combination", "label": 0},
            {"text": "fluoxetine tramadol causes serotonin syndrome", "label": 4},
            {"text": "simvastatin amiodarone increases myopathy risk", "label": 3},
            {"text": "omeprazole reduces clopidogrel efficacy", "label": 2},
            {"text": "digoxin amiodarone toxicity risk elevated", "label": 3},
            {"text": "lithium NSAIDs increases lithium levels", "label": 3},
            {"text": "benzodiazepines opioids respiratory depression", "label": 4},
            {"text": "metronidazole alcohol disulfiram reaction", "label": 4},
            {"text": "ACE inhibitors potassium hyperkalemia risk", "label": 2},
            {"text": "amlodipine atorvastatin safe combination", "label": 0},
            {"text": "gabapentin pregabalin CNS depression additive", "label": 2},
            {"text": "warfarin vitamin K antagonism reduced effect", "label": 2},
            {"text": "insulin metformin hypoglycemia risk", "label": 1},
            {"text": "aspirin ibuprofen GI bleeding increased", "label": 3},
        ] * 10  # 150 samples
    
    # Create temp directory
    work_dir = tempfile.mkdtemp()
    model_dir = os.path.join(work_dir, 'model')
    
    try:
        print(f"Training samples: {len(training_data)}")
        print(f"Model: {model_name}")
        print(f"Epochs: {epochs}, Batch size: {batch_size}")
        print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        
        # Load dataset
        dataset = Dataset.from_list(training_data)
        
        # Load model and tokenizer
        print(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=5  # DDI severity: none, minor, moderate, major, contraindicated
        )
        
        # Tokenize dataset
        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=128
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=model_dir,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            warmup_steps=50,
            weight_decay=0.01,
            logging_steps=10,
            save_strategy='no',  # Don't save checkpoints (avoids tensor contiguity issues)
            fp16=torch.cuda.is_available(),
            report_to='none',
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
        
        # Get metrics
        metrics = {
            'train_loss': float(train_result.training_loss),
            'epochs': epochs,
            'model_name': model_name,
            'samples': len(training_data),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
        }
        
        print(f"Training complete! Loss: {metrics['train_loss']:.4f}")
        
        return {
            'status': 'success',
            'metrics': metrics,
            'message': 'Model trained successfully'
        }
        
    except Exception as e:
        import traceback
        return {
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }
    finally:
        # Cleanup
        shutil.rmtree(work_dir, ignore_errors=True)


def handler(job):
    """RunPod serverless handler."""
    job_input = job.get('input', {})
    return train_ddi_model(job_input)


# RunPod serverless entrypoint
runpod.serverless.start({'handler': handler})
