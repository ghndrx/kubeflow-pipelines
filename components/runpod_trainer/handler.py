"""
RunPod Serverless Handler for DDI Model Training

This runs on RunPod GPU instances and trains the Bio_ClinicalBERT model
for drug-drug interaction detection using real DrugBank data via TDC.
"""
import os
import json
import runpod
from typing import Dict, Any, List, Optional


# DrugBank DDI type mapping to severity categories
# TDC DrugBank has 86 interaction types - we map to 5 severity levels
DDI_SEVERITY_MAP = {
    # 0 = No significant interaction / safe
    'no known interaction': 0,
    
    # 1 = Minor interaction (mechanism-based, low clinical impact)
    'the metabolism of drug1 can be increased': 1,
    'the metabolism of drug1 can be decreased': 1,
    'the absorption of drug1 can be affected': 1,
    'the bioavailability of drug1 can be affected': 1,
    'drug1 may affect the excretion rate': 1,
    
    # 2 = Moderate interaction (effect-based, monitor patient)
    'the serum concentration of drug1 can be increased': 2,
    'the serum concentration of drug1 can be decreased': 2,
    'the therapeutic efficacy of drug1 can be decreased': 2,
    'the therapeutic efficacy of drug1 can be increased': 2,
    'the protein binding of drug1 can be affected': 2,
    
    # 3 = Major interaction (significant risk, avoid if possible)
    'the risk or severity of adverse effects can be increased': 3,
    'the risk of bleeding can be increased': 3,
    'the risk of hypotension can be increased': 3,
    'the risk of hypertension can be increased': 3,
    'the risk of hypoglycemia can be increased': 3,
    'the risk of hyperglycemia can be increased': 3,
    'the risk of QTc prolongation can be increased': 3,
    'the risk of cardiotoxicity can be increased': 3,
    'the risk of nephrotoxicity can be increased': 3,
    'the risk of hepatotoxicity can be increased': 3,
    
    # 4 = Contraindicated (avoid combination)
    'the risk of serotonin syndrome can be increased': 4,
    'the risk of rhabdomyolysis can be increased': 4,
    'the risk of severe hypotension can be increased': 4,
    'the risk of life-threatening arrhythmias can be increased': 4,
}


def get_severity_label(ddi_type: str) -> int:
    """Map DDI type string to severity label (0-4)."""
    ddi_lower = ddi_type.lower()
    
    # Check exact matches first
    for pattern, label in DDI_SEVERITY_MAP.items():
        if pattern in ddi_lower:
            return label
    
    # Default heuristics based on keywords
    if any(x in ddi_lower for x in ['contraindicated', 'life-threatening', 'fatal', 'death']):
        return 4
    elif any(x in ddi_lower for x in ['severe', 'serious', 'major', 'toxic']):
        return 3
    elif any(x in ddi_lower for x in ['increased', 'decreased', 'risk', 'adverse']):
        return 2
    elif any(x in ddi_lower for x in ['may', 'can', 'affect', 'metabolism']):
        return 1
    else:
        return 0  # Unknown/no interaction


def load_drugbank_ddi(max_samples: int = 50000) -> List[Dict[str, Any]]:
    """
    Load DrugBank DDI dataset from TDC (Therapeutics Data Commons).
    
    Returns list of {"text": "drug1 drug2 interaction_description", "label": severity}
    """
    from tdc.multi_pred import DDI
    import pandas as pd
    
    print("Loading DrugBank DDI dataset from TDC...")
    
    # Load the DrugBank DDI dataset
    data = DDI(name='DrugBank')
    df = data.get_data()
    
    print(f"Total DDI pairs in DrugBank: {len(df)}")
    
    # Sample if dataset is too large
    if len(df) > max_samples:
        print(f"Sampling {max_samples} examples...")
        df = df.sample(n=max_samples, random_state=42)
    
    # Convert to training format
    training_data = []
    for _, row in df.iterrows():
        drug1 = row['Drug1']
        drug2 = row['Drug2']
        ddi_type = row['Y']  # Interaction type string
        
        # Create text input
        text = f"{drug1} {drug2} {ddi_type}"
        
        # Map to severity label
        label = get_severity_label(ddi_type)
        
        training_data.append({
            'text': text,
            'label': label
        })
    
    # Log label distribution
    label_counts = {}
    for item in training_data:
        label_counts[item['label']] = label_counts.get(item['label'], 0) + 1
    
    print(f"Label distribution: {label_counts}")
    print(f"Total training samples: {len(training_data)}")
    
    return training_data


def train_ddi_model(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train DDI detection model.
    
    Expected input:
    {
        "model_name": "emilyalsentzer/Bio_ClinicalBERT",
        "use_drugbank": true,  # Use real DrugBank data
        "max_samples": 50000,  # Max samples to use
        "training_data": [...],  # Or provide inline data
        "epochs": 3,
        "learning_rate": 2e-5,
        "batch_size": 16,
        "eval_split": 0.1  # Validation split ratio
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
    from sklearn.model_selection import train_test_split
    import tempfile
    import shutil
    
    # Extract parameters
    model_name = job_input.get('model_name', 'emilyalsentzer/Bio_ClinicalBERT')
    use_drugbank = job_input.get('use_drugbank', True)
    max_samples = job_input.get('max_samples', 50000)
    training_data = job_input.get('training_data', None)
    epochs = job_input.get('epochs', 3)
    learning_rate = job_input.get('learning_rate', 2e-5)
    batch_size = job_input.get('batch_size', 16)
    eval_split = job_input.get('eval_split', 0.1)
    
    # Load data
    if use_drugbank and not training_data:
        print("Loading real DrugBank DDI dataset...")
        training_data = load_drugbank_ddi(max_samples=max_samples)
    elif not training_data:
        print("No training data provided, using sample DDI dataset...")
        training_data = [
            {"text": "warfarin aspirin the risk of bleeding can be increased", "label": 3},
            {"text": "metformin lisinopril no known interaction", "label": 0},
            {"text": "fluoxetine tramadol the risk of serotonin syndrome can be increased", "label": 4},
            {"text": "simvastatin amiodarone the risk of rhabdomyolysis can be increased", "label": 4},
            {"text": "omeprazole clopidogrel the therapeutic efficacy of drug1 can be decreased", "label": 2},
        ] * 30  # 150 samples
    
    # Create temp directory
    work_dir = tempfile.mkdtemp()
    model_dir = os.path.join(work_dir, 'model')
    
    try:
        print(f"Training samples: {len(training_data)}")
        print(f"Model: {model_name}")
        print(f"Epochs: {epochs}, Batch size: {batch_size}")
        print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        
        # Split into train/eval
        if eval_split > 0 and len(training_data) > 100:
            train_data, eval_data = train_test_split(
                training_data, 
                test_size=eval_split, 
                random_state=42,
                stratify=[d['label'] for d in training_data]
            )
            print(f"Train: {len(train_data)}, Eval: {len(eval_data)}")
        else:
            train_data = training_data
            eval_data = None
        
        # Create datasets
        train_dataset = Dataset.from_list(train_data)
        eval_dataset = Dataset.from_list(eval_data) if eval_data else None
        
        # Load model and tokenizer
        print(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=5  # DDI severity: none(0), minor(1), moderate(2), major(3), contraindicated(4)
        )
        
        # Tokenize datasets
        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=256  # Longer for drug names + interaction text
            )
        
        tokenized_train = train_dataset.map(tokenize_function, batched=True)
        tokenized_eval = eval_dataset.map(tokenize_function, batched=True) if eval_dataset else None
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=model_dir,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_ratio=0.1,
            weight_decay=0.01,
            logging_steps=50,
            eval_strategy='epoch' if tokenized_eval else 'no',
            save_strategy='no',  # Don't save checkpoints
            fp16=torch.cuda.is_available(),
            report_to='none',
            load_best_model_at_end=False,
        )
        
        # Compute metrics function
        def compute_metrics(eval_pred):
            from sklearn.metrics import accuracy_score, f1_score
            predictions, labels = eval_pred
            predictions = predictions.argmax(-1)
            return {
                'accuracy': accuracy_score(labels, predictions),
                'f1_macro': f1_score(labels, predictions, average='macro'),
                'f1_weighted': f1_score(labels, predictions, average='weighted'),
            }
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            compute_metrics=compute_metrics if tokenized_eval else None,
        )
        
        # Train
        print("Starting training...")
        train_result = trainer.train()
        
        # Get metrics
        metrics = {
            'train_loss': float(train_result.training_loss),
            'epochs': epochs,
            'model_name': model_name,
            'samples': len(train_data),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
            'data_source': 'DrugBank' if use_drugbank else 'custom'
        }
        
        # Run evaluation if we have eval data
        if tokenized_eval:
            print("Running evaluation...")
            eval_result = trainer.evaluate()
            metrics.update({
                'eval_loss': float(eval_result['eval_loss']),
                'eval_accuracy': float(eval_result['eval_accuracy']),
                'eval_f1_macro': float(eval_result['eval_f1_macro']),
                'eval_f1_weighted': float(eval_result['eval_f1_weighted']),
            })
        
        print(f"Training complete! Loss: {metrics['train_loss']:.4f}")
        if 'eval_accuracy' in metrics:
            print(f"Eval accuracy: {metrics['eval_accuracy']:.4f}, F1: {metrics['eval_f1_weighted']:.4f}")
        
        return {
            'status': 'success',
            'metrics': metrics,
            'message': 'Model trained successfully on DrugBank DDI data'
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
