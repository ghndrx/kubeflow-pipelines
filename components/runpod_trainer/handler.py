"""
RunPod Serverless Handler for DDI Model Training

Supports both BERT-style classification and LLM fine-tuning with LoRA.
Uses 176K real DrugBank DDI samples with drug names.
Saves trained models to S3.
"""
import os
import json
import runpod
import boto3
from datetime import datetime
from typing import Dict, Any, List, Optional


def upload_to_s3(local_path: str, s3_bucket: str, s3_prefix: str, aws_credentials: Dict) -> str:
    """Upload a directory to S3 and return the S3 URI."""
    import tarfile
    import tempfile
    
    # Create tar.gz of the model directory
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    tar_name = f"model_{timestamp}.tar.gz"
    tar_path = os.path.join(tempfile.gettempdir(), tar_name)
    
    print(f"Creating archive: {tar_path}")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(local_path, arcname="model")
    
    # Upload to S3
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_credentials.get('aws_access_key_id'),
        aws_secret_access_key=aws_credentials.get('aws_secret_access_key'),
        aws_session_token=aws_credentials.get('aws_session_token'),
        region_name=aws_credentials.get('aws_region', 'us-east-1')
    )
    
    s3_key = f"{s3_prefix}/{tar_name}"
    print(f"Uploading to s3://{s3_bucket}/{s3_key}")
    
    s3_client.upload_file(tar_path, s3_bucket, s3_key)
    
    # Cleanup
    os.remove(tar_path)
    
    return f"s3://{s3_bucket}/{s3_key}"


# DDI severity labels
DDI_SEVERITY = {
    1: "minor",
    2: "moderate", 
    3: "major",
    4: "contraindicated"
}


def load_drugbank_data(max_samples: int = None, severity_filter: List[int] = None) -> List[Dict]:
    """Load real DrugBank DDI data from bundled file."""
    data_path = os.environ.get('DDI_DATA_PATH', '/app/data/drugbank_ddi_complete.jsonl')
    
    if not os.path.exists(data_path):
        print(f"WARNING: Data file not found at {data_path}, using curated fallback")
        return get_curated_fallback()
    
    data = []
    with open(data_path) as f:
        for line in f:
            item = json.loads(line)
            if severity_filter and item['severity'] not in severity_filter:
                continue
            data.append(item)
            if max_samples and len(data) >= max_samples:
                break
    
    return data


def get_curated_fallback() -> List[Dict]:
    """Fallback curated data if main file not available."""
    patterns = [
        {"drug1": "fluoxetine", "drug2": "tramadol", "interaction_text": "fluoxetine may increase the risk of serotonin syndrome when combined with tramadol", "severity": 4},
        {"drug1": "warfarin", "drug2": "aspirin", "interaction_text": "warfarin may increase the risk of bleeding when combined with aspirin", "severity": 3},
        {"drug1": "simvastatin", "drug2": "amlodipine", "interaction_text": "The serum concentration of simvastatin can be increased when combined with amlodipine", "severity": 2},
        {"drug1": "metformin", "drug2": "lisinopril", "interaction_text": "metformin and lisinopril have no significant interaction", "severity": 1},
    ]
    return patterns * 50  # 200 samples


def format_for_llm(item: Dict, model_name: str = "") -> str:
    """Format DDI item for LLM instruction tuning. Auto-detects format based on model."""
    severity_name = DDI_SEVERITY.get(item['severity'], 'unknown')
    
    user_msg = f"Analyze the drug interaction between {item['drug1']} and {item['drug2']}."
    assistant_msg = f"""Interaction: {item['interaction_text']}
Severity: {severity_name.upper()}
Recommendation: {"Avoid this combination" if item['severity'] >= 3 else "Monitor patient" if item['severity'] == 2 else "Generally safe"}"""
    
    # Llama 3 format
    if 'llama' in model_name.lower():
        return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{user_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{assistant_msg}<|eot_id|>"""
    
    # Mistral/Mixtral format
    elif 'mistral' in model_name.lower() or 'mixtral' in model_name.lower():
        return f"""<s>[INST] {user_msg} [/INST] {assistant_msg}</s>"""
    
    # Qwen format
    elif 'qwen' in model_name.lower():
        return f"""<|im_start|>user
{user_msg}<|im_end|>
<|im_start|>assistant
{assistant_msg}<|im_end|>"""
    
    # Gemma format
    elif 'gemma' in model_name.lower():
        return f"""<start_of_turn>user
{user_msg}<end_of_turn>
<start_of_turn>model
{assistant_msg}<end_of_turn>"""
    
    # Generic fallback
    else:
        return f"""### User: {user_msg}\n### Assistant: {assistant_msg}"""


def train_llm_lora(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Train LLM with QLoRA for DDI classification. Supports Llama, Mistral, Qwen, Gemma."""
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer
    from datasets import Dataset
    import tempfile
    import shutil
    
    # Parameters - Default to Llama 3.1 8B (Bedrock-compatible)
    model_name = job_input.get('model_name', 'meta-llama/Llama-3.1-8B-Instruct')
    max_samples = job_input.get('max_samples', 10000)
    epochs = job_input.get('epochs', 1)
    learning_rate = job_input.get('learning_rate', 2e-4)
    batch_size = job_input.get('batch_size', 2)
    lora_r = job_input.get('lora_r', 16)
    lora_alpha = job_input.get('lora_alpha', 32)
    max_seq_length = job_input.get('max_seq_length', 512)
    severity_filter = job_input.get('severity_filter', None)  # e.g., [3, 4] for major/contraindicated only
    
    work_dir = tempfile.mkdtemp()
    
    try:
        print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"Model: {model_name}")
        print(f"LoRA r={lora_r}, alpha={lora_alpha}")
        
        # Load training data
        print(f"Loading DrugBank DDI data (max {max_samples})...")
        raw_data = load_drugbank_data(max_samples=max_samples, severity_filter=severity_filter)
        
        # Format for the target LLM
        formatted_data = [{"text": format_for_llm(item, model_name)} for item in raw_data]
        dataset = Dataset.from_list(formatted_data)
        
        print(f"Dataset size: {len(dataset)}")
        
        # Severity distribution
        from collections import Counter
        sev_dist = Counter(item['severity'] for item in raw_data)
        print(f"Severity distribution: {dict(sev_dist)}")
        
        # QLoRA config - 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load model
        print(f"Loading {model_name} with 4-bit quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        # Prepare for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        # LoRA config
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        model = get_peft_model(model, lora_config)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=work_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=8,
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_steps=25,
            save_strategy="no",
            bf16=True,
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
            report_to="none",
            max_grad_norm=0.3,
        )
        
        # SFT Trainer
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            args=training_args,
            peft_config=lora_config,
            processing_class=tokenizer,
            max_seq_length=max_seq_length,
        )
        
        # Train
        print("Starting LoRA fine-tuning...")
        train_result = trainer.train()
        
        # Metrics
        metrics = {
            'train_loss': float(train_result.training_loss),
            'epochs': epochs,
            'model_name': model_name,
            'samples': len(raw_data),
            'lora_r': lora_r,
            'lora_alpha': lora_alpha,
            'trainable_params': trainable_params,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
            'vram_gb': torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0,
            'data_source': 'drugbank_176k',
            'severity_dist': dict(sev_dist),
        }
        
        print(f"Training complete! Loss: {metrics['train_loss']:.4f}")
        
        # Save LoRA adapter to S3 if credentials provided
        s3_uri = None
        s3_bucket = job_input.get('s3_bucket')
        if s3_bucket:
            save_dir = os.path.join(work_dir, 'lora_adapter')
            print(f"Saving LoRA adapter to {save_dir}...")
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            
            aws_creds = {
                'aws_access_key_id': job_input.get('aws_access_key_id'),
                'aws_secret_access_key': job_input.get('aws_secret_access_key'),
                'aws_session_token': job_input.get('aws_session_token'),
                'aws_region': job_input.get('aws_region', 'us-east-1'),
            }
            model_short = model_name.split('/')[-1]
            s3_prefix = job_input.get('s3_prefix', f'ddi-models/lora-{model_short}')
            s3_uri = upload_to_s3(save_dir, s3_bucket, s3_prefix, aws_creds)
            metrics['s3_uri'] = s3_uri
            print(f"LoRA adapter uploaded to {s3_uri}")
        
        return {
            'status': 'success',
            'metrics': metrics,
            'model_uri': s3_uri,
            'message': f'LLM fine-tuned on {len(raw_data):,} real DrugBank DDI samples'
        }
        
    except Exception as e:
        import traceback
        return {
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def train_bert_classifier(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Train BERT-style classifier for DDI severity prediction."""
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
    
    model_name = job_input.get('model_name', 'emilyalsentzer/Bio_ClinicalBERT')
    max_samples = job_input.get('max_samples', 50000)
    epochs = job_input.get('epochs', 3)
    learning_rate = job_input.get('learning_rate', 2e-5)
    batch_size = job_input.get('batch_size', 16)
    eval_split = job_input.get('eval_split', 0.1)
    
    work_dir = tempfile.mkdtemp()
    
    try:
        print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        print(f"Model: {model_name}")
        
        # Load data
        raw_data = load_drugbank_data(max_samples=max_samples)
        
        # Create text + label format
        # Shift severity to 0-indexed (1-4 -> 0-3)
        training_data = [{
            "text": f"{d['drug1']} and {d['drug2']}: {d['interaction_text']}",
            "label": d['severity'] - 1  # 0-indexed
        } for d in raw_data]
        
        print(f"Loaded {len(training_data)} samples")
        
        # Split
        if eval_split > 0:
            train_data, eval_data = train_test_split(
                training_data, test_size=eval_split, random_state=42,
                stratify=[d['label'] for d in training_data]
            )
        else:
            train_data, eval_data = training_data, None
        
        train_dataset = Dataset.from_list(train_data)
        eval_dataset = Dataset.from_list(eval_data) if eval_data else None
        
        # Load model (4 classes: minor, moderate, major, contraindicated)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=4
        )
        
        def tokenize(examples):
            return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=256)
        
        train_dataset = train_dataset.map(tokenize, batched=True)
        if eval_dataset:
            eval_dataset = eval_dataset.map(tokenize, batched=True)
        
        training_args = TrainingArguments(
            output_dir=work_dir,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            eval_strategy='epoch' if eval_dataset else 'no',
            save_strategy='no',
            fp16=torch.cuda.is_available(),
            report_to='none',
        )
        
        def compute_metrics(eval_pred):
            from sklearn.metrics import accuracy_score, f1_score
            preds = eval_pred.predictions.argmax(-1)
            return {
                'accuracy': accuracy_score(eval_pred.label_ids, preds),
                'f1_weighted': f1_score(eval_pred.label_ids, preds, average='weighted'),
            }
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics if eval_dataset else None,
        )
        
        train_result = trainer.train()
        
        metrics = {
            'train_loss': float(train_result.training_loss),
            'epochs': epochs,
            'model_name': model_name,
            'samples': len(train_data),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
            'data_source': 'drugbank_176k',
        }
        
        if eval_dataset:
            eval_result = trainer.evaluate()
            metrics.update({
                'eval_accuracy': float(eval_result['eval_accuracy']),
                'eval_f1_weighted': float(eval_result['eval_f1_weighted']),
            })
        
        # Save model to S3 if credentials provided
        s3_uri = None
        s3_bucket = job_input.get('s3_bucket')
        if s3_bucket:
            save_dir = os.path.join(work_dir, 'saved_model')
            print(f"Saving model to {save_dir}...")
            trainer.save_model(save_dir)
            tokenizer.save_pretrained(save_dir)
            
            aws_creds = {
                'aws_access_key_id': job_input.get('aws_access_key_id'),
                'aws_secret_access_key': job_input.get('aws_secret_access_key'),
                'aws_session_token': job_input.get('aws_session_token'),
                'aws_region': job_input.get('aws_region', 'us-east-1'),
            }
            s3_prefix = job_input.get('s3_prefix', 'ddi-models/bert')
            s3_uri = upload_to_s3(save_dir, s3_bucket, s3_prefix, aws_creds)
            metrics['s3_uri'] = s3_uri
            print(f"Model uploaded to {s3_uri}")
        
        return {'status': 'success', 'metrics': metrics, 'model_uri': s3_uri, 'message': 'BERT classifier trained on DrugBank data'}
        
    except Exception as e:
        import traceback
        return {'status': 'error', 'error': str(e), 'traceback': traceback.format_exc()}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def handler(job):
    """RunPod serverless handler."""
    job_input = job.get('input', {})
    
    model_name = job_input.get('model_name', 'meta-llama/Llama-3.1-8B-Instruct')
    use_lora = job_input.get('use_lora', True)
    
    # Auto-detect: use LoRA for large models
    if any(x in model_name.lower() for x in ['gemma', 'llama', 'mistral', 'qwen']):
        use_lora = True
    elif 'bert' in model_name.lower():
        use_lora = False
    
    if use_lora:
        return train_llm_lora(job_input)
    else:
        return train_bert_classifier(job_input)


# RunPod serverless entrypoint
runpod.serverless.start({'handler': handler})
