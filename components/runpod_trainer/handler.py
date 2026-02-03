"""
RunPod Serverless Handler for DDI Model Training

Supports both BERT-style classification and LLM fine-tuning with LoRA.
Default: Gemma 3 12B with QLoRA for DDI severity classification.
"""
import os
import json
import runpod
from typing import Dict, Any, List, Optional


# DDI severity labels
DDI_LABELS = {
    0: "no_interaction",
    1: "minor",
    2: "moderate", 
    3: "major",
    4: "contraindicated"
}

LABEL_DESCRIPTIONS = {
    0: "No clinically significant interaction",
    1: "Minor interaction - minimal clinical significance",
    2: "Moderate interaction - may require monitoring or dose adjustment",
    3: "Major interaction - avoid combination if possible, high risk",
    4: "Contraindicated - do not use together, life-threatening risk"
}


def get_ddi_training_data(max_samples: int = 5000) -> List[Dict[str, Any]]:
    """Generate DDI training data formatted for instruction tuning."""
    import random
    random.seed(42)
    
    # Real drug interaction patterns based on clinical data
    ddi_patterns = [
        # Contraindicated (4)
        {"drugs": ["fluoxetine", "tramadol"], "type": "serotonin syndrome risk", "label": 4},
        {"drugs": ["fluoxetine", "phenelzine"], "type": "serotonin syndrome risk", "label": 4},
        {"drugs": ["simvastatin", "itraconazole"], "type": "rhabdomyolysis risk", "label": 4},
        {"drugs": ["methotrexate", "trimethoprim"], "type": "severe bone marrow suppression", "label": 4},
        {"drugs": ["warfarin", "miconazole"], "type": "severe bleeding risk", "label": 4},
        {"drugs": ["cisapride", "erythromycin"], "type": "QT prolongation cardiac arrest", "label": 4},
        {"drugs": ["pimozide", "clarithromycin"], "type": "QT prolongation risk", "label": 4},
        {"drugs": ["ergotamine", "ritonavir"], "type": "ergot toxicity risk", "label": 4},
        {"drugs": ["sildenafil", "nitroglycerin"], "type": "severe hypotension", "label": 4},
        {"drugs": ["linezolid", "sertraline"], "type": "serotonin syndrome", "label": 4},
        {"drugs": ["maoi", "meperidine"], "type": "hypertensive crisis", "label": 4},
        {"drugs": ["metronidazole", "disulfiram"], "type": "psychosis risk", "label": 4},
        
        # Major (3)
        {"drugs": ["warfarin", "aspirin"], "type": "increased bleeding risk", "label": 3},
        {"drugs": ["digoxin", "amiodarone"], "type": "digoxin toxicity elevated", "label": 3},
        {"drugs": ["lithium", "ibuprofen"], "type": "lithium toxicity risk", "label": 3},
        {"drugs": ["metformin", "iodinated contrast"], "type": "lactic acidosis risk", "label": 3},
        {"drugs": ["potassium chloride", "lisinopril"], "type": "hyperkalemia risk", "label": 3},
        {"drugs": ["oxycodone", "alprazolam"], "type": "respiratory depression", "label": 3},
        {"drugs": ["theophylline", "ciprofloxacin"], "type": "theophylline toxicity", "label": 3},
        {"drugs": ["phenytoin", "fluconazole"], "type": "phenytoin toxicity", "label": 3},
        {"drugs": ["carbamazepine", "verapamil"], "type": "carbamazepine toxicity", "label": 3},
        {"drugs": ["cyclosporine", "ketoconazole"], "type": "nephrotoxicity risk", "label": 3},
        {"drugs": ["methotrexate", "ibuprofen"], "type": "methotrexate toxicity", "label": 3},
        {"drugs": ["quinidine", "digoxin"], "type": "digoxin toxicity", "label": 3},
        {"drugs": ["clopidogrel", "omeprazole"], "type": "reduced antiplatelet effect", "label": 3},
        {"drugs": ["warfarin", "rifampin"], "type": "reduced anticoagulation", "label": 3},
        {"drugs": ["dabigatran", "rifampin"], "type": "reduced anticoagulant effect", "label": 3},
        
        # Moderate (2)
        {"drugs": ["simvastatin", "amlodipine"], "type": "increased statin exposure", "label": 2},
        {"drugs": ["metformin", "cimetidine"], "type": "increased metformin levels", "label": 2},
        {"drugs": ["levothyroxine", "calcium carbonate"], "type": "reduced thyroid absorption", "label": 2},
        {"drugs": ["gabapentin", "aluminum hydroxide"], "type": "reduced gabapentin absorption", "label": 2},
        {"drugs": ["furosemide", "gentamicin"], "type": "ototoxicity risk", "label": 2},
        {"drugs": ["prednisone", "naproxen"], "type": "GI bleeding risk", "label": 2},
        {"drugs": ["metoprolol", "verapamil"], "type": "bradycardia risk", "label": 2},
        {"drugs": ["sertraline", "tramadol"], "type": "seizure threshold lowered", "label": 2},
        {"drugs": ["losartan", "potassium supplements"], "type": "hyperkalemia risk", "label": 2},
        {"drugs": ["alprazolam", "ketoconazole"], "type": "increased sedation", "label": 2},
        {"drugs": ["atorvastatin", "grapefruit juice"], "type": "increased statin levels", "label": 2},
        {"drugs": ["ciprofloxacin", "ferrous sulfate"], "type": "reduced antibiotic absorption", "label": 2},
        {"drugs": ["warfarin", "acetaminophen"], "type": "slight INR increase", "label": 2},
        {"drugs": ["insulin", "propranolol"], "type": "masked hypoglycemia", "label": 2},
        {"drugs": ["digoxin", "spironolactone"], "type": "increased digoxin levels", "label": 2},
        
        # Minor (1)
        {"drugs": ["aspirin", "ibuprofen"], "type": "reduced cardioprotection", "label": 1},
        {"drugs": ["metformin", "vitamin B12"], "type": "reduced B12 absorption long-term", "label": 1},
        {"drugs": ["amoxicillin", "ethinyl estradiol"], "type": "theoretical reduced efficacy", "label": 1},
        {"drugs": ["omeprazole", "vitamin B12"], "type": "reduced absorption", "label": 1},
        {"drugs": ["caffeine", "ciprofloxacin"], "type": "increased caffeine effect", "label": 1},
        {"drugs": ["calcium carbonate", "ferrous sulfate"], "type": "timing interaction", "label": 1},
        {"drugs": ["atorvastatin", "niacin"], "type": "monitoring recommended", "label": 1},
        {"drugs": ["lisinopril", "aspirin"], "type": "possible reduced effect", "label": 1},
        {"drugs": ["hydrochlorothiazide", "calcium"], "type": "hypercalcemia monitoring", "label": 1},
        {"drugs": ["metoprolol", "clonidine"], "type": "withdrawal monitoring", "label": 1},
        
        # No interaction (0)
        {"drugs": ["amlodipine", "atorvastatin"], "type": "safe combination", "label": 0},
        {"drugs": ["metformin", "lisinopril"], "type": "complementary therapy", "label": 0},
        {"drugs": ["omeprazole", "levothyroxine"], "type": "can be used together with spacing", "label": 0},
        {"drugs": ["aspirin", "atorvastatin"], "type": "standard combination", "label": 0},
        {"drugs": ["metoprolol", "lisinopril"], "type": "common combination", "label": 0},
        {"drugs": ["gabapentin", "acetaminophen"], "type": "no interaction", "label": 0},
        {"drugs": ["sertraline", "omeprazole"], "type": "generally safe", "label": 0},
        {"drugs": ["metformin", "glipizide"], "type": "complementary", "label": 0},
        {"drugs": ["hydrochlorothiazide", "lisinopril"], "type": "synergistic", "label": 0},
        {"drugs": ["pantoprazole", "amlodipine"], "type": "no known interaction", "label": 0},
    ]
    
    training_data = []
    
    for pattern in ddi_patterns:
        drug1, drug2 = pattern["drugs"]
        interaction_type = pattern["type"]
        label = pattern["label"]
        label_name = DDI_LABELS[label]
        label_desc = LABEL_DESCRIPTIONS[label]
        
        # Create instruction-tuning format
        prompts = [
            f"Analyze the drug-drug interaction between {drug1} and {drug2}.",
            f"What is the severity of combining {drug1} with {drug2}?",
            f"A patient is taking {drug1}. They need to start {drug2}. Assess the interaction risk.",
            f"Evaluate the interaction: {drug1} + {drug2}",
            f"Drug interaction check: {drug1} and {drug2}",
        ]
        
        for prompt in prompts:
            response = f"Severity: {label_name.upper()}\nInteraction: {interaction_type}\nRecommendation: {label_desc}"
            
            training_data.append({
                "instruction": prompt,
                "response": response,
                "label": label,
                "label_name": label_name
            })
    
    # Shuffle and replicate to reach target size
    random.shuffle(training_data)
    
    while len(training_data) < max_samples:
        training_data.extend(training_data[:min(len(training_data), max_samples - len(training_data))])
    
    return training_data[:max_samples]


def format_for_gemma(example: Dict) -> str:
    """Format example for Gemma instruction tuning."""
    return f"""<start_of_turn>user
{example['instruction']}<end_of_turn>
<start_of_turn>model
{example['response']}<end_of_turn>"""


def train_gemma_lora(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Train Gemma 3 with QLoRA for DDI classification."""
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
    
    # Parameters
    model_name = job_input.get('model_name', 'google/gemma-3-12b-it')
    max_samples = job_input.get('max_samples', 2000)
    epochs = job_input.get('epochs', 1)
    learning_rate = job_input.get('learning_rate', 2e-4)
    batch_size = job_input.get('batch_size', 4)
    lora_r = job_input.get('lora_r', 16)
    lora_alpha = job_input.get('lora_alpha', 32)
    max_seq_length = job_input.get('max_seq_length', 512)
    
    work_dir = tempfile.mkdtemp()
    
    try:
        print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        print(f"Model: {model_name}")
        print(f"LoRA r={lora_r}, alpha={lora_alpha}")
        print(f"Samples: {max_samples}, Epochs: {epochs}")
        
        # Load training data
        print("Loading DDI training data...")
        training_data = get_ddi_training_data(max_samples=max_samples)
        
        # Format for Gemma
        formatted_data = [{"text": format_for_gemma(ex)} for ex in training_data]
        dataset = Dataset.from_list(formatted_data)
        
        print(f"Dataset size: {len(dataset)}")
        
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
        model.print_trainable_parameters()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=work_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_steps=10,
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
            'samples': len(training_data),
            'lora_r': lora_r,
            'lora_alpha': lora_alpha,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
            'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'quantization': '4-bit QLoRA',
        }
        
        print(f"Training complete! Loss: {metrics['train_loss']:.4f}")
        
        return {
            'status': 'success',
            'metrics': metrics,
            'message': f'Gemma 3 12B fine-tuned with QLoRA on DDI data'
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
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def train_bert_classifier(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Train BERT-style classifier (original approach)."""
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
    max_samples = job_input.get('max_samples', 5000)
    epochs = job_input.get('epochs', 3)
    learning_rate = job_input.get('learning_rate', 2e-5)
    batch_size = job_input.get('batch_size', 16)
    eval_split = job_input.get('eval_split', 0.1)
    
    work_dir = tempfile.mkdtemp()
    
    try:
        print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        print(f"Model: {model_name}")
        
        # Get data in BERT format
        raw_data = get_ddi_training_data(max_samples=max_samples)
        training_data = [{"text": d["instruction"], "label": d["label"]} for d in raw_data]
        
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
        
        # Load model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=5
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
        }
        
        if eval_dataset:
            eval_result = trainer.evaluate()
            metrics.update({
                'eval_accuracy': float(eval_result['eval_accuracy']),
                'eval_f1_weighted': float(eval_result['eval_f1_weighted']),
            })
        
        return {'status': 'success', 'metrics': metrics, 'message': 'BERT classifier trained'}
        
    except Exception as e:
        import traceback
        return {'status': 'error', 'error': str(e), 'traceback': traceback.format_exc()}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def handler(job):
    """RunPod serverless handler."""
    job_input = job.get('input', {})
    
    # Choose training mode
    model_name = job_input.get('model_name', 'google/gemma-3-12b-it')
    use_lora = job_input.get('use_lora', True)
    
    # Auto-detect: use LoRA for large models
    if 'gemma' in model_name.lower() or 'llama' in model_name.lower() or 'mistral' in model_name.lower():
        use_lora = True
    elif 'bert' in model_name.lower():
        use_lora = False
    
    if use_lora:
        return train_gemma_lora(job_input)
    else:
        return train_bert_classifier(job_input)


# RunPod serverless entrypoint
runpod.serverless.start({'handler': handler})
