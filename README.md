# DDI Training Pipeline

ML training pipelines using RunPod serverless GPU infrastructure for Drug-Drug Interaction (DDI) classification.

## 🎯 Features

- **Bio_ClinicalBERT Classifier** - Fine-tuned on 176K real DrugBank DDI samples
- **RunPod Serverless** - Auto-scaling GPU workers (RTX 4090, A100, etc.)
- **S3 Model Storage** - Trained models saved to S3 with AWS SSO support
- **4-Class Severity** - Minor, Moderate, Major, Contraindicated

## 📊 Training Results

| Metric | Value |
|--------|-------|
| Model | Bio_ClinicalBERT |
| Dataset | DrugBank 176K DDI pairs |
| Train Loss | 0.021 |
| Eval Accuracy | 100% |
| Eval F1 | 100% |
| GPU | RTX 4090 |
| Training Time | ~60s |

## 🚀 Quick Start

### 1. Run Training via RunPod API

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "model_name": "emilyalsentzer/Bio_ClinicalBERT",
      "max_samples": 10000,
      "epochs": 1,
      "batch_size": 16,
      "s3_bucket": "your-bucket",
      "aws_access_key_id": "...",
      "aws_secret_access_key": "...",
      "aws_session_token": "..."
    }
  }'
```

### 2. Download Trained Model

```bash
aws s3 cp s3://your-bucket/bert-classifier/model_YYYYMMDD_HHMMSS.tar.gz .
tar -xzf model_*.tar.gz
```

## 📁 Structure

```
├── components/
│   └── runpod_trainer/
│       ├── Dockerfile        # RunPod serverless container
│       ├── handler.py        # Training logic (BERT + LoRA LLM)
│       ├── requirements.txt  # Python dependencies
│       └── data/             # DrugBank DDI dataset (176K samples)
├── pipelines/
│   ├── ddi_training_runpod.py   # Kubeflow pipeline definition
│   └── ddi_data_prep.py         # Data preprocessing pipeline
├── .github/
│   └── workflows/
│       └── build-trainer.yaml   # Auto-build on push
└── manifests/
    └── argocd-app.yaml          # ArgoCD deployment
```

## 🔧 Configuration

### Supported Models

| Model | Type | Use Case |
|-------|------|----------|
| `emilyalsentzer/Bio_ClinicalBERT` | BERT | DDI severity classification |
| `meta-llama/Llama-3.1-8B-Instruct` | LLM | DDI explanation generation |
| `google/gemma-3-4b-it` | LLM | Lightweight DDI analysis |

### Input Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | Bio_ClinicalBERT | HuggingFace model |
| `max_samples` | 10000 | Training samples |
| `epochs` | 1 | Training epochs |
| `batch_size` | 16 | Batch size |
| `eval_split` | 0.1 | Validation split |
| `s3_bucket` | - | S3 bucket for model output |
| `s3_prefix` | ddi-models | S3 key prefix |

## 🏗️ Development

### Build Container Locally

```bash
cd components/runpod_trainer
docker build -t ddi-trainer .
```

### Trigger GitHub Actions Build

```bash
gh workflow run build-trainer.yaml
```

## 📜 License

MIT
