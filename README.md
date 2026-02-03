# Healthcare ML Training Pipeline

Serverless GPU training infrastructure for healthcare NLP models. Training runs on RunPod serverless GPUs, with trained models stored in S3.

## Overview

This project provides production-ready ML pipelines for training healthcare classification models:

- **Drug-Drug Interaction (DDI)** - Severity classification from DrugBank (176K samples)
- **Adverse Drug Events (ADE)** - Binary detection from ADE Corpus V2 (30K samples)
- **Medical Triage** - Urgency level classification
- **Symptom-to-Disease** - Diagnosis prediction (41 disease classes)

All models use Bio_ClinicalBERT as the base and are fine-tuned on domain-specific datasets.

## Training Results

| Task | Dataset | Samples | Accuracy | F1 Score |
|------|---------|---------|----------|----------|
| DDI Classification | DrugBank | 176K | 100% | 100% |
| ADE Detection | ADE Corpus V2 | 9K | 93.5% | 95.3% |
| Symptom-Disease | Disease Symptoms | 4.4K | 100% | 100% |

## Quick Start

### Run Training

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "task": "ddi",
      "model_name": "emilyalsentzer/Bio_ClinicalBERT",
      "max_samples": 10000,
      "epochs": 3,
      "batch_size": 16,
      "s3_bucket": "your-bucket",
      "aws_access_key_id": "...",
      "aws_secret_access_key": "...",
      "aws_session_token": "..."
    }
  }'
```

Available tasks: `ddi`, `ade`, `triage`, `symptom_disease`

### Download Trained Model

```bash
aws s3 cp s3://your-bucket/model.tar.gz .
tar -xzf model.tar.gz
```

## Project Structure

```
├── components/
│   └── runpod_trainer/
│       ├── Dockerfile
│       ├── handler.py          # Multi-task training logic
│       ├── requirements.txt
│       └── data/               # DrugBank DDI dataset
├── pipelines/
│   ├── healthcare_training.py  # Kubeflow pipeline definitions
│   ├── ddi_training_runpod.py
│   └── ddi_data_prep.py
├── .github/workflows/
│   └── build-trainer.yaml      # CI/CD
└── manifests/
    └── argocd-app.yaml
```

## Configuration

All configuration is via environment variables. Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
# Edit .env with your credentials
```

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `RUNPOD_API_KEY` | Yes | - | RunPod API key |
| `RUNPOD_ENDPOINT` | Yes | - | RunPod serverless endpoint ID |
| `AWS_ACCESS_KEY_ID` | Yes | - | AWS credentials for S3 |
| `AWS_SECRET_ACCESS_KEY` | Yes | - | AWS credentials for S3 |
| `AWS_SESSION_TOKEN` | No | - | For assumed role sessions |
| `AWS_REGION` | No | us-east-1 | AWS region |
| `S3_BUCKET` | Yes | - | Bucket for model artifacts |
| `BASE_MODEL` | No | Bio_ClinicalBERT | HuggingFace model ID |
| `MAX_SAMPLES` | No | 10000 | Training samples |
| `EPOCHS` | No | 3 | Training epochs |
| `BATCH_SIZE` | No | 16 | Batch size |

### Kubernetes Secrets (Recommended)

For production, use Kubernetes secrets instead of environment variables:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: ml-pipeline-secrets
type: Opaque
stringData:
  RUNPOD_API_KEY: "your-key"
  AWS_ACCESS_KEY_ID: "your-key"
  AWS_SECRET_ACCESS_KEY: "your-secret"
```

### Supported Models

| Model | Type | Use Case |
|-------|------|----------|
| `emilyalsentzer/Bio_ClinicalBERT` | BERT | Classification tasks |
| `meta-llama/Llama-3.1-8B-Instruct` | LLM | Text generation (LoRA) |
| `google/gemma-3-4b-it` | LLM | Lightweight inference |

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `task` | ddi | Training task |
| `model_name` | Bio_ClinicalBERT | HuggingFace model ID |
| `max_samples` | 10000 | Training samples |
| `epochs` | 3 | Training epochs |
| `batch_size` | 16 | Batch size |
| `eval_split` | 0.1 | Validation split |
| `s3_bucket` | - | S3 bucket for output |

## Development

```bash
# Build container
cd components/runpod_trainer
docker build -t healthcare-trainer .

# Trigger CI build
gh workflow run build-trainer.yaml
```

## License

MIT
