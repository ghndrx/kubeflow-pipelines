# Kubeflow Pipelines - GitOps Repository

This repository contains ML pipeline definitions managed via ArgoCD.

## Structure

```
.
├── pipelines/           # Pipeline Python definitions
│   └── examples/        # Example pipelines
├── components/          # Reusable pipeline components
├── experiments/         # Experiment configurations
├── runs/               # Scheduled/triggered runs
└── manifests/          # K8s manifests for ArgoCD
```

## Usage

1. **Add a pipeline**: Create a Python file in `pipelines/`
2. **Push to main**: ArgoCD auto-deploys
3. **Monitor**: Check Kubeflow UI at https://kubeflow.walleye-frog.ts.net

## Quick Start

```python
from kfp import dsl

@dsl.component
def hello_world() -> str:
    return "Hello from Kubeflow!"

@dsl.pipeline(name="hello-pipeline")
def hello_pipeline():
    hello_world()
```

## Environment

- **Kubeflow**: https://kubeflow.walleye-frog.ts.net
- **MinIO**: https://minio.walleye-frog.ts.net
- **ArgoCD**: https://argocd.walleye-frog.ts.net
