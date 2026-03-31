# Cloud Deployment & Architecture Planning

## Local Development Setup

### Environment
- **OS Support:** Windows, macOS, Linux
- **Python:** 3.8+
- **GPU:** NVIDIA CUDA 11.8+ (optional, for faster training)
- **RAM:** 16GB minimum (32GB recommended)

### Local Workflow
```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run training (single machine)
python scripts/04_finetune.py --batch_size 4 --num_epochs 3

# Inference (CPU or GPU)
python scripts/05_evaluate.py --model_path models/finetuned-trocr
```

## Cloud Deployment Architecture

### Option 1: Azure ML (Recommended for Enterprise)

```
┌─────────────────────────────────────────────────┐
│         Azure ML Workspace                      │
├─────────────────────────────────────────────────┤
│ • Compute Instance (for development)            │
│ • Compute Cluster (for training jobs)           │
│ • Model Registry (versioning)                   │
└─────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────┐
│    Azure Blob Storage                           │
│  (training data, model artifacts)               │
└─────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────┐
│    Azure Container Registry                     │
│  (Docker images for inference)                  │
└─────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────┐
│    Deployment Options:                          │
│  • Azure Container Instances (quick)            │
│  • Azure App Service (REST API)                 │
│  • Azure Batch (bulk processing)                │
└─────────────────────────────────────────────────┘
```

### Training on Cloud (Azure ML)

**Step 1: Prepare training script**
```python
# scripts/train_azureml.py
# Same as 04_finetune.py but with Azure logging
from azureml.core import Run
run = Run.get_context()
run.log("epoch", epoch)
run.log("train_loss", loss)
```

**Step 2: Create compute cluster**
```bash
# Using Azure CLI
az ml compute create -f compute.yaml
```

**Step 3: Submit training job**
```bash
# Using AML CLI
az ml job create -f train_job.yaml
```

### Model Artifacts Storage

```
Azure Blob Storage Structure:
├── models/
│   ├── baseline/
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   └── preprocessor_config.json
│   ├── finetuned/
│   │   ├── checkpoint-epoch-1/
│   │   ├── checkpoint-epoch-2/
│   │   └── checkpoint-epoch-3/
│   └── production/
│       └── v1.0/
├── data/
│   ├── training/
│   ├── validation/
│   └── test/
└── results/
    ├── metrics/
    ├── predictions/
    └── logs/
```

### Inference Deployment

**Option A: REST API (Recommended)**
```python
# inference_api.py (FastAPI)
from fastapi import FastAPI, UploadFile
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

app = FastAPI()
model = VisionEncoderDecoderModel.from_pretrained("model_path")

@app.post("/predict")
async def predict(image: UploadFile):
    pixel_values = processor(image, return_tensors="pt").pixel_values
    ids = model.generate(pixel_values)
    text = processor.batch_decode(ids)[0]
    return {"text": text}
```

**Deployment:**
```dockerfile
# Dockerfile
FROM pytorch/pytorch:2.0-cuda11.8-runtime-ubuntu22.04
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "inference_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and push to ACR
docker build -t ocr:latest .
az acr build --registry <registry-name> --image ocr:latest .

# Deploy to Azure Container Instances
az container create \
  --resource-group myresourcegroup \
  --name ocr-api \
  --image <registry>.azurecr.io/ocr:latest \
  --cpu 1 \
  --memory 2 \
  --registry-login-server <registry>.azurecr.io
```

**Option B: Batch Processing**
```python
# batch_inference.py
# Process large dataset of images
# Load from Blob Storage, process, save results
```

## Performance & Cost Considerations

### Training
```
Resource Requirements:
- Compute: GPU (Standard_NC6s_v3 = 1x Tesla V100)
- Training time: ~2-4 hours per epoch (batch_size=8)
- Cost: ~$0.90/hour (spot pricing)
- Total for 3 epochs: ~$2.50-$3.60

Better option for production:
- Use Spot instances (70% discount)
- Distributed training on multiple GPUs
```

### Inference
```
Latency targets:
- Per-image: 100-500ms (depending on image size)
- Batch inference: 5-20ms per image

Deployment option recommendations:
- Low volume (< 100/day): Azure Container Instances (on-demand)
- Medium volume (100-1000/day): App Service with auto-scale
- High volume (> 1000/day): Kubernetes (AKS) or Batch

Cost per prediction:
- API call overhead: ~$0.0001
- Model inference: ~$0.00001 (GPU time)
- Total: ~$0.0001 per image
```

## Monitoring & Maintenance

### Logging
```
Application Insights:
├── Request latency
├── Error rates
├── Model accuracy drift
└── Resource utilization
```

### Model Versioning
```
Azure ML Model Registry:
- Version: Timestamp-based (2024.01.15.v1)
- Properties: accuracy, latency, size
- Tags: production, staging, deprecated
- Lineage: training data, hyperparameters
```

### Continuous Monitoring
```python
# Monitor predictions for drift
def check_model_drift():
    recent_predictions = get_recent_predictions()
    accuracy = calculate_accuracy(recent_predictions)
    if accuracy < baseline_accuracy * 0.95:
        alert("Model accuracy dropped!")
        trigger_retraining()
```

## Data Pipeline (Cloud Native)

```
Data Ingestion
  ↓
Azure Data Factory (orchestration)
  ↓
Data Validation & Cleaning
  ↓
Azure Blob Storage
  ↓
Azure ML Training Job
  ↓
Model Registry
  ↓
Inference API
  ↓
Results → Blob Storage
```

## Security Considerations

- **Secrets Management:** Azure Key Vault (API keys, connection strings)
- **Network:** Virtual Network, Private Endpoints
- **Authentication:** Managed Identity for Azure resources
- **Encryption:** Data at rest (Blob Storage encryption), in transit (HTTPS)
- **Compliance:** Audit logs, data retention policies

## Scaling Strategy

### Horizontal Scaling (Multiple Instances)
```yaml
# Azure Container Instances scale set
containers:
  - name: ocr-api
    image: ocr:latest
    resources:
      requests:
        cpu: 1
        memory: 2Gi
  - replicas: 3  # Scale to 3 instances
    rules:
      - cpu: 80%
        scale_up_to: 5
```

### Vertical Scaling (Larger Machines)
```
For training:
- Upgrade GPU (V100 → A100)
- Add more CPUs
- Increase memory for larger batches

For inference:
- Use optimized models (quantization, distillation)
- Cache model in memory
- Use batch processing
```

## Disaster Recovery

- **Backup:** Model checkpoints → Blob Storage (geo-replicated)
- **Version Control:** Git for code, Model Registry for artifacts
- **Rollback:** Ability to deploy previous model version in < 5 minutes
- **Testing:** Stage environment mirrors production

## Real-world Deployment Checklist

- [ ] Model quantization / optimization
- [ ] Load testing (throughput, latency)
- [ ] Error handling and retry logic
- [ ] Logging and monitoring setup
- [ ] API documentation (Swagger/OpenAPI)
- [ ] Rate limiting / throttling
- [ ] Cost estimation and alerting
- [ ] Compliance review
- [ ] Documentation for operations team
- [ ] Runbooks for common issues

## Timeline Example

```
Week 1: Infrastructure setup, CI/CD pipeline
Week 2-3: Model training on cloud, evaluation
Week 4: API development, testing, deployment
Week 5: Monitoring, optimization, production launch
```

## When NOT to Deploy to Cloud

- Model updates are rare (< quarterly)
- Inference latency must be < 50ms (edge device better)
- Data cannot leave on-premises (regulatory requirement)
- Very low volume (< 10 predictions/day)
- Cost of cloud exceeds on-premises
