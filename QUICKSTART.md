# End-to-End OCR System - Quick Start Guide

## Project Summary

This is a **complete OCR pipeline** for handwritten text recognition that demonstrates:
- ✅ Proper data handling and quality auditing
- ✅ Baseline model evaluation
- ✅ Iterative fine-tuning with measurable improvements
- ✅ Detailed error analysis and categorization
- ✅ LLM-based post-processing for correction
- ✅ Cloud deployment planning

## What You Get

1. **Data Audit System** - Understand your data before training
2. **Baseline Model** - Pretrained TrOCR for comparison
3. **Fine-tuning Pipeline** - Train on your dataset
4. **Evaluation Framework** - CER/WER metrics with error analysis
5. **Improvement Tracking** - Compare baseline vs improved models
6. **LLM Integration** - Post-process OCR for better results
7. **Cloud Ready** - Deployment architecture and guides

## Getting Started (5 minutes)

### 1. Setup Environment

```bash
# Navigate to project
cd ocr-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Setup

**Option A: Using IAM Dataset (Recommended)**

The IAM Handwriting Database is a public dataset with 13,353 handwritten text samples.

```bash
# Download from: http://www.fki.inf.unibe.ch/databases/iam-handwriting-database
# 1. Register on their website
# 2. Download "lines.txt" and "lines.tgz"
# 3. Extract to: data/raw/iam/

# After extraction, your structure should be:
# data/raw/iam/
#   ├── lines.txt          (metadata)
#   └── lines/             (images organized by writer ID)
#       ├── a01/
#       ├── a02/
#       └── ...
```

**Option B: Using Your Own Data**

Create JSON files with format:
```json
[
  {
    "image_id": "sample_001",
    "image_path": "/path/to/image.png",
    "text": "Handwritten text"
  }
]
```

### 3. Run the Pipeline (Step by Step)

```bash
# Step 1: Understand your data
python scripts/01_data_audit.py --data_dir data/raw/iam

# Expected output:
# - results/data_audit_report.txt
# - results/data_audit.json

# Step 2: Prepare data for training
python scripts/02_data_prep.py --data_dir data/raw/iam --output_dir data/processed

# Expected output:
# - data/processed/train.json (70%)
# - data/processed/val.json (15%)
# - data/processed/test.json (15%)

# Step 3: Evaluate baseline model
python scripts/03_baseline.py --data_split data/processed/test.json

# Expected output:
# - results/baseline_metrics.json
# - results/baseline_report.txt
# - results/baseline_predictions.json

# Step 4: Fine-tune model (takes 1-2 hours on GPU)
python scripts/04_finetune.py \
  --train_data data/processed/train.json \
  --val_data data/processed/val.json \
  --output_dir models/finetuned-trocr \
  --num_epochs 3 \
  --batch_size 4

# Expected output:
# - models/finetuned-trocr/ (trained model)
# - models/finetuned-trocr/training_history.json

# Step 5: Evaluate fine-tuned model
python scripts/05_evaluate.py \
  --model_path models/finetuned-trocr \
  --test_data data/processed/test.json

# Expected output:
# - results/finetuned_metrics.json

# Step 6: Compare improvements
python scripts/06_improvements.py \
  --baseline_metrics results/baseline_metrics.json \
  --improved_metrics results/finetuned_metrics.json

# Expected output:
# - results/improvement_comparison.json
# - results/improvement_report.txt

# Step 7: Apply LLM post-processing
python scripts/07_llm_correction.py \
  --predictions_path results/finetuned_predictions.json \
  --output_corrected results/llm_corrected.json

# Expected output:
# - results/llm_corrected.json
# - results/llm_postprocessing_report.txt
```

## Key Metrics Explained

### Character Error Rate (CER)
Measures fine-grained errors at character level:
```
CER = (Insertions + Deletions + Substitutions) / Total_Characters × 100%

Example:
Reference: "hello"
Prediction: "helo"
Errors: 1 deletion
CER = 1/5 = 20%
```

### Word Error Rate (WER)
Measures errors at word level:
```
WER = (Word_Insertions + Word_Deletions + Word_Substitutions) / Total_Words × 100%

Example:
Reference: "hello world"
Prediction: "hello word"
Errors: 1 substitution
WER = 1/2 = 50%
```

## Expected Results

Based on IAM dataset with TrOCR-small:

| Model | CER | WER | Accuracy |
|-------|-----|-----|----------|
| Baseline (pretrained) | 8-12% | 15-20% | 80-85% |
| Fine-tuned (3 epochs) | 4-6% | 8-12% | 88-92% |

*Actual results depend on data quality and training parameters*

## Common Issues & Solutions

### Issue: CUDA out of memory
```bash
# Solution: Reduce batch size
python scripts/04_finetune.py --batch_size 2 --num_epochs 2
```

### Issue: Dataset not found
```bash
# Make sure directory structure is correct:
# data/raw/iam/lines.txt (exists)
# data/raw/iam/lines/ (directory with images)
```

### Issue: Slow inference
```bash
# Solutions:
# 1. Use GPU: CUDA automatically detected
# 2. Increase batch size for batch processing
# 3. Use quantized model (not implemented yet)
```

## Understanding Results

### Data Audit (Step 1)
**What to look for:**
- ✓ Number of samples (1000+ is good)
- ✓ Balanced data distribution
- ✓ No corrupted images
- ✓ Consistent text format

**Actionable insights:**
- If many corrupted images → filter them out
- If unbalanced → use weighted sampling
- If inconsistent format → apply normalization

### Baseline Report (Step 3)
**What to look for:**
- Error breakdown by type
- Which error types are most common
- Sample predictions to understand failure modes

**Actionable insights:**
- High "merged_words" errors → improve spacing handling
- High "missing_characters" → model needs better training
- Random errors → check image quality

### Improvement Comparison (Step 6)
**What to look for:**
- CER/WER improvements
- Which error types improved most
- Whether improvements are significant

**Actionable insights:**
- If < 2% improvement → try different approach
- If > 10% improvement → good! Consider deploying
- If accuracy >90% → model is production-ready

## File Structure Reference

```
ocr-system/
├── data/
│   ├── raw/              ← Put IAM dataset here
│   │   └── iam/
│   │       ├── lines.txt
│   │       └── lines/
│   └── processed/        ← Generated splits (train/val/test)
├── scripts/
│   ├── 01_data_audit.py
│   ├── 02_data_prep.py
│   ├── 03_baseline.py
│   ├── 04_finetune.py
│   ├── 05_evaluate.py
│   ├── 06_improvements.py
│   ├── 07_llm_correction.py
│   └── utils.py          ← Shared utilities
├── models/
│   └── finetuned-trocr/  ← Trained model saved here
├── results/              ← All metrics, predictions, reports
├── requirements.txt
├── README.md
├── DEPLOYMENT.md
└── QUICKSTART.md         ← This file
```

## Next Steps After Completing Pipeline

### 1. **Investigate Remaining Errors**
Review the "error_examples" in evaluation reports to understand what's still failing.

### 2. **Try Improvements**
- Adjust normalization rules for your domain
- Experiment with different model sizes (base, large)
- Use different learning rates and epochs
- Filter low-confidence predictions

### 3. **Domain-Specific Tuning**
- If working with specific handwriting (e.g., medical): get domain data
- Create custom LLM prompts for your domain
- Collect user feedback on incorrect predictions

### 4. **Deploy to Cloud**
See [DEPLOYMENT.md](DEPLOYMENT.md) for:
- Setting up Azure ML
- Creating REST API
- Scaling to production
- Monitoring and maintenance

### 5. **Continuous Improvement**
- Collect user feedback on incorrect predictions
- Retrain periodically with new data
- Monitor for model drift
- A/B test new model versions

## Performance Tips

### For Training
- Use GPU (Tesla T4, V100, or A100 recommended)
- Increase batch size as much as memory allows
- Use learning rate scheduler
- Monitor validation loss for early stopping

### For Inference
- Batch process images when possible
- Cache model in memory
- Use quantized models for edge devices
- Consider image resizing for small handwriting

## When NOT to Use This Approach

- ❌ Digital text extraction → Use PDF parsing
- ❌ Printed documents at scale → Use document scanning service
- ❌ Real-time requirements (< 10ms) → Use edge AI
- ❌ Perfect accuracy needed → Consider hybrid human+AI
- ❌ Very small dataset (< 100 samples) → Use pre-trained only

## Additional Resources

- [TrOCR Paper](https://arxiv.org/abs/2109.10282)
- [IAM Handwriting Database](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database)
- [Hugging Face TrOCR Docs](https://huggingface.co/docs/transformers/model_doc/trocr)
- [CER/WER Metrics](https://en.wikipedia.org/wiki/Word_error_rate)

## Support & Troubleshooting

For issues with:
- **Dataset loading:** Check data/raw/iam structure
- **Model training:** Check GPU memory and batch size
- **Metrics calculation:** Verify text encoding (UTF-8)
- **Deployment:** See DEPLOYMENT.md

---

**Ready to start?** Begin with:
```bash
python scripts/01_data_audit.py --data_dir data/raw/iam
```

**Next:** Follow the 7-step pipeline in "Run the Pipeline" section above.

Good luck! 🚀
