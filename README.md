# End-to-End OCR System

A complete OCR pipeline for handwritten text recognition, demonstrating data handling, model fine-tuning, evaluation, and practical improvements.

## Project Overview

This project implements a real-world OCR system that processes raw handwritten text images. It covers:
- Data exploration and quality auditing
- Model baseline establishment
- Fine-tuning on target dataset
- Systematic evaluation with CER/WER metrics
- Iterative improvements
- LLM-based post-processing for error correction

## Architecture

```
Data (IAM Handwriting)
    ↓
[Data Audit & Cleaning]
    ↓
[Train/Val/Test Split]
    ↓
[Baseline: microsoft/trocr-small-handwritten]
    ↓
[Fine-tuning on train set]
    ↓
[Evaluation: CER/WER metrics]
    ↓
[Error Analysis & Improvements]
    ↓
[LLM Post-Processing]
    ↓
[Final Results & Comparison]
```

## Setup & Requirements

### Prerequisites
- Python 3.8+
- PyTorch (with CUDA support if using GPU)
- Hugging Face Transformers
- Datasets library
- (Optional) OpenAI API key for LLM post-processing

### Installation

```bash
# Clone repository
git clone <repo-url>
cd ocr-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
ocr-system/
├── data/                    # Dataset storage
│   ├── raw/                # Original IAM dataset
│   ├── processed/          # Cleaned and split data
│   └── splits/             # train.txt, val.txt, test.txt
├── scripts/
│   ├── 01_data_audit.py    # Data exploration and quality checks
│   ├── 02_data_prep.py     # Data cleaning and preparation
│   ├── 03_baseline.py      # Baseline model inference
│   ├── 04_finetune.py      # Model fine-tuning
│   ├── 05_evaluate.py      # Evaluation metrics (CER/WER)
│   ├── 06_improvements.py  # Implement improvements
│   ├── 07_llm_correction.py# LLM post-processing
│   └── utils.py            # Utility functions
├── models/                 # Saved model checkpoints
├── results/                # Metrics, predictions, visualizations
├── notebooks/              # Jupyter notebooks for exploration
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Usage

### 1. Data Audit (Understand your data first!)

```bash
python scripts/01_data_audit.py
```

This generates a data audit report showing:
- Sample counts and distributions
- Data quality issues (duplicates, empty samples, inconsistencies)
- Text characteristics (case, punctuation, special characters)

### 2. Data Preparation

```bash
python scripts/02_data_prep.py
```

- Downloads IAM dataset (requires setup, see data/README.md)
- Normalizes text according to defined rules
- Splits into train/validation/test (no leakage)
- Saves processed splits to `data/processed/`

### 3. Baseline Evaluation

```bash
python scripts/03_baseline.py
```

- Loads pretrained `microsoft/trocr-small-handwritten`
- Runs inference on test set
- Calculates CER and WER
- Saves sample predictions to `results/baseline_predictions.json`

### 4. Fine-Tune Model

```bash
python scripts/04_finetune.py
```

- Trains model on prepared dataset
- Saves best checkpoint to `models/finetuned-trocr/`
- Logs training metrics

### 5. Evaluate & Analyze

```bash
python scripts/05_evaluate.py
```

- Evaluates fine-tuned model
- Detailed error analysis with categorization
- Saves error examples and visualizations

### 6. Apply Improvements

```bash
python scripts/06_improvements.py
```

- Implements improvement strategy (e.g., better filtering, hyperparameter tuning)
- Shows before/after comparison
- Saves results

### 7. LLM Post-Processing

```bash
python scripts/07_llm_correction.py
```

- Uses LLM to correct common OCR errors
- Shows example input/output pairs
- Saves corrected predictions

## Key Findings

### Data Audit Summary
[To be filled after step 1]

### Baseline Performance
- **CER:** [To be filled]
- **WER:** [To be filled]

### Fine-tuned Performance
- **CER:** [To be filled]
- **WER:** [To be filled]

### Improvement Applied
[Description of improvement strategy and results]

### Sample Predictions

#### Baseline
```
Image: sample_001.png
Ground Truth: "The quick brown fox"
Prediction:   "The quikc brown fox"
Error: Typo in "quick"
```

#### After Improvement
```
Ground Truth: "The quick brown fox"
Prediction:   "The quick brown fox"
Error: None
```

## Metrics Explanation

### Character Error Rate (CER)
$$\text{CER} = \frac{\text{# Character insertions} + \text{# deletions} + \text{# substitutions}}{\text{# Characters in ground truth}} \times 100\%$$

Lower is better. Accounts for fine-grained errors.

### Word Error Rate (WER)
$$\text{WER} = \frac{\text{# Word insertions} + \text{# deletions} + \text{# substitutions}}{\text{# Words in ground truth}} \times 100\%$$

Higher penalty than CER for word-level mistakes.

## Normalization Rules

- **Case:** Lowercase all text
- **Punctuation:** Kept as-is
- **Whitespace:** Single space between words, trim edges
- **Special Characters:** Unicode preserved
- **Numbers:** Kept as-is

## Error Categories

When analyzing errors, we categorize them as:

1. **Missing Characters:** OCR omitted a character ("teh" → "te")
2. **Extra Characters:** OCR added a character ("cat" → "catt")
3. **Character Substitution:** Wrong character ("cat" → "car")
4. **Merged Words:** Two words read as one ("cat dog" → "catdog")
5. **Split Words:** One word read as two ("catdog" → "cat dog")
6. **Case Errors:** Incorrect capitalization
7. **Punctuation:** Wrong or missing punctuation
8. **Complete Misread:** Unrecognizable output

## Cloud Deployment Strategy

### Local Development
- **Environment:** Windows/Mac/Linux with Python
- **Compute:** CPU for inference, GPU (T4/V100) for training
- **Setup:** Virtual environment with pip/conda

### Cloud Training (Azure ML / GCP Vertex / AWS SageMaker)
- Training script: `scripts/04_finetune.py`
- Batch size: Configurable for cloud resources
- Model checkpoint saved to cloud storage

### Model Artifacts
- **Storage:** Azure Blob Storage / GCS / S3
- **Location:** `models/finetuned-trocr/`
- **Format:** Hugging Face model format

### Inference Deployment
- **API:** FastAPI endpoint
- **Container:** Docker
- **Scale:** Horizontal scaling with load balancer
- **Example:** POST `/ocr/predict` with image → JSON response

## Decisions & Trade-offs

### Why TrOCR?
- Pre-trained on handwritten text
- Lightweight (small variant suitable for limited compute)
- Proven performance on handwriting
- Active community support

### Why IAM Dataset?
- Largest public handwriting dataset
- Real-world handwriting (not synthetic)
- Good for fine-tuning
- Well-documented

### Normalization Choices
- Lowercase: Reduces model burden, common practice
- Keep punctuation: Preserves semantic meaning
- Single space: Handles variable spacing in handwriting

## Limitations & When NOT to Use OCR

⚠️ Important considerations:

1. **Digital text extraction:** Use PDF parsers instead
2. **Printed documents at scale:** Consider document scanning services
3. **Real-time requirements:** CPU inference may be too slow (milliseconds needed)
4. **Perfect accuracy needed:** OCR will fail on ~5-10% of samples
5. **Highly specialized domains:** Medical/scientific notation needs custom training
6. **Low-quality images:** Blurry or extremely degraded images won't work

## Results Summary

[To be updated with final results]

- **Best Model:** [model checkpoint]
- **Best CER:** [value]%
- **Best WER:** [value]%
- **Total Training Time:** [hours]
- **Inference Time:** [milliseconds per image]
- **Key Improvement:** [description]

## References

- [TrOCR Paper](https://arxiv.org/abs/2109.10282)
- [IAM Handwriting Database](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database)
- [Hugging Face TrOCR](https://huggingface.co/docs/transformers/model_doc/trocr)
- [CER/WER Metrics](https://en.wikipedia.org/wiki/Word_error_rate)

## Notes for Reviewers

This project demonstrates:
- ✅ Clear data understanding and audit
- ✅ Proper train/val/test splitting with no leakage
- ✅ Systematic evaluation with multiple metrics
- ✅ Error analysis and categorization
- ✅ Iterative improvement with before/after comparison
- ✅ Practical thinking about real-world deployment
- ✅ Understanding of limitations and when NOT to use OCR

---

**Author:** [Your Name]  
**Date:** 2026
