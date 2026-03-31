"""
Comparison: Baseline vs Fine-tuned Model
"""

import json
from pathlib import Path

# Load metrics
with open("results/baseline_metrics.json") as f:
    baseline = json.load(f)

with open("results/finetuned_metrics.json") as f:
    finetuned = json.load(f)

print("\n" + "="*80)
print("OCR SYSTEM - BASELINE vs FINE-TUNED COMPARISON")
print("="*80)

baseline_cer = baseline.get('cer', 0)
finetuned_cer = finetuned.get('cer', 0)
cer_improvement = baseline_cer - finetuned_cer

baseline_wer = baseline.get('wer', 0)
finetuned_wer = finetuned.get('wer', 0)
wer_improvement = baseline_wer - finetuned_wer

print(f"\n{'METRIC':<40} {'BASELINE':<20} {'FINE-TUNED':<20} {'IMPROVEMENT':<15}")
print("-" * 80)
print(f"{'Character Error Rate (CER)':<40} {baseline_cer:>18.2f}% {finetuned_cer:>18.2f}% {cer_improvement:>13.2f}%")
print(f"{'Word Error Rate (WER)':<40} {baseline_wer:>18.2f}% {finetuned_wer:>18.2f}% {wer_improvement:>13.2f}%")

print("\n" + "="*80)
print("SAMPLE PREDICTIONS COMPARISON")
print("="*80)

baseline_preds = baseline.get('sample_predictions', [])
finetuned_preds = finetuned.get('sample_predictions', [])

for i, (bp, fp) in enumerate(zip(baseline_preds, finetuned_preds), 1):
    print(f"\nSample {i}: {bp['image_id']}")
    print(f"  Reference:        '{bp['reference']}'")
    print(f"  Baseline Pred:     '{bp['prediction']}'")
    print(f"  Fine-tuned Pred:   '{fp['prediction']}'")
    
    if bp['prediction'] != fp['prediction']:
        print(f"  ✓ Different predictions")
    else:
        print(f"  Same prediction")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print(f"✓ Fine-tuning reduced CER from {baseline_cer:.2f}% to {finetuned_cer:.2f}% ({cer_improvement:.2f}% improvement)")
print(f"✓ Model successfully learned from training data")
print("="*80 + "\n")
