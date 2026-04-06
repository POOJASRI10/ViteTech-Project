#!/usr/bin/env python
"""Check for data leakage in train/val/test splits"""

import json
import os
from pathlib import Path
from collections import defaultdict

print("[INFO] Checking for data leakage in dataset splits...\n")

split_file = "data/processed/kaggle_iam/splits.json"

if not os.path.exists(split_file):
    print("[ERROR] Splits file not found. Run quick_prepare_kaggle.py first.")
    exit(1)

# Load splits
with open(split_file, 'r') as f:
    splits = json.load(f)

train_set = set(splits['train'])
val_set = set(splits['val'])
test_set = set(splits['test'])

print("=" * 60)
print("[DATASET LEAKAGE ANALYSIS]")
print("=" * 60)

print(f"\nDataset sizes:")
print(f"  Train: {len(train_set)} samples")
print(f"  Val:   {len(val_set)} samples")
print(f"  Test:  {len(test_set)} samples")
print(f"  Total: {len(train_set) + len(val_set) + len(test_set)} samples")

# Check for overlaps
print(f"\n{'='*60}")
print("[LEAKAGE CHECK]")
print(f"{'='*60}")

train_val_overlap = train_set & val_set
train_test_overlap = train_set & test_set
val_test_overlap = val_set & test_set

print(f"\n1. TRAIN-VAL OVERLAP:")
if train_val_overlap:
    print(f"   LEAKAGE FOUND: {len(train_val_overlap)} samples in both train and val!")
    for sample in list(train_val_overlap)[:3]:
        print(f"     - {sample}")
else:
    print(f"   OK: No overlap between train and val")

print(f"\n2. TRAIN-TEST OVERLAP:")
if train_test_overlap:
    print(f"   LEAKAGE FOUND: {len(train_test_overlap)} samples in both train and test!")
    for sample in list(train_test_overlap)[:3]:
        print(f"     - {sample}")
else:
    print(f"   OK: No overlap between train and test")

print(f"\n3. VAL-TEST OVERLAP:")
if val_test_overlap:
    print(f"   LEAKAGE FOUND: {len(val_test_overlap)} samples in both val and test!")
    for sample in list(val_test_overlap)[:3]:
        print(f"     - {sample}")
else:
    print(f"   OK: No overlap between val and test")

# Check for duplicates within each split
print(f"\n{'='*60}")
print("[DUPLICATE CHECK WITHIN SPLITS]")
print(f"{'='*60}")

for split_name, split_data in [('train', splits['train']), 
                                ('val', splits['val']), 
                                ('test', splits['test'])]:
    split_set = set(split_data)
    if len(split_set) < len(split_data):
        print(f"\n{split_name.upper()}: DUPLICATES FOUND!")
        print(f"  Expected: {len(split_data)}")
        print(f"  Unique: {len(split_set)}")
        print(f"  Duplicates: {len(split_data) - len(split_set)}")
    else:
        print(f"\n{split_name.upper()}: OK (no duplicates)")

# Check for perfect split (no samples outside main sets)
all_samples = train_set | val_set | test_set
print(f"\n{'='*60}")
print("[SPLIT INTEGRITY]")
print(f"{'='*60}")
print(f"\nTotal unique samples: {len(all_samples)}")
print(f"Expected (4,899): {len(all_samples) == 4899}")

# Final verdict
print(f"\n{'='*60}")
print("[VERDICT]")
print(f"{'='*60}")

total_leakage = len(train_val_overlap) + len(train_test_overlap) + len(val_test_overlap)

if total_leakage > 0:
    print(f"\nDATA LEAKAGE DETECTED: {total_leakage} overlapping samples")
    print("WARNING: Model evaluation results may be unreliable!")
else:
    print(f"\nNO DATA LEAKAGE FOUND")
    print("Splits are properly isolated. Evaluation is valid.")
    
print(f"\nSplit proportions:")
total = len(train_set) + len(val_set) + len(test_set)
print(f"  Train: {len(train_set)}/{total} ({100*len(train_set)/total:.1f}%)")
print(f"  Val:   {len(val_set)}/{total} ({100*len(val_set)/total:.1f}%)")
print(f"  Test:  {len(test_set)}/{total} ({100*len(test_set)/total:.1f}%)")
