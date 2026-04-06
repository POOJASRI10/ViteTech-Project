#!/usr/bin/env python
"""Quick dataset preparation for Kaggle IAM dataset"""

import os
import json
from pathlib import Path
import random

print("[INFO] Preparing Kaggle IAM Handwriting dataset...")

kaggle_data_dir = "data/raw/kaggle_iam/data_subset"
output_dir = "data/processed/kaggle_iam"

os.makedirs(output_dir, exist_ok=True)

# Find all PNG files
png_files = list(Path(kaggle_data_dir).rglob("*.png"))
print(f"[INFO] Found {len(png_files)} PNG images")

if not png_files:
    print("[ERROR] No PNG files found!")
    exit(1)

# Create simple train/val/test split
random.shuffle(png_files)
n_train = int(0.7 * len(png_files))
n_val = int(0.15 * len(png_files))

train_files = png_files[:n_train]
val_files = png_files[n_train:n_train+n_val]
test_files = png_files[n_train+n_val:]

print(f"[INFO] Split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")

# Save metadata
splits = {
    'train': [str(f) for f in train_files],
    'val': [str(f) for f in val_files],
    'test': [str(f) for f in test_files]
}

with open(os.path.join(output_dir, "splits.json"), 'w') as f:
    json.dump(splits, f, indent=2)

print(f"[SUCCESS] Data preparation complete!")
print(f"[SUCCESS] Metadata saved to: {output_dir}/splits.json")
print(f"\n[NEXT] Run baseline on Kaggle dataset:")
print(f"  python scripts/03_baseline.py --data_dir {kaggle_data_dir} --split_dir {output_dir}")
