#!/usr/bin/env python
"""Check for duplicates in Kaggle IAM dataset"""

import os
from pathlib import Path
from PIL import Image
import hashlib

print("[INFO] Checking for duplicates in Kaggle IAM dataset...")

data_dir = "data/raw/kaggle_iam/data_subset"

if not os.path.exists(data_dir):
    print(f"[ERROR] Directory not found: {data_dir}")
    exit(1)

# Find all PNG files
png_files = list(Path(data_dir).rglob("*.png"))
print(f"[INFO] Found {len(png_files)} PNG images")

if not png_files:
    print("[ERROR] No PNG files found!")
    exit(1)

# Check for file name duplicates
file_names = [f.name for f in png_files]
duplicate_names = [name for name in file_names if file_names.count(name) > 1]

if duplicate_names:
    print(f"[WARNING] Found {len(set(duplicate_names))} duplicate file names:")
    for name in set(duplicate_names):
        count = file_names.count(name)
        print(f"  - {name}: appears {count} times")
else:
    print("[PASS] No duplicate file names found")

# Check for identical images (by hash)
print("\n[INFO] Checking for identical images (this may take a moment)...")
file_hashes = {}
duplicate_images = 0

for i, png_file in enumerate(png_files):
    if (i + 1) % 500 == 0:
        print(f"  Processed {i + 1}/{len(png_files)}...")
    
    try:
        with open(png_file, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        
        if file_hash in file_hashes:
            duplicate_images += 1
            print(f"[DUPLICATE] {png_file} matches {file_hashes[file_hash]}")
        else:
            file_hashes[file_hash] = str(png_file)
    except Exception as e:
        print(f"[ERROR] Reading {png_file}: {e}")

print(f"\n[RESULTS]")
print(f"  - Total images: {len(png_files)}")
print(f"  - Unique hashes: {len(file_hashes)}")
print(f"  - Duplicate images found: {duplicate_images}")

if duplicate_images == 0:
    print(f"\n[PASS] NO DUPLICATES FOUND - Dataset is clean!")
else:
    print(f"\n[WARNING] {duplicate_images} duplicate images detected")

# Summary
print(f"\n[SUMMARY]")
print(f"  Duplicate file names: {'NO' if not duplicate_names else 'YES'}")
print(f"  Duplicate images: {'NO' if duplicate_images == 0 else 'YES'}")
print(f"  Data quality: {'GOOD - Ready for training' if duplicate_images == 0 else 'NEEDS CLEANING'}")
