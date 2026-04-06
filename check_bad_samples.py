#!/usr/bin/env python
"""Check for bad samples in Kaggle IAM dataset"""

import os
from pathlib import Path
from PIL import Image
import numpy as np

print("[INFO] Checking for bad samples in Kaggle IAM dataset...")

data_dir = "data/raw/kaggle_iam/data_subset"

if not os.path.exists(data_dir):
    print(f"[ERROR] Directory not found: {data_dir}")
    exit(1)

# Find all PNG files
png_files = list(Path(data_dir).rglob("*.png"))
print(f"[INFO] Analyzing {len(png_files)} PNG images...")

bad_samples = {
    'corrupted': [],
    'tiny': [],
    'blank': [],
    'very_dark': [],
    'read_error': []
}

stats = {
    'total': len(png_files),
    'ok': 0,
    'width_min': float('inf'),
    'width_max': 0,
    'height_min': float('inf'),
    'height_max': 0,
}

for i, png_file in enumerate(png_files):
    if (i + 1) % 500 == 0:
        print(f"  Processed {i + 1}/{len(png_files)}...")
    
    try:
        # Try to open and verify image
        img = Image.open(png_file)
        width, height = img.size
        
        # Check dimensions
        stats['width_min'] = min(stats['width_min'], width)
        stats['width_max'] = max(stats['width_max'], width)
        stats['height_min'] = min(stats['height_min'], height)
        stats['height_max'] = max(stats['height_max'], height)
        
        # Check for tiny images (too small to contain text)
        if width < 100 or height < 20:
            bad_samples['tiny'].append({
                'file': str(png_file),
                'size': f"{width}x{height}"
            })
            continue
        
        # Check if blank/mostly white
        img_array = np.array(img.convert('L'))
        mean_pixel = np.mean(img_array)
        
        if mean_pixel > 250:  # Almost completely white
            bad_samples['blank'].append({
                'file': str(png_file),
                'mean_pixel': f"{mean_pixel:.1f}"
            })
            continue
        
        if mean_pixel < 30:  # Almost completely black
            bad_samples['very_dark'].append({
                'file': str(png_file),
                'mean_pixel': f"{mean_pixel:.1f}"
            })
            continue
        
        stats['ok'] += 1
        
    except Exception as e:
        bad_samples['read_error'].append({
            'file': str(png_file),
            'error': str(e)[:50]
        })

# Print results
print(f"\n{'='*60}")
print("[RESULTS]")
print(f"{'='*60}")

print(f"\nImage Dimensions:")
print(f"  Width:  {stats['width_min']}-{stats['width_max']} pixels")
print(f"  Height: {stats['height_min']}-{stats['height_max']} pixels")

print(f"\nSample Status:")
print(f"  OK samples:       {stats['ok']:,} ({100*stats['ok']/stats['total']:.1f}%)")
print(f"  Bad samples:      {stats['total'] - stats['ok']:,} ({100*(stats['total']-stats['ok'])/stats['total']:.1f}%)")

print(f"\nBad Sample Breakdown:")
print(f"  Corrupted:        {len(bad_samples['corrupted'])}")
print(f"  Tiny images:      {len(bad_samples['tiny'])}")
print(f"  Blank/white:      {len(bad_samples['blank'])}")
print(f"  Very dark/black:  {len(bad_samples['very_dark'])}")
print(f"  Read errors:      {len(bad_samples['read_error'])}")

# Show examples
print(f"\n{'='*60}")
if len(bad_samples['tiny']) > 0:
    print(f"\n[EXAMPLES] Tiny images:")
    for item in bad_samples['tiny'][:3]:
        print(f"  - {item['file'][-50:]}: {item['size']}")

if len(bad_samples['blank']) > 0:
    print(f"\n[EXAMPLES] Blank/white images:")
    for item in bad_samples['blank'][:3]:
        print(f"  - {item['file'][-50:]}: pixel value {item['mean_pixel']}")

if len(bad_samples['very_dark']) > 0:
    print(f"\n[EXAMPLES] Very dark images:")
    for item in bad_samples['very_dark'][:3]:
        print(f"  - {item['file'][-50:]}: pixel value {item['mean_pixel']}")

if len(bad_samples['read_error']) > 0:
    print(f"\n[EXAMPLES] Read errors:")
    for item in bad_samples['read_error'][:3]:
        print(f"  - {item['file'][-50:]}: {item['error']}")

# Overall assessment
print(f"\n{'='*60}")
print("[ASSESSMENT]")
bad_count = sum(len(v) for v in bad_samples.values())
if bad_count == 0:
    print("✓ PERFECT: No bad samples detected!")
    print("✓ Dataset is clean and ready for training")
    quality = "EXCELLENT"
elif bad_count <= 50:
    print(f"✓ GOOD: Only {bad_count} bad samples (<1.3%)")
    print("✓ Dataset quality is acceptable")
    quality = "GOOD"
elif bad_count <= 100:
    print(f"⚠ MODERATE: {bad_count} bad samples (2%)")
    print("⚠ Consider removing bad samples")
    quality = "MODERATE"
else:
    print(f"✗ POOR: {bad_count} bad samples (>2%)")
    print("✗ Recommend data cleaning")
    quality = "POOR"

print(f"\nData Quality: {quality}")
print(f"Usable samples: {stats['ok']:,} / {stats['total']:,}")
print(f"Recommendation: {'Ready for training' if quality in ['EXCELLENT', 'GOOD'] else 'Clean before training'}")
