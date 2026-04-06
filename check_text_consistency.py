#!/usr/bin/env python
"""Analyze text consistency in Kaggle IAM dataset"""

import os
import json
from pathlib import Path
import re

print("[INFO] Analyzing text consistency in Kaggle IAM dataset...")

# Check if we have the splits file with metadata
split_file = "data/processed/kaggle_iam/splits.json"

if not os.path.exists(split_file):
    print(f"[INFO] Splits file not found, creating analysis from filenames...")
    
    # Check what metadata exists
    forms_file = "data/raw/kaggle_iam/forms_for_parsing.txt"
    
    if os.path.exists(forms_file):
        print(f"\n[INFO] Found forms_for_parsing.txt - analyzing text content...")
        
        with open(forms_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        print(f"Total lines in metadata: {len(lines)}")
        
        # Sample analysis
        text_samples = []
        case_patterns = {
            'all_uppercase': 0,
            'all_lowercase': 0,
            'mixed_case': 0,
            'first_cap': 0
        }
        
        punctuation_count = 0
        space_variations = set()
        
        for line in lines[:1000]:  # Sample first 1000
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Try to extract text (format may vary)
            parts = line.split('\t')
            if len(parts) > 0:
                text = parts[-1] if parts else ""
                
                if text and len(text) > 0:
                    text_samples.append(text)
                    
                    # Check case
                    if text.isupper():
                        case_patterns['all_uppercase'] += 1
                    elif text.islower():
                        case_patterns['all_lowercase'] += 1
                    elif text[0].isupper():
                        case_patterns['first_cap'] += 1
                    else:
                        case_patterns['mixed_case'] += 1
                    
                    # Check punctuation
                    if any(p in text for p in '.,!?;:\'"()[]{}'):
                        punctuation_count += 1
                    
                    # Check spacing
                    if '  ' in text:
                        space_variations.add('double_space')
                    if '\t' in text:
                        space_variations.add('tabs')
        
        print(f"\n{'='*60}")
        print("[TEXT CHARACTERISTICS ANALYSIS]")
        print(f"{'='*60}")
        
        print(f"\nSample size analyzed: {len(text_samples)} texts")
        
        if text_samples:
            print(f"\nCASE PATTERNS:")
            total_texts = sum(case_patterns.values())
            if total_texts > 0:
                print(f"  Uppercase: {case_patterns['all_uppercase']} ({100*case_patterns['all_uppercase']/total_texts:.1f}%)")
                print(f"  Lowercase: {case_patterns['all_lowercase']} ({100*case_patterns['all_lowercase']/total_texts:.1f}%)")
                print(f"  First cap: {case_patterns['first_cap']} ({100*case_patterns['first_cap']/total_texts:.1f}%)")
                print(f"  Mixed case: {case_patterns['mixed_case']} ({100*case_patterns['mixed_case']/total_texts:.1f}%)")
            
            print(f"\nPUNCTUATION:")
            print(f"  With punctuation: {punctuation_count} ({100*punctuation_count/len(text_samples):.1f}%)")
            print(f"  Without punctuation: {len(text_samples)-punctuation_count} ({100*(len(text_samples)-punctuation_count)/len(text_samples):.1f}%)")
            
            print(f"\nSPACING VARIATIONS:")
            if space_variations:
                for var in space_variations:
                    print(f"  - {var}: YES")
            else:
                print(f"  - All standard single spaces")
            
            print(f"\nSAMPLE TEXTS:")
            for text in text_samples[:5]:
                print(f"  '{text}'")
        
        print(f"\n{'='*60}")
        print("[ASSESSMENT]")
        print(f"{'='*60}")
        
        if case_patterns['all_lowercase'] > total_texts * 0.5:
            consistency = "HIGH - Mostly lowercase"
        elif case_patterns['mixed_case'] > total_texts * 0.5:
            consistency = "MEDIUM - Mixed case patterns"
        else:
            consistency = "LOW - Variable case patterns"
        
        print(f"Case Consistency: {consistency}")
        print(f"Punctuation: {'CONSISTENT' if punctuation_count/len(text_samples) > 0.8 else 'VARIABLE'}")
        print(f"Spacing: {'CONSISTENT' if not space_variations else 'VARIABLE'}")
        
else:
    print("[INFO] Analyzing from splits metadata...")
    with open(split_file, 'r') as f:
        splits = json.load(f)
    
    print(f"Dataset splits created:")
    print(f"  Train: {len(splits['train'])} samples")
    print(f"  Val: {len(splits['val'])} samples")
    print(f"  Test: {len(splits['test'])} samples")
    
    print("\n[INFO] For detailed text analysis, metadata/transcriptions file needed")
    print("[INFO] Check forms_for_parsing.txt for text content")

print("\n[SUMMARY]")
print("IAM dataset is expected to have:")
print("  - Mixed case (real-world variation)")
print("  - Punctuation (natural text)")
print("  - Variable spacing (handwritten)")
print("  - Natural language patterns")
print("\nThis is NORMAL and expected for real handwriting!")
