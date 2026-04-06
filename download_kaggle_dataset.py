import kagglehub
import os

print("[INFO] Downloading IAM Handwriting Top 50 dataset from Kaggle...")

try:
    # Download latest version
    path = kagglehub.dataset_download("tejasreddy/iam-handwriting-top50")
    print(f"[SUCCESS] Dataset downloaded to: {path}")
    
    # Check what's inside
    if os.path.exists(path):
        files = os.listdir(path)
        print(f"\n[INFO] Files in dataset ({len(files)} items):")
        for f in files[:10]:  # Show first 10
            print(f"  - {f}")
        if len(files) > 10:
            print(f"  ... and {len(files) - 10} more")
    
    print("\n[NEXT STEPS]:")
    print(f"1. Dataset location: {path}")
    print("2. Update project to use this dataset")
    print("3. Run data preparation: python scripts/02_data_prep.py")
    
except Exception as e:
    print(f"[ERROR] Failed to download: {e}")
    print("\nNote: Make sure you have Kaggle API credentials configured")
    print("Configure at: ~/.kaggle/kaggle.json")
