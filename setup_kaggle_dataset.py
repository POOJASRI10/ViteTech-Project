import os
import shutil
from pathlib import Path

kaggle_dataset_path = r"C:\Users\POOJA SHETTY\.cache\kagglehub\datasets\tejasreddy\iam-handwriting-top50\versions\2"
project_data_path = r"data\raw\kaggle_iam"

print("[INFO] Setting up Kaggle IAM Handwriting dataset...")
print(f"[INFO] Source: {kaggle_dataset_path}")
print(f"[INFO] Destination: {project_data_path}")

# Create destination directory
os.makedirs(project_data_path, exist_ok=True)

# Copy dataset
data_subset_src = os.path.join(kaggle_dataset_path, "data_subset")
forms_txt_src = os.path.join(kaggle_dataset_path, "forms_for_parsing.txt")

if os.path.exists(data_subset_src):
    print(f"\n[PROGRESS] Copying dataset files...")
    # Count files
    total_files = len(list(Path(data_subset_src).rglob("*")))
    print(f"[INFO] Total items: {total_files}")
    
    # Copy
    dest_subset = os.path.join(project_data_path, "data_subset")
    if not os.path.exists(dest_subset):
        shutil.copytree(data_subset_src, dest_subset)
        print(f"[SUCCESS] Copied data_subset")
    else:
        print(f"[INFO] data_subset already exists")

if os.path.exists(forms_txt_src):
    dest_forms = os.path.join(project_data_path, "forms_for_parsing.txt")
    if not os.path.exists(dest_forms):
        shutil.copy2(forms_txt_src, dest_forms)
        print(f"[SUCCESS] Copied forms_for_parsing.txt")

# Check what we have
dataset_dir = os.path.join(project_data_path, "data_subset")
if os.path.exists(dataset_dir):
    subdirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    print(f"\n[INFO] Dataset structure:")
    print(f"  - Subdirectories: {len(subdirs)}")
    print(f"  - Sample subdirs: {subdirs[:3]}")
    
    # Count images
    img_count = len(list(Path(dataset_dir).rglob("*.png")))
    print(f"  - Total PNG images: {img_count}")

print(f"\n[SUCCESS] Dataset setup complete!")
print(f"[NEXT] Run: python scripts/02_data_prep.py --data_dir {project_data_path} --output_dir data/processed/kaggle_iam")
