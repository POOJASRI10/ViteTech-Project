"""Create a small demo dataset for testing"""
from PIL import Image, ImageDraw
import os
import json

# Create output directory - using IAM format structure
os.makedirs("data/raw/demo/lines/demo", exist_ok=True)

# Create sample images
sample_texts = [
    "The quick brown fox",
    "Handwriting recognition",
    "OCR system test",
    "Machine learning model",
    "Python programming",
    "Deep learning networks",
    "Computer vision tasks",
    "Text recognition demo",
]

# Create lines.txt metadata in IAM format
metadata_lines = ["# IAM DEMO DATASET\n", "# image_id status col row pos gray num_components text\n"]

for i, text in enumerate(sample_texts):
    # Create image ID matching IAM format (demo-XXXX)
    image_id = f"demo-{i:04d}"
    
    # Create image
    img = Image.new('RGB', (400, 64), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((10, 20), text, fill='black')
    
    # Save image in subdirectory matching IAM structure
    img_path = f"data/raw/demo/lines/demo/{image_id}.png"
    img.save(img_path)
    
    # Add metadata line in IAM format: image_id status col row pos gray num_components text
    metadata_lines.append(f"{image_id} ok 0 0 0 0 0 {text}\n")

# Save metadata
with open("data/raw/demo/lines.txt", 'w', encoding='utf-8') as f:
    f.writelines(metadata_lines)

print(f"✓ Created demo dataset with {len(sample_texts)} samples")
print(f"  Location: data/raw/demo/")
print(f"  - lines.txt: Metadata")
print(f"  - lines/demo/: Image files")
