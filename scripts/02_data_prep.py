"""
Step 2: Data Preparation - Clean, normalize, and split data

This script:
- Loads the IAM dataset
- Normalizes text according to defined rules
- Verifies train/val/test splits with no data leakage
- Saves processed data for training
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import argparse
from sklearn.model_selection import train_test_split

import pandas as pd
from PIL import Image

from utils import TextNormalizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreparator:
    """Prepare dataset for training"""
    
    def __init__(self, data_dir: str, output_dir: str):
        """
        Initialize data preparer.
        
        Args:
            data_dir: Path to raw IAM dataset
            output_dir: Path for processed output
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.samples = []
        self.train_split = []
        self.val_split = []
        self.test_split = []
    
    def load_iam_dataset(self) -> bool:
        """Load IAM handwriting dataset"""
        logger.info("Loading IAM Handwriting dataset...")
        
        txt_file = self.data_dir / "lines.txt"
        if not txt_file.exists():
            logger.error(f"Metadata file not found: {txt_file}")
            return False
        
        try:
            with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # Skip header comments
            data_lines = [l.strip() for l in lines if l.strip() and not l.startswith('#')]
            
            logger.info(f"Processing {len(data_lines)} samples...")
            
            for line in data_lines:
                try:
                    parts = line.split(' ')
                    if len(parts) < 8:
                        continue
                    
                    image_id = parts[0]
                    status = parts[1]
                    text = ' '.join(parts[8:])
                    
                    # Only use OK status
                    if status != "ok":
                        continue
                    
                    # Verify image exists
                    # Convert image_id like "a01-000u" to path like "lines/a01/a01-000u.png"
                    parts_id = image_id.split('-')
                    if len(parts_id) >= 2:
                        subdir = parts_id[0]
                        image_path = self.data_dir / "lines" / subdir / f"{image_id}.png"
                        
                        if image_path.exists():
                            self.samples.append({
                                "image_id": image_id,
                                "image_path": str(image_path),
                                "text": text,
                                "text_normalized": TextNormalizer.normalize(text),
                            })
                except Exception as e:
                    logger.debug(f"Error processing line: {e}")
                    continue
            
            logger.info(f"Successfully loaded {len(self.samples)} valid samples")
            return len(self.samples) > 0
        
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return False
    
    def verify_images(self, max_check: int = 100) -> Dict[str, Any]:
        """Verify a sample of images can be opened"""
        logger.info(f"Verifying images (checking {min(max_check, len(self.samples))} samples)...")
        
        verification = {
            "total_samples": len(self.samples),
            "images_verified": 0,
            "images_valid": 0,
            "images_corrupted": 0,
            "size_stats": {},
        }
        
        sizes = []
        for sample in self.samples[:max_check]:
            try:
                img = Image.open(sample["image_path"])
                verification["images_verified"] += 1
                verification["images_valid"] += 1
                sizes.append(img.size)
            except Exception as e:
                verification["images_verified"] += 1
                verification["images_corrupted"] += 1
                logger.warning(f"Cannot open image {sample['image_id']}: {e}")
        
        if sizes:
            widths, heights = zip(*sizes)
            verification["size_stats"] = {
                "min_width": min(widths),
                "max_width": max(widths),
                "mean_width": float(sum(widths) / len(widths)),
                "min_height": min(heights),
                "max_height": max(heights),
                "mean_height": float(sum(heights) / len(heights)),
            }
        
        return verification
    
    def check_no_leakage(self, train: List[Dict], val: List[Dict], test: List[Dict]) -> Dict[str, Any]:
        """Verify no data leakage between splits"""
        train_ids = set(s["image_id"] for s in train)
        val_ids = set(s["image_id"] for s in val)
        test_ids = set(s["image_id"] for s in test)
        
        leakage_check = {
            "train_val_overlap": len(train_ids & val_ids),
            "train_test_overlap": len(train_ids & test_ids),
            "val_test_overlap": len(val_ids & test_ids),
        }
        
        no_leakage = all(v == 0 for v in leakage_check.values())
        leakage_check["no_leakage"] = no_leakage
        
        if not no_leakage:
            logger.warning("⚠️  Data leakage detected!")
        else:
            logger.info("✓ No data leakage detected")
        
        return leakage_check
    
    def split_dataset(self, train_ratio: float = 0.7, val_ratio: float = 0.15, 
                     test_ratio: float = 0.15, random_state: int = 42) -> Dict[str, Any]:
        """
        Split dataset into train/val/test.
        
        Args:
            train_ratio: Proportion for training (default: 0.7)
            val_ratio: Proportion for validation (default: 0.15)
            test_ratio: Proportion for testing (default: 0.15)
            random_state: Random seed for reproducibility
        
        Returns:
            Dictionary with split statistics
        """
        logger.info(f"Splitting dataset: train={train_ratio}, val={val_ratio}, test={test_ratio}")
        
        if not self.samples:
            logger.error("No samples loaded")
            return {}
        
        # Ensure ratios sum to 1.0
        total_ratio = train_ratio + val_ratio + test_ratio
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio
        
        # First split: train + temp
        train_temp, test = train_test_split(
            self.samples,
            test_size=test_ratio,
            random_state=random_state
        )
        
        # Second split: train and val
        train, val = train_test_split(
            train_temp,
            test_size=val_ratio / (val_ratio + train_ratio),
            random_state=random_state
        )
        
        self.train_split = train
        self.val_split = val
        self.test_split = test
        
        split_stats = {
            "total_samples": len(self.samples),
            "train_samples": len(train),
            "val_samples": len(val),
            "test_samples": len(test),
            "train_ratio": len(train) / len(self.samples),
            "val_ratio": len(val) / len(self.samples),
            "test_ratio": len(test) / len(self.samples),
        }
        
        # Check for leakage
        split_stats["leakage_check"] = self.check_no_leakage(train, val, test)
        
        return split_stats
    
    def save_splits(self, image_dir_in_split: bool = True) -> Dict[str, str]:
        """
        Save train/val/test splits as JSON files.
        
        Each file contains list of samples with format:
        {
            "image_id": "a01-000u",
            "image_path": "...",
            "text_original": "...",
            "text_normalized": "..."
        }
        
        Args:
            image_dir_in_split: Whether to include image_path in each sample
        
        Returns:
            Dictionary with paths to saved files
        """
        logger.info("Saving dataset splits...")
        
        output_paths = {}
        
        for split_name, split_data in [
            ("train", self.train_split),
            ("val", self.val_split),
            ("test", self.test_split),
        ]:
            # Prepare data with original and normalized text
            prepared = []
            for sample in split_data:
                item = {
                    "image_id": sample["image_id"],
                    "text": sample["text"],
                    "text_normalized": sample["text_normalized"],
                }
                if image_dir_in_split:
                    item["image_path"] = sample["image_path"]
                
                prepared.append(item)
            
            # Save to JSON
            output_path = self.output_dir / f"{split_name}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(prepared, f, indent=2, ensure_ascii=False)
            
            output_paths[split_name] = str(output_path)
            logger.info(f"Saved {split_name} split: {output_path} ({len(prepared)} samples)")
        
        return output_paths
    
    def save_text_splits(self) -> Dict[str, str]:
        """
        Save text-only splits (used for some model training approaches).
        
        Format: One line per sample with normalized text
        """
        logger.info("Saving text-only splits...")
        
        output_paths = {}
        
        for split_name, split_data in [
            ("train", self.train_split),
            ("val", self.val_split),
            ("test", self.test_split),
        ]:
            output_path = self.output_dir / f"{split_name}_text.txt"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for sample in split_data:
                    f.write(sample["text_normalized"] + "\n")
            
            output_paths[split_name] = str(output_path)
            logger.info(f"Saved text-only {split_name}: {output_path}")
        
        return output_paths
    
    def generate_preparation_report(self) -> str:
        """Generate readable preparation report"""
        report = []
        report.append("=" * 80)
        report.append("DATA PREPARATION REPORT")
        report.append("=" * 80)
        report.append(f"Timestamp: {pd.Timestamp.now().isoformat()}")
        report.append("")
        
        report.append("DATASET STATISTICS")
        report.append("-" * 80)
        total = len(self.samples)
        report.append(f"Total samples loaded: {total}")
        report.append("")
        
        if self.train_split:
            report.append("SPLIT DISTRIBUTION")
            report.append("-" * 80)
            train_cnt = len(self.train_split)
            val_cnt = len(self.val_split)
            test_cnt = len(self.test_split)
            
            report.append(f"Training samples: {train_cnt} ({train_cnt/total*100:.1f}%)")
            report.append(f"Validation samples: {val_cnt} ({val_cnt/total*100:.1f}%)")
            report.append(f"Test samples: {test_cnt} ({test_cnt/total*100:.1f}%)")
            report.append("")
        
        report.append("TEXT NORMALIZATION")
        report.append("-" * 80)
        report.append(TextNormalizer.rules_summary())
        report.append("")
        
        # Example samples
        if self.samples:
            report.append("SAMPLE TRANSFORMATIONS")
            report.append("-" * 80)
            for i, sample in enumerate(self.samples[:3]):
                report.append(f"\nSample {i+1}:")
                report.append(f"  Original: '{sample['text']}'")
                report.append(f"  Normalized: '{sample['text_normalized']}'")
        
        report.append("\n" + "=" * 80)
        report.append("NEXT STEPS:")
        report.append("1. Run 03_baseline.py to generate baseline predictions")
        report.append("2. Evaluate baseline model performance")
        report.append("3. Run 04_finetune.py to fine-tune the model")
        
        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for training")
    parser.add_argument("--data_dir", type=str, default="data/raw/iam",
                       help="Path to raw IAM dataset")
    parser.add_argument("--output_dir", type=str, default="data/processed",
                       help="Path for processed output")
    parser.add_argument("--train_ratio", type=float, default=0.7,
                       help="Training set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.15,
                       help="Validation set ratio")
    parser.add_argument("--test_ratio", type=float, default=0.15,
                       help="Test set ratio")
    parser.add_argument("--random_state", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Prepare data
    preparator = DataPreparator(args.data_dir, args.output_dir)
    
    if not preparator.load_iam_dataset():
        logger.error("Failed to load dataset")
        return
    
    # Verify images
    verification = preparator.verify_images()
    logger.info(f"Image verification: {verification['images_valid']}/{verification['images_verified']} valid")
    
    # Split dataset
    split_stats = preparator.split_dataset(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_state
    )
    
    logger.info(f"Split statistics: {split_stats}")
    
    # Save splits
    preparator.save_splits()
    preparator.save_text_splits()
    
    # Generate report
    report = preparator.generate_preparation_report()
    print(report)
    
    # Save report
    report_path = Path(args.output_dir) / "preparation_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"Report saved to {report_path}")
    
    logger.info("✓ Data preparation complete!")


if __name__ == "__main__":
    main()

import pandas as pd
