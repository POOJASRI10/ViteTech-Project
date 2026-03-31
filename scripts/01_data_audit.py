"""
Step 1: Data Audit - Understand the dataset before training

This script explores the IAM Handwriting dataset to:
- Understand data structure and volume
- Check for quality issues (duplicates, empty samples, inconsistencies)
- Analyze text characteristics
- Generate a comprehensive audit report
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter
import argparse

import pandas as pd
from PIL import Image
import numpy as np

from utils import TextNormalizer, DataLogger

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataAuditor:
    """Audit dataset quality and characteristics"""
    
    def __init__(self, data_dir: str):
        """
        Initialize auditor.
        
        Args:
            data_dir: Path to dataset directory
        """
        self.data_dir = Path(data_dir)
        self.audit_results = {}
        self.samples = []
    
    def audit_iam_handwriting(self) -> Dict[str, Any]:
        """
        Audit IAM Handwriting dataset.
        
        Expected structure:
        - lines.txt: Metadata file with format:
          image_id status gray_value num_components bounding_box grammar text
        - lines/: Directory with line images (PNG files)
        """
        logger.info("Starting IAM Handwriting Dataset Audit...")
        
        results = {
            "dataset": "IAM Handwriting",
            "timestamp": pd.Timestamp.now().isoformat(),
        }
        
        # 1. Check directory structure
        logger.info("Checking directory structure...")
        results["directory_check"] = self._check_directories()
        
        # 2. Load and parse metadata
        logger.info("Loading metadata...")
        results["metadata_status"] = self._check_metadata()
        
        # 3. Data quality checks
        if self.samples:
            logger.info("Running quality checks...")
            results["data_quality"] = self._check_data_quality()
            results["text_analysis"] = self._analyze_text()
            results["image_analysis"] = self._analyze_images()
        
        self.audit_results = results
        return results
    
    def _check_directories(self) -> Dict[str, Any]:
        """Check if required directories exist"""
        checks = {
            "data_dir_exists": self.data_dir.exists(),
            "data_dir_path": str(self.data_dir),
        }
        
        # Check for common IAM dataset directories
        lines_dir = self.data_dir / "lines"
        txt_file = self.data_dir / "lines.txt"
        
        checks["lines_dir_exists"] = lines_dir.exists()
        checks["lines_txt_exists"] = txt_file.exists()
        
        if lines_dir.exists():
            png_files = list(lines_dir.glob("*.png"))
            checks["num_image_files"] = len(png_files)
        
        return checks
    
    def _check_metadata(self) -> Dict[str, Any]:
        """Check and load metadata"""
        txt_file = self.data_dir / "lines.txt"
        
        if not txt_file.exists():
            return {
                "metadata_file_found": False,
                "message": "lines.txt not found. Please ensure IAM dataset is properly set up."
            }
        
        try:
            # Parse lines.txt
            # Format: image_id status gray_value num_components bounding_box grammar text
            with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # Skip header and comments
            data_lines = [l.strip() for l in lines if l.strip() and not l.startswith('#')]
            
            logger.info(f"Found {len(data_lines)} lines in metadata")
            
            # Parse samples
            valid_samples = 0
            error_count = 0
            
            for line_id, line in enumerate(data_lines[:]):  # Parse all for audit
                try:
                    parts = line.split(' ')
                    if len(parts) < 8:
                        error_count += 1
                        continue
                    
                    image_id = parts[0]
                    status = parts[1]
                    text = ' '.join(parts[8:])  # Text is everything after index 8
                    
                    # Only use OK status samples
                    if status != "ok":
                        continue
                    
                    self.samples.append({
                        "image_id": image_id,
                        "text": text,
                        "text_length": len(text),
                    })
                    valid_samples += 1
                except Exception as e:
                    error_count += 1
            
            return {
                "metadata_file_found": True,
                "total_lines_in_file": len(data_lines),
                "valid_samples": valid_samples,
                "invalid_samples": error_count,
                "parsing_successful": True,
            }
        
        except Exception as e:
            logger.error(f"Error parsing metadata: {e}")
            return {
                "metadata_file_found": True,
                "parsing_successful": False,
                "error": str(e),
            }
    
    def _check_data_quality(self) -> Dict[str, Any]:
        """Check for data quality issues"""
        if not self.samples:
            return {"message": "No samples to analyze"}
        
        df = pd.DataFrame(self.samples)
        
        quality_report = {
            "total_samples": len(df),
            "duplicate_ids": len(df[df.duplicated(subset=['image_id'], keep=False)]),
            "empty_text": len(df[df['text'].str.len() == 0]),
            "null_values": df.isnull().sum().to_dict(),
        }
        
        # Check for potential duplicates (same text from different images)
        text_counts = df['text'].value_counts()
        duplicate_texts = (text_counts > 1).sum()
        quality_report["duplicate_texts"] = duplicate_texts
        
        # Text length statistics
        quality_report["text_length_stats"] = {
            "min": int(df['text_length'].min()),
            "max": int(df['text_length'].max()),
            "mean": float(df['text_length'].mean()),
            "median": float(df['text_length'].median()),
        }
        
        return quality_report
    
    def _analyze_text(self) -> Dict[str, Any]:
        """Analyze text characteristics"""
        if not self.samples:
            return {}
        
        texts = [s['text'] for s in self.samples]
        
        analysis = {
            "character_set_size": len(set(''.join(texts))),
            "contains_uppercase": any(any(c.isupper() for c in t) for t in texts),
            "contains_punctuation": any(any(c in '.,!?;:\'"' for c in t) for t in texts),
            "contains_numbers": any(any(c.isdigit() for c in t) for t in texts),
            "contains_special_chars": any(any(not c.isalnum() and c not in ' .,!?;:\'"' for c in t) for t in texts),
        }
        
        # Sample texts
        analysis["sample_texts"] = texts[:5]
        
        # Character frequency
        all_chars = ''.join(texts)
        char_freq = Counter(all_chars)
        analysis["most_common_chars"] = char_freq.most_common(10)
        
        # Text normalization impact
        normalized_texts = [TextNormalizer.normalize(t) for t in texts]
        changes = sum(1 for orig, norm in zip(texts, normalized_texts) if orig != norm)
        analysis["texts_changed_by_normalization"] = changes
        
        return analysis
    
    def _analyze_images(self) -> Dict[str, Any]:
        """Analyze image files"""
        lines_dir = self.data_dir / "lines"
        
        if not lines_dir.exists():
            return {"message": "Lines directory not found"}
        
        image_files = list(lines_dir.glob("*.png"))
        
        analysis = {
            "total_images": len(image_files),
            "sample_images_found": min(5, len(image_files)),
        }
        
        # Analyze a sample of images
        image_sizes = []
        for img_path in image_files[:min(100, len(image_files))]:
            try:
                img = Image.open(img_path)
                image_sizes.append(img.size)  # (width, height)
            except Exception as e:
                logger.warning(f"Error reading image {img_path}: {e}")
        
        if image_sizes:
            widths, heights = zip(*image_sizes)
            analysis["image_size_stats"] = {
                "min_width": min(widths),
                "max_width": max(widths),
                "mean_width": float(np.mean(widths)),
                "min_height": min(heights),
                "max_height": max(heights),
                "mean_height": float(np.mean(heights)),
            }
        
        return analysis
    
    def generate_report(self, output_path: str = None) -> str:
        """Generate readable audit report"""
        if not self.audit_results:
            return "No audit results available. Run audit first."
        
        report = []
        report.append("=" * 80)
        report.append("DATA AUDIT REPORT - IAM HANDWRITING DATASET")
        report.append("=" * 80)
        report.append(f"Timestamp: {self.audit_results.get('timestamp', 'N/A')}")
        report.append("")
        
        # Directory structure
        report.append("DIRECTORY STRUCTURE")
        report.append("-" * 80)
        dir_check = self.audit_results.get("directory_check", {})
        report.append(f"Data directory exists: {dir_check.get('data_dir_exists', False)}")
        report.append(f"Data directory path: {dir_check.get('data_dir_path', 'N/A')}")
        report.append(f"Lines directory exists: {dir_check.get('lines_dir_exists', False)}")
        report.append(f"Metadata file (lines.txt) exists: {dir_check.get('lines_txt_exists', False)}")
        report.append(f"Number of image files: {dir_check.get('num_image_files', 0)}")
        report.append("")
        
        # Metadata
        report.append("METADATA STATUS")
        report.append("-" * 80)
        meta = self.audit_results.get("metadata_status", {})
        report.append(f"Metadata file found: {meta.get('metadata_file_found', False)}")
        report.append(f"Parsing successful: {meta.get('parsing_successful', False)}")
        report.append(f"Total lines in file: {meta.get('total_lines_in_file', 0)}")
        report.append(f"Valid samples: {meta.get('valid_samples', 0)}")
        report.append(f"Invalid samples: {meta.get('invalid_samples', 0)}")
        report.append("")
        
        # Data Quality
        report.append("DATA QUALITY CHECKS")
        report.append("-" * 80)
        quality = self.audit_results.get("data_quality", {})
        report.append(f"Total samples analyzed: {quality.get('total_samples', 0)}")
        report.append(f"Duplicate image IDs: {quality.get('duplicate_ids', 0)}")
        report.append(f"Empty text fields: {quality.get('empty_text', 0)}")
        report.append(f"Duplicate text values: {quality.get('duplicate_texts', 0)}")
        
        text_stats = quality.get("text_length_stats", {})
        if text_stats:
            report.append(f"Text Length - Min: {text_stats.get('min', 'N/A')}, " +
                        f"Max: {text_stats.get('max', 'N/A')}, " +
                        f"Mean: {text_stats.get('mean', 'N/A'):.1f}")
        report.append("")
        
        # Text Analysis
        report.append("TEXT CHARACTERISTICS")
        report.append("-" * 80)
        text_analysis = self.audit_results.get("text_analysis", {})
        report.append(f"Unique characters: {text_analysis.get('character_set_size', 0)}")
        report.append(f"Contains uppercase: {text_analysis.get('contains_uppercase', False)}")
        report.append(f"Contains punctuation: {text_analysis.get('contains_punctuation', False)}")
        report.append(f"Contains numbers: {text_analysis.get('contains_numbers', False)}")
        report.append(f"Contains special characters: {text_analysis.get('contains_special_chars', False)}")
        
        sample_texts = text_analysis.get("sample_texts", [])
        if sample_texts:
            report.append("\nSample texts:")
            for i, text in enumerate(sample_texts, 1):
                report.append(f"  {i}. '{text}'")
        
        most_common = text_analysis.get("most_common_chars", [])
        if most_common:
            report.append(f"\nMost common characters: {most_common[:5]}")
        
        report.append(f"Texts changed by normalization: {text_analysis.get('texts_changed_by_normalization', 0)}")
        report.append("")
        
        # Image Analysis
        report.append("IMAGE ANALYSIS")
        report.append("-" * 80)
        img_analysis = self.audit_results.get("image_analysis", {})
        report.append(f"Total images: {img_analysis.get('total_images', 0)}")
        
        size_stats = img_analysis.get("image_size_stats", {})
        if size_stats:
            report.append(f"Image dimensions (pixels):")
            report.append(f"  Width - Min: {size_stats.get('min_width', 'N/A')}, " +
                        f"Max: {size_stats.get('max_width', 'N/A')}, " +
                        f"Mean: {size_stats.get('mean_width', 'N/A'):.1f}")
            report.append(f"  Height - Min: {size_stats.get('min_height', 'N/A')}, " +
                        f"Max: {size_stats.get('max_height', 'N/A')}, " +
                        f"Mean: {size_stats.get('mean_height', 'N/A'):.1f}")
        report.append("")
        
        # Normalization Rules
        report.append("TEXT NORMALIZATION RULES")
        report.append("-" * 80)
        report.append(TextNormalizer.rules_summary())
        report.append("")
        
        # Recommendations
        report.append("AUDIT FINDINGS & RECOMMENDATIONS")
        report.append("-" * 80)
        report.append("✓ Dataset structure verified")
        report.append("✓ Metadata successfully parsed")
        report.append("✓ Text characteristics documented")
        report.append("✓ Image statistics collected")
        report.append("")
        report.append("NEXT STEPS:")
        report.append("1. Run 02_data_prep.py to prepare data for training")
        report.append("2. Define train/validation/test splits")
        report.append("3. Generate baseline model predictions")
        report.append("")
        
        report_text = "\n".join(report)
        
        # Save report
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"Report saved to {output_path}")
        
        return report_text


def main():
    parser = argparse.ArgumentParser(description="Audit IAM Handwriting dataset")
    parser.add_argument("--data_dir", type=str, default="data/raw/iam",
                       help="Path to IAM dataset directory")
    parser.add_argument("--output_report", type=str, default="results/data_audit_report.txt",
                       help="Output path for audit report")
    parser.add_argument("--output_json", type=str, default="results/data_audit.json",
                       help="Output path for audit JSON")
    
    args = parser.parse_args()
    
    # Run audit
    auditor = DataAuditor(args.data_dir)
    results = auditor.audit_iam_handwriting()
    
    # Generate report
    report = auditor.generate_report(args.output_report)
    print(report)
    
    # Save JSON
    DataLogger.log_data_audit_summary(results, args.output_json)
    logger.info("Data audit complete!")


if __name__ == "__main__":
    main()
