"""
Step 3: Baseline Model - Run inference with pretrained TrOCR model

This script:
- Loads pretrained microsoft/trocr-small-handwritten
- Runs inference on test dataset
- Calculates CER and WER metrics
- Saves predictions for analysis
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import argparse

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import numpy as np
from tqdm import tqdm
import pandas as pd

from utils import MetricsCalculator, ErrorAnalyzer, ResultsWriter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BaselineOCRModel:
    """Baseline OCR model using pretrained TrOCR"""
    
    def __init__(self, model_name: str = "microsoft/trocr-small-handwritten", 
                 device: str = None):
        """
        Initialize baseline model.
        
        Args:
            model_name: Hugging Face model identifier
            device: Device to use ('cuda' or 'cpu'). Auto-detected if None.
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        # Load processor and model
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("Model loaded successfully")
    
    def predict(self, image_path: str) -> str:
        """
        Predict text from image.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Predicted text
        """
        try:
            # Load and prepare image
            image = Image.open(image_path).convert("RGB")
            
            # Process image
            pixel_values = self.processor(image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)
            
            # Generate prediction
            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values)
            
            # Decode prediction
            predicted_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return predicted_text
        
        except Exception as e:
            logger.error(f"Error predicting on {image_path}: {e}")
            return ""
    
    def batch_predict(self, image_paths: List[str], batch_size: int = 8) -> List[str]:
        """
        Predict text for multiple images.
        
        Args:
            image_paths: List of image paths
            batch_size: Batch size for processing
        
        Returns:
            List of predictions
        """
        predictions = []
        
        logger.info(f"Running inference on {len(image_paths)} images...")
        
        with torch.no_grad():
            for i in tqdm(range(0, len(image_paths), batch_size), desc="Predicting"):
                batch_paths = image_paths[i:i+batch_size]
                
                # Load images
                images = []
                for path in batch_paths:
                    try:
                        img = Image.open(path).convert("RGB")
                        images.append(img)
                    except Exception as e:
                        logger.warning(f"Cannot load {path}: {e}")
                        images.append(None)
                
                # Process valid images
                valid_images = [img for img in images if img is not None]
                
                if not valid_images:
                    predictions.extend([""] * len(batch_paths))
                    continue
                
                # Batch process
                pixel_values = self.processor(valid_images, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(self.device)
                
                # Generate
                generated_ids = self.model.generate(pixel_values)
                batch_predictions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                
                # Map back to original batch (including None images)
                pred_idx = 0
                for img in images:
                    if img is not None:
                        predictions.append(batch_predictions[pred_idx])
                        pred_idx += 1
                    else:
                        predictions.append("")
        
        return predictions


class BaselineEvaluator:
    """Evaluate baseline model"""
    
    def __init__(self, model: BaselineOCRModel):
        """Initialize evaluator"""
        self.model = model
    
    def evaluate_on_split(self, data_path: str, split_name: str = "test") -> Dict[str, Any]:
        """
        Evaluate model on a dataset split.
        
        Args:
            data_path: Path to split JSON file
            split_name: Name of split for reporting
        
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating on {split_name} split...")
        
        # Load split
        with open(data_path, 'r', encoding='utf-8') as f:
            split_data = json.load(f)
        
        logger.info(f"Loaded {len(split_data)} samples from {split_name} split")
        
        # Extract image paths and references
        image_paths = [s["image_path"] for s in split_data]
        references = [s["text_normalized"] for s in split_data]
        
        # Generate predictions
        predictions = self.model.batch_predict(image_paths)
        
        # Calculate metrics
        metrics = MetricsCalculator.calculate_metrics(predictions, references)
        metrics["model"] = self.model.model_name
        metrics["split"] = split_name
        
        # Error analysis
        error_analysis = ErrorAnalyzer.analyze_predictions(predictions, references)
        metrics["error_analysis"] = error_analysis
        
        # Collect prediction examples
        examples = []
        for i, (ref, pred, sample) in enumerate(zip(references, predictions, split_data)):
            examples.append({
                "image_id": sample["image_id"],
                "reference": ref,
                "prediction": pred,
                "error_type": ErrorAnalyzer.categorize_error(ref, pred),
            })
        
        metrics["sample_predictions"] = examples
        
        return metrics
    
    def generate_report(self, metrics: Dict[str, Any]) -> str:
        """Generate readable evaluation report"""
        report = []
        report.append("=" * 80)
        report.append(f"BASELINE MODEL EVALUATION - {metrics.get('split', 'test').upper()}")
        report.append("=" * 80)
        report.append(f"Model: {metrics.get('model', 'unknown')}")
        report.append(f"Timestamp: {pd.Timestamp.now().isoformat()}")
        report.append("")
        
        report.append("PERFORMANCE METRICS")
        report.append("-" * 80)
        report.append(f"Character Error Rate (CER): {metrics.get('cer', 0):.2f}%")
        report.append(f"Word Error Rate (WER): {metrics.get('wer', 0):.2f}%")
        report.append(f"Samples evaluated: {metrics.get('num_samples', 0)}")
        report.append("")
        
        # Accuracy
        error_analysis = metrics.get("error_analysis", {})
        accuracy = error_analysis.get("accuracy", 0)
        report.append(f"Exact Match Accuracy: {accuracy:.2f}%")
        report.append("")
        
        # Error breakdown
        error_counts = error_analysis.get("error_counts", {})
        if error_counts:
            report.append("ERROR BREAKDOWN")
            report.append("-" * 80)
            total_errors = sum(v for k, v in error_counts.items() if k != "exact_match")
            
            for error_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
                pct = count / metrics.get('num_samples', 1) * 100 if count > 0 else 0
                report.append(f"{error_type.replace('_', ' ').title()}: {count} ({pct:.1f}%)")
            report.append("")
        
        # Sample predictions
        report.append("SAMPLE PREDICTIONS (First 10)")
        report.append("-" * 80)
        
        samples = metrics.get("sample_predictions", [])[:10]
        for i, sample in enumerate(samples, 1):
            report.append(f"\n{i}. Image: {sample.get('image_id', 'unknown')}")
            report.append(f"   Reference: '{sample.get('reference', '')}'")
            report.append(f"   Prediction: '{sample.get('prediction', '')}'")
            report.append(f"   Error Type: {sample.get('error_type', 'unknown')}")
        
        report.append("\n" + "=" * 80)
        report.append("NEXT STEPS:")
        report.append("1. Run 04_finetune.py to fine-tune model on training data")
        report.append("2. Evaluate fine-tuned model")
        report.append("3. Compare baseline vs fine-tuned performance")
        
        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline OCR model")
    parser.add_argument("--data_split", type=str, default="data/processed/test.json",
                       help="Path to test split JSON file")
    parser.add_argument("--model_name", type=str, default="microsoft/trocr-small-handwritten",
                       help="Pretrained model name")
    parser.add_argument("--output_metrics", type=str, default="results/baseline_metrics.json",
                       help="Output path for metrics")
    parser.add_argument("--output_predictions", type=str, default="results/baseline_predictions.json",
                       help="Output path for predictions")
    parser.add_argument("--output_report", type=str, default="results/baseline_report.txt",
                       help="Output path for report")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for inference")
    
    args = parser.parse_args()
    
    # Load model
    model = BaselineOCRModel(model_name=args.model_name)
    
    # Evaluate
    evaluator = BaselineEvaluator(model)
    metrics = evaluator.evaluate_on_split(args.data_split, split_name="test")
    
    # Generate report
    report = evaluator.generate_report(metrics)
    print(report)
    
    # Save results
    ResultsWriter.save_metrics(metrics, args.output_metrics)
    ResultsWriter.save_predictions(
        metrics.get("sample_predictions", []),
        args.output_predictions,
        model_name=args.model_name
    )
    
    # Save report
    os.makedirs(os.path.dirname(args.output_report), exist_ok=True)
    with open(args.output_report, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info("✓ Baseline evaluation complete!")


if __name__ == "__main__":
    main()

import pandas as pd
