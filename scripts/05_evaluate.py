"""
Step 5: Evaluation - Evaluate fine-tuned model with detailed error analysis
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
from tqdm import tqdm

from utils import MetricsCalculator, ErrorAnalyzer, ResultsWriter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate OCR model"""
    
    def __init__(self, model_path: str, device: str = None):
        """Load model"""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = TrOCRProcessor.from_pretrained(model_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
    
    def predict_batch(self, image_paths: List[str], batch_size: int = 8) -> List[str]:
        """Batch prediction"""
        predictions = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(image_paths), batch_size), desc="Predicting"):
                batch_paths = image_paths[i:i+batch_size]
                images = []
                
                for path in batch_paths:
                    try:
                        img = Image.open(path).convert("RGB")
                        images.append(img)
                    except:
                        images.append(None)
                
                valid_images = [img for img in images if img is not None]
                if not valid_images:
                    predictions.extend([""] * len(batch_paths))
                    continue
                
                pixel_values = self.processor(valid_images, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(self.device)
                generated_ids = self.model.generate(pixel_values)
                batch_preds = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                
                pred_idx = 0
                for img in images:
                    if img is not None:
                        predictions.append(batch_preds[pred_idx])
                        pred_idx += 1
                    else:
                        predictions.append("")
        
        return predictions
    
    def evaluate(self, data_path: str, model_name: str) -> Dict[str, Any]:
        """Full evaluation"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        image_paths = [s["image_path"] for s in data]
        references = [s["text_normalized"] for s in data]
        
        predictions = self.predict_batch(image_paths)
        
        # Calculate metrics
        metrics = MetricsCalculator.calculate_metrics(predictions, references)
        metrics["model"] = model_name
        metrics["num_samples"] = len(data)
        
        # Error analysis
        error_analysis = ErrorAnalyzer.analyze_predictions(predictions, references)
        metrics["error_analysis"] = error_analysis
        
        # Collect examples with different error types
        examples = []
        error_by_type = {}
        
        for ref, pred, sample in zip(references, predictions, data):
            error_type = ErrorAnalyzer.categorize_error(ref, pred)
            if error_type not in error_by_type:
                error_by_type[error_type] = []
            
            if len(error_by_type[error_type]) < 3:
                examples.append({
                    "image_id": sample["image_id"],
                    "reference": ref,
                    "prediction": pred,
                    "error_type": error_type,
                })
                error_by_type[error_type].append(True)
        
        metrics["sample_predictions"] = examples
        return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/finetuned-trocr")
    parser.add_argument("--test_data", type=str, default="data/processed/test.json")
    parser.add_argument("--output_metrics", type=str, default="results/finetuned_metrics.json")
    parser.add_argument("--model_name", type=str, default="finetuned-trocr")
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(args.model_path)
    metrics = evaluator.evaluate(args.test_data, args.model_name)
    
    # Print summary
    print("\n" + "=" * 80)
    print(f"MODEL EVALUATION - {args.model_name.upper()}")
    print("=" * 80)
    print(f"CER: {metrics.get('cer', 0):.2f}%")
    print(f"WER: {metrics.get('wer', 0):.2f}%")
    print(f"Accuracy: {metrics['error_analysis'].get('accuracy', 0):.2f}%")
    print("\nError Breakdown:")
    for error_type, count in metrics['error_analysis'].get('error_counts', {}).items():
        pct = count / metrics['num_samples'] * 100 if count > 0 else 0
        print(f"  {error_type}: {count} ({pct:.1f}%)")
    
    # Save results
    ResultsWriter.save_metrics(metrics, args.output_metrics)
    logger.info("✓ Evaluation complete!")


if __name__ == "__main__":
    main()
