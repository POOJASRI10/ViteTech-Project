"""
Step 5: Evaluate Fine-tuned Model - Generate predictions with fine-tuned model
"""

import json
import logging
from pathlib import Path

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from tqdm import tqdm

from utils import MetricsCalculator, ErrorAnalyzer, ResultsWriter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_finetuned_model(model_path: str, test_data_path: str, output_dir: str):
    """Evaluate fine-tuned model"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load fine-tuned model
    logger.info(f"Loading model from {model_path}")
    processor = TrOCRProcessor.from_pretrained(model_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path).to(device)
    model.eval()
    
    # Load test data
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    
    logger.info(f"Loaded {len(test_data)} test samples")
    
    predictions = []
    all_references = []
    all_predictions = []
    
    with torch.no_grad():
        with tqdm(test_data, desc="Evaluating") as pbar:
            for sample in pbar:
                try:
                    image_path = sample["image_path"]
                    image = Image.open(image_path).convert("RGB")
                    reference_text = sample.get("text_normalized", "")
                    
                    # Inference
                    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
                    generated_ids = model.generate(pixel_values, max_length=128)
                    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    
                    all_references.append(reference_text)
                    all_predictions.append(generated_text)
                    
                    predictions.append({
                        "image_id": Path(image_path).stem,
                        "reference": reference_text,
                        "prediction": generated_text
                    })
                    
                    pbar.set_postfix({"pred": generated_text[:30]})
                
                except Exception as e:
                    logger.debug(f"Error: {e}")
                    continue
    
    # Calculate metrics
    metrics = MetricsCalculator.calculate_metrics(all_references, all_predictions)
    error_analysis = ErrorAnalyzer.analyze_predictions(all_predictions, all_references)
    
    # Prepare results
    results = {
        "model": "microsoft/trocr-small-handwritten (fine-tuned)",
        "num_samples": len(predictions),
        **metrics,
        "error_analysis": error_analysis,
        "sample_predictions": predictions[:10]
    }
    
    # Save results
    metrics_path = f"{output_dir}/finetuned_metrics.json"
    predictions_path = f"{output_dir}/finetuned_predictions.json"
    
    ResultsWriter.save_metrics(results, metrics_path)
    ResultsWriter.save_predictions(predictions, predictions_path, "fine-tuned-trocr")
    
    logger.info(f"✓ Metrics saved to {metrics_path}")
    logger.info(f"✓ Predictions saved to {predictions_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("FINE-TUNED MODEL EVALUATION")
    print("="*80)
    print(f"Character Error Rate (CER): {metrics.get('cer', 0):.2f}%")
    print(f"Word Error Rate (WER): {metrics.get('wer', 0):.2f}%")
    print(f"Samples Evaluated: {len(predictions)}")
    print("="*80)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/finetuned_trocr")
    parser.add_argument("--test_data", type=str, default="data/processed/demo/test.json")
    parser.add_argument("--output_dir", type=str, default="results")
    
    args = parser.parse_args()
    
    evaluate_finetuned_model(args.model_path, args.test_data, args.output_dir)
