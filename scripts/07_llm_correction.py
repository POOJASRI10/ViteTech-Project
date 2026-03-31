"""
Step 7: LLM Post-Processing - Use LLM to improve OCR output

This script demonstrates using an LLM to:
- Correct common OCR errors
- Normalize text
- Extract structured output
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMCorrector:
    """Use LLM to post-process OCR output"""
    
    def __init__(self, api_key: str = None):
        """Initialize LLM corrector"""
        self.api_key = api_key
        
        # Try to import OpenAI
        try:
            import openai
            self.openai = openai
            if api_key:
                self.openai.api_key = api_key
            self.has_openai = True
        except ImportError:
            self.has_openai = False
            logger.warning("OpenAI not installed. Using mock corrections.")
    
    def correct_ocr_output(self, ocr_text: str, use_mock: bool = True) -> str:
        """
        Correct OCR output using LLM.
        
        Args:
            ocr_text: Raw OCR output
            use_mock: Use mock correction (no API key needed)
        
        Returns:
            Corrected text
        """
        if use_mock or not self.has_openai:
            return self._mock_correct(ocr_text)
        
        return self._openai_correct(ocr_text)
    
    def _mock_correct(self, text: str) -> str:
        """Mock correction with heuristic rules"""
        corrected = text
        
        # Common OCR mistakes
        corrections = {
            "rn": "m",  # "m" often read as "rn"
            "l1": "ll",  # "1" mistaken for "l"
            "O0": "00",  # "0" mistaken for "O"
            "ii": "u",   # "u" mistaken as "ii"
        }
        
        # Apply corrections carefully (not in all contexts)
        # This is simplified - real LLM would understand context
        
        # Simple rule: fix common substitutions
        if "tne" in corrected:
            corrected = corrected.replace("tne", "the")
        if "tnis" in corrected:
            corrected = corrected.replace("tnis", "this")
        if "adn" in corrected:
            corrected = corrected.replace("adn", "and")
        
        return corrected
    
    def _openai_correct(self, text: str) -> str:
        """Correct using OpenAI API"""
        try:
            response = self.openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an OCR correction system. " +
                                 "Fix OCR errors but preserve the original meaning. " +
                                 "Return only the corrected text."
                    },
                    {
                        "role": "user",
                        "content": f"Fix OCR errors in this text:\n{text}"
                    }
                ],
                temperature=0.2,
            )
            
            corrected = response.choices[0].message.content.strip()
            return corrected
        
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return self._mock_correct(text)
    
    def batch_correct(self, ocr_texts: List[str], use_mock: bool = True) -> List[str]:
        """Correct multiple texts"""
        return [self.correct_ocr_output(text, use_mock) for text in ocr_texts]


class PostProcessingDemo:
    """Demonstrate LLM post-processing"""
    
    @staticmethod
    def load_predictions(predictions_path: str) -> List[str]:
        """Load predictions"""
        with open(predictions_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both formats
        if isinstance(data, dict):
            predictions = [p.get("prediction", "") for p in data.get("predictions", [])]
        else:
            predictions = [p.get("prediction", "") for p in data]
        
        return predictions
    
    @staticmethod
    def generate_demo_report(original: List[str], corrected: List[str]) -> str:
        """Generate demo report"""
        report = []
        report.append("=" * 80)
        report.append("LLM POST-PROCESSING DEMONSTRATION")
        report.append("=" * 80)
        report.append("")
        
        report.append("APPROACH")
        report.append("-" * 80)
        report.append("1. Raw OCR output often contains errors")
        report.append("2. LLM understands context and common patterns")
        report.append("3. Can fix specific errors while preserving meaning")
        report.append("4. Can normalize formatting and structure")
        report.append("")
        
        report.append("EXAMPLE CORRECTIONS (First 10 samples)")
        report.append("-" * 80)
        
        improvements = 0
        for i, (orig, corr) in enumerate(zip(original[:10], corrected[:10])):
            if orig != corr:
                improvements += 1
                report.append(f"\n{i+1}. Original: '{orig}'")
                report.append(f"   Corrected: '{corr}'")
        
        report.append("")
        report.append(f"Samples improved: {improvements}/{min(10, len(original))}")
        
        report.append("")
        report.append("USE CASES FOR LLM POST-PROCESSING")
        report.append("-" * 80)
        report.append("✓ Correcting common OCR mistakes")
        report.append("✓ Normalizing text format")
        report.append("✓ Extracting structured data from OCR output")
        report.append("✓ Handling context-specific corrections")
        report.append("✓ Grammar and spell checking")
        report.append("")
        
        report.append("CONSIDERATIONS")
        report.append("-" * 80)
        report.append("- Adds latency (API calls slower than model inference)")
        report.append("- Requires API key and incurs costs")
        report.append("- Should be used selectively (high-confidence errors)")
        report.append("- Works best with domain-specific prompts")
        report.append("- Can hallucinate incorrect corrections")
        report.append("")
        
        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_path", type=str, default="results/finetuned_predictions.json")
    parser.add_argument("--output_corrected", type=str, default="results/llm_corrected.json")
    parser.add_argument("--output_report", type=str, default="results/llm_postprocessing_report.txt")
    parser.add_argument("--use_mock", action="store_true", default=True,
                       help="Use mock LLM (no API key needed)")
    parser.add_argument("--api_key", type=str, default=None,
                       help="OpenAI API key (optional)")
    
    args = parser.parse_args()
    
    # Initialize corrector
    corrector = LLMCorrector(api_key=args.api_key)
    
    # Load predictions
    try:
        original_predictions = PostProcessingDemo.load_predictions(args.predictions_path)
    except FileNotFoundError:
        logger.warning(f"Predictions file not found: {args.predictions_path}")
        # Use demo data
        original_predictions = [
            "tne quick brown fox",
            "tnis is a test",
            "handwriting recognitio",
            "ocr systern",
            "example text",
        ]
    
    # Correct
    logger.info(f"Correcting {len(original_predictions)} predictions...")
    corrected_predictions = corrector.batch_correct(original_predictions, use_mock=True)
    
    # Generate report
    report = PostProcessingDemo.generate_demo_report(original_predictions, corrected_predictions)
    print(report)
    
    # Save corrected predictions
    output_data = {
        "original": original_predictions,
        "corrected": corrected_predictions,
        "method": "LLM (mock)",
    }
    
    Path(args.output_corrected).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_corrected, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # Save report
    Path(args.output_report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_report, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info("✓ LLM post-processing complete!")


if __name__ == "__main__":
    main()
