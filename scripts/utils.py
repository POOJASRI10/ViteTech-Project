"""
Utility functions for OCR system
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

import numpy as np
import pandas as pd
from jiwer import cer, wer
from editdistance import eval as edit_distance


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextNormalizer:
    """Handles text normalization with consistent rules"""
    
    @staticmethod
    def normalize(text: str, lowercase: bool = True, keep_punctuation: bool = True) -> str:
        """
        Normalize text according to defined rules.
        
        Args:
            text: Input text
            lowercase: Convert to lowercase (default: True)
            keep_punctuation: Keep punctuation marks (default: True)
        
        Returns:
            Normalized text
        """
        if not isinstance(text, str):
            return ""
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        # Normalize whitespace (single space between words)
        text = " ".join(text.split())
        
        # Lowercase
        if lowercase:
            text = text.lower()
        
        return text
    
    @staticmethod
    def rules_summary() -> str:
        """Return string describing normalization rules"""
        return """
Normalization Rules Applied:
- Lowercase: All text converted to lowercase
- Whitespace: Normalized to single spaces between words
- Punctuation: Preserved as-is
- Special characters: Unicode preserved
- Leading/trailing: Trimmed
        """


class MetricsCalculator:
    """Calculate OCR evaluation metrics"""
    
    @staticmethod
    def calculate_cer(predictions: List[str], references: List[str]) -> float:
        """Calculate Character Error Rate"""
        try:
            return cer(references, predictions) * 100
        except Exception as e:
            logger.error(f"Error calculating CER: {e}")
            return 0.0
    
    @staticmethod
    def calculate_wer(predictions: List[str], references: List[str]) -> float:
        """Calculate Word Error Rate"""
        try:
            return wer(references, predictions) * 100
        except Exception as e:
            logger.error(f"Error calculating WER: {e}")
            return 0.0
    
    @staticmethod
    def calculate_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate all metrics at once"""
        return {
            "cer": MetricsCalculator.calculate_cer(predictions, references),
            "wer": MetricsCalculator.calculate_wer(predictions, references),
            "num_samples": len(predictions),
        }


class ErrorAnalyzer:
    """Analyze and categorize OCR errors"""
    
    @staticmethod
    def get_edit_operations(reference: str, hypothesis: str) -> List[Tuple[str, int, str, str]]:
        """
        Get detailed edit operations between reference and hypothesis.
        
        Returns list of (operation_type, position, expected, got)
        """
        # Simple implementation using edit distance
        # For production, use more sophisticated alignment
        operations = []
        
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        
        # Word-level alignment
        for i, (ref_word, hyp_word) in enumerate(zip(ref_words, hyp_words)):
            if ref_word != hyp_word:
                if len(hyp_word) < len(ref_word):
                    operations.append(("deletion", i, ref_word, hyp_word))
                elif len(hyp_word) > len(ref_word):
                    operations.append(("insertion", i, ref_word, hyp_word))
                else:
                    operations.append(("substitution", i, ref_word, hyp_word))
        
        if len(ref_words) > len(hyp_words):
            operations.append(("deletion", len(hyp_words), 
                             " ".join(ref_words[len(hyp_words):]), ""))
        elif len(hyp_words) > len(ref_words):
            operations.append(("insertion", len(ref_words), "", 
                             " ".join(hyp_words[len(ref_words):])))
        
        return operations
    
    @staticmethod
    def categorize_error(reference: str, hypothesis: str) -> str:
        """
        Categorize error type.
        
        Categories:
        - exact_match: Perfect prediction
        - merged_words: Multiple words read as one
        - split_words: One word read as two
        - missing_characters: Characters omitted
        - extra_characters: Extra characters added
        - substitution: Character/word substitution
        - case_error: Case mismatch only
        - other: Other errors
        """
        if reference == hypothesis:
            return "exact_match"
        
        ref_lower = reference.lower()
        hyp_lower = hypothesis.lower()
        
        # Check if only case differs
        if ref_lower == hyp_lower:
            return "case_error"
        
        ref_words = ref_lower.split()
        hyp_words = hyp_lower.split()
        
        # Check for merged words
        if len(hyp_words) < len(ref_words):
            return "merged_words"
        
        # Check for split words
        if len(hyp_words) > len(ref_words):
            return "split_words"
        
        # Check for missing/extra characters
        ref_len = len(ref_lower)
        hyp_len = len(hyp_lower)
        
        if hyp_len < ref_len:
            return "missing_characters"
        elif hyp_len > ref_len:
            return "extra_characters"
        
        return "substitution"
    
    @staticmethod
    def analyze_predictions(predictions: List[str], references: List[str]) -> Dict[str, Any]:
        """
        Comprehensive error analysis.
        
        Returns:
            Dictionary with error statistics and categorization
        """
        error_counts = {
            "exact_match": 0,
            "merged_words": 0,
            "split_words": 0,
            "missing_characters": 0,
            "extra_characters": 0,
            "substitution": 0,
            "case_error": 0,
            "other": 0,
        }
        
        error_examples = {k: [] for k in error_counts.keys()}
        
        for pred, ref in zip(predictions, references):
            category = ErrorAnalyzer.categorize_error(ref, pred)
            error_counts[category] += 1
            
            if category != "exact_match" and len(error_examples[category]) < 5:
                error_examples[category].append({
                    "reference": ref,
                    "prediction": pred,
                })
        
        total = len(predictions)
        accuracy = error_counts["exact_match"] / total * 100 if total > 0 else 0
        
        return {
            "total_samples": total,
            "accuracy": accuracy,
            "error_counts": error_counts,
            "error_examples": error_examples,
        }


class ResultsWriter:
    """Handle saving results to disk"""
    
    @staticmethod
    def save_predictions(predictions: List[Dict[str, str]], 
                        output_path: str, model_name: str):
        """Save predictions to JSON file"""
        output = {
            "model": model_name,
            "timestamp": pd.Timestamp.now().isoformat(),
            "num_predictions": len(predictions),
            "predictions": predictions,
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved predictions to {output_path}")
    
    @staticmethod
    def save_metrics(metrics: Dict[str, Any], output_path: str):
        """Save metrics to JSON file"""
        metrics['timestamp'] = pd.Timestamp.now().isoformat()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Saved metrics to {output_path}")
    
    @staticmethod
    def save_comparison(baseline_metrics: Dict, improved_metrics: Dict, 
                       output_path: str):
        """Save comparison between baseline and improved model"""
        comparison = {
            "baseline": baseline_metrics,
            "improved": improved_metrics,
            "differences": {
                "cer_improvement": baseline_metrics.get("cer", 0) - improved_metrics.get("cer", 0),
                "wer_improvement": baseline_metrics.get("wer", 0) - improved_metrics.get("wer", 0),
            },
            "timestamp": pd.Timestamp.now().isoformat(),
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2)
        
        logger.info(f"Saved comparison to {output_path}")


class DataLogger:
    """Logging utilities for data operations"""
    
    @staticmethod
    def log_data_audit_summary(audit_results: Dict[str, Any], output_path: str):
        """Save data audit results"""
        summary = {
            "timestamp": pd.Timestamp.now().isoformat(),
            **audit_results,
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Data audit saved to {output_path}")


if __name__ == "__main__":
    # Test utilities
    print(TextNormalizer.rules_summary())
    
    # Test metrics
    refs = ["hello world", "the quick brown fox"]
    hyps = ["hello word", "the quick brown fax"]
    
    metrics = MetricsCalculator.calculate_metrics(hyps, refs)
    print(f"Metrics: {metrics}")
    
    # Test error analysis
    analysis = ErrorAnalyzer.analyze_predictions(hyps, refs)
    print(f"Analysis: {analysis}")
