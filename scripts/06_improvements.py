"""
Step 6: Improvements - Implement improvements over baseline

This script demonstrates iterative improvements:
- Better data filtering
- Hyperparameter tuning
- Model checkpoint selection
- Text normalization refinements
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any
import argparse

from utils import ResultsWriter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImprovementAnalyzer:
    """Analyze and compare improvements"""
    
    @staticmethod
    def load_metrics(path: str) -> Dict[str, Any]:
        """Load metrics from JSON"""
        with open(path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def compare_models(baseline_path: str, improved_path: str) -> Dict[str, Any]:
        """Compare baseline and improved model"""
        baseline = ImprovementAnalyzer.load_metrics(baseline_path)
        improved = ImprovementAnalyzer.load_metrics(improved_path)
        
        comparison = {
            "baseline": baseline,
            "improved": improved,
            "improvements": {
                "cer_reduction": baseline.get("cer", 0) - improved.get("cer", 0),
                "wer_reduction": baseline.get("wer", 0) - improved.get("wer", 0),
                "accuracy_gain": improved['error_analysis'].get('accuracy', 0) - 
                                baseline['error_analysis'].get('accuracy', 0),
            },
            "improvement_percentage": {
                "cer": ((baseline.get("cer", 1) - improved.get("cer", 0)) / 
                       baseline.get("cer", 1) * 100) if baseline.get("cer", 0) > 0 else 0,
                "wer": ((baseline.get("wer", 1) - improved.get("wer", 0)) / 
                       baseline.get("wer", 1) * 100) if baseline.get("wer", 0) > 0 else 0,
            }
        }
        
        return comparison
    
    @staticmethod
    def generate_improvement_report(comparison: Dict[str, Any]) -> str:
        """Generate comparison report"""
        report = []
        report.append("=" * 80)
        report.append("MODEL IMPROVEMENT ANALYSIS")
        report.append("=" * 80)
        report.append("")
        
        report.append("PERFORMANCE COMPARISON")
        report.append("-" * 80)
        
        baseline = comparison.get("baseline", {})
        improved = comparison.get("improved", {})
        improvements = comparison.get("improvements", {})
        imp_pct = comparison.get("improvement_percentage", {})
        
        report.append(f"{'Metric':<20} {'Baseline':<15} {'Improved':<15} {'Change':<15}")
        report.append("-" * 80)
        
        baseline_cer = baseline.get("cer", 0)
        improved_cer = improved.get("cer", 0)
        report.append(f"{'CER':<20} {baseline_cer:<15.2f}% {improved_cer:<15.2f}% " +
                     f"{improvements.get('cer_reduction', 0):<15.2f}%")
        
        baseline_wer = baseline.get("wer", 0)
        improved_wer = improved.get("wer", 0)
        report.append(f"{'WER':<20} {baseline_wer:<15.2f}% {improved_wer:<15.2f}% " +
                     f"{improvements.get('wer_reduction', 0):<15.2f}%")
        
        baseline_acc = baseline['error_analysis'].get('accuracy', 0)
        improved_acc = improved['error_analysis'].get('accuracy', 0)
        report.append(f"{'Accuracy':<20} {baseline_acc:<15.2f}% {improved_acc:<15.2f}% " +
                     f"{improvements.get('accuracy_gain', 0):<15.2f}%")
        
        report.append("")
        report.append("IMPROVEMENT SUMMARY")
        report.append("-" * 80)
        report.append(f"CER Improvement: {imp_pct.get('cer', 0):.1f}%")
        report.append(f"WER Improvement: {imp_pct.get('wer', 0):.1f}%")
        report.append(f"Accuracy Gain: {improvements.get('accuracy_gain', 0):.2f} percentage points")
        
        report.append("")
        report.append("ERROR DISTRIBUTION")
        report.append("-" * 80)
        
        baseline_errors = baseline['error_analysis'].get('error_counts', {})
        improved_errors = improved['error_analysis'].get('error_counts', {})
        
        report.append(f"{'Error Type':<25} {'Baseline':<15} {'Improved':<15}")
        report.append("-" * 80)
        
        for error_type in baseline_errors.keys():
            base_count = baseline_errors.get(error_type, 0)
            imp_count = improved_errors.get(error_type, 0)
            report.append(f"{error_type:<25} {base_count:<15} {imp_count:<15}")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_metrics", type=str, default="results/baseline_metrics.json")
    parser.add_argument("--improved_metrics", type=str, default="results/finetuned_metrics.json")
    parser.add_argument("--output_comparison", type=str, default="results/improvement_comparison.json")
    parser.add_argument("--output_report", type=str, default="results/improvement_report.txt")
    
    args = parser.parse_args()
    
    # Compare models
    comparison = ImprovementAnalyzer.compare_models(args.baseline_metrics, args.improved_metrics)
    
    # Generate report
    report = ImprovementAnalyzer.generate_improvement_report(comparison)
    print(report)
    
    # Save results
    ResultsWriter.save_comparison(
        comparison.get("baseline", {}),
        comparison.get("improved", {}),
        args.output_comparison
    )
    
    Path(args.output_report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_report, 'w') as f:
        f.write(report)
    
    logger.info("✓ Improvement analysis complete!")


if __name__ == "__main__":
    main()
