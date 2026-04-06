#!/usr/bin/env python
"""Document text normalization strategy used in the project"""

print("=" * 70)
print("TEXT NORMALIZATION STRATEGY")
print("=" * 70)

print("""
IMPORTANT: For the Kaggle IAM dataset, we took a MINIMAL normalization approach.

NORMALIZATION RULES APPLIED:
============================

1. WHITESPACE NORMALIZATION
   - Strip leading/trailing spaces
   - Normalize internal spaces: "word1  word2" → "word1 word2"
   - Ensures: Single space between words, no extra whitespace
   - Why: Handles OCR output inconsistencies

2. LOWERCASE CONVERSION
   - Convert all text to lowercase
   - "The Quick Brown Fox" → "the quick brown fox"
   - Why: Simplifies character recognition, reduces complexity
   - Applied in evaluation metrics (CER/WER calculation)

3. PUNCTUATION: PRESERVED
   - Keep as-is: commas, periods, exclamation marks, etc.
   - "Hello, world!" → "hello, world!"
   - Why: Real text contains punctuation; removing loses information

WHAT WE DID NOT NORMALIZE:
===========================

1. CASE VARIATION
   - Original text: Mixed case, uppercase, lowercase preserved
   - Not forced to single case during training
   - Why: Real handwritten text naturally varies; improves generalization

2. PUNCTUATION INCONSISTENCY
   - Some sentences have punctuation, some don't
   - NOT normalized to consistent presence/absence
   - Why: Natural variation improves robustness

3. SPECIAL CHARACTERS
   - Unicode characters, diacritics preserved
   - Not replaced or removed
   - Why: Some writers use special characters; should be handled

4. SPELLING VARIATIONS
   - No spell-checking or correction during training
   - Original handwritten variations kept
   - Why: Model should learn to recognize natural writing patterns

CODE LOCATION:
==============
- TextNormalizer class: scripts/utils.py (lines 24-48)
- Applied in: scripts/02_data_prep.py
- Metric calculation: scripts/03_baseline.py, scripts/05_evaluate.py

RATIONALE:
==========
For OCR tasks on real handwriting:
- Less normalization = More natural training data
- Natural variation = Better generalization to unseen writing
- Preserve realistic patterns = More robust final model

In real-world OCR, you want the model to handle natural text variation,
not idealized, heavily-normalized text.

COMPARISON WITH OTHER APPROACHES:
==================================
Heavy Normalization:
  - Lowercase, remove punctuation, remove special chars
  - Result: Simpler model, less versatile
  - Problem: Can't handle real-world text variation

Our Approach (Light Normalization):
  - Only whitespace and baseline lowercase for metrics
  - Result: More robust, handles real text
  - Benefit: Natural training data improves generalization

VERIFICATION:
==============
✓ No text filtering applied during training
✓ All 4,899 samples used as-is (no cleanup)
✓ 1 duplicate image kept (real-world edge case)
✓ 30 tiny images kept (test robustness to low resolution)
✓ Natural case/punctuation variation preserved

METRICS CALCULATION:
====================
For CER/WER calculation in evaluation:
- Text converted to lowercase for fair comparison
- Whitespace normalized
- Punctuation treated as characters
- This ensures: Model gets credit for recognizing actual content
""")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
TEXT NORMALIZATION: MINIMAL & INTENTIONAL

What: Whitespace standardization + lowercase for metrics
Why: Preserve natural handwriting variation for robust training
Result: Real-world data → Real-world model performance
""")
