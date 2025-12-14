"""
Evaluate GPT-4o Predictions (Offline)

Reads predictions from JSONL file and computes metrics.
No API calls needed - can run offline.

Usage:
    python eval_gpt4o_results.py
"""

import os
import json
import pandas as pd
import numpy as np
import re
import unicodedata
from rouge_score import rouge_scorer

# ======================
# CONFIG
# ======================
PREDICTIONS_JSONL = "/kaggle/working/gpt4o_predictions.jsonl"
OUTPUT_CSV = "/kaggle/working/eval_gpt4o_results.csv"
OUTPUT_SUMMARY = "/kaggle/working/eval_gpt4o_summary.txt"

# ======================
# TEXT NORMALIZATION
# ======================
def normalize_text(s):
    """Same normalization as eval script"""
    if s is None or not s:
        return ""
    s = s.lower().strip()
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ======================
# PARSER
# ======================
def parse_gpt4o_response(text: str):
    """Parse GPT-4o output"""
    answer = ""
    reasoning = ""
    
    # Method 1: "Answer: X\nReasoning: Y"
    answer_match = re.search(r'Answer:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    reasoning_match = re.search(r'Reasoning:\s*(.+?)$', text, re.IGNORECASE | re.DOTALL)
    
    if answer_match:
        answer = answer_match.group(1).strip()
        answer = re.split(r'\s*Reasoning:', answer, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    
    # Method 2: Line-based
    if not answer or not reasoning:
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        for line in lines:
            lower_line = line.lower()
            if lower_line.startswith('answer:'):
                answer = line.split(':', 1)[1].strip()
            elif lower_line.startswith('reasoning:'):
                reasoning = line.split(':', 1)[1].strip()
    
    # Method 3: First line as answer
    if not answer and text:
        lines = text.strip().split('\n')
        answer = lines[0].strip()
        if len(lines) > 1:
            reasoning = '\n'.join(lines[1:]).strip()
    
    return {
        'answer': answer,
        'reasoning': reasoning,
        'valid_format': bool(answer),
        'raw': text
    }

# ======================
# TOKEN F1
# ======================
def token_f1(prediction, ground_truth):
    """Token-level F1 score"""
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()

    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0

    common = set(pred_tokens) & set(gt_tokens)
    if len(common) == 0:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)

# ======================
# MAIN EVALUATION
# ======================
def main():
    print("="*70)
    print("EVALUATE GPT-4o PREDICTIONS (OFFLINE)")
    print("="*70)
    
    # Check input file
    if not os.path.exists(PREDICTIONS_JSONL):
        print(f"[ERROR] Predictions file not found: {PREDICTIONS_JSONL}")
        print(f"[INFO] Run generate_gpt4o_predictions.py first!")
        return
    
    # Load predictions
    print(f"[INFO] Loading predictions from: {PREDICTIONS_JSONL}")
    predictions = []
    with open(PREDICTIONS_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            predictions.append(json.loads(line))
    
    print(f"[INFO] Loaded {len(predictions)} predictions\n")
    
    # Parse and evaluate
    print("[INFO] Parsing and computing metrics...")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeLsum"], use_stemmer=True)
    
    refs, hyps = [], []
    records = []
    format_valid_count = 0
    error_count = 0
    
    rouge1_list, rougel_list, f1_list = [], [], []
    exact_matches = []
    
    for pred in predictions:
        # Parse
        parsed = parse_gpt4o_response(pred['raw_output'])
        
        # Track
        if parsed['valid_format']:
            format_valid_count += 1
        if pred.get('error'):
            error_count += 1
        
        # Ground truth
        gt_answer = pred['ground_truth']
        pred_answer = parsed['answer']
        
        refs.append(gt_answer)
        hyps.append(pred_answer)
        
        # Compute metrics
        ref_n = normalize_text(gt_answer)
        hyp_n = normalize_text(pred_answer)
        
        # ROUGE
        if hyp_n:
            scores = scorer.score(ref_n, hyp_n)
            rouge1_list.append(scores["rouge1"].fmeasure)
            rougel_list.append(scores["rougeLsum"].fmeasure)
        else:
            rouge1_list.append(0.0)
            rougel_list.append(0.0)
        
        # Token F1
        f1 = token_f1(pred_answer, gt_answer)
        f1_list.append(f1)
        
        # Exact match
        em = int(ref_n == hyp_n)
        exact_matches.append(em)
        
        # Record
        records.append({
            "img_id": pred['img_id'],
            "question": pred['question'],
            "ground_truth": gt_answer,
            "predicted_raw": pred['raw_output'],
            "predicted_answer": pred_answer,
            "predicted_reasoning": parsed['reasoning'],
            "valid_format": parsed['valid_format'],
            "exact_match": em,
            "token_f1": f1,
            "rouge1": rouge1_list[-1],
            "rougel": rougel_list[-1],
            "error": pred.get('error', ''),
            "tokens_used": pred.get('total_tokens', 0)
        })
    
    # Aggregate metrics
    avg_rouge1 = np.mean(rouge1_list)
    avg_rougel = np.mean(rougel_list)
    avg_f1 = np.mean(f1_list)
    acc = np.mean(exact_matches)
    
    # ======================
    # PRINT RESULTS
    # ======================
    summary = []
    summary.append("="*70)
    summary.append("GPT-4o EVALUATION RESULTS")
    summary.append("="*70)
    summary.append(f"Total Samples:       {len(predictions)}")
    summary.append(f"Valid Format:        {format_valid_count}/{len(predictions)} ({100*format_valid_count/len(predictions):.2f}%)")
    summary.append(f"Errors:              {error_count}")
    summary.append(f"-" * 70)
    summary.append(f"Accuracy (EM):       {acc*100:.2f}%")
    summary.append(f"Token F1:            {avg_f1:.4f}")
    summary.append(f"ROUGE-1 F1:          {avg_rouge1:.4f}")
    summary.append(f"ROUGE-L F1:          {avg_rougel:.4f}")
    summary.append("="*70)
    
    # Print to console
    for line in summary:
        print(line)
    
    # Save summary
    with open(OUTPUT_SUMMARY, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary))
    print(f"\n[INFO] Summary saved to: {OUTPUT_SUMMARY}")
    
    # ======================
    # SAVE DETAILED CSV
    # ======================
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[INFO] Detailed results saved to: {OUTPUT_CSV}")
    
    # ======================
    # SAMPLE PREDICTIONS
    # ======================
    print("\n" + "="*70)
    print("SAMPLE PREDICTIONS")
    print("="*70)
    
    # Correct samples
    correct_samples = [r for r in records if r['exact_match'] == 1]
    if correct_samples:
        print(f"\n✅ CORRECT PREDICTIONS ({len(correct_samples)} total):")
        for i, r in enumerate(correct_samples[:5], 1):
            print(f"\n{i}. ID: {r['img_id']}")
            print(f"   Q: {r['question']}")
            print(f"   GT: {r['ground_truth']}")
            print(f"   PRED: {r['predicted_answer']}")
    
    # Incorrect samples
    incorrect_samples = [r for r in records if r['exact_match'] == 0]
    if incorrect_samples:
        print(f"\n❌ INCORRECT PREDICTIONS ({len(incorrect_samples)} total):")
        for i, r in enumerate(incorrect_samples[:5], 1):
            print(f"\n{i}. ID: {r['img_id']}")
            print(f"   Q: {r['question']}")
            print(f"   GT: {r['ground_truth']}")
            print(f"   PRED: {r['predicted_answer']}")
            print(f"   Token F1: {r['token_f1']:.3f}")
    
    # Error samples
    error_samples = [r for r in records if r['error']]
    if error_samples:
        print(f"\n⚠️ ERROR SAMPLES ({len(error_samples)} total):")
        for i, r in enumerate(error_samples[:3], 1):
            print(f"\n{i}. ID: {r['img_id']}")
            print(f"   Q: {r['question']}")
            print(f"   ERROR: {r['error']}")
    
    # Token usage
    total_tokens = sum(r['tokens_used'] for r in records)
    print(f"\n" + "="*70)
    print(f"TOTAL TOKEN USAGE: {total_tokens:,}")
    print("="*70)

if __name__ == "__main__":
    main()
