"""
EVALUATION SCRIPT: DINOv2 + BARTpho VQA (Answer-Only Model)
============================================================
Evaluate trained model with answer generation metrics only
(No reasoning - simplified architecture)
"""

import torch
import json
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from model_dinov2_bartpho_2 import DINOv2BARTphoVQA
from dataset import VQAGenDataset
import numpy as np
from collections import defaultdict
import re


def normalize_text(s):
    """Normalize Vietnamese text for comparison"""
    s = s.lower().strip()
    s = re.sub(r"[^0-9a-zA-ZÀ-ỹ\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def compute_f1(pred_tokens, gt_tokens):
    """Compute token-level F1 score"""
    common = set(pred_tokens) & set(gt_tokens)
    if len(common) == 0:
        return 0.0
    precision = len(common) / len(pred_tokens) if pred_tokens else 0
    recall = len(common) / len(gt_tokens) if gt_tokens else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def lcs_length(a, b):
    """Compute longest common subsequence length"""
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def compute_rouge(pred, gt):
    """Compute ROUGE-1 and ROUGE-L scores"""
    pred_tokens = pred.split()
    gt_tokens = gt.split()
    
    # ROUGE-1 (unigram overlap)
    rouge1 = compute_f1(pred_tokens, gt_tokens)
    
    # ROUGE-L (LCS-based)
    lcs = lcs_length(pred_tokens, gt_tokens)
    if lcs == 0:
        rougel = 0.0
    else:
        r_lcs = lcs / len(gt_tokens) if gt_tokens else 0
        p_lcs = lcs / len(pred_tokens) if pred_tokens else 0
        if r_lcs + p_lcs == 0:
            rougel = 0.0
        else:
            rougel = 2 * r_lcs * p_lcs / (r_lcs + p_lcs)
    
    return rouge1, rougel


def evaluate_model(
    model,
    test_dataset,
    batch_size=4,
    device='cuda',
    max_length=32,
    num_beams=4,
    repetition_penalty=2.0,
    length_penalty=0.8,
    no_repeat_ngram_size=3
):
    """
    Evaluate model with answer generation only
    
    Args:
        model: DINOv2BARTphoVQA model
        test_dataset: Dataset with (pixel_values, input_ids, attention_mask, labels)
        batch_size: Batch size for evaluation
        device: Device to run on
        max_length: Max answer length
        num_beams: Beam search width
        repetition_penalty: Penalty for repetition
        length_penalty: Length penalty
        no_repeat_ngram_size: N-gram size for no-repeat
        
    Returns:
        results: List of dicts with predictions
        stats: Dict with metric statistics
    """
    model.eval()
    model = model.to(device)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    results = []
    
    # Metrics tracking
    exact_matches = []
    rouge1_list = []
    rougel_list = []
    f1_list = []
    has_gt = False
    
    print("[INFO] Generating predictions...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            pixel_values = batch[0].to(device)
            input_ids = batch[1].to(device)
            attention_mask = batch[2].to(device)
            labels = batch[3].to(device) if len(batch) > 3 else None
            
            # Generate answers
            answer_texts = model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                early_stopping=True
            )
            
            # Process each sample in batch
            for i in range(len(answer_texts)):
                pred_answer = answer_texts[i]
                
                # Decode question
                question_ids = input_ids[i]
                question = model.tokenizer.decode(question_ids.cpu().tolist(), skip_special_tokens=True)
                
                result_row = {
                    'sample_id': batch_idx * batch_size + i,
                    'question': question,
                    'pred_answer': pred_answer
                }
                
                # Compute metrics if ground truth available
                if labels is not None:
                    has_gt = True
                    # Decode GT answer from labels
                    answer_label_ids = labels[i]
                    # Filter out -100 and pad tokens
                    answer_label_ids = answer_label_ids[answer_label_ids != -100]
                    answer_label_ids = answer_label_ids[answer_label_ids != model.config.pad_token_id]
                    gt_answer = model.tokenizer.decode(answer_label_ids.cpu().tolist(), skip_special_tokens=True)
                    result_row['gt_answer'] = gt_answer
                    
                    # Normalize
                    pred_norm = normalize_text(pred_answer)
                    gt_norm = normalize_text(gt_answer)
                    
                    # Exact Match
                    is_correct = (pred_norm == gt_norm)
                    exact_matches.append(1.0 if is_correct else 0.0)
                    result_row['exact_match'] = is_correct
                    
                    # Token F1
                    pred_tokens = pred_norm.split()
                    gt_tokens = gt_norm.split()
                    f1 = compute_f1(pred_tokens, gt_tokens)
                    f1_list.append(f1)
                    result_row['token_f1'] = f1
                    
                    # ROUGE
                    rouge1, rougel = compute_rouge(pred_norm, gt_norm)
                    rouge1_list.append(rouge1)
                    rougel_list.append(rougel)
                    result_row['rouge1'] = rouge1
                    result_row['rougel'] = rougel
                
                results.append(result_row)
    
    # Compute statistics
    stats = {'total_samples': len(results)}
    
    if has_gt:
        stats['exact_match_acc'] = np.mean(exact_matches) * 100.0
        stats['token_f1'] = np.mean(f1_list) * 100.0
        stats['rouge1'] = np.mean(rouge1_list) * 100.0
        stats['rougel'] = np.mean(rougel_list) * 100.0
    
    return results, stats


def main():
    import argparse
    import os
    import csv
    from transformers import AutoImageProcessor
    
    parser = argparse.ArgumentParser(description='Evaluate DINOv2 + BARTpho VQA (Answer-Only)')
    parser.add_argument('--mode', type=str, default='val', choices=['test', 'val'],
                       help='Evaluation mode: test (no GT) or val (with GT metrics)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.pth file)')
    parser.add_argument('--csv_path', type=str, default='/kaggle/input/vivqa/ViVQA-main/ViVQA-main/train.csv',
                       help='Path to CSV file (train.csv for val mode, test.csv for test mode)')
    parser.add_argument('--image_folder', type=str, default='/kaggle/input/vivqa/drive-download-20220309T020508Z-001/train',
                       help='Path to image folder')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split ratio (for val mode)')
    parser.add_argument('--output_csv', type=str, default='/kaggle/working/predictions.csv',
                       help='Output CSV path')
    parser.add_argument('--max_length', type=int, default=32, help='Max answer length')
    parser.add_argument('--num_beams', type=int, default=4, help='Beam search width')
    parser.add_argument('--repetition_penalty', type=float, default=2.0, help='Repetition penalty')
    parser.add_argument('--length_penalty', type=float, default=0.8, help='Length penalty')
    parser.add_argument('--no_repeat_ngram_size', type=int, default=3, help='No repeat n-gram size')
    args = parser.parse_args()
    
    print("="*80)
    print(f"EVALUATION: DINOv2 + BARTpho VQA - Answer Only ({args.mode.upper()} mode)")
    print("="*80)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[INFO] Device: {device}")
    
    # Load model
    print("\n[INFO] Loading model...")
    model = DINOv2BARTphoVQA(
        dinov2_model_name='facebook/dinov2-base',
        bartpho_model_name='vinai/bartpho-syllable',
        num_cross_attn_layers=3,
        num_heads=16,
        dropout=0.1,
        gradient_checkpointing=False  # Disable for inference
    )
    
    # Load checkpoint
    print(f"[INFO] Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    print("[INFO] ✓ Checkpoint loaded successfully")
    
    model = model.to(device)
    model.eval()
    
    # Load dataset
    print(f"\n[INFO] Loading dataset from {args.csv_path}")
    vision_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    
    if args.mode == 'val':
        # Val mode: use validation split from training data
        from torch.utils.data import random_split
        
        full_dataset = VQAGenDataset(
            csv_path=args.csv_path,
            image_folder=args.image_folder,
            vision_processor=vision_processor
        )
        
        # Split into train/val (same as training)
        val_size = int(len(full_dataset) * args.val_split)
        train_size = len(full_dataset) - val_size
        torch.manual_seed(42)  # Same seed as training
        _, test_dataset = random_split(full_dataset, [train_size, val_size])
        
        print(f"[INFO] Using validation split: {len(test_dataset)} samples")
    
    else:  # test mode
        # Test mode: load test.csv (no labels expected in dataset)
        test_dataset = VQAGenDataset(
            csv_path=args.csv_path,
            image_folder=args.image_folder,
            vision_processor=vision_processor
        )
        print(f"[INFO] Loaded {len(test_dataset)} test samples")
    
    # Evaluate
    print("\n[INFO] Starting evaluation...")
    results, stats = evaluate_model(
        model=model,
        test_dataset=test_dataset,
        batch_size=args.batch_size,
        device=device,
        max_length=args.max_length,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
        length_penalty=args.length_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size
    )
    
    # Save results to CSV
    print(f"\n[INFO] Saving results to {args.output_csv}")
    os.makedirs(os.path.dirname(args.output_csv) if os.path.dirname(args.output_csv) else '.', exist_ok=True)
    
    # Determine fieldnames based on available data
    fieldnames = ['sample_id', 'question', 'pred_answer']
    if results and 'gt_answer' in results[0]:
        fieldnames.extend(['gt_answer', 'exact_match', 'token_f1', 'rouge1', 'rougel'])
    
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    
    print(f"[INFO] ✓ Predictions saved to {args.output_csv}")
    
    # Print statistics
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"Total samples: {stats['total_samples']}")
    
    if 'exact_match_acc' in stats:
        print(f"\n📊 Answer Generation Metrics:")
        print(f"  • Exact Match Accuracy: {stats['exact_match_acc']:.2f}%")
        print(f"  • Token F1 Score:       {stats['token_f1']:.2f}%")
        print(f"  • ROUGE-1:              {stats['rouge1']:.2f}%")
        print(f"  • ROUGE-L:              {stats['rougel']:.2f}%")
    else:
        print("\n(No ground truth available - test mode)")
    
    print("="*80)
    
    # Save stats to JSON
    stats_path = args.output_csv.replace('.csv', '_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] ✓ Statistics saved to {stats_path}")


if __name__ == '__main__':
    main()
