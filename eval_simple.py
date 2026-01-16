"""
EVALUATION SCRIPT: SimpleFusionVQA
===================================
Evaluate SimpleFusionVQA model với:
✅ Simple fusion architecture (DINOv2 + BARTpho)
✅ Generate answers from fused representations
✅ Metrics: Accuracy, F1, ROUGE-1, ROUGE-L
✅ Evaluate answer quality

Usage:
    # Validation mode (evaluate on train.csv validation split)
    python eval_simple.py --mode val --checkpoint checkpoints/simple_best.pt --batch_size 8
    
    # Test mode (evaluate on test.csv with metrics)
    python eval_simple.py --mode test --checkpoint checkpoints/simple_best.pt --batch_size 8
    
    # Custom paths
    python eval_simple.py --mode test --checkpoint best.pt \
        --csv_path data/test.csv --image_dir data/test_images

Data Augmentation:
    Errors are saved to errors_simple.csv for:
    - LLM-based question refinement
    - Synthetic data generation
    - Hard negative mining
    Note: This is data augmentation, not knowledge distillation.
    Knowledge distillation = training small model from large model's outputs.
"""

import torch
import json
import csv
import pandas as pd
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from model import SimpleFusionVQA
from transformers import AutoImageProcessor
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
    
    # ROUGE-1 (unigram F1)
    rouge1 = compute_f1(pred_tokens, gt_tokens)
    
    # ROUGE-L (LCS-based F1)
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


def evaluate_simple_vqa(
    model,
    test_dataset,
    batch_size=4,
    device='cuda',
    max_answer_len=32,
    num_beams=4
):
    """
    Evaluate SimpleFusionVQA model
    
    Args:
        model: SimpleFusionVQA model
        test_dataset: VQA dataset with GT
        batch_size: Batch size for evaluation
        device: Device to use
        max_answer_len: Max length for answer generation
        num_beams: Number of beams for beam search
    
    Returns:
        results: List of prediction dicts
        stats: Dict of evaluation metrics
    """
    model.eval()
    model = model.to(device)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    results = []
    
    # Metrics tracking
    answer_exact_matches = []
    answer_rouge1_list = []
    answer_rougel_list = []
    answer_f1_list = []
    
    # Type-based error tracking
    type_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'errors': []})
    
    # Track all errors for CSV export (for data augmentation)
    all_errors = []
    
    has_gt = False
    
    print("[INFO] Generating predictions with SimpleFusionVQA...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            # Generate answers using SimpleFusionVQA.generate()
            answer_text = model.generate(
                pixel_values=batch['pixel_values'],
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_length=max_answer_len,
                num_beams=num_beams
            )
            
            # Check if we have ground truth
            gt_answer_labels = batch.get('labels', None)
            
            for i in range(len(answer_text)):
                pred_answer = answer_text[i]
                
                # Get metadata
                if 'img_id' in batch:
                    img_id = batch['img_id'][i] if isinstance(batch['img_id'], list) else batch['img_id']
                else:
                    img_id = f"sample_{i}"
                
                # Get question text
                if 'question' in batch:
                    question = batch['question'][i] if isinstance(batch['question'], list) else batch['question']
                else:
                    question_ids = batch['input_ids'][i] if batch['input_ids'].dim() > 1 else batch['input_ids']
                    question = model.tokenizer.decode(question_ids.cpu().tolist(), skip_special_tokens=True)
                
                # Get type if available
                if 'type' in batch:
                    q_type = batch['type'][i] if isinstance(batch['type'], list) else batch['type']
                    q_type = str(q_type)  # Convert to string for consistency
                else:
                    q_type = 'unknown'
                
                result_row = {
                    'img_id': str(img_id),
                    'question': question,
                    'pred_answer': pred_answer,
                    'type': q_type
                }
                
                # ===== EVALUATE ANSWER =====
                if gt_answer_labels is not None:
                    has_gt = True
                    # Decode GT answer from labels tensor
                    answer_label_ids = gt_answer_labels[i] if gt_answer_labels.dim() > 1 else gt_answer_labels
                    # Filter out -100 (padding mask) before decode
                    answer_label_ids = answer_label_ids[answer_label_ids != -100]
                    gt_answer = model.tokenizer.decode(answer_label_ids.cpu().tolist(), skip_special_tokens=True)
                    result_row['gt_answer'] = gt_answer
                    
                    # Normalize
                    pred_norm = normalize_text(pred_answer)
                    gt_norm = normalize_text(gt_answer)
                    
                    # Exact Match
                    is_correct = (pred_norm == gt_norm)
                    answer_exact_matches.append(1.0 if is_correct else 0.0)
                    
                    # Token F1
                    pred_tokens = pred_norm.split()
                    gt_tokens = gt_norm.split()
                    f1 = compute_f1(pred_tokens, gt_tokens)
                    answer_f1_list.append(f1)
                    
                    # ROUGE
                    rouge1, rougel = compute_rouge(pred_norm, gt_norm)
                    answer_rouge1_list.append(rouge1)
                    answer_rougel_list.append(rougel)
                    
                    # Track by type
                    type_stats[q_type]['total'] += 1
                    if is_correct:
                        type_stats[q_type]['correct'] += 1
                    else:
                        # Store error sample (limit to 5 per type for display)
                        if len(type_stats[q_type]['errors']) < 5:
                            type_stats[q_type]['errors'].append({
                                'img_id': str(img_id),
                                'question': question,
                                'pred': pred_answer,
                                'gt': gt_answer
                            })
                        
                        # Store ALL errors for CSV export
                        all_errors.append({
                            'img_id': str(img_id),
                            'question': question,
                            'type': q_type,
                            'ground_truth': gt_answer,
                            'prediction': pred_answer,
                            'pred_normalized': pred_norm,
                            'gt_normalized': gt_norm,
                            'token_f1': f1,
                            'rouge1': rouge1,
                            'rougel': rougel
                        })
                
                results.append(result_row)
    
    # Compute statistics
    stats = {'total_samples': len(results)}
    
    # Answer metrics
    if has_gt:
        stats['answer_exact_match_acc'] = np.mean(answer_exact_matches) * 100.0
        stats['answer_token_f1'] = np.mean(answer_f1_list) * 100.0
        stats['answer_rouge1'] = np.mean(answer_rouge1_list) * 100.0
        stats['answer_rougel'] = np.mean(answer_rougel_list) * 100.0
        
        # Type-based statistics
        stats['type_stats'] = dict(type_stats)
        stats['all_errors'] = all_errors
    
    return results, stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate SimpleFusionVQA')
    parser.add_argument('--mode', type=str, default='val', choices=['test', 'val'],
                       help='Evaluation mode: val (train.csv split) or test (test.csv with metrics)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint')
    parser.add_argument('--csv_path', type=str, default=None,
                       help='Path to CSV file (overrides default based on mode)')
    parser.add_argument('--image_dir', type=str, default=None,
                       help='Path to image directory (overrides default based on mode)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--output_csv', type=str, default='predictions_simple.csv',
                       help='Output CSV path')
    parser.add_argument('--error_csv', type=str, default='errors_simple.csv',
                       help='Output CSV for errors (for data augmentation/improvement)')
    parser.add_argument('--max_answer_len', type=int, default=32,
                       help='Max answer length')
    parser.add_argument('--num_beams', type=int, default=4,
                       help='Number of beams for beam search')
    args = parser.parse_args()
    
    # Config based on mode
    if args.mode == 'val':
        # Use train.csv with validation split for metrics
        CONFIG = {
            'checkpoint_path': args.checkpoint,
            'csv_path': args.csv_path or '/kaggle/input/vivqa/ViVQA-main/ViVQA-main/train.csv',
            'image_dir': args.image_dir or '/kaggle/input/vivqa/drive-download-20220309T020508Z-001/train',
            'output_csv': args.output_csv,
            'error_csv': args.error_csv,
            'batch_size': args.batch_size,
            'use_val_split': True,
            'val_split': 0.1,
        }
    else:  # test mode
        CONFIG = {
            'checkpoint_path': args.checkpoint,
            'csv_path': args.csv_path or '/kaggle/input/vivqa/ViVQA-main/ViVQA-main/test.csv',
            'image_dir': args.image_dir or '/kaggle/input/vivqa/drive-download-20220309T020508Z-001/test',
            'output_csv': args.output_csv,
            'error_csv': args.error_csv,
            'batch_size': args.batch_size,
            'use_val_split': False,
        }
    
    print("="*70)
    print(f"SIMPLEFUSIONVQA EVALUATION ({args.mode.upper()} mode)")
    print("="*70)
    print(f"Max Answer Length: {args.max_answer_len}")
    print(f"Beam Search: {args.num_beams} beams")
    print("="*70)
    
    # Load model
    print("\n[INFO] Loading SimpleFusionVQA model...")
    model = SimpleFusionVQA(
        dinov2_model_name='facebook/dinov2-base',
        bartpho_model_name='vinai/bartpho-syllable',
        num_fusion_layers=6,  # Match training config
        num_heads=8,
        dropout=0.3,  # Match training dropout
        image_dropout_prob=0.0,  # No dropout during inference
        gradient_checkpointing=False  # Disable for inference
    )
    
    # Load checkpoint
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(CONFIG['checkpoint_path'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"[INFO] ✓ Loaded checkpoint from {CONFIG['checkpoint_path']}")
    if 'epoch' in checkpoint:
        print(f"      Epoch: {checkpoint['epoch']}")
    if 'best_val_loss' in checkpoint:
        print(f"      Best Val Loss: {checkpoint['best_val_loss']:.4f}")
    
    # Load dataset based on mode
    # Create vision_processor
    vision_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    
    # Define dataset class (used by both val and test modes)
    class VQACSVDataset(torch.utils.data.Dataset):
        def __init__(self, csv_path, image_dir, vision_processor, tokenizer, max_question_len=64, max_answer_len=32):
            self.data = pd.read_csv(csv_path)
            self.image_dir = image_dir
            self.vision_processor = vision_processor
            self.tokenizer = tokenizer
            self.max_question_len = max_question_len
            self.max_answer_len = max_answer_len

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            row = self.data.iloc[idx]
            img_id = str(row['img_id'])
            question = row['question']
            answer = row['answer']
            # Get type as is (keep numeric)
            q_type = str(row.get('type', 'unknown'))
            
            from PIL import Image
            img_path = f"{self.image_dir}/{img_id}.jpg"
            try:
                image = Image.open(img_path).convert('RGB')
            except:
                image = Image.new('RGB', (224, 224), color='white')
            
            pixel_values = self.vision_processor(images=image, return_tensors='pt')['pixel_values'][0]
            
            question_enc = self.tokenizer(
                question,
                max_length=self.max_question_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            answer_enc = self.tokenizer(
                answer,
                max_length=self.max_answer_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            labels = answer_enc['input_ids'][0].clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            return {
                'pixel_values': pixel_values,
                'input_ids': question_enc['input_ids'][0],
                'attention_mask': question_enc['attention_mask'][0],
                'labels': labels,
                'img_id': img_id,
                'question': question,
                'type': q_type
            }
    
    if args.mode == 'val':
        print("\n[INFO] Loading validation dataset with ground truth from CSV...")
        
        full_dataset = VQACSVDataset(
            csv_path=CONFIG['csv_path'],
            image_dir=CONFIG['image_dir'],
            vision_processor=vision_processor,
            tokenizer=model.tokenizer
        )
        
        if CONFIG.get('use_val_split', False):
            # Split into val (same as training)
            total_size = len(full_dataset)
            val_size = int(total_size * CONFIG['val_split'])
            train_size = total_size - val_size
            torch.manual_seed(42)  # Same seed as training
            _, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
            print(f"[INFO] Using validation split: {len(test_dataset)} samples")
        else:
            test_dataset = full_dataset
            print(f"[INFO] Using full dataset: {len(test_dataset)} samples")
    else:
        print("\n[INFO] Loading test dataset from CSV (with GT for metrics)...")
        
        # Use same dataset class as val mode (test.csv also has answer column)
        test_dataset = VQACSVDataset(
            csv_path=CONFIG['csv_path'],
            image_dir=CONFIG['image_dir'],
            vision_processor=vision_processor,
            tokenizer=model.tokenizer
        )
        print(f"[INFO] Loaded {len(test_dataset)} test samples")
    
    # Evaluate
    results, stats = evaluate_simple_vqa(
        model=model,
        test_dataset=test_dataset,
        batch_size=CONFIG['batch_size'],
        device=device,
        max_answer_len=args.max_answer_len,
        num_beams=args.num_beams
    )

    # Save results to CSV
    import os
    os.makedirs(os.path.dirname(CONFIG['output_csv']) if os.path.dirname(CONFIG['output_csv']) else '.', exist_ok=True)
    
    # Determine fieldnames based on available data
    fieldnames = ['img_id', 'question', 'pred_answer', 'type']
    if results and 'gt_answer' in results[0]:
        fieldnames.append('gt_answer')
    
    with open(CONFIG['output_csv'], 'w', newline='', encoding='utf-8') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\n[INFO] ✓ Predictions saved to {CONFIG['output_csv']}")
    
    # Save errors to separate CSV for data augmentation
    if 'all_errors' in stats and len(stats['all_errors']) > 0:
        error_csv_path = args.error_csv
        error_fieldnames = ['img_id', 'question', 'type', 'ground_truth', 'prediction', 
                           'pred_normalized', 'gt_normalized', 'token_f1', 'rouge1', 'rougel']
        
        with open(error_csv_path, 'w', newline='', encoding='utf-8') as csvf:
            writer = csv.DictWriter(csvf, fieldnames=error_fieldnames)
            writer.writeheader()
            for error in stats['all_errors']:
                writer.writerow(error)
        
        print(f"[INFO] ✓ {len(stats['all_errors'])} errors saved to {error_csv_path}")
        print(f"       → Use this file for data augmentation (LLM refinement, image generation, etc.)")
    
    # Print results
    print("\n" + "="*70)
    print("========== SimpleFusionVQA Evaluation Results ==========")
    print("="*70)
    print(f"Total samples: {stats['total_samples']}")
    
    if 'answer_exact_match_acc' in stats:
        print("\n[ANSWER METRICS]")
        print(f"Accuracy (EM): {stats['answer_exact_match_acc']:.2f}%")
        print(f"ROUGE-1 F1: {stats['answer_rouge1']/100:.4f}")  # Convert % to decimal
        print(f"ROUGE-L F1: {stats['answer_rougel']/100:.4f}")
        print(f"Token F1: {stats['answer_token_f1']/100:.4f}")
    
    if not ('answer_exact_match_acc' in stats):
        print("\n(No ground truth found in dataset)")
    
    # Print sample predictions
    print("\n===== Sample Predictions =====")
    for i in range(min(10, len(results))):
        print(f"Q{i+1} | GT: {results[i].get('gt_answer', 'N/A')} || Pred: {results[i]['pred_answer']}")
    
    # Print type-based error statistics
    if 'type_stats' in stats:
        print("\n" + "="*70)
        print("===== Error Analysis by Question Type =====")
        print("="*70)
        
        # Sort types by error rate (descending)
        type_error_rates = []
        for q_type, data in stats['type_stats'].items():
            if data['total'] > 0:
                error_rate = (data['total'] - data['correct']) / data['total'] * 100
                type_error_rates.append({
                    'type': q_type,
                    'total': data['total'],
                    'correct': data['correct'],
                    'errors': data['total'] - data['correct'],
                    'error_rate': error_rate,
                    'accuracy': data['correct'] / data['total'] * 100,
                    'samples': data['errors']
                })
        
        # Sort by error rate (highest first)
        type_error_rates.sort(key=lambda x: x['error_rate'], reverse=True)
        
        print("\n[Type Statistics]")
        print(f"{'Type':<20} {'Total':<10} {'Correct':<10} {'Errors':<10} {'Accuracy':<12} {'Error Rate'}")
        print("-" * 80)
        for item in type_error_rates:
            print(f"{item['type']:<20} {item['total']:<10} {item['correct']:<10} {item['errors']:<10} "
                  f"{item['accuracy']:>10.2f}% {item['error_rate']:>10.2f}%")
        
        # Print error samples for worst types (top 3)
        print("\n[Error Samples - Top 3 Worst Types]")
        for idx, item in enumerate(type_error_rates[:3]):
            if len(item['samples']) > 0:
                print(f"\n{idx+1}. Type: '{item['type']}' (Error Rate: {item['error_rate']:.2f}%)")
                print("-" * 70)
                for i, sample in enumerate(item['samples'], 1):
                    print(f"  Sample {i}:")
                    print(f"    Question: {sample['question']}")
                    print(f"    Ground Truth: {sample['gt']}")
                    print(f"    Prediction: {sample['pred']}")
                    if i < len(item['samples']):
                        print()
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
