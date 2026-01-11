"""
EVALUATION SCRIPT: Latent Reasoning VQA (FixedLatentReasoningVQA)
===================================================================
Evaluate latent reasoning model với:
✅ Latent reasoning bottleneck (implicit reasoning)
✅ Generate answers from latent representations
✅ Metrics: Accuracy, F1, ROUGE-1, ROUGE-L
✅ Evaluate answer quality
"""

import torch
import json
import csv
import pandas as pd
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from model import FixedLatentReasoningVQA  # Changed to latent reasoning model
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


def evaluate_autoregressive_cot(
    model,
    test_dataset,
    batch_size=4,
    device='cuda',
    max_reasoning_len=128,
    max_answer_len=32,
    num_beams=4,
    evaluate_reasoning=True
):
    """
    Evaluate autoregressive CoT model
    
    Args:
        model: DINOv2BARTphoVQA model
        test_dataset: VQADistillationDataset with GT
        batch_size: Batch size for evaluation
        device: Device to use
        max_reasoning_len: Max length for reasoning generation
        max_answer_len: Max length for answer generation
        num_beams: Number of beams for beam search
        evaluate_reasoning: Whether to evaluate reasoning quality (if GT available)
    
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
    
    reasoning_rouge1_list = []
    reasoning_rougel_list = []
    reasoning_f1_list = []
    
    has_gt = False
    has_gt_reasoning = False
    
    print("[INFO] Generating predictions with autoregressive CoT...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            # Generate answers using FixedLatentReasoningVQA.generate()
            # Note: This model doesn't generate separate reasoning text (it's latent)
            answer_text = model.generate(
                pixel_values=batch['pixel_values'],
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_length=max_answer_len,
                num_beams=num_beams
            )
            
            # Latent reasoning model doesn't output explicit reasoning text
            reasoning_text = ['[LATENT REASONING]'] * len(answer_text)
            
            # Check if we have ground truth
            gt_answer_labels = batch.get('labels', None)
            gt_reasoning_labels = batch.get('reasoning_labels', None)
            
            for i in range(len(answer_text)):
                pred_answer = answer_text[i]
                pred_reasoning = reasoning_text[i]
                
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
                
                result_row = {
                    'img_id': str(img_id),
                    'question': question,
                    'pred_answer': pred_answer,
                    'pred_reasoning': pred_reasoning
                }
                
                # ===== EVALUATE ANSWER =====
                if gt_answer_labels is not None:
                    has_gt = True
                    # Decode GT answer from labels tensor
                    answer_label_ids = gt_answer_labels[i] if gt_answer_labels.dim() > 1 else gt_answer_labels
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
                
                # ===== EVALUATE REASONING =====
                # Note: Latent reasoning model doesn't produce explicit reasoning text
                # Reasoning is implicit in latent representations
                # Skip reasoning evaluation for this model type
                pass
                
                results.append(result_row)
    
    # Compute statistics
    stats = {'total_samples': len(results)}
    
    # Answer metrics
    if has_gt:
        stats['answer_exact_match_acc'] = np.mean(answer_exact_matches) * 100.0
        stats['answer_token_f1'] = np.mean(answer_f1_list) * 100.0
        stats['answer_rouge1'] = np.mean(answer_rouge1_list) * 100.0
        stats['answer_rougel'] = np.mean(answer_rougel_list) * 100.0
    
    # Reasoning metrics
    if has_gt_reasoning:
        stats['reasoning_token_f1'] = np.mean(reasoning_f1_list) * 100.0
        stats['reasoning_rouge1'] = np.mean(reasoning_rouge1_list) * 100.0
        stats['reasoning_rougel'] = np.mean(reasoning_rougel_list) * 100.0
    
    return results, stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate Latent Reasoning VQA (FixedLatentReasoningVQA)')
    parser.add_argument('--mode', type=str, default='val', choices=['test', 'val'],
                       help='Evaluation mode: test (no GT) or val (with GT metrics)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--output_csv', type=str, default='predictions_autoregressive_cot.csv',
                       help='Output CSV path')
    parser.add_argument('--max_reasoning_len', type=int, default=128,
                       help='Max reasoning length')
    parser.add_argument('--max_answer_len', type=int, default=32,
                       help='Max answer length')
    parser.add_argument('--num_beams', type=int, default=4,
                       help='Number of beams for beam search')
    parser.add_argument('--no_eval_reasoning', action='store_true',
                       help='Skip reasoning evaluation (faster)')
    args = parser.parse_args()
    
    # Config based on mode
    if args.mode == 'val':
        # Use train.csv with validation split for metrics
        CONFIG = {
            'checkpoint_path': args.checkpoint,
            'csv_path': '/kaggle/input/vivqa/ViVQA-main/ViVQA-main/train.csv',
            'image_dir': '/kaggle/input/vivqa/drive-download-20220309T020508Z-001/train',
            'output_csv': args.output_csv,
            'batch_size': args.batch_size,
            'use_val_split': True,
            'val_split': 0.1,
        }
    else:  # test mode
        CONFIG = {
            'checkpoint_path': args.checkpoint,
            'csv_path': '/kaggle/input/vivqa/ViVQA-main/ViVQA-main/test.csv',
            'image_dir': '/kaggle/input/vivqa/drive-download-20220309T020508Z-001/test',
            'output_csv': args.output_csv,
            'batch_size': args.batch_size,
            'use_val_split': False,
        }
    
    print("="*70)
    print(f"LATENT REASONING VQA EVALUATION ({args.mode.upper()} mode)")
    print("="*70)
    print(f"Max Reasoning Length: {args.max_reasoning_len}")
    print(f"Max Answer Length: {args.max_answer_len}")
    print(f"Beam Search: {args.num_beams} beams")
    print(f"Evaluate Reasoning: {not args.no_eval_reasoning}")
    print("="*70)
    
    # Load model
    print("\n[INFO] Loading FixedLatentReasoningVQA model...")
    model = FixedLatentReasoningVQA(
        dinov2_model_name='facebook/dinov2-base',
        bartpho_model_name='vinai/bartpho-syllable',
        num_reasoning_tokens=6,
        latent_dim=256,
        num_reasoning_layers=2,
        num_fusion_layers=2,
        free_bits=0.5,
        ortho_weight=0.1,
        image_dropout_prob=0.1,
        token_dropout_prob=0.3,
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
    if args.mode == 'val':
        print("\n[INFO] Loading validation dataset with ground truth from CSV...")
        
        # Use same CSV-based dataset as example
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
                    'question': question
                }
        
        full_dataset = VQACSVDataset(
            csv_path=CONFIG['csv_path'],
            image_dir=CONFIG['image_dir'],
            vision_processor=model.vision_processor,
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
        print("\n[INFO] Loading test dataset from CSV (no GT)...")
        
        class VQATestCSVDataset(torch.utils.data.Dataset):
            def __init__(self, csv_path, image_dir, vision_processor, tokenizer, max_question_len=64):
                self.samples = []
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        self.samples.append(row)
                self.image_dir = image_dir
                self.vision_processor = vision_processor
                self.tokenizer = tokenizer
                self.max_question_len = max_question_len

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                item = self.samples[idx]
                img_id = item['img_id']
                img_path = f"{img_id}.jpg"
                full_img_path = f"{self.image_dir}/{img_path}"
                from PIL import Image
                image = Image.open(full_img_path).convert('RGB')
                pixel_values = self.vision_processor(images=image, return_tensors='pt')['pixel_values'][0]
                question = item['question']
                question_enc = self.tokenizer(
                    question,
                    max_length=self.max_question_len,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                return {
                    'pixel_values': pixel_values,
                    'input_ids': question_enc['input_ids'][0],
                    'attention_mask': question_enc['attention_mask'][0],
                    'img_id': img_id,
                    'question': question
                }
        
        test_dataset = VQATestCSVDataset(
            csv_path=CONFIG['test_csv'],
            image_dir=CONFIG['image_dir'],
            vision_processor=model.vision_processor,
            tokenizer=model.tokenizer
        )
        print(f"[INFO] Loaded {len(test_dataset)} test samples")
    
    # Evaluate
    results, stats = evaluate_autoregressive_cot(
        model=model,
        test_dataset=test_dataset,
        batch_size=CONFIG['batch_size'],
        device=device,
        max_reasoning_len=args.max_reasoning_len,
        max_answer_len=args.max_answer_len,
        num_beams=args.num_beams,
        evaluate_reasoning=not args.no_eval_reasoning
    )

    # Save results to CSV
    import os
    os.makedirs(os.path.dirname(CONFIG['output_csv']) if os.path.dirname(CONFIG['output_csv']) else '.', exist_ok=True)
    
    # Determine fieldnames based on available data
    fieldnames = ['img_id', 'question', 'pred_answer', 'pred_reasoning']
    if results and 'gt_answer' in results[0]:
        fieldnames.append('gt_answer')
    if results and 'gt_reasoning' in results[0]:
        fieldnames.append('gt_reasoning')
    
    with open(CONFIG['output_csv'], 'w', newline='', encoding='utf-8') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\n[INFO] ✓ Predictions saved to {CONFIG['output_csv']}")
    
    # Print results (format giống file example)
    print("\n" + "="*70)
    print("========== Test Results ==========")
    print("="*70)
    print(f"Total samples: {stats['total_samples']}")
    
    if 'answer_exact_match_acc' in stats:
        print("\n[ANSWER METRICS]")
        print(f"Accuracy (EM): {stats['answer_exact_match_acc']:.2f}%")
        print(f"ROUGE-1 F1: {stats['answer_rouge1']/100:.4f}")  # Convert % to decimal
        print(f"ROUGE-L F1: {stats['answer_rougel']/100:.4f}")
        print(f"Token F1: {stats['answer_token_f1']/100:.4f}")
    
    if 'reasoning_token_f1' in stats:
        print("\n[REASONING METRICS]")
        print(f"Token F1 Score:       {stats['reasoning_token_f1']:.2f}%")
        print(f"ROUGE-1:              {stats['reasoning_rouge1']:.2f}%")
        print(f"ROUGE-L:              {stats['reasoning_rougel']:.2f}%")
    
    if not ('answer_exact_match_acc' in stats):
        print("\n(No ground truth available - test mode)")
    
    # Print sample predictions
    print("\n===== Sample Predictions =====")
    for i in range(min(10, len(results))):
        print(f"Q{i+1} | GT: {results[i].get('gt_answer', 'N/A')} || Pred: {results[i]['pred_answer']}")
    
    print("="*70)


if __name__ == '__main__':
    main()
