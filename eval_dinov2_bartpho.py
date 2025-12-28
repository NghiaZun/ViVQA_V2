"""
EVALUATION SCRIPT: DINOv2 + BARTpho VQA
========================================
Evaluate trained model với reasoning quality metrics
"""

import torch
import json
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from model_dinov2_bartpho import DINOv2BARTphoVQA
from train_dinov2_bartpho import VQADistillationDataset
import numpy as np
from collections import defaultdict


def evaluate_model(
    model,
    test_dataset,
    batch_size=4,
    device='cuda'
):
    """
    Evaluate model với reasoning và answer generation
    """
    model.eval()
    model = model.to(device)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    import re
    def normalize_text(s):
        s = s.lower().strip()
        s = re.sub(r"[^0-9a-zA-ZÀ-ỹ\s]", "", s)
        s = re.sub(r"\s+", " ", s)
        return s

    def compute_f1(pred_tokens, gt_tokens):
        common = set(pred_tokens) & set(gt_tokens)
        if len(common) == 0:
            return 0.0
        precision = len(common) / len(pred_tokens) if pred_tokens else 0
        recall = len(common) / len(gt_tokens) if gt_tokens else 0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def compute_rouge(pred, gt):
        pred_tokens = pred.split()
        gt_tokens = gt.split()
        rouge1 = compute_f1(pred_tokens, gt_tokens)
        def lcs_length(a, b):
            m, n = len(a), len(b)
            dp = [[0]*(n+1) for _ in range(m+1)]
            for i in range(1, m+1):
                for j in range(1, n+1):
                    if a[i-1] == b[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            return dp[m][n]
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

    results = []
    
    # Metrics tracking
    exact_matches = []
    rouge1_list = []
    rougel_list = []
    f1_list = []
    has_gt = False
    
    print("[INFO] Generating predictions...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            reasoning_text, answer_text, _ = model.generate(
                pixel_values=batch['pixel_values'],
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_reasoning_len=128,
                max_answer_len=32,
                num_beams=4,  # Use beam search for better quality
                repetition_penalty=1.2,
                length_penalty=1.0
            )
            
            # Check if we have ground truth (from VQADistillationDataset)
            gt_answer_labels = batch.get('labels', None)
            gt_reasoning_labels = batch.get('reasoning_labels', None)
            
            for i in range(len(answer_text)):
                pred_answer = answer_text[i]
                pred_reasoning = reasoning_text[i]
                
                # Get img_id (fallback to index if not available)
                if 'img_id' in batch:
                    img_id = batch['img_id'][i] if isinstance(batch['img_id'], list) else batch['img_id']
                else:
                    img_id = f"sample_{i}"
                
                # Decode question from input_ids
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
                
                # Compute metrics if GT available (decode from labels)
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
                    exact_matches.append(1.0 if is_correct else 0.0)
                    
                    # Token F1
                    pred_tokens = pred_norm.split()
                    gt_tokens = gt_norm.split()
                    f1 = compute_f1(pred_tokens, gt_tokens)
                    f1_list.append(f1)
                    
                    # ROUGE
                    rouge1, rougel = compute_rouge(pred_norm, gt_norm)
                    rouge1_list.append(rouge1)
                    rougel_list.append(rougel)
                
                # Decode GT reasoning if available
                if gt_reasoning_labels is not None:
                    reasoning_label_ids = gt_reasoning_labels[i] if gt_reasoning_labels.dim() > 1 else gt_reasoning_labels
                    gt_reasoning = model.tokenizer.decode(reasoning_label_ids.cpu().tolist(), skip_special_tokens=True)
                    result_row['gt_reasoning'] = gt_reasoning
                
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
    parser = argparse.ArgumentParser(description='Evaluate DINOv2 + BARTpho VQA')
    parser.add_argument('--mode', type=str, default='test', choices=['test', 'val'],
                       help='Evaluation mode: test (no GT) or val (with GT metrics)')
    parser.add_argument('--checkpoint', type=str, default='/kaggle/input/test-3/transformers/default/1/best_model_stage3.pt',
                       help='Path to checkpoint')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--output_csv', type=str, default='/kaggle/working/predictions.csv',
                       help='Output CSV path')
    args = parser.parse_args()
    
    # Config based on mode
    if args.mode == 'val':
        CONFIG = {
            'checkpoint_path': args.checkpoint,
            'train_json': '/kaggle/input/teacher-5-12/teacher_outputs_train.jsonl',
            'image_dir': '/kaggle/input/vivqa/drive-download-20220309T020508Z-001/train',
            'output_csv': args.output_csv,
            'batch_size': args.batch_size,
            'use_val_split': True,
            'val_split': 0.1,
        }
    else:  # test mode
        CONFIG = {
            'checkpoint_path': args.checkpoint,
            'test_csv': '/kaggle/input/vivqa/ViVQA-main/ViVQA-main/test.csv',
            'image_dir': '/kaggle/input/vivqa/drive-download-20220309T020508Z-001/test',
            'output_csv': args.output_csv,
            'batch_size': args.batch_size,
            'use_val_split': False,
        }
    
    print("="*70)
    print(f"EVALUATION: DINOv2 + BARTpho VQA ({args.mode.upper()} mode)")
    print("="*70)
    
    # Load model
    print("\n[INFO] Loading model...")
    model = DINOv2BARTphoVQA(
        dinov2_model_name='facebook/dinov2-base',
        bartpho_model_name='vinai/bartpho-syllable',
        num_cross_attn_layers=3,
        use_reasoning_quality_check=True,
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
        print("\n[INFO] Loading validation dataset with ground truth...")
        full_dataset = VQADistillationDataset(
            json_path=CONFIG['train_json'],
            image_dir=CONFIG['image_dir'],
            vision_processor=model.vision_processor,
            tokenizer=model.tokenizer,
            augment=False
        )
        
        # Split into val (same as training)
        total_size = len(full_dataset)
        val_size = int(total_size * CONFIG['val_split'])
        train_size = total_size - val_size
        torch.manual_seed(42)  # Same seed as training
        _, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        print(f"[INFO] Using validation split: {len(test_dataset)} samples")
    else:
        print("\n[INFO] Loading test dataset from CSV (no GT)...")
        import csv
        
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
                # Lấy img_id làm image_id, và lấy question đúng trường
                img_id = item['img_id']
                # Nếu file ảnh là img_id.jpg/png, sửa lại tên file cho đúng
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
    results, stats = evaluate_model(
        model=model,
        test_dataset=test_dataset,
        batch_size=CONFIG['batch_size'],
        device=device
    )

    # Save results to CSV
    import csv
    import os
    os.makedirs(os.path.dirname(CONFIG['output_csv']) if os.path.dirname(CONFIG['output_csv']) else '.', exist_ok=True)
    
    # Determine fieldnames based on available data
    fieldnames = ['img_id', 'question', 'pred_answer', 'pred_reasoning']
    if results and 'gt_answer' in results[0]:
        fieldnames.extend(['gt_answer', 'gt_reasoning'])
    
    with open(CONFIG['output_csv'], 'w', newline='', encoding='utf-8') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\n[INFO] ✓ Predictions saved to {CONFIG['output_csv']}")
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Total samples: {stats['total_samples']}")
    
    if 'exact_match_acc' in stats:
        print(f"Exact Match Accuracy: {stats['exact_match_acc']:.2f}%")
        print(f"Token F1 Score:       {stats['token_f1']:.2f}%")
        print(f"ROUGE-1:              {stats['rouge1']:.2f}%")
        print(f"ROUGE-L:              {stats['rougel']:.2f}%")
    else:
        print("(No ground truth available - test mode)")
    print("="*70)


if __name__ == '__main__':
    main()
