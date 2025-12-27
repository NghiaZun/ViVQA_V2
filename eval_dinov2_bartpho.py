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
                num_beams=4
            )
            for i in range(len(answer_text)):
                results.append({
                    'image_id': batch['image_id'][i] if isinstance(batch['image_id'], list) else batch['image_id'],
                    'question': batch['question'][i] if isinstance(batch['question'], list) else batch['question'],
                    'pred_answer': answer_text[i],
                    'pred_reasoning': reasoning_text[i]
                })
    stats = {'total_samples': len(results)}
    return results, stats


def main():
    # Config
    CONFIG = {
        'checkpoint_path': '/kaggle/input/test-3/transformers/default/1/best_model_stage3.pt',
        'test_csv': '/kaggle/input/vivqa/ViVQA-main/ViVQA-main/test.csv',  # Đọc file test gốc CSV
        'image_dir': '/kaggle/input/vivqa/drive-download-20220309T020508Z-001/test',
        'output_csv': '/kaggle/working/predictions.csv',
        'batch_size': 4,
    }
    
    print("="*70)
    print("EVALUATION: DINOv2 + BARTpho VQA")
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
    
    # Load test data từ CSV gốc
    print("\n[INFO] Loading test dataset from CSV...")
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
            img_path = item['image'] if 'image' in item else item['image_path']
            img_path = img_path.split('/')[-1]
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
                'image_id': item.get('image', item.get('image_path', '')),
                'question': question
            }

    test_dataset = VQATestCSVDataset(
        csv_path=CONFIG['test_csv'],
        image_dir=CONFIG['image_dir'],
        vision_processor=model.vision_processor,
        tokenizer=model.tokenizer
    )
    
    # Evaluate
    results, stats = evaluate_model(
        model=model,
        test_dataset=test_dataset,
        batch_size=CONFIG['batch_size'],
        device=device
    )

    # Save results to CSV only
    with open(CONFIG['output_csv'], 'w', newline='', encoding='utf-8') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=['image_id','question','pred_answer','pred_reasoning'])
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\n[INFO] ✓ Predictions saved to {CONFIG['output_csv']}")
    print("\nStatistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
