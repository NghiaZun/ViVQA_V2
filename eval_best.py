"""
Evaluation script for the saved best model.
Saves predictions to /kaggle/working/predictions.csv and prints accuracy.
Metrics: Exact Match, ROUGE-1, ROUGE-L, F1

IMPORTANT: 
- Uses the SAME model architecture as new_train.py (ChainOfThoughtVQAModel)
- Uses the SAME config paths as new_train.py for consistency
- Supports both validation split and full dataset evaluation
"""

import os
import sys
import csv
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from new_train import VQADistillationDataset
from model_cot import create_cot_model


def normalize_text(s):
    import re
    s = s.lower().strip()
    s = re.sub(r"[^0-9a-zA-ZÀ-ỹ\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def compute_f1(pred_tokens, gt_tokens):
    """Token-level F1 score"""
    common = set(pred_tokens) & set(gt_tokens)
    if len(common) == 0:
        return 0.0
    precision = len(common) / len(pred_tokens) if pred_tokens else 0
    recall = len(common) / len(gt_tokens) if gt_tokens else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_rouge(pred, gt):
    """
    Compute ROUGE-1 and ROUGE-L (simple implementation)
    Returns: (rouge1_f1, rougel_f1)
    """
    pred_tokens = pred.split()
    gt_tokens = gt.split()
    
    # ROUGE-1: unigram overlap F1
    rouge1 = compute_f1(pred_tokens, gt_tokens)
    
    # ROUGE-L: Longest Common Subsequence
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


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ===== SAME CONFIG AS new_train.py =====
    print(f"[INFO] Loading model with SAME config as training...")
    model = create_cot_model(
        clip_model='openai/clip-vit-base-patch32',
        text_encoder='vinai/phobert-base',
        decoder='VietAI/vit5-base',
        hidden_dim=768,
        fusion='co_attention',  # SAME as training
        use_reasoning_attention=True  # SAME as training
    )

    ckpt = args.checkpoint
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    print(f"[INFO] Loading checkpoint: {ckpt}")
    checkpoint = torch.load(ckpt, map_location=device)
    
    # Load model state
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"     Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"     Best Val Loss: {checkpoint.get('best_val_loss', 'N/A'):.4f}")
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()

    print("[INFO] Creating dataset (SAME as training)...")
    full_dataset = VQADistillationDataset(
        json_path=args.jsonl,
        image_dir=args.image_dir,
        clip_processor=model.clip_processor,
        text_tokenizer=model.text_tokenizer,
        decoder_tokenizer=model.decoder_tokenizer,
        augment=False  # No augmentation for evaluation
    )
    
    # Use validation split if requested (SAME split as training)
    if args.use_val_split:
        total_size = len(full_dataset)
        val_size = int(0.1 * total_size)  # SAME 10% split as training
        train_size = total_size - val_size
        
        # SAME random split as training (fixed seed)
        torch.manual_seed(42)
        _, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        dataset = val_dataset
        print(f"[INFO] Using validation split: {len(dataset)} samples (10% of {total_size})")
    else:
        dataset = full_dataset
        print(f"[INFO] Using full dataset: {len(dataset)} samples")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    out_path = args.output_csv
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Metrics tracking
    total = 0
    exact_matches = []
    rouge1_list = []
    rougel_list = []
    f1_list = []

    print(f"[INFO] Running inference on {len(dataset)} samples...")
    with open(out_path, 'w', newline='', encoding='utf-8') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=['img_id','question','pred_answer','gt_answer','pred_reasoning','gt_reasoning'])
        writer.writeheader()

        with torch.no_grad():
            for i, item in enumerate(loader):
                pixel_values = item['pixel_values'].to(device)
                input_ids = item['input_ids'].unsqueeze(0).to(device) if item['input_ids'].dim()==1 else item['input_ids'].to(device)
                attention_mask = item['attention_mask'].unsqueeze(0).to(device) if item['attention_mask'].dim()==1 else item['attention_mask'].to(device)

                # Use model.generate_answer for greedy decode
                try:
                    answer_text = model.generate_answer(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, return_reasoning=False)
                except Exception:
                    # Fallback: forward and argmax
                    out = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
                    answer_ids = torch.argmax(out.answer_logits, dim=-1)
                    answer_text = model.decoder_tokenizer.decode(answer_ids[0], skip_special_tokens=True)

                # Decode GT answers
                gt_answer = model.decoder_tokenizer.decode(item['labels'].tolist(), skip_special_tokens=True)
                gt_reasoning = model.decoder_tokenizer.decode(item.get('reasoning_labels', item['labels']).tolist(), skip_special_tokens=True)

                pred_norm = normalize_text(answer_text)
                gt_norm = normalize_text(gt_answer)

                # Exact match
                is_correct = (pred_norm == gt_norm)
                exact_matches.append(1.0 if is_correct else 0.0)
                
                # Token-level F1
                pred_tokens = pred_norm.split()
                gt_tokens = gt_norm.split()
                f1 = compute_f1(pred_tokens, gt_tokens)
                f1_list.append(f1)
                
                # ROUGE
                rouge1, rougel = compute_rouge(pred_norm, gt_norm)
                rouge1_list.append(rouge1)
                rougel_list.append(rougel)
                
                total += 1

                writer.writerow({
                    'img_id': item.get('img_id', [''])[0],
                    'question': model.text_tokenizer.decode(item['input_ids'].tolist() if isinstance(item['input_ids'], torch.Tensor) else item['input_ids'][0], skip_special_tokens=True),
                    'pred_answer': answer_text,
                    'gt_answer': gt_answer,
                    'pred_reasoning': '',
                    'gt_reasoning': gt_reasoning
                })

                if (i+1) % 100 == 0:
                    print(f"  Processed {i+1}/{len(dataset)}")

    # Compute final metrics
    acc = np.mean(exact_matches) * 100.0 if exact_matches else 0.0
    avg_rouge1 = np.mean(rouge1_list) * 100.0 if rouge1_list else 0.0
    avg_rougel = np.mean(rougel_list) * 100.0 if rougel_list else 0.0
    avg_f1 = np.mean(f1_list) * 100.0 if f1_list else 0.0
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Total samples: {total}")
    print(f"Exact Match Accuracy: {acc:.2f}%")
    print(f"Token F1 Score:       {avg_f1:.2f}%")
    print(f"ROUGE-1:              {avg_rouge1:.2f}%")
    print(f"ROUGE-L:              {avg_rougel:.2f}%")
    print("="*70)
    print(f"[INFO] Predictions saved to: {out_path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Evaluate trained Chain-of-Thought VQA model')
    p.add_argument('--checkpoint', type=str, default='/kaggle/working/checkpoints/best_model.pt',
                   help='Path to checkpoint (default: best_model.pt from training)')
    p.add_argument('--jsonl', type=str, default='/kaggle/input/teacher-5-12/teacher_outputs_train.jsonl',
                   help='JSONL file (SAME path as training)')
    p.add_argument('--image_dir', type=str, default='/kaggle/input/vivqa/drive-download-20220309T020508Z-001/train',
                   help='Image directory (SAME path as training)')
    p.add_argument('--output_csv', type=str, default='/kaggle/working/predictions.csv',
                   help='Output CSV path for predictions')
    p.add_argument('--batch_size', type=int, default=4,
                   help='Batch size for inference (default: 4)')
    p.add_argument('--use_val_split', action='store_true',
                   help='Evaluate only on validation split (10% as in training)')
    args = p.parse_args()
    main(args)
