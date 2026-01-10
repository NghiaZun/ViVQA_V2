"""
EVALUATION SCRIPT: Latent Reasoning VQA
========================================
Evaluate trained latent reasoning model trên test set

Metrics:
1. Exact Match Accuracy (EM)
2. Answer loss
3. Reasoning sensitivity analysis (ablation)
4. Multiple reasoning samples diversity

Usage:
    python eval_latent_reasoning.py \
        --checkpoint /path/to/best_model.pt \
        --test_csv /path/to/test.csv \
        --test_images /path/to/test_images \
        --num_reasoning_samples 5
"""

import os
import argparse
from typing import Dict, List
import json

import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor
from tqdm import tqdm
import pandas as pd
import numpy as np

from dataset import VQAGenDataset
from model_latent_reasoning import LatentReasoningVQA


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--test_csv', type=str, required=True,
                      help='Path to test CSV')
    parser.add_argument('--test_images', type=str, required=True,
                      help='Path to test images folder')
    
    # Generation
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_length', type=int, default=32)
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument('--num_reasoning_samples', type=int, default=1,
                      help='Number of reasoning samples for diversity analysis')
    
    # Output
    parser.add_argument('--output_json', type=str, default='eval_results_latent.json')
    parser.add_argument('--output_csv', type=str, default='predictions_latent.csv')
    
    return parser.parse_args()


def normalize_answer(s: str) -> str:
    """Normalize answer text cho exact match"""
    return s.lower().strip()


def compute_exact_match(predicted: str, ground_truth: str) -> float:
    """Compute exact match (0 or 1)"""
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)
    return 1.0 if pred_norm == gt_norm else 0.0


def evaluate_model(
    model: LatentReasoningVQA,
    dataloader: DataLoader,
    device: torch.device,
    num_reasoning_samples: int = 1,
    max_length: int = 32,
    num_beams: int = 4
) -> Dict:
    """
    Evaluate model trên test set
    
    Returns:
        results: Dict containing metrics
    """
    model.eval()
    
    all_predictions = []
    all_ground_truths = []
    all_exact_matches = []
    
    # For diversity analysis (if num_reasoning_samples > 1)
    all_diverse_predictions = []
    
    print("\n" + "="*80)
    print("RUNNING EVALUATION")
    print("="*80)
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", ncols=100)
        
        for batch_idx, (pixel_values, input_ids, attention_mask, labels) in enumerate(pbar):
            # Move to device
            pixel_values = pixel_values.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            # Generate answers
            if num_reasoning_samples == 1:
                # Single sample (deterministic)
                predictions = model.generate(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    num_beams=num_beams,
                    num_reasoning_samples=1
                )
                all_predictions.extend(predictions)
            else:
                # Multiple samples for diversity analysis
                diverse_preds = []
                for _ in range(num_reasoning_samples):
                    predictions = model.generate(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=max_length,
                        num_beams=num_beams,
                        num_reasoning_samples=1
                    )
                    diverse_preds.append(predictions)
                
                # Use first sample as primary prediction
                all_predictions.extend(diverse_preds[0])
                
                # Store all samples for diversity analysis
                # Transpose: [num_samples, batch] -> [batch, num_samples]
                batch_diverse = [[diverse_preds[i][j] for i in range(num_reasoning_samples)]
                               for j in range(len(diverse_preds[0]))]
                all_diverse_predictions.extend(batch_diverse)
            
            # Decode ground truth
            batch_size = labels.size(0)
            for i in range(batch_size):
                label_ids = labels[i]
                # Remove padding
                label_ids = label_ids[label_ids != -100]
                ground_truth = model.tokenizer.decode(label_ids, skip_special_tokens=True).strip()
                all_ground_truths.append(ground_truth)
            
            # Update progress
            current_em = np.mean([
                compute_exact_match(pred, gt)
                for pred, gt in zip(all_predictions, all_ground_truths)
            ])
            pbar.set_postfix({'EM': f'{current_em*100:.2f}%'})
    
    # ========================================================================
    # COMPUTE METRICS
    # ========================================================================
    
    print("\n" + "="*80)
    print("COMPUTING METRICS")
    print("="*80)
    
    # Exact match
    for pred, gt in zip(all_predictions, all_ground_truths):
        em = compute_exact_match(pred, gt)
        all_exact_matches.append(em)
    
    exact_match_acc = np.mean(all_exact_matches)
    
    results = {
        'exact_match': exact_match_acc,
        'num_samples': len(all_predictions),
        'num_reasoning_samples': num_reasoning_samples
    }
    
    print(f"\nExact Match Accuracy: {exact_match_acc*100:.2f}%")
    print(f"Total samples: {len(all_predictions)}")
    
    # Diversity analysis (if multiple reasoning samples)
    if num_reasoning_samples > 1 and len(all_diverse_predictions) > 0:
        print("\n" + "-"*80)
        print("DIVERSITY ANALYSIS")
        print("-"*80)
        
        # Compute diversity: unique answers per sample
        diversities = []
        for diverse_preds in all_diverse_predictions:
            unique_preds = len(set(diverse_preds))
            diversity = unique_preds / num_reasoning_samples
            diversities.append(diversity)
        
        avg_diversity = np.mean(diversities)
        results['answer_diversity'] = avg_diversity
        
        print(f"Average diversity: {avg_diversity:.2%}")
        print(f"  (1.0 = all samples different, 0.0 = all samples same)")
        
        # Best of N accuracy (oracle)
        best_of_n_matches = []
        for i, diverse_preds in enumerate(all_diverse_predictions):
            gt = all_ground_truths[i]
            # Check if any prediction matches
            match = max([compute_exact_match(pred, gt) for pred in diverse_preds])
            best_of_n_matches.append(match)
        
        best_of_n_acc = np.mean(best_of_n_matches)
        results['best_of_n_accuracy'] = best_of_n_acc
        
        print(f"Best-of-{num_reasoning_samples} accuracy (oracle): {best_of_n_acc*100:.2f}%")
        print(f"  (upper bound if we had perfect answer selection)")
    
    print("="*80 + "\n")
    
    return results, all_predictions, all_ground_truths


def main():
    """Main evaluation function"""
    args = parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # ========================================================================
    # 1. LOAD MODEL
    # ========================================================================
    print("\n[1/3] Loading model...")
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Get model config from checkpoint
    cfg = checkpoint.get('config', None)
    
    # Initialize model
    model = LatentReasoningVQA(
        dinov2_model_name='facebook/dinov2-base',
        bartpho_model_name='vinai/bartpho-syllable',
        num_cross_attn_layers=cfg.num_cross_attn_layers if cfg else 3,
        num_reasoning_tokens=cfg.num_reasoning_tokens if cfg else 12,
        num_reasoning_layers=cfg.num_reasoning_layers if cfg else 2,
        use_stochastic_reasoning=cfg.use_stochastic_reasoning if cfg else True,
        latent_dim=cfg.latent_dim if cfg else 512,
        gradient_checkpointing=False  # Disable for inference
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']+1}")
    
    # ========================================================================
    # 2. LOAD DATASET
    # ========================================================================
    print("\n[2/3] Loading test dataset...")
    
    vision_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    
    test_dataset = VQAGenDataset(
        csv_path=args.test_csv,
        image_folder=args.test_images,
        vision_processor=vision_processor,
        tokenizer_name='vinai/bartpho-syllable',
        max_q_len=32,
        max_a_len=32
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # ========================================================================
    # 3. EVALUATE
    # ========================================================================
    print("\n[3/3] Evaluating...")
    
    results, predictions, ground_truths = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
        num_reasoning_samples=args.num_reasoning_samples,
        max_length=args.max_length,
        num_beams=args.num_beams
    )
    
    # ========================================================================
    # 4. SAVE RESULTS
    # ========================================================================
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Save metrics JSON
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved metrics: {args.output_json}")
    
    # Save predictions CSV
    df = pd.DataFrame({
        'prediction': predictions,
        'ground_truth': ground_truths,
        'exact_match': [compute_exact_match(p, g) for p, g in zip(predictions, ground_truths)]
    })
    df.to_csv(args.output_csv, index=False, encoding='utf-8')
    print(f"✓ Saved predictions: {args.output_csv}")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\nFinal Exact Match: {results['exact_match']*100:.2f}%")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
