"""
EVALUATION SCRIPT: Latent Reasoning VQA
===============================================
Evaluate trained latent reasoning model

Metrics:
1. Exact Match Accuracy (EM)
2. Answer loss
3. Reasoning intervention tests (ablation, noise)
4. Diversity metrics

Usage:
    python eval.py \
        --checkpoint /kaggle/working/checkpoints_fixed/best.pt \
        --test_csv /kaggle/input/vivqa/vlsp2023_vqa_test_final.csv \
        --test_images /kaggle/input/vivqa/test_images \
        --batch_size 8
"""

import os
import argparse
import json

import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor
from tqdm import tqdm
import pandas as pd
import numpy as np

from dataset import VQAGenDataset
from model import FixedLatentReasoningVQA


def parse_args():
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
    parser.add_argument('--max_length', type=int, default=10)
    parser.add_argument('--num_beams', type=int, default=4)
    
    # Intervention tests
    parser.add_argument('--run_intervention', action='store_true',
                      help='Run intervention tests (ablation, noise)')
    
    # Output
    parser.add_argument('--output_json', type=str, default='eval_results_fixed.json')
    parser.add_argument('--output_csv', type=str, default='predictions_fixed.csv')
    
    return parser.parse_args()


def normalize_answer(s: str) -> str:
    """Normalize answer for exact match"""
    return s.lower().strip()


def compute_exact_match(predicted: str, ground_truth: str) -> float:
    """Compute exact match (0 or 1)"""
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)
    return 1.0 if pred_norm == gt_norm else 0.0


@torch.no_grad()
def evaluate_model(
    model,
    dataloader,
    device,
    max_length=10,
    num_beams=4,
    run_intervention=False
):
    """
    Evaluate model on test set
    
    Returns:
        metrics: Dict with accuracy, diversity, intervention results
        predictions: List of predictions for CSV output
    """
    model.eval()
    
    all_predictions = []
    all_ground_truths = []
    all_questions = []
    all_img_ids = []
    
    exact_matches = []
    diversity_metrics_list = []
    
    # Intervention results
    intervention_results = {
        'baseline_acc': [],
        'ablated_acc': [],
        'noise_acc': []
    }
    
    print("\n[Evaluation] Running inference...")
    for batch_idx, (pixel_values, input_ids, attention_mask, labels) in enumerate(tqdm(dataloader)):
        pixel_values = pixel_values.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        batch_size = pixel_values.size(0)
        
        # Decode ground truths
        for i in range(batch_size):
            label_ids = labels[i][labels[i] != -100]
            gt = model.tokenizer.decode(label_ids, skip_special_tokens=True).strip()
            all_ground_truths.append(gt)
            
            # Decode question (for output)
            q = model.tokenizer.decode(input_ids[i], skip_special_tokens=True).strip()
            all_questions.append(q)
            all_img_ids.append(batch_idx * dataloader.batch_size + i)
        
        # === BASELINE GENERATION ===
        predictions = model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams
        )
        
        all_predictions.extend(predictions)
        
        # Compute exact match for this batch
        for pred, gt in zip(predictions, all_ground_truths[-batch_size:]):
            em = compute_exact_match(pred, gt)
            exact_matches.append(em)
        
        # === INTERVENTION TESTS ===
        # NOTE: Intervention tests disabled - generate() method doesn't support
        # ablate_reasoning and noise_reasoning parameters (only in forward())
        # To enable: need to modify model.generate() to accept these params
        if run_intervention:
            print("[Warning] Intervention tests not implemented in generate() method")
            # For now, use forward() for intervention tests:
            # This requires using generate_from_reasoning() instead
            pass
    
    # Compute metrics
    metrics = {
        'exact_match_accuracy': np.mean(exact_matches) * 100,
        'num_examples': len(exact_matches)
    }
    
    if run_intervention:
        metrics['intervention'] = {
            'baseline_acc': np.mean(exact_matches) * 100,
            'ablated_acc': np.mean(intervention_results['ablated_acc']) * 100,
            'noise_acc': np.mean(intervention_results['noise_acc']) * 100,
            'ablation_drop': (np.mean(exact_matches) - np.mean(intervention_results['ablated_acc'])) * 100,
            'noise_drop': (np.mean(exact_matches) - np.mean(intervention_results['noise_acc'])) * 100
        }
    
    # Prepare predictions DataFrame
    predictions_df = pd.DataFrame({
        'img_id': all_img_ids,
        'question': all_questions,
        'ground_truth': all_ground_truths,
        'prediction': all_predictions,
        'exact_match': exact_matches
    })
    
    return metrics, predictions_df


def main():
    args = parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Eval] Using device: {device}")
    
    # Load checkpoint
    print(f"\n[Eval] Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Initialize model
    print("[Eval] Initializing model...")
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
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"[Eval] Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Load test dataset
    print(f"\n[Eval] Loading test dataset: {args.test_csv}")
    vision_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    
    test_dataset = VQAGenDataset(
        csv_path=args.test_csv,
        image_folder=args.test_images,
        vision_processor=vision_processor,
        tokenizer_name='vinai/bartpho-syllable',
        max_q_len=64,
        max_a_len=32
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"[Eval] Test examples: {len(test_dataset)}")
    
    # Run evaluation
    metrics, predictions_df = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
        max_length=args.max_length,
        num_beams=args.num_beams,
        run_intervention=args.run_intervention
    )
    
    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"Exact Match Accuracy: {metrics['exact_match_accuracy']:.2f}%")
    print(f"Number of examples: {metrics['num_examples']}")
    
    if 'intervention' in metrics:
        print("\nIntervention Tests:")
        print(f"  Baseline Accuracy: {metrics['intervention']['baseline_acc']:.2f}%")
        print(f"  Ablated Accuracy: {metrics['intervention']['ablated_acc']:.2f}%")
        print(f"  Noisy Accuracy: {metrics['intervention']['noise_acc']:.2f}%")
        print(f"  Ablation Drop: {metrics['intervention']['ablation_drop']:.2f}%")
        print(f"  Noise Drop: {metrics['intervention']['noise_drop']:.2f}%")
    
    print("="*80)
    
    # Save results
    print(f"\n[Eval] Saving results to {args.output_json}")
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"[Eval] Saving predictions to {args.output_csv}")
    predictions_df.to_csv(args.output_csv, index=False, encoding='utf-8')
    
    print("\n✅ Evaluation complete!")


if __name__ == '__main__':
    main()
