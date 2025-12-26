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
    
    results = []
    reasoning_confidences = []
    
    print("[INFO] Generating predictions...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Generate reasoning và answer
            reasoning_text, answer_text, confidence = model.generate(
                pixel_values=batch['pixel_values'],
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_reasoning_len=128,
                max_answer_len=32,
                num_beams=4
            )
            
            results.extend([{
                'reasoning': r,
                'answer': a
            } for r, a in zip(reasoning_text, answer_text)])
            
            if confidence is not None:
                reasoning_confidences.extend(confidence.cpu().numpy())
    
    # Statistics
    stats = {
        'total_samples': len(results),
        'avg_reasoning_confidence': np.mean(reasoning_confidences) if reasoning_confidences else None,
        'std_reasoning_confidence': np.std(reasoning_confidences) if reasoning_confidences else None,
    }
    
    return results, stats


def main():
    # Config
    CONFIG = {
        'checkpoint_path': '/kaggle/working/checkpoints_dinov2_bartpho/best_model_stage4.pt',
        'test_json': '/kaggle/input/teacher-5-12/teacher_outputs_train.jsonl',  # Replace với test set
        'image_dir': '/kaggle/input/vivqa/drive-download-20220309T020508Z-001/train',
        'output_json': '/kaggle/working/predictions.json',
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
    
    # Load test data
    print("\n[INFO] Loading test dataset...")
    test_dataset = VQADistillationDataset(
        json_path=CONFIG['test_json'],
        image_dir=CONFIG['image_dir'],
        vision_processor=model.vision_processor,
        tokenizer=model.tokenizer,
        augment=False
    )
    
    # Evaluate
    results, stats = evaluate_model(
        model=model,
        test_dataset=test_dataset,
        batch_size=CONFIG['batch_size'],
        device=device
    )
    
    # Save results
    with open(CONFIG['output_json'], 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n[INFO] ✓ Predictions saved to {CONFIG['output_json']}")
    print("\nStatistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
