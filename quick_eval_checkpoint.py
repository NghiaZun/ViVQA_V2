"""
Quick evaluation script to test checkpoint accuracy
Run this to verify model performance at current epoch
"""

import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from model_optimal import OptimalVQAModel
from transformers import CLIPProcessor
from train_optimal import CurriculumVQADataset

def quick_eval(
    checkpoint_path="/kaggle/working/latest_checkpoint_optimal.pt",
    checkpoint_dir="/kaggle/input/base-model/transformers/default/1/checkpoints",
    test_csv="/kaggle/input/vivqa/ViVQA-main/ViVQA-main/train.csv",
    image_dir="/kaggle/input/vivqa/drive-download-20220309T020508Z-001/train",
    teacher_jsonl="/kaggle/input/teacher-final/teacher_outputs_train.jsonl",
    num_samples=500,  # Test on subset for speed
):
    """Quick evaluation on validation set"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model
    print("\n[INFO] Loading model...")
    model = OptimalVQAModel(
        vision_model_name="openai/clip-vit-large-patch14",
        phobert_dir=os.path.join(checkpoint_dir, "phobert_tokenizer"),
        vit5_dir=os.path.join(checkpoint_dir, "vit5_tokenizer"),
        hidden_dim=768,
        num_fusion_layers=4,
        num_heads=12,
        dropout=0.1,
        use_lora=True,
        use_type_routing=True
    ).to(device)
    
    # Load checkpoint
    if os.path.exists(checkpoint_path):
        print(f"[INFO] Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        best_val_loss = checkpoint.get('best_val_loss', 'unknown')
        print(f"✅ Checkpoint from epoch {epoch}, best val loss: {best_val_loss}")
    else:
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    vision_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    # Load dataset
    print("\n[INFO] Loading dataset...")
    dataset = CurriculumVQADataset(
        test_csv, image_dir, teacher_jsonl,
        vision_processor, model.text_tokenizer, model.decoder_tokenizer,
        max_len=128, use_augmentation=False, difficulty_level='all'
    )
    
    # Use validation split (last 10%)
    n_val = int(len(dataset) * 0.1)
    _, val_dataset = torch.utils.data.random_split(
        dataset, [len(dataset) - n_val, n_val]
    )
    
    # Limit samples for speed
    if num_samples and num_samples < len(val_dataset):
        val_dataset = torch.utils.data.Subset(val_dataset, range(num_samples))
    
    val_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False, num_workers=2
    )
    
    print(f"[INFO] Evaluating on {len(val_dataset)} samples...")
    
    # Evaluate
    model.eval()
    correct = 0
    total = 0
    samples_shown = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Generate
            output_ids = model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=96,
                num_beams=4,
                length_penalty=1.2,
                early_stopping=True
            )
            
            # Decode
            predictions = model.decoder_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            gt_answers = model.decoder_tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            # Check accuracy
            for pred, gt in zip(predictions, gt_answers):
                pred_answer = pred.split("Answer:")[-1].split("\n")[0].strip().lower()
                gt_answer = gt.split("Answer:")[-1].split("\n")[0].strip().lower()
                
                # Normalize
                pred_answer = ' '.join(pred_answer.split())
                gt_answer = ' '.join(gt_answer.split())
                
                is_correct = (pred_answer == gt_answer or 
                            gt_answer in pred_answer or 
                            pred_answer in gt_answer)
                
                if is_correct:
                    correct += 1
                total += 1
                
                # Show first 5 samples
                if samples_shown < 5:
                    status = "✅ CORRECT" if is_correct else "❌ WRONG"
                    print(f"\nSample {samples_shown + 1} {status}:")
                    print(f"  Prediction: {pred_answer}")
                    print(f"  Ground Truth: {gt_answer}")
                    samples_shown += 1
    
    # Results
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Total samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("="*70)
    
    return accuracy

if __name__ == "__main__":
    quick_eval()
