"""
Evaluate OptimalVQAModel checkpoint with Answer/Reasoning parser
Tests checkpoint from train_optimal.py
"""
import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import CLIPProcessor
from rouge_score import rouge_scorer
import re
import unicodedata
from PIL import Image

from model_optimal import OptimalVQAModel

# ======================
# TEXT NORMALIZATION
# ======================
def normalize_text(s):
    if s is None or not s:
        return ""
    s = s.lower().strip()
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ======================
# PARSER - EXTRACT ANSWER FROM "Answer: X Reasoning: Y"
# ======================
def parse_answer_reasoning(text: str):
    """
    Parse model output: "Answer: X Reasoning: Y"
    Returns dict with answer, reasoning
    """
    answer = ""
    reasoning = ""
    
    # Method 1: Split by "Answer:" then by "Reasoning:"
    if "Answer:" in text:
        text = text.split("Answer:")[-1]
    
    # Remove Reasoning part
    if "\nReasoning:" in text:
        parts = text.split("\nReasoning:")
        answer = parts[0].strip()
        reasoning = parts[1].strip() if len(parts) > 1 else ""
    elif "Reasoning:" in text:
        parts = text.split("Reasoning:")
        answer = parts[0].strip()
        reasoning = parts[1].strip() if len(parts) > 1 else ""
    else:
        answer = text.split("\n")[0].strip()
    
    return {
        'answer': answer,
        'reasoning': reasoning,
        'valid_format': bool(answer),
        'raw': text
    }


# ======================
# TOKEN LEVEL F1
# ======================
def token_f1(prediction, ground_truth):
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()

    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0

    common = set(pred_tokens) & set(gt_tokens)
    if len(common) == 0:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


# === CONFIG ===
TEST_CSV_PATH = "/kaggle/input/vivqa/ViVQA-main/ViVQA-main/test.csv"
IMAGE_FOLDER = "/kaggle/input/vivqa/drive-download-20220309T020508Z-001/test"
CHECKPOINT_PATH = "/kaggle/working/best_optimal_model.pt"  # ← CHANGE THIS to your checkpoint
CHECKPOINT_DIR = "/kaggle/input/base-model/transformers/default/1/checkpoints"
BATCH_SIZE = 4  # Lower for ViT-Large
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === LOAD MODEL ===
print(f"[INFO] Device: {DEVICE}")
print("[INFO] Loading OptimalVQAModel...")

model = OptimalVQAModel(
    vision_model_name="openai/clip-vit-large-patch14",
    phobert_dir=os.path.join(CHECKPOINT_DIR, "phobert_tokenizer"),
    vit5_dir=os.path.join(CHECKPOINT_DIR, "vit5_tokenizer"),
    hidden_dim=768,
    num_fusion_layers=4,
    num_heads=12,
    dropout=0.1,
    use_lora=True,
    use_type_routing=True
).to(DEVICE)

# Load checkpoint
print(f"[INFO] Loading checkpoint: {CHECKPOINT_PATH}")
if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    
    # Handle different formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', '?')
        val_loss = checkpoint.get('best_val_loss', '?')
        print(f"[INFO] Loaded checkpoint from epoch {epoch}, val_loss: {val_loss}")
    else:
        model.load_state_dict(checkpoint)
        print(f"[INFO] Loaded state dict directly")
else:
    print(f"[ERROR] Checkpoint not found: {CHECKPOINT_PATH}")
    exit(1)

model.eval()
print(f"[INFO] Total parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# === VISION PROCESSOR ===
vision_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# === LOAD TEST DATA ===
print("[INFO] Loading test data...")
test_df = pd.read_csv(TEST_CSV_PATH)
print(f"[INFO] Test samples: {len(test_df)}")

# === EVAL LOOP ===
print(f"[INFO] Running evaluation...")
refs, hyps = [], []
records = []
format_valid_count = 0

with torch.no_grad():
    for idx in tqdm(range(len(test_df)), desc="Evaluating"):
        row = test_df.iloc[idx]
        img_id = row['img_id']
        question = str(row['question'])
        gt_answer = str(row['answer'])
        
        # Load image
        img_path = os.path.join(IMAGE_FOLDER, f"{img_id}.jpg")
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = Image.new("RGB", (224, 224), (255, 255, 255))
        
        # Process inputs
        vision_inputs = vision_processor(images=image, return_tensors="pt")
        pixel_values = vision_inputs["pixel_values"].to(DEVICE)
        
        text_inputs = model.text_tokenizer(
            question, max_length=64, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        input_ids = text_inputs["input_ids"].to(DEVICE)
        attention_mask = text_inputs["attention_mask"].to(DEVICE)

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
        
        # Decode prediction
        pred_raw = model.decoder_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Parse prediction
        parsed = parse_answer_reasoning(pred_raw)

        # Store for metrics
        refs.append(gt_answer)
        hyps.append(parsed['answer'])
        
        if parsed['valid_format']:
            format_valid_count += 1
        
        records.append({
            "img_id": img_id,
            "question": question,
            "ground_truth": gt_answer,
            "predicted_raw": parsed['raw'][:200],  # Truncate for CSV
            "predicted_answer": parsed['answer'],
            "predicted_reasoning": parsed['reasoning'][:100] if parsed['reasoning'] else "",
            "valid_format": parsed['valid_format']
        })


# === METRICS ===
scorer = rouge_scorer.RougeScorer(["rouge1", "rougeLsum"], use_stemmer=True)

rouge1_list, rougel_list, f1_list = [], [], []
exact_matches = []

for ref, hyp in zip(refs, hyps):
    ref_n = normalize_text(ref)
    hyp_n = normalize_text(hyp)

    # ROUGE scores
    if hyp_n:
        scores = scorer.score(ref_n, hyp_n)
        rouge1_list.append(scores["rouge1"].fmeasure)
        rougel_list.append(scores["rougeLsum"].fmeasure)
    else:
        rouge1_list.append(0.0)
        rougel_list.append(0.0)
    
    # Token F1
    f1_list.append(token_f1(hyp, ref))
    
    # Exact match
    exact_matches.append(int(ref_n == hyp_n))

avg_rouge1 = np.mean(rouge1_list)
avg_rougel = np.mean(rougel_list)
avg_f1 = np.mean(f1_list)
acc = np.mean(exact_matches)


# === PRINT RESULTS ===
print("\n" + "="*70)
print("OPTIMAL MODEL TEST RESULTS")
print("="*70)
print(f"Checkpoint:          {CHECKPOINT_PATH}")
print(f"Total Samples:       {len(records)}")
print(f"Valid Format:        {format_valid_count}/{len(records)} ({100*format_valid_count/len(records):.2f}%)")
print(f"Accuracy (EM):       {acc*100:.2f}%")
print(f"Token F1:            {avg_f1:.4f}")
print(f"ROUGE-1 F1:          {avg_rouge1:.4f}")
print(f"ROUGE-L F1:          {avg_rougel:.4f}")
print("="*70)


# === SAVE CSV ===
out_csv = "/kaggle/working/eval_optimal_results.csv"
df = pd.DataFrame(records)
df.to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"\n[INFO] Detailed results saved to: {out_csv}")

# Save summary
summary_path = "/kaggle/working/eval_optimal_summary.txt"
with open(summary_path, 'w') as f:
    f.write(f"Checkpoint: {CHECKPOINT_PATH}\n")
    f.write(f"Accuracy: {acc*100:.2f}%\n")
    f.write(f"Token F1: {avg_f1:.4f}\n")
    f.write(f"ROUGE-1: {avg_rouge1:.4f}\n")
    f.write(f"ROUGE-L: {avg_rougel:.4f}\n")
print(f"[INFO] Summary saved to: {summary_path}")


# === PRINT SAMPLES ===
print("\n" + "="*70)
print("SAMPLE PREDICTIONS")
print("="*70)

# Show 5 correct predictions
correct_samples = [r for r, em in zip(records, exact_matches) if em == 1]
if correct_samples:
    print(f"\n✅ CORRECT PREDICTIONS ({len(correct_samples)} total):")
    for i, r in enumerate(correct_samples[:5], 1):
        print(f"\n{i}. Question: {r['question']}")
        print(f"   GT: {r['ground_truth']}")
        print(f"   PRED: {r['predicted_answer']}")
        if r['predicted_reasoning']:
            print(f"   REASONING: {r['predicted_reasoning']}...")

# Show 5 incorrect predictions
incorrect_samples = [r for r, em in zip(records, exact_matches) if em == 0]
if incorrect_samples:
    print(f"\n❌ INCORRECT PREDICTIONS ({len(incorrect_samples)} total):")
    for i, r in enumerate(incorrect_samples[:5], 1):
        print(f"\n{i}. Question: {r['question']}")
        print(f"   GT: {r['ground_truth']}")
        print(f"   PRED: {r['predicted_answer']}")
        if r['predicted_reasoning']:
            print(f"   REASONING: {r['predicted_reasoning']}...")

print("\n" + "="*70)
print(f"\n🎯 FINAL ACCURACY: {acc*100:.2f}%")
print("="*70)
