"""
Enhanced evaluation script with Answer/Reasoning parser
Evaluates model that outputs: "Answer: X Reasoning: Y" format
"""
import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, BlipProcessor

from rouge_score import rouge_scorer
import re
import unicodedata

from model import VQAGenModel

# Import TypeAwareVQAModel from training script
import sys
import importlib.util
spec = importlib.util.spec_from_file_location("train_module", "train_type_aware_distill.py")
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)
TypeAwareVQAModel = train_module.TypeAwareVQAModel


# ======================
# TEXT NORMALIZATION
# ======================
def normalize_text(s):
    if s is None or not s:
        return ""
    s = s.lower().strip()
    s = unicodedata.normalize("NFC", s)
    # Keep Vietnamese characters
    s = re.sub(r"[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ======================
# PARSER - SUPPORTS MULTIPLE FORMATS
# ======================
def parse_answer_reasoning(text: str):
    """
    Parse model output in multiple formats:
    1. "Answer: X Reasoning: Y" (standard)
    2. "X. Giải thích: Reasoning (Type): Y" (Vietnamese)
    Returns dict with answer, reasoning, and validity flag
    """
    answer = ""
    reasoning = ""
    
    # Method 1: Standard format "Answer: X"
    answer_match = re.search(r'Answer:\s*(.+?)(?:\s+Reasoning:|$)', text, re.IGNORECASE | re.DOTALL)
    reasoning_match = re.search(r'Reasoning:\s*(.+?)$', text, re.IGNORECASE | re.DOTALL)
    
    if answer_match:
        answer = answer_match.group(1).strip()
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    
    # Method 2: Vietnamese format "X. Giải thích: Reasoning (Type): Y"
    if not answer or not reasoning:
        vn_match = re.search(r'^(.+?)\.\s*Giải thích:\s*Reasoning\s*\([^)]+\):\s*(.+)$', text, re.DOTALL)
        if vn_match:
            answer = vn_match.group(1).strip()
            reasoning = vn_match.group(2).strip()
    
    # Method 3: Simpler - "X. Reasoning: Y" or "X. Giải thích: Y"
    if not answer or not reasoning:
        simple_match = re.search(r'^(.+?)\.\s*(?:Giải thích|Reasoning)[:\s]+(.+)$', text, re.IGNORECASE | re.DOTALL)
        if simple_match:
            answer = simple_match.group(1).strip()
            reasoning = simple_match.group(2).strip()
            # Remove "Reasoning (Type):" prefix if exists
            reasoning = re.sub(r'^Reasoning\s*\([^)]+\):\s*', '', reasoning, flags=re.IGNORECASE)
    
    # Method 4: Line-based fallback
    if not answer or not reasoning:
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        for line in lines:
            lower_line = line.lower()
            if lower_line.startswith('answer:'):
                answer = line.split(':', 1)[1].strip()
            elif lower_line.startswith('reasoning:'):
                reasoning = line.split(':', 1)[1].strip()
    
    # Method 5: If answer contains "Reasoning:", split it
    if "Reasoning:" in answer or "reasoning:" in answer:
        parts = re.split(r'\s+reasoning:\s*', answer, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) == 2:
            answer = parts[0].strip()
            reasoning = parts[1].strip()
    
    # Method 6: Last resort - if answer not found but text exists, use first sentence
    if not answer and text:
        first_part = text.split('.')[0].strip()
        if len(first_part) < 100:  # Reasonable answer length
            answer = first_part
    
    return {
        'answer': answer,
        'reasoning': reasoning,
        'valid_format': bool(answer and reasoning),
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
MODEL_PATH = "/kaggle/input/best-model/transformers/default/1/vqa_type_aware_best.pt"  # ✅ Best model (epoch 14)
TOKENIZER_DIR = "/kaggle/input/student/transformers/default/1/checkpoints"  # Directory with phobert/vit5 tokenizers
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === LOAD MODEL ===
print(f"[INFO] Device: {DEVICE}")
print("[INFO] Loading TypeAwareVQAModel (with type classifier)...")

# Load base model
base_model = VQAGenModel(
    vision_model_name="Salesforce/blip-vqa-base",
    phobert_dir=os.path.join(TOKENIZER_DIR, "phobert_tokenizer"),
    vit5_dir=os.path.join(TOKENIZER_DIR, "vit5_tokenizer")
)

# Wrap with TypeAwareVQAModel
model = TypeAwareVQAModel(base_model)

# Load checkpoint
print(f"[INFO] Loading checkpoint from: {MODEL_PATH}")
state_dict = torch.load(MODEL_PATH, map_location='cpu')

# Handle different checkpoint formats
if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
    # Full checkpoint with optimizer etc.
    model.load_state_dict(state_dict['model_state_dict'])
    print(f"[INFO] Loaded from full checkpoint (epoch {state_dict.get('epoch', '?')})")
elif isinstance(state_dict, dict) and 'model' in state_dict:
    # Legacy format
    model.load_state_dict(state_dict['model'])
else:
    # Direct state dict
    model.load_state_dict(state_dict)

model = model.to(DEVICE)
model.eval()
print("[INFO] Model loaded successfully!")
print(f"[INFO] Total parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# === TOKENIZERS ===
# Access tokenizers from base_model (wrapped inside TypeAwareVQAModel)
q_tokenizer = model.base_model.text_tokenizer
a_tokenizer = model.base_model.decoder_tokenizer

# === LOAD TEST DATA ===
print("[INFO] Loading test data...")
test_df = pd.read_csv(TEST_CSV_PATH)
vision_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

# === EVAL LOOP ===
print(f"[INFO] Running evaluation on {len(test_df)} samples...")
refs, hyps = [], []
records = []
format_valid_count = 0

from PIL import Image

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
        pixel_values = vision_processor(image, return_tensors="pt").pixel_values.to(DEVICE)
        q_enc = q_tokenizer(question, truncation=True, padding="max_length", max_length=64, return_tensors="pt")
        input_ids = q_enc.input_ids.to(DEVICE)
        attention_mask = q_enc.attention_mask.to(DEVICE)

        # Generate
        output_ids = model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=96,
            num_beams=4,
            early_stopping=True
        )
        
        # Decode prediction
        pred_raw = a_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Parse prediction
        parsed = parse_answer_reasoning(pred_raw)

        # Store for metrics
        refs.append(gt_answer)
        hyps.append(parsed['answer'])  # Use only answer for comparison
        
        if parsed['valid_format']:
            format_valid_count += 1
        
        records.append({
            "ground_truth": gt_answer,
            "predicted_raw": parsed['raw'],
            "predicted_answer": parsed['answer'],
            "predicted_reasoning": parsed['reasoning'],
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
print("\n" + "="*60)
print("TEST RESULTS")
print("="*60)
print(f"Total Samples:       {len(records)}")
print(f"Valid Format:        {format_valid_count}/{len(records)} ({100*format_valid_count/len(records):.2f}%)")
print(f"Accuracy (EM):       {acc*100:.2f}%")
print(f"Token F1:            {avg_f1:.4f}")
print(f"ROUGE-1 F1:          {avg_rouge1:.4f}")
print(f"ROUGE-L F1:          {avg_rougel:.4f}")
print("="*60)


# === SAVE CSV ===
out_csv = "/kaggle/working/eval_vqa_parsed_results.csv"
df = pd.DataFrame(records)
df.to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"\n[INFO] Results saved to: {out_csv}")


# === PRINT SAMPLES ===
print("\n" + "="*60)
print("SAMPLE PREDICTIONS")
print("="*60)

# Show 5 correct predictions
correct_samples = [r for r, em in zip(records, exact_matches) if em == 1]
if correct_samples:
    print("\n✅ CORRECT PREDICTIONS:")
    for i, r in enumerate(correct_samples[:5], 1):
        print(f"\n{i}. GT: {r['ground_truth']}")
        print(f"   PRED: {r['predicted_answer']}")
        print(f"   REASONING: {r['predicted_reasoning'][:80]}..." if len(r['predicted_reasoning']) > 80 else f"   REASONING: {r['predicted_reasoning']}")
        print(f"   Format OK: {'✅' if r['valid_format'] else '❌'}")

# Show 5 incorrect predictions
incorrect_samples = [r for r, em in zip(records, exact_matches) if em == 0]
if incorrect_samples:
    print("\n❌ INCORRECT PREDICTIONS:")
    for i, r in enumerate(incorrect_samples[:5], 1):
        print(f"\n{i}. GT: {r['ground_truth']}")
        print(f"   PRED: {r['predicted_answer']}")
        print(f"   REASONING: {r['predicted_reasoning'][:80]}..." if len(r['predicted_reasoning']) > 80 else f"   REASONING: {r['predicted_reasoning']}")
        print(f"   Format OK: {'✅' if r['valid_format'] else '❌'}")

# Show samples with invalid format
invalid_format = [r for r in records if not r['valid_format']]
if invalid_format:
    print(f"\n⚠️ INVALID FORMAT SAMPLES ({len(invalid_format)} total):")
    for i, r in enumerate(invalid_format[:3], 1):
        print(f"\n{i}. RAW OUTPUT: {r['predicted_raw'][:150]}...")
        print(f"   GT: {r['ground_truth']}")

print("\n" + "="*60)
