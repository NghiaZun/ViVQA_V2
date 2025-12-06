"""
Quick test script - Infer 5 samples with TypeAwareVQAModel
"""
import os
import torch
from PIL import Image
import pandas as pd
from transformers import BlipProcessor

# Import models
from model import VQAGenModel
import importlib.util
spec = importlib.util.spec_from_file_location("train_module", "train_type_aware_distill.py")
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)
TypeAwareVQAModel = train_module.TypeAwareVQAModel

# === CONFIG ===
TEST_CSV = "/kaggle/input/vivqa/ViVQA-main/ViVQA-main/test.csv"
IMAGE_DIR = "/kaggle/input/vivqa/drive-download-20220309T020508Z-001/test"
MODEL_PATH = "/kaggle/input/best-model/transformers/default/1/vqa_type_aware_best.pt"  # ✅ Best model (epoch 14)
TOKENIZER_DIR = "/kaggle/input/student/transformers/default/1/checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[INFO] Device: {DEVICE}")

# === LOAD MODEL ===
print("[INFO] Loading TypeAwareVQAModel...")
base_model = VQAGenModel(
    vision_model_name="Salesforce/blip-vqa-base",
    phobert_dir=os.path.join(TOKENIZER_DIR, "phobert_tokenizer"),
    vit5_dir=os.path.join(TOKENIZER_DIR, "vit5_tokenizer")
)
model = TypeAwareVQAModel(base_model)

# Load checkpoint
print(f"[INFO] Loading checkpoint: {MODEL_PATH}")
state_dict = torch.load(MODEL_PATH, map_location='cpu')
if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
    model.load_state_dict(state_dict['model_state_dict'])
    print(f"[INFO] Loaded from epoch {state_dict.get('epoch', '?')}, val_loss={state_dict.get('best_val_loss', '?'):.4f}")
else:
    model.load_state_dict(state_dict)

model = model.to(DEVICE)
model.eval()
print(f"[INFO] Model ready! ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")

# === LOAD DATA ===
df = pd.read_csv(TEST_CSV)
vision_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
q_tokenizer = model.base_model.text_tokenizer
a_tokenizer = model.base_model.decoder_tokenizer

# === TEST 5 SAMPLES ===
print("\n" + "="*70)
print("TESTING 5 SAMPLES")
print("="*70)

for i in range(5):
    row = df.iloc[i]
    img_id = row['img_id']
    question = row['question']
    gt_answer = row['answer']
    
    # Load image
    img_path = os.path.join(IMAGE_DIR, f"{img_id}.jpg")
    try:
        image = Image.open(img_path).convert("RGB")
    except:
        print(f"\n[ERROR] Cannot load image: {img_path}")
        continue
    
    # Prepare inputs
    pixel_values = vision_processor(image, return_tensors="pt").pixel_values.to(DEVICE)
    q_enc = q_tokenizer(question, truncation=True, padding="max_length", max_length=64, return_tensors="pt")
    input_ids = q_enc.input_ids.to(DEVICE)
    attention_mask = q_enc.attention_mask.to(DEVICE)
    
    # Generate
    print(f"\n{'='*70}")
    print(f"Sample {i+1} - Image ID: {img_id}")
    print(f"{'='*70}")
    print(f"Question: {question}")
    print(f"GT Answer: {gt_answer}")
    
    with torch.no_grad():
        output_ids = model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=96,
            num_beams=4,
            early_stopping=True,
            do_sample=False
        )
    
    # Decode
    prediction = a_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    print(f"\n🤖 MODEL OUTPUT:")
    print(f"{prediction}")
    
    # Parse answer and reasoning
    import re
    answer_match = re.search(r'Answer:\s*(.+?)(?:\s+Reasoning:|$)', prediction, re.IGNORECASE | re.DOTALL)
    reasoning_match = re.search(r'Reasoning:\s*(.+?)$', prediction, re.IGNORECASE | re.DOTALL)
    
    if answer_match:
        pred_answer = answer_match.group(1).strip()
        print(f"\n✅ Parsed Answer: {pred_answer}")
        
        # Check if correct
        if pred_answer.lower().strip() == gt_answer.lower().strip():
            print(f"   ✅ CORRECT!")
        else:
            print(f"   ❌ WRONG (GT: {gt_answer})")
    else:
        print(f"\n⚠️  Could not parse answer")
    
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
        print(f"\n💭 Reasoning: {reasoning}")
    else:
        print(f"\n⚠️  No reasoning found")

print("\n" + "="*70)
print("DONE!")
print("="*70)
