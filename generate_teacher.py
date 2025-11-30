"""
teacher_generate.py – Generate teacher REASONING for ground truth answers
Author: Nghia-Duong (Option 2: GT-guided)

Strategy: Teacher nhìn image + question + GT answer → sinh reasoning giải thích
Ưu điểm: Reasoning khớp 100% với GT, student học cách GIẢI THÍCH đúng
"""

import os
import json
import re
import pandas as pd
from PIL import Image
import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq
from utils_prompt import SYSTEM_PROMPT, build_fewshot_prompt

# ===========================
# CONFIG
# ===========================
CSV_PATH = "/kaggle/input/vivqa/ViVQA-main/ViVQA-main/train.csv"  # Original dataset
IMAGE_DIR = "/kaggle/input/vivqa/drive-download-20220309T020508Z-001/train"
MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"  # Changed from 2.5 → 2 for 10x speed
OUT_JSONL = "/kaggle/working/teacher_outputs_gt_guided.jsonl"

# Reasoning type keywords để auto-classify
REASONING_KEYWORDS = {
    "COUNTING": ["bao nhiêu", "mấy", "số lượng", "đếm", "có bao nhiêu"],
    "SPATIAL": ["ở đâu", "vị trí", "phía", "trên", "dưới", "trong", "ngoài", "cạnh", "giữa"],
    "CAUSAL": ["tại sao", "vì sao", "lý do", "nguyên nhân", "để làm gì"],
    "OBJECT": ["cái gì", "con gì", "là gì", "vật gì", "đồ gì"],
    "INTENT": ["mục đích", "ý định", "để làm gì", "dùng để"],
    "COMMONSENSE": ["nên", "thường", "có thể", "phải"],
    "DESCRIPTIVE": []  # Default fallback
}

REASONING_WEIGHTS = {
    "CAUSAL": 5.0,
    "DESCRIPTIVE": 4.0,
    "INTENT": 4.0,
    "OBJECT": 2.0,
    "COUNTING": 2.0,
    "SPATIAL": 1.5,
    "COMMONSENSE": 1.0
}

def infer_reasoning_type(question: str) -> str:
    """Auto-classify reasoning type based on question keywords"""
    q_lower = question.lower().strip()
    
    for rtype, keywords in REASONING_KEYWORDS.items():
        if rtype == "DESCRIPTIVE":
            continue
        for kw in keywords:
            if kw in q_lower:
                return rtype
    
    return "DESCRIPTIVE"  # Default

# ===========================
# LOAD MODEL
# ===========================
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

# ===========================
# PARSE OUTPUT
# ===========================
def parse_structured_output(text: str):
    answer, reasoning, reasoning_type = "", "", ""
    lines = text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if line.startswith('Answer:'):
            answer = line.split(':', 1)[1].strip()
        elif line.startswith('Type:'):
            reasoning_type = line.split(':', 1)[1].strip().upper()
        elif line.startswith('Reasoning:'):
            reasoning = line.split(':', 1)[1].strip()
    
    return answer, reasoning, reasoning_type

# ===========================
# TEACHER GENERATION
# ===========================
def call_teacher_qwen(image_path: str, question: str, ground_truth: str):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[WARN] Cannot open image {image_path}: {e}")
        return {"answer": "", "reasoning": "", "reasoning_type": "", "raw": ""}

    # Simple format prompt
    user_prompt = f"""Câu hỏi: {question}
Đáp án: {ground_truth}

Hãy giải thích ngắn gọn TẠI SAO đáp án là "{ground_truth}" dựa vào hình ảnh.

Format:
Answer: {ground_truth}
Type: [COUNTING/SPATIAL/CAUSAL/OBJECT/INTENT/COMMONSENSE/DESCRIPTIVE]
Reasoning: (1 câu giải thích)

Chọn Type phù hợp:
- COUNTING: đếm số lượng
- SPATIAL: vị trí
- CAUSAL: nguyên nhân
- OBJECT: nhận diện vật
- DESCRIPTIVE: mô tả màu sắc/hình dạng
- COMMONSENSE: kiến thức chung
- INTENT: mục đích
"""

    enhanced_system_prompt = "Bạn là VQA model. Trả lời ngắn gọn theo format cho sẵn."

    messages = [
        {"role": "system", "content": enhanced_system_prompt},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": user_prompt}
        ]}
    ]

    try:
        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=80,  # Reduced from 200 → 2.5x faster
                do_sample=False,     # Greedy decoding → faster
                temperature=1.0
            )

        gen = processor.batch_decode(
            output[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0].strip()

        answer, reasoning, reasoning_type = parse_structured_output(gen)
        
        # Fallback: Nếu Qwen không output type, dùng heuristic
        if not reasoning_type:
            reasoning_type = infer_reasoning_type(question)
            # Debug để kiếm lỗi
            if len(results) < 5:  # Only log first 5 samples
                print(f"\n[FALLBACK] Question: {question[:30]}... → Type: {reasoning_type}")
                print(f"  Raw: {gen[:80]}...")
        
        return {
            "answer": answer,
            "reasoning": reasoning,
            "reasoning_type": reasoning_type,
            "raw": gen,
            "reasoning_weight": REASONING_WEIGHTS.get(reasoning_type, 1.0)
        }

    except Exception as e:
        print(f"[WARN] Generation failed: {e}")
        return {"answer": "", "reasoning": "", "reasoning_type": "", "raw": "", "reasoning_weight": 1.0}

# ===========================
# MAIN LOOP WITH AUTO-SAVE
# ===========================
df = pd.read_csv(CSV_PATH)
results = []

print(f"[INFO] Processing {len(df)} samples | Auto-save every 100")
print(f"[INFO] Using Qwen2-VL (faster than 2.5) → ETA: ~4h")

for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Teacher Generating")):
    image_id = str(row.get("img_id", row.get("image_id", ""))).strip()
    image_path = os.path.join(IMAGE_DIR, f"{image_id}.jpg")
    if not os.path.exists(image_path):
        continue

    q = str(row["question"]).strip()
    gt_answer = str(row["answer"]).strip()
    
    res = call_teacher_qwen(image_path, q, gt_answer)
    
    if res["answer"]:
        results.append({
            "img_id": image_id,
            "image_path": image_path,
            "question": q,
            "reasoning_type": res["reasoning_type"],
            "teacher_answer": res["answer"],
            "teacher_reasoning": res["reasoning"],
            "teacher_raw": res["raw"],
            "reasoning_weight": res["reasoning_weight"]
        })
    
    if (idx + 1) % 100 == 0:
        with open(OUT_JSONL, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

# Final save
with open(OUT_JSONL, "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"[DONE] {len(results)} samples → {OUT_JSONL}")
