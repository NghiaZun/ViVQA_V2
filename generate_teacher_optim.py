"""
teacher_generate.py – STABLE version với tối ưu đơn giản
Author: Nghia-Duong (stable + faster)
"""

import os
import json
import re
import pandas as pd
from PIL import Image
import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq

# ===========================
# CONFIG
# ===========================
CSV_PATH = "/kaggle/input/vivqa/ViVQA-main/ViVQA-main/train.csv"  # GT-guided dataset
IMAGE_DIR = "/kaggle/input/vivqa/drive-download-20220309T020508Z-001/train"
MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
OUT_JSONL = "/kaggle/working/teacher_outputs_gt_guided.jsonl"

# Reasoning type keywords for auto-classification
REASONING_KEYWORDS = {
    "COUNTING": ["bao nhiêu", "mấy", "số lượng", "đếm"],
    "SPATIAL": ["ở đâu", "vị trí", "phía", "trên", "dưới", "trong", "ngoài"],
    "CAUSAL": ["tại sao", "vì sao", "lý do", "nguyên nhân"],
    "OBJECT": ["cái gì", "con gì", "là gì", "vật gì"],
    "INTENT": ["mục đích", "ý định", "dùng để"],
    "COMMONSENSE": ["nên", "thường", "có thể", "phải"],
    "DESCRIPTIVE": []
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
    """Auto-classify reasoning type from question"""
    q_lower = question.lower().strip()
    for rtype, keywords in REASONING_KEYWORDS.items():
        if rtype == "DESCRIPTIVE":
            continue
        for kw in keywords:
            if kw in q_lower:
                return rtype
    return "DESCRIPTIVE"

# ===========================
# LOAD MODEL - ĐƠN GIẢN HÓA
# ===========================
device = "cuda:0"  # Chỉ dùng GPU đầu tiên cho ổn định
print(f"[INFO] Using device: {device}")

processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",  # Để tự động chọn, nhưng sẽ ưu tiên GPU 0
    trust_remote_code=True,
    low_cpu_mem_usage=True
)
model.eval()

# ===========================
# PARSE OUTPUT - SIMPLE FORMAT
# ===========================
def parse_structured_output(text: str, question: str = ""):
    """Parse simple format: Answer: X / Type: Y / Reasoning: Z với validation"""
    answer, reasoning, reasoning_type = "", "", ""
    lines = text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if line.startswith('Answer:'):
            answer = line.split(':', 1)[1].strip()
        elif line.startswith('Type:'):
            reasoning_type = line.split(':', 1)[1].strip().upper()
            # Clean type: chỉ lấy keyword đầu tiên
            reasoning_type = reasoning_type.split()[0] if reasoning_type else ""
        elif line.startswith('Reasoning:'):
            reasoning = line.split(':', 1)[1].strip()
    
    # VALIDATION: Lọc reasoning xấu
    bad_patterns = [
        "(1 câu giải thích)", "(1 câu)", "...", 
        "Chọn Type", "Format:", "BẮT BUỘC"
    ]
    if reasoning and any(bad in reasoning for bad in bad_patterns):
        reasoning = ""  # Mark as invalid
    
    # Fallback to heuristic if no type found
    if not reasoning_type and question:
        reasoning_type = infer_reasoning_type(question)
    
    # Validate type
    valid_types = ["COUNTING", "SPATIAL", "CAUSAL", "OBJECT", "DESCRIPTIVE", "COMMONSENSE", "INTENT"]
    if reasoning_type not in valid_types:
        reasoning_type = infer_reasoning_type(question)
    
    return answer, reasoning, reasoning_type

# ===========================
# TEACHER GENERATION - GT-GUIDED + OPTIMIZED
# ===========================
@torch.no_grad()
def call_teacher_qwen(image_path: str, question: str, ground_truth: str):
    """GT-guided: Teacher explains WHY answer is ground_truth"""
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return None

    # IMPROVED: Cleaner prompt với examples để model follow tốt hơn
    user_prompt = f"""Dựa vào hình ảnh, trả lời câu hỏi với format chính xác:

Câu hỏi: {question}
Đáp án đúng: {ground_truth}

Viết giải thích TẠI SAO đáp án là "{ground_truth}".

BẮT BUỘC format (3 dòng):
Answer: {ground_truth}
Type: [COUNTING hoặc SPATIAL hoặc CAUSAL hoặc OBJECT hoặc DESCRIPTIVE hoặc COMMONSENSE hoặc INTENT]
Reasoning: [Giải thích dựa vào hình ảnh, 1 câu hoàn chỉnh]

Ví dụ:
Answer: màu xanh lá
Type: DESCRIPTIVE
Reasoning: Hình ảnh cho thấy chiếc xe buýt có màu xanh lá.

Bây giờ trả lời:"""

    enhanced_system_prompt = "Bạn là trợ lý VQA chuyên nghiệp. Luôn tuân thủ format 3 dòng: Answer, Type, Reasoning."

    messages = [
        {"role": "system", "content": enhanced_system_prompt},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": user_prompt}
        ]}
    ]

    try:
        text_prompt = processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = processor(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(device)

        # Mixed precision + optimized generation
        with torch.amp.autocast('cuda'):
            output = model.generate(
                **inputs,
                max_new_tokens=100,       # Tăng lên để đủ chỗ cho reasoning đầy đủ
                min_new_tokens=30,        # Đảm bảo sinh đủ 3 dòng
                do_sample=False,          # Greedy = faster + deterministic
                temperature=1.0,
                use_cache=True,
                repetition_penalty=1.1,   # Tránh lặp lại
                pad_token_id=processor.tokenizer.pad_token_id
            )

        gen = processor.batch_decode(
            output[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0].strip()

        answer, reasoning, reasoning_type = parse_structured_output(gen, question)

        # QUALITY CHECK: Đảm bảo reasoning hợp lệ
        if not reasoning or len(reasoning) < 10:
            # Retry với prompt đơn giản hơn nếu lần đầu fail
            return None
        
        # Clean raw output: loại bỏ phần hướng dẫn thừa
        clean_raw = "\n".join([
            f"Answer: {answer}",
            f"Type: {reasoning_type}",
            f"Reasoning: {reasoning}"
        ])

        return {
            "answer": answer,
            "reasoning": reasoning,
            "reasoning_type": reasoning_type,
            "raw": clean_raw,  # Lưu clean version thay vì raw
            "reasoning_weight": REASONING_WEIGHTS.get(reasoning_type, 1.0)
        }

    except Exception as e:
        print(f"[ERROR] Generation failed for {image_path}: {e}")
        return None

# ===========================
# MAIN LOOP - CẢI THIỆN
# ===========================
df = pd.read_csv(CSV_PATH)
results = []

# RESUME từ checkpoint nếu có
processed_ids = set()
if os.path.exists(OUT_JSONL):
    print(f"[INFO] 🔄 Found existing checkpoint: {OUT_JSONL}")
    with open(OUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                results.append(r)
                processed_ids.add(r["img_id"])
            except:
                continue
    print(f"[INFO] ✅ Resumed with {len(results)} existing samples")

# Periodic save để tránh mất dữ liệu
SAVE_INTERVAL = 50  # Save thường xuyên hơn (mỗi 50 samples)
failed_samples = 0  # Track failed generations

print(f"[INFO] Total samples to process: {len(df)} | Already done: {len(processed_ids)}")
print(f"[INFO] Quality filters enabled: reasoning validation + format check")

try:
    for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="GT-Guided Teacher")):
        image_id = str(row.get("img_id", row.get("image_id", ""))).strip()
        
        # SKIP nếu đã xử lý rồi
        if image_id in processed_ids:
            continue
        
        image_path = os.path.join(IMAGE_DIR, f"{image_id}.jpg")
        
        if not os.path.exists(image_path):
            continue

        q = str(row["question"]).strip()
        gt_answer = str(row["answer"]).strip()  # Ground truth

        res = call_teacher_qwen(image_path, q, gt_answer)

        if res and res["answer"] and res["reasoning"]:  # STRICTER: Phải có cả answer VÀ reasoning
            new_entry = {
                "img_id": image_id,
                "image_path": image_path,
                "question": q,
                "reasoning_type": res["reasoning_type"],
                "teacher_answer": res["answer"],
                "teacher_reasoning": res["reasoning"],
                "teacher_raw": res["raw"],
                "reasoning_weight": res["reasoning_weight"]
            }
            results.append(new_entry)
            processed_ids.add(image_id)
            
            # APPEND mode: Save ngay lập tức sau mỗi sample thành công
            with open(OUT_JSONL, "a", encoding="utf-8") as f:
                f.write(json.dumps(new_entry, ensure_ascii=False) + "\n")
        else:
            failed_samples += 1
            if failed_samples <= 5:  # Log first 5 failures
                print(f"\n[SKIP] Failed sample: {image_id} | Q: {q[:40]}...")
        
        # Progress report định kỳ
        if len(results) % SAVE_INTERVAL == 0 and len(results) > 0:
            print(f"\n[INFO] 💾 Progress: {len(results)} samples saved | Failed: {failed_samples}")
        
        # Memory management mỗi 100 samples
        if idx % 100 == 0:
            torch.cuda.empty_cache()
            import gc
            gc.collect()  # Python garbage collection

except KeyboardInterrupt:
    print(f"\n[WARN] ⚠️ Interrupted by user! Saving progress...")
    print(f"[INFO] Saved {len(results)} samples before interruption")
finally:
    # Final report (file đã được save liên tục rồi, không cần save lại)
    print(f"\n[INFO] ✅ Completed! Total saved: {len(results)}/{len(df)} teacher samples → {OUT_JSONL}")
    if len(results) > 0:
        print(f"[INFO] Success rate: {len(results)/len(df)*100:.1f}% | Failed: {failed_samples}")
        print(f"[INFO] Average reasoning length: {sum(len(r['teacher_reasoning']) for r in results)/len(results):.1f} chars")
    else:
        print(f"[WARN] No valid samples generated!")
