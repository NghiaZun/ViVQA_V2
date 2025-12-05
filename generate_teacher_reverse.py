"""
teacher_generate.py – STABLE version với tối ưu đơn giản (REVERSE ORDER)
Author: Nghia-Duong (stable + faster)
Chạy từ cuối CSV lên đầu
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
OUT_JSONL = "/kaggle/working/teacher_outputs_gt_guided_reverse.jsonl"

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
# TEACHER GENERATION - GT-GUIDED + OPTIMIZED + RETRY
# ===========================
@torch.no_grad()
def call_teacher_qwen(image_path: str, question: str, ground_truth: str, max_retries=3):
    """GT-guided: Teacher explains WHY answer is ground_truth
    
    Args:
        image_path: Path to image
        question: Question text
        ground_truth: Ground truth answer
        max_retries: Number of retry attempts with same prompt (default: 3)
    
    Returns:
        Dict with answer, reasoning, type, raw, weight or None
    """
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

        # QUALITY CHECK: Nếu parse fail → RETRY với CÙNG prompt chuẩn (max 3 lần)
        retry_count = 0
        while (not reasoning or len(reasoning) < 5 or not reasoning_type) and retry_count < max_retries:
            retry_count += 1
            
            try:
                print(f"[RETRY {retry_count}/{max_retries}] Retrying generation for same prompt...")
                
                # RETRY với CÙNG prompt và inputs (model có thể generate khác)
                with torch.amp.autocast('cuda'):
                    retry_output = model.generate(
                        **inputs,
                        max_new_tokens=100,
                        min_new_tokens=30,
                        do_sample=False,  # Giữ deterministic
                        temperature=1.0,
                        use_cache=True,
                        repetition_penalty=1.1,
                        pad_token_id=processor.tokenizer.pad_token_id
                    )
                
                retry_gen = processor.batch_decode(
                    retry_output[:, inputs.input_ids.shape[1]:],
                    skip_special_tokens=True
                )[0].strip()
                
                answer_retry, reasoning_retry, type_retry = parse_structured_output(retry_gen, question)
                
                # Nếu retry thành công → dùng kết quả retry
                if reasoning_retry and len(reasoning_retry) >= 5 and type_retry:
                    answer, reasoning, reasoning_type = answer_retry, reasoning_retry, type_retry
                    print(f"[SUCCESS] Retry {retry_count} succeeded!")
                    break  # Thoát khỏi retry loop
                else:
                    print(f"[RETRY {retry_count}] Still invalid, trying again...")
                    
            except Exception as retry_error:
                print(f"[ERROR] Retry {retry_count} failed: {retry_error}")
                continue  # Thử retry tiếp
        
        # FINAL FALLBACK: Nếu vẫn không có reasoning hợp lệ
        if not reasoning or len(reasoning) < 5:
            reasoning = f"Dựa vào hình ảnh, câu trả lời là {ground_truth}."
            if not reasoning_type:
                reasoning_type = infer_reasoning_type(question)
        
        # Validate answer không empty
        if not answer or not answer.strip():
            answer = ground_truth  # Force answer = GT
        
        # Clean raw output
        clean_raw = "\n".join([
            f"Answer: {answer}",
            f"Type: {reasoning_type}",
            f"Reasoning: {reasoning}"
        ])

        return {
            "answer": answer,
            "reasoning": reasoning,
            "reasoning_type": reasoning_type,
            "raw": clean_raw,
            "reasoning_weight": REASONING_WEIGHTS.get(reasoning_type, 1.0)
        }

    except Exception as e:
        print(f"[ERROR] Generation completely failed: {e}")
        return None

# ===========================
# MAIN LOOP - CẢI THIỆN (REVERSE ORDER)
# ===========================
df = pd.read_csv(CSV_PATH)
results = []

# RESUME từ checkpoint hoặc output file
processed_ids = set()

# Tự động tìm file teacher_outputs_merged_reverse trong /kaggle/input
def find_teacher_merged_reverse_file():
    """Đào sâu tìm file teacher_outputs_merged_reverse* trong /kaggle/input"""
    kaggle_input = "/kaggle/input"
    if not os.path.exists(kaggle_input):
        return None
    
    print(f"[INFO] 🔍 Searching for teacher_outputs_merged_reverse* in {kaggle_input}...")
    
    for root, dirs, files in os.walk(kaggle_input):
        for file in files:
            if file.startswith("teacher_outputs_merged_reverse") and file.endswith(".jsonl"):
                found_path = os.path.join(root, file)
                print(f"[INFO] ✅ Found merged reverse file: {found_path}")
                return found_path
    
    print(f"[INFO] ⚠️  No teacher_outputs_merged_reverse*.jsonl found in {kaggle_input}")
    return None

# Tìm file merged reverse hoặc fallback về output file hiện tại
resume_from = None
merged_reverse_file = find_teacher_merged_reverse_file()

if merged_reverse_file:
    resume_from = merged_reverse_file
    print(f"[INFO] 🔄 Resuming from merged reverse checkpoint: {merged_reverse_file}")
elif os.path.exists(OUT_JSONL):
    resume_from = OUT_JSONL
    print(f"[INFO] 🔄 Found existing output: {OUT_JSONL}")

if resume_from:
    with open(resume_from, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                results.append(r)
                img_id = str(r.get("img_id", "")).strip()
                if img_id:
                    processed_ids.add(img_id)
            except Exception as e:
                continue
    print(f"[INFO] ✅ Resumed with {len(results)} existing samples (unique IDs: {len(processed_ids)})")
    
    # Nếu resume từ checkpoint khác với output file, cần merge
    if resume_from != OUT_JSONL and os.path.exists(OUT_JSONL):
        print(f"[WARN] ⚠️  Both checkpoint and output file exist!")
        print(f"[WARN] Consider running merge_teacher_outputs.py first to avoid duplicates")
else:
    print(f"[INFO] Starting fresh - no existing data found")

# Periodic save để tránh mất dữ liệu
SAVE_INTERVAL = 50  # Save thường xuyên hơn (mỗi 50 samples)
failed_samples = 0  # Track failed generations

print(f"[INFO] Total samples to process: {len(df)} | Already done: {len(processed_ids)}")
print(f"[INFO] Processing in REVERSE order (from end to start)")
print(f"[INFO] Quality filters enabled: reasoning validation + format check")

try:
    # Chạy từ cuối lên đầu (reverse order)
    for idx, (_, row) in enumerate(tqdm(df.iloc[::-1].iterrows(), total=len(df), desc="GT-Guided Teacher (Reverse)")):
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

        # ALWAYS save - even with fallback reasoning (GT-guided guarantee!)
        if res:
            # Teacher generated successfully (có fallback nếu reasoning ngắn)
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
            # Teacher generation hoàn toàn failed - tạo minimal fallback
            failed_samples += 1
            if failed_samples <= 5:  # Log first 5 failures
                print(f"\n[WARN] Teacher failed for {image_id}, creating fallback...")
            
            # Create minimal fallback entry (GT-guided: answer = GT)
            reasoning_type = infer_reasoning_type(q)
            fallback_entry = {
                "img_id": image_id,
                "image_path": image_path,
                "question": q,
                "reasoning_type": reasoning_type,
                "teacher_answer": gt_answer,  # GT-guided
                "teacher_reasoning": f"Dựa vào hình ảnh, câu trả lời là {gt_answer}.",
                "teacher_raw": f"Answer: {gt_answer}\nType: {reasoning_type}\nReasoning: Dựa vào hình ảnh, câu trả lời là {gt_answer}.",
                "reasoning_weight": REASONING_WEIGHTS.get(reasoning_type, 1.0),
                "_fallback": True  # Flag to track fallback entries
            }
            results.append(fallback_entry)
            processed_ids.add(image_id)
            
            # Save fallback entry
            with open(OUT_JSONL, "a", encoding="utf-8") as f:
                f.write(json.dumps(fallback_entry, ensure_ascii=False) + "\n")
        
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
        # Count fallback entries
        fallback_count = sum(1 for r in results if r.get('_fallback', False))
        teacher_generated = len(results) - fallback_count
        
        print(f"[INFO] Coverage: {len(results)/len(df)*100:.1f}%")
        print(f"[INFO] Teacher-generated: {teacher_generated} ({teacher_generated/len(results)*100:.1f}%)")
        print(f"[INFO] Fallback entries: {fallback_count} ({fallback_count/len(results)*100:.1f}%)")
        print(f"[INFO] Average reasoning length: {sum(len(r['teacher_reasoning']) for r in results)/len(results):.1f} chars")
    else:
        print(f"[WARN] No valid samples generated!")
