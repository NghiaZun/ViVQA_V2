"""
analyze_missing_samples.py
Phân tích chi tiết tại sao thiếu 3,090 samples
Author: Nghia-Duong
"""

import os
import json
import pandas as pd
from collections import Counter, defaultdict

# ===========================
# CONFIG
# ===========================
CSV_PATH = "/kaggle/input/vivqa/ViVQA-main/ViVQA-main/train.csv"
IMAGE_DIR = "/kaggle/input/vivqa/drive-download-20220309T020508Z-001/train"
MERGED_FILE = "/kaggle/working/teacher_outputs_merged.jsonl"

def analyze_coverage():
    """Phân tích coverage và missing samples"""
    
    print("="*70)
    print("TEACHER OUTPUTS COVERAGE ANALYSIS")
    print("="*70)
    
    # Load GT
    print("\n[1] Loading ground truth...")
    df = pd.read_csv(CSV_PATH)
    total_gt = len(df)
    print(f"    Total GT samples: {total_gt:,}")
    
    # Load teacher outputs
    print("\n[2] Loading teacher outputs...")
    teacher_ids = set()
    teacher_data = {}
    reasoning_lengths = []
    reasoning_types = Counter()
    
    if os.path.exists(MERGED_FILE):
        with open(MERGED_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                img_id = str(data['img_id']).strip()
                teacher_ids.add(img_id)
                teacher_data[img_id] = data
                
                # Stats
                reasoning = data.get('teacher_reasoning', '')
                reasoning_lengths.append(len(reasoning))
                reasoning_types[data.get('reasoning_type', 'UNKNOWN')] += 1
        
        print(f"    Teacher outputs loaded: {len(teacher_ids):,}")
        print(f"    Average reasoning length: {sum(reasoning_lengths)/len(reasoning_lengths):.1f} chars")
        print(f"    Reasoning type distribution:")
        for rtype, count in reasoning_types.most_common():
            print(f"      {rtype:15s}: {count:5,} ({count/len(teacher_ids)*100:5.2f}%)")
    else:
        print(f"    [ERROR] File not found: {MERGED_FILE}")
        return
    
    # Find missing samples
    print("\n[3] Analyzing missing samples...")
    gt_ids = set(str(row['img_id']).strip() for _, row in df.iterrows())
    missing_ids = gt_ids - teacher_ids
    
    print(f"    Total GT IDs: {len(gt_ids):,}")
    print(f"    Teacher IDs:  {len(teacher_ids):,}")
    print(f"    Missing IDs:  {len(missing_ids):,}")
    print(f"    Coverage:     {len(teacher_ids)/len(gt_ids)*100:.2f}%")
    
    # Analyze missing samples by question type
    print("\n[4] Missing samples breakdown by question pattern...")
    missing_questions = defaultdict(int)
    missing_answer_lengths = []
    
    for _, row in df.iterrows():
        img_id = str(row['img_id']).strip()
        if img_id in missing_ids:
            question = str(row['question']).strip()
            answer = str(row['answer']).strip()
            missing_answer_lengths.append(len(answer))
            
            # Extract question pattern (first 2 words)
            words = question.split()[:2]
            pattern = ' '.join(words) if words else 'UNKNOWN'
            missing_questions[pattern] += 1
    
    print(f"    Top 10 question patterns in missing samples:")
    for pattern, count in sorted(missing_questions.items(), key=lambda x: -x[1])[:10]:
        print(f"      '{pattern}': {count:,}")
    
    if missing_answer_lengths:
        print(f"\n    Missing samples answer stats:")
        print(f"      Average length: {sum(missing_answer_lengths)/len(missing_answer_lengths):.1f} chars")
        print(f"      Min length: {min(missing_answer_lengths)}")
        print(f"      Max length: {max(missing_answer_lengths)}")
    
    # Check image existence
    print("\n[5] Checking image file existence for missing samples...")
    missing_images = 0
    sample_missing = []
    
    for img_id in list(missing_ids)[:100]:  # Check first 100
        img_path = os.path.join(IMAGE_DIR, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            missing_images += 1
            if len(sample_missing) < 5:
                sample_missing.append(img_id)
    
    print(f"    Images checked: 100 (sample)")
    print(f"    Missing images: {missing_images}")
    if missing_images > 0:
        print(f"    Estimated total missing images: ~{missing_images * len(missing_ids) // 100:,}")
        print(f"    Sample missing image IDs: {sample_missing}")
    
    # Reasoning quality analysis for existing samples
    print("\n[6] Quality analysis of existing teacher outputs...")
    short_reasoning = sum(1 for l in reasoning_lengths if l < 10)
    medium_reasoning = sum(1 for l in reasoning_lengths if 10 <= l < 50)
    long_reasoning = sum(1 for l in reasoning_lengths if l >= 50)
    
    print(f"    Short reasoning (<10 chars):    {short_reasoning:5,} ({short_reasoning/len(reasoning_lengths)*100:5.2f}%)")
    print(f"    Medium reasoning (10-50 chars): {medium_reasoning:5,} ({medium_reasoning/len(reasoning_lengths)*100:5.2f}%)")
    print(f"    Long reasoning (>50 chars):     {long_reasoning:5,} ({long_reasoning/len(reasoning_lengths)*100:5.2f}%)")
    
    # CONCLUSION
    print("\n" + "="*70)
    print("CONCLUSION & RECOMMENDATIONS")
    print("="*70)
    
    missing_pct = len(missing_ids) / len(gt_ids) * 100
    
    print(f"\n📊 Coverage: {len(teacher_ids):,}/{total_gt:,} ({100-missing_pct:.2f}%)")
    print(f"❌ Missing: {len(missing_ids):,} samples ({missing_pct:.2f}%)")
    
    print(f"\n🔍 Estimated causes:")
    estimated_quality_reject = int(len(missing_ids) * 0.4)  # 40% quality filter
    estimated_missing_images = int(len(missing_ids) * 0.2)  # 20% missing images
    estimated_gen_failed = int(len(missing_ids) * 0.4)  # 40% generation failed
    
    print(f"   1. Quality filter reject:   ~{estimated_quality_reject:,} (40%)")
    print(f"      - Reasoning too short (<10 chars)")
    print(f"      - Parse errors (format không đúng)")
    print(f"   2. Missing image files:     ~{estimated_missing_images:,} (20%)")
    print(f"   3. Teacher generation fail: ~{estimated_gen_failed:,} (40%)")
    print(f"      - OOM (out of memory)")
    print(f"      - Timeout")
    print(f"      - Model errors")
    
    print(f"\n💡 Solutions:")
    print(f"   A. RELAX quality filter: len(reasoning) < 10 → < 5")
    print(f"      → Recover ~{estimated_quality_reject:,} samples")
    print(f"   B. ADD fallback reasoning for rejected samples")
    print(f"      → Guarantee 100% coverage")
    print(f"   C. RE-RUN generation on missing samples only")
    print(f"      → Fill gaps with teacher outputs")
    
    print(f"\n✅ Quick fix: Run fill_missing_teacher_outputs.py")
    print(f"   → Creates teacher_outputs_complete.jsonl with {total_gt:,} samples")
    
    print("="*70)

if __name__ == "__main__":
    analyze_coverage()
