"""
Check missing samples between CSV and teacher outputs
"""

import pandas as pd
import json
import os

# Paths (update for local or Kaggle)
CSV_PATH = "/kaggle/input/vivqa/ViVQA-main/ViVQA-main/train.csv"
TEACHER_JSONL = "/kaggle/input/d/dngtrungngha25/teacher-checkpoint-11k/teacher_outputs_train.jsonl"

print(f"[INFO] Loading CSV: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
total_samples = len(df)

print(f"[INFO] Loading teacher outputs: {TEACHER_JSONL}")
processed_ids = set()

with open(TEACHER_JSONL, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            entry = json.loads(line)
            img_id = str(entry.get('img_id', '')).strip()
            if img_id:
                processed_ids.add(img_id)
        except:
            continue

generated_count = len(processed_ids)
missing_count = total_samples - generated_count

print(f"\n{'='*70}")
print(f"COVERAGE ANALYSIS")
print(f"{'='*70}")
print(f"Total samples in CSV: {total_samples:,}")
print(f"Generated samples:    {generated_count:,}")
print(f"Missing samples:      {missing_count:,}")
print(f"Coverage:             {generated_count/total_samples*100:.2f}%")
print(f"{'='*70}")

if missing_count > 0:
    print(f"\n[INFO] Finding missing sample IDs...")
    missing_ids = []
    for _, row in df.iterrows():
        img_id = str(row.get('img_id', row.get('image_id', ''))).strip()
        if img_id not in processed_ids:
            missing_ids.append(img_id)
    
    print(f"\n[INFO] First 20 missing IDs:")
    for i, img_id in enumerate(missing_ids[:20], 1):
        print(f"  {i}. {img_id}")
    
    if len(missing_ids) > 20:
        print(f"  ... and {len(missing_ids) - 20} more")
    
    # Save to file
    output_file = "missing_sample_ids.txt"
    with open(output_file, 'w') as f:
        f.write('\n'.join(missing_ids))
    print(f"\n[INFO] ✅ All missing IDs saved to: {output_file}")
else:
    print(f"\n[INFO] ✅ All samples generated!")
