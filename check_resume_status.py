"""
check_resume_status.py
Xem chi tiết samples nào đã xử lý, samples nào còn thiếu
Author: Nghia-Duong
"""

import os
import json
import pandas as pd

# ===========================
# CONFIG
# ===========================
CSV_PATH = "/kaggle/input/vivqa/ViVQA-main/ViVQA-main/train.csv"
MERGED_FILE = "/kaggle/working/teacher_outputs_merged.jsonl"

def check_resume_status():
    """Kiểm tra chi tiết resume status"""
    
    print("="*70)
    print("RESUME STATUS CHECK")
    print("="*70)
    
    # Load GT
    print("\n[1] Loading ground truth CSV...")
    df = pd.read_csv(CSV_PATH)
    total_samples = len(df)
    all_gt_ids = set(str(row['img_id']).strip() for _, row in df.iterrows())
    print(f"    Total GT samples: {total_samples:,}")
    print(f"    Unique GT IDs: {len(all_gt_ids):,}")
    
    # Load existing teacher outputs
    print("\n[2] Loading existing teacher outputs...")
    processed_ids = set()
    teacher_generated = 0
    fallback_count = 0
    
    if os.path.exists(MERGED_FILE):
        with open(MERGED_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                img_id = str(data['img_id']).strip()
                processed_ids.add(img_id)
                
                if data.get('_fallback', False):
                    fallback_count += 1
                else:
                    teacher_generated += 1
        
        print(f"    Processed IDs: {len(processed_ids):,}")
        print(f"    Teacher-generated: {teacher_generated:,}")
        print(f"    Fallback entries: {fallback_count:,}")
    else:
        print(f"    [WARN] Checkpoint file not found: {MERGED_FILE}")
        print(f"    Will start from scratch")
    
    # Find remaining samples
    print("\n[3] Analyzing remaining samples...")
    remaining_ids = all_gt_ids - processed_ids
    
    print(f"    ✅ Processed: {len(processed_ids):,}/{total_samples:,} ({len(processed_ids)/total_samples*100:.2f}%)")
    print(f"    ❌ Remaining: {len(remaining_ids):,}/{total_samples:,} ({len(remaining_ids)/total_samples*100:.2f}%)")
    
    # Show sample processed IDs
    if processed_ids:
        sample_processed = sorted(list(processed_ids))[:10]
        print(f"\n    Sample processed IDs: {sample_processed}")
    
    # Show sample remaining IDs
    if remaining_ids:
        sample_remaining = sorted(list(remaining_ids))[:10]
        print(f"    Sample remaining IDs: {sample_remaining}")
        
        # Find gaps in processed range
        print(f"\n[4] Finding gaps in ID ranges...")
        all_ids_sorted = sorted([int(x) for x in all_gt_ids if x.isdigit()])
        processed_nums = sorted([int(x) for x in processed_ids if x.isdigit()])
        
        if processed_nums:
            min_processed = min(processed_nums)
            max_processed = max(processed_nums)
            print(f"    Processed range: {min_processed} → {max_processed}")
            
            # Find gaps
            gaps = []
            for i in range(len(processed_nums) - 1):
                gap_size = processed_nums[i+1] - processed_nums[i] - 1
                if gap_size > 0:
                    gaps.append((processed_nums[i]+1, processed_nums[i+1]-1, gap_size))
            
            if gaps:
                print(f"    Found {len(gaps)} gaps in processed IDs:")
                for start, end, size in gaps[:5]:  # Show first 5 gaps
                    print(f"      Gap: {start} → {end} ({size} samples)")
    
    # Resume readiness
    print("\n" + "="*70)
    print("RESUME READINESS")
    print("="*70)
    
    if len(remaining_ids) == 0:
        print("\n✅ ALL DONE! No samples remaining to process")
        print(f"   Total: {len(processed_ids):,}/{total_samples:,} (100%)")
        print(f"\n   Ready to train with: {MERGED_FILE}")
    else:
        print(f"\n📋 READY TO RESUME!")
        print(f"   Will process: {len(remaining_ids):,} remaining samples")
        print(f"   Already done: {len(processed_ids):,} samples (will be skipped)")
        print(f"\n   How resume works:")
        print(f"   1. Load processed_ids from: {MERGED_FILE}")
        print(f"   2. Loop through all {total_samples:,} samples in CSV")
        print(f"   3. Skip {len(processed_ids):,} samples (already in processed_ids)")
        print(f"   4. Process only {len(remaining_ids):,} new samples")
        print(f"   5. Append to output file (no duplicates!)")
        
        print(f"\n   Estimated time:")
        samples_per_hour = 500  # Conservative estimate
        hours = len(remaining_ids) / samples_per_hour
        print(f"   ~{hours:.1f} hours ({samples_per_hour} samples/hour)")
    
    print("="*70)

if __name__ == "__main__":
    check_resume_status()
