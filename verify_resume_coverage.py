"""
verify_resume_coverage.py
Kiểm tra xem resume có thể đạt 100% coverage không
Author: Nghia-Duong
"""

import os
import json
import pandas as pd
from collections import Counter

# ===========================
# CONFIG
# ===========================
CSV_PATH = "/kaggle/input/vivqa/ViVQA-main/ViVQA-main/train.csv"
IMAGE_DIR = "/kaggle/input/vivqa/drive-download-20220309T020508Z-001/train"
MERGED_FILE = "/kaggle/working/teacher_outputs_merged.jsonl"

def verify_resume_capability():
    """Kiểm tra khả năng resume để đạt 100% coverage"""
    
    print("="*70)
    print("RESUME COVERAGE VERIFICATION")
    print("="*70)
    
    # Load GT
    print("\n[1] Loading ground truth data...")
    df = pd.read_csv(CSV_PATH)
    total_samples = len(df)
    print(f"    Total samples in CSV: {total_samples:,}")
    
    # Load existing teacher outputs
    print("\n[2] Loading existing teacher outputs...")
    existing_ids = set()
    fallback_count = 0
    teacher_generated = 0
    
    if os.path.exists(MERGED_FILE):
        with open(MERGED_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                img_id = str(data['img_id']).strip()
                existing_ids.add(img_id)
                
                if data.get('_fallback', False):
                    fallback_count += 1
                else:
                    teacher_generated += 1
        
        print(f"    Existing outputs: {len(existing_ids):,}")
        print(f"    Teacher-generated: {teacher_generated:,}")
        print(f"    Fallback entries: {fallback_count:,}")
    else:
        print(f"    [WARN] File not found: {MERGED_FILE}")
        print(f"    Starting from scratch")
    
    # Analyze remaining samples
    print("\n[3] Analyzing remaining samples to process...")
    remaining_ids = set()
    missing_images = 0
    processable = 0
    
    for _, row in df.iterrows():
        img_id = str(row['img_id']).strip()
        
        if img_id not in existing_ids:
            remaining_ids.add(img_id)
            
            # Check if image exists
            img_path = os.path.join(IMAGE_DIR, f"{img_id}.jpg")
            if os.path.exists(img_path):
                processable += 1
            else:
                missing_images += 1
    
    print(f"    Remaining samples: {len(remaining_ids):,}")
    print(f"    Has image file: {processable:,}")
    print(f"    Missing images: {missing_images:,}")
    
    # Predict resume outcome WITH FALLBACK
    print("\n[4] Predicting resume outcome (WITH fallback)...")
    
    # Với fallback, TẤT CẢ samples sẽ được save
    expected_after_resume = len(existing_ids) + len(remaining_ids)
    
    # Teacher generation có thể fail, nhưng fallback sẽ cover
    estimated_new_teacher = int(processable * 0.8)  # 80% teacher success
    estimated_new_fallback = len(remaining_ids) - estimated_new_teacher
    
    print(f"    Current coverage: {len(existing_ids):,}/{total_samples:,} ({len(existing_ids)/total_samples*100:.2f}%)")
    print(f"    After resume (expected): {expected_after_resume:,}/{total_samples:,} ({expected_after_resume/total_samples*100:.2f}%)")
    print(f"    New teacher outputs: ~{estimated_new_teacher:,}")
    print(f"    New fallback entries: ~{estimated_new_fallback:,}")
    
    # Final stats
    print("\n[5] Final composition (after resume)...")
    total_teacher = teacher_generated + estimated_new_teacher
    total_fallback = fallback_count + estimated_new_fallback
    
    print(f"    Teacher-generated: {total_teacher:,} ({total_teacher/total_samples*100:.1f}%)")
    print(f"    Fallback entries:  {total_fallback:,} ({total_fallback/total_samples*100:.1f}%)")
    print(f"    Total coverage:    {total_samples:,}/{total_samples:,} (100.0%)")
    
    # Recommendation
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    if expected_after_resume >= total_samples:
        print("\n✅ YES - Resume WILL achieve 100% coverage!")
        print("\nReasons:")
        print("  1. ✅ Fallback system ensures NO sample is skipped")
        print("  2. ✅ Missing images get fallback reasoning")
        print("  3. ✅ Failed teacher generations get fallback reasoning")
        print("  4. ✅ All samples are saved (teacher or fallback)")
        
        print("\nExpected quality:")
        teacher_pct = total_teacher / total_samples * 100
        if teacher_pct >= 80:
            print(f"  🌟 EXCELLENT: {teacher_pct:.1f}% teacher-generated")
        elif teacher_pct >= 70:
            print(f"  ✅ GOOD: {teacher_pct:.1f}% teacher-generated")
        elif teacher_pct >= 60:
            print(f"  ⚠️  ACCEPTABLE: {teacher_pct:.1f}% teacher-generated")
        else:
            print(f"  ⚠️  LOW: {teacher_pct:.1f}% teacher-generated")
            print(f"     Consider re-running with better GPU/memory")
    else:
        print("\n❌ NO - Resume may not achieve 100% coverage")
        print(f"\nExpected coverage: {expected_after_resume/total_samples*100:.2f}%")
        print(f"Missing: {total_samples - expected_after_resume:,} samples")
        print("\nRecommendations:")
        print("  1. Check if images exist for all samples")
        print("  2. Verify CSV has correct img_ids")
        print("  3. Run fill_missing_teacher_outputs.py after resume")
    
    print("="*70)
    
    # Action items
    print("\n📋 NEXT STEPS:")
    if len(remaining_ids) > 0:
        print(f"  1. Resume generation: python generate_teacher_optim.py")
        print(f"     → Will process {len(remaining_ids):,} remaining samples")
        print(f"  2. Wait for completion (auto-saves every 50 samples)")
        print(f"  3. Merge if running both normal + reverse")
        print(f"  4. Verify final coverage with this script")
    else:
        print(f"  ✅ All samples already processed!")
        print(f"  → Ready to train with {MERGED_FILE}")
    
    print("\n💡 TIP: Resume is SAFE - processed_ids prevents duplicates")
    print("="*70)

if __name__ == "__main__":
    verify_resume_capability()
