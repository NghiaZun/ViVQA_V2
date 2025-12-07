"""
Check duplicate img_ids with different questions/answers
"""

import pandas as pd

CSV_PATH = "/kaggle/input/vivqa/ViVQA-main/ViVQA-main/train.csv"

print(f"[INFO] Loading CSV: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

print(f"\n{'='*70}")
print(f"DUPLICATE IMG_ID ANALYSIS")
print(f"{'='*70}")
print(f"Total rows:        {len(df):,}")
print(f"Unique img_ids:    {df['img_id'].nunique():,}")
print(f"Duplicate rows:    {len(df) - df['img_id'].nunique():,}")
print(f"{'='*70}")

# Find img_ids that appear multiple times
dup_counts = df['img_id'].value_counts()
dup_ids = dup_counts[dup_counts > 1]

print(f"\n[INFO] Number of img_ids appearing multiple times: {len(dup_ids):,}")
print(f"[INFO] Total duplicate rows: {dup_counts[dup_counts > 1].sum() - len(dup_ids):,}")

if len(dup_ids) > 0:
    print(f"\n[INFO] Top 10 most duplicated img_ids:")
    for img_id, count in dup_ids.head(10).items():
        print(f"  img_id {img_id}: appears {count} times")
    
    # Analyze duplicates: same question or different?
    print(f"\n[INFO] Analyzing duplicate types...")
    
    same_question_count = 0
    diff_question_count = 0
    
    for img_id in dup_ids.index:
        rows = df[df['img_id'] == img_id]
        unique_questions = rows['question'].nunique()
        
        if unique_questions == 1:
            same_question_count += 1
        else:
            diff_question_count += 1
    
    print(f"  - Same image + SAME question: {same_question_count} img_ids (TRUE duplicates)")
    print(f"  - Same image + DIFFERENT questions: {diff_question_count} img_ids (Valid multiple QA)")
    
    # Show examples
    print(f"\n{'='*70}")
    print(f"EXAMPLE 1: Same image, DIFFERENT questions (VALID)")
    print(f"{'='*70}")
    
    # Find an example with different questions
    for img_id in dup_ids.index[:10]:
        rows = df[df['img_id'] == img_id]
        if rows['question'].nunique() > 1:
            print(f"\nImage ID: {img_id} (appears {len(rows)} times)")
            for idx, row in rows.iterrows():
                print(f"  Q: {row['question']}")
                print(f"  A: {row['answer']}")
                print(f"  Type: {row.get('type', 'N/A')}")
                print()
            break
    
    # Show example of true duplicate (same question)
    print(f"{'='*70}")
    print(f"EXAMPLE 2: Same image, SAME question (TRUE DUPLICATE)")
    print(f"{'='*70}")
    
    for img_id in dup_ids.index[:100]:
        rows = df[df['img_id'] == img_id]
        if rows['question'].nunique() == 1:
            print(f"\nImage ID: {img_id} (appears {len(rows)} times)")
            print(f"Question: {rows.iloc[0]['question']}")
            print(f"\nAnswers:")
            for idx, row in rows.iterrows():
                print(f"  - {row['answer']} (Type: {row.get('type', 'N/A')})")
            break
    
    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    total_dup_rows = sum(dup_counts[dup_counts > 1]) - len(dup_ids)
    print(f"CSV has {len(df):,} rows but only {df['img_id'].nunique():,} unique images")
    print(f"This means {total_dup_rows:,} rows are duplicates")
    print(f"\nInterpretation:")
    print(f"  ✅ If different questions → VALID (multiple QA pairs per image)")
    print(f"  ❌ If same questions → TRUE DUPLICATES (data error)")
    print(f"{'='*70}")
else:
    print(f"\n[INFO] ✅ No duplicate img_ids found!")
