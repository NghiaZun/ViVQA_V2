"""
Print ALL duplicate (img_id, question) pairs in CSV
Để verify có thực sự 109 duplicates không
"""

import pandas as pd
from collections import Counter

# ===========================
# CONFIG
# ===========================
CSV_PATH = "/kaggle/input/vivqa/ViVQA-main/ViVQA-main/train.csv"

print(f"{'='*70}")
print("DUPLICATE PAIRS DETECTION")
print(f"{'='*70}")

# Load CSV
df = pd.read_csv(CSV_PATH)
print(f"\n[INFO] Loaded CSV: {len(df)} total rows")

# Count all pairs
all_pairs = []
for idx, row in df.iterrows():
    img_id = str(row.get("img_id", row.get("image_id", ""))).strip()
    question = str(row["question"]).strip()
    answer = str(row["answer"]).strip()
    pair = (img_id, question)
    all_pairs.append({
        "csv_row": idx + 2,  # +2 vì header là row 1, data bắt đầu row 2
        "img_id": img_id,
        "question": question,
        "answer": answer,
        "pair": pair
    })

# Count occurrences
pair_counter = Counter([item["pair"] for item in all_pairs])

# Find duplicates
duplicates = {pair: count for pair, count in pair_counter.items() if count > 1}

print(f"\n[RESULT] Total unique pairs: {len(pair_counter)}")
print(f"[RESULT] Duplicate pairs found: {len(duplicates)}")
print(f"[RESULT] Total duplicate occurrences: {sum(count - 1 for count in duplicates.values())}")

if duplicates:
    print(f"\n{'='*70}")
    print(f"ALL {len(duplicates)} DUPLICATE PAIRS (showing all occurrences)")
    print(f"{'='*70}\n")
    
    for idx, (pair, count) in enumerate(sorted(duplicates.items(), key=lambda x: x[1], reverse=True), 1):
        img_id, question = pair
        print(f"{idx}. img_id={img_id}, appears {count} times")
        print(f"   Question: {question[:100]}{'...' if len(question) > 100 else ''}")
        
        # Find all rows with this pair
        matching_rows = [item for item in all_pairs if item["pair"] == pair]
        print(f"   Rows in CSV:")
        for match in matching_rows:
            print(f"      - Row {match['csv_row']}: answer='{match['answer']}'")
        print()

    print(f"{'='*70}")
    print("STATISTICS")
    print(f"{'='*70}")
    
    # Count by duplication level
    dup_levels = Counter([count for count in duplicates.values()])
    for level in sorted(dup_levels.keys(), reverse=True):
        count = dup_levels[level]
        print(f"   Pairs appearing {level} times: {count} pairs")
    
    # Verify calculation
    total_duplicate_rows = sum((count - 1) * len([p for p in duplicates if duplicates[p] == count]) 
                                for count in dup_levels.keys())
    
    print(f"\n[VERIFY] Calculation:")
    print(f"   Total CSV rows: {len(df)}")
    print(f"   Unique pairs: {len(pair_counter)}")
    print(f"   Duplicate rows: {len(df) - len(pair_counter)}")
    print(f"   Expected: {len(df)} = {len(pair_counter)} + {len(df) - len(pair_counter)}")
    
    # Check if any duplicates have different answers
    print(f"\n{'='*70}")
    print("DUPLICATE PAIRS WITH DIFFERENT ANSWERS")
    print(f"{'='*70}\n")
    
    different_answers_found = False
    for pair in duplicates:
        matching_rows = [item for item in all_pairs if item["pair"] == pair]
        unique_answers = set(item["answer"] for item in matching_rows)
        
        if len(unique_answers) > 1:
            different_answers_found = True
            img_id, question = pair
            print(f"⚠️  img_id={img_id}")
            print(f"   Question: {question[:80]}...")
            print(f"   Different answers found:")
            for ans in unique_answers:
                rows_with_ans = [item["csv_row"] for item in matching_rows if item["answer"] == ans]
                print(f"      - '{ans}' (rows: {rows_with_ans})")
            print()
    
    if not different_answers_found:
        print("✅ All duplicate pairs have the SAME answer (consistent)")
    
else:
    print(f"\n✅ No duplicate pairs found - CSV is clean!")

print(f"\n{'='*70}")
