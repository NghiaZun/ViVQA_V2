"""
Verify Dataset Integrity - Check CSV vs Images consistency
Kiểm tra xem có phải 109 samples thiếu do ảnh bị thiếu/corrupt không
"""

import os
import json
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ===========================
# CONFIG
# ===========================
CSV_PATH = "/kaggle/input/vivqa/ViVQA-main/ViVQA-main/train.csv"
IMAGE_DIR = "/kaggle/input/vivqa/drive-download-20220309T020508Z-001/train"
TEACHER_JSONL = "/kaggle/working/teacher_outputs_gt_guided.jsonl"
OUTPUT_REPORT = "/kaggle/working/dataset_integrity_report.json"

print(f"{'='*70}")
print("DATASET INTEGRITY CHECK")
print(f"{'='*70}")

# ===========================
# 1. LOAD CSV
# ===========================
print(f"\n[1/5] Loading CSV...")
df = pd.read_csv(CSV_PATH)
print(f"   ✅ Total rows in CSV: {len(df)}")

# Check for unique (img_id, question) pairs
csv_pairs = []
duplicate_pairs = []
for idx, row in df.iterrows():
    img_id = str(row.get("img_id", row.get("image_id", ""))).strip()
    question = str(row["question"]).strip()
    pair = (img_id, question)
    
    if pair in csv_pairs:
        duplicate_pairs.append(pair)
    else:
        csv_pairs.append(pair)

print(f"   ✅ Unique (img_id, question) pairs: {len(csv_pairs)}")
if duplicate_pairs:
    print(f"   ⚠️  Duplicate pairs found: {len(duplicate_pairs)}")
else:
    print(f"   ✅ No duplicates in CSV")

# ===========================
# 2. CHECK IMAGES
# ===========================
print(f"\n[2/5] Checking image files...")

unique_img_ids = df['img_id'].unique() if 'img_id' in df.columns else df['image_id'].unique()
print(f"   ✅ Unique image IDs in CSV: {len(unique_img_ids)}")

missing_images = []
corrupt_images = []
valid_images = []

for img_id in tqdm(unique_img_ids, desc="Validating images"):
    img_id_str = str(img_id).strip()
    image_path = os.path.join(IMAGE_DIR, f"{img_id_str}.jpg")
    
    # Check if file exists
    if not os.path.exists(image_path):
        missing_images.append(img_id_str)
        continue
    
    # Try to open image
    try:
        img = Image.open(image_path)
        img.convert("RGB")  # Test conversion
        img.verify()  # Verify image integrity
        valid_images.append(img_id_str)
    except Exception as e:
        corrupt_images.append({
            "img_id": img_id_str,
            "error": str(e)
        })

print(f"\n   IMAGE VALIDATION RESULTS:")
print(f"   ✅ Valid images: {len(valid_images)}")
print(f"   ❌ Missing images: {len(missing_images)}")
print(f"   ⚠️  Corrupt images: {len(corrupt_images)}")

if missing_images:
    print(f"\n   Missing image IDs (first 10):")
    for i, img_id in enumerate(missing_images[:10]):
        print(f"      {i+1}. {img_id}")

if corrupt_images:
    print(f"\n   Corrupt images (first 10):")
    for i, info in enumerate(corrupt_images[:10]):
        print(f"      {i+1}. {info['img_id']}: {info['error']}")

# ===========================
# 3. LOAD TEACHER OUTPUTS
# ===========================
print(f"\n[3/5] Loading teacher outputs...")

if not os.path.exists(TEACHER_JSONL):
    print(f"   ⚠️  Teacher output file not found: {TEACHER_JSONL}")
    teacher_pairs = set()
else:
    teacher_pairs = set()
    teacher_img_ids = set()
    
    with open(TEACHER_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            img_id = str(data.get("img_id", "")).strip()
            question = str(data.get("question", "")).strip()
            teacher_pairs.add((img_id, question))
            teacher_img_ids.add(img_id)
    
    print(f"   ✅ Teacher outputs: {len(teacher_pairs)} pairs")
    print(f"   ✅ Unique images in teacher: {len(teacher_img_ids)}")

# ===========================
# 4. FIND MISSING SAMPLES
# ===========================
print(f"\n[4/5] Analyzing missing samples...")

missing_samples_by_reason = {
    "image_not_found": [],
    "image_corrupt": [],
    "valid_image_but_missing": [],
    "unknown": []
}

csv_pairs_set = set(csv_pairs)
missing_pairs = csv_pairs_set - teacher_pairs

print(f"   Total missing pairs: {len(missing_pairs)}")

for img_id, question in tqdm(missing_pairs, desc="Analyzing missing"):
    image_path = os.path.join(IMAGE_DIR, f"{img_id}.jpg")
    
    # Check reason
    if not os.path.exists(image_path):
        missing_samples_by_reason["image_not_found"].append({
            "img_id": img_id,
            "question": question[:80]
        })
    elif img_id in [c["img_id"] for c in corrupt_images]:
        missing_samples_by_reason["image_corrupt"].append({
            "img_id": img_id,
            "question": question[:80]
        })
    else:
        # Image exists and valid, but not in teacher outputs
        try:
            img = Image.open(image_path)
            img.convert("RGB")
            missing_samples_by_reason["valid_image_but_missing"].append({
                "img_id": img_id,
                "question": question[:80],
                "image_path": image_path
            })
        except Exception as e:
            missing_samples_by_reason["unknown"].append({
                "img_id": img_id,
                "question": question[:80],
                "error": str(e)
            })

print(f"\n   MISSING SAMPLES BREAKDOWN:")
print(f"   ❌ Image not found: {len(missing_samples_by_reason['image_not_found'])}")
print(f"   ⚠️  Image corrupt: {len(missing_samples_by_reason['image_corrupt'])}")
print(f"   ⚠️  Valid image but missing: {len(missing_samples_by_reason['valid_image_but_missing'])}")
print(f"   ❓ Unknown reason: {len(missing_samples_by_reason['unknown'])}")

# ===========================
# 5. CALCULATE AFFECTED SAMPLES
# ===========================
print(f"\n[5/5] Calculating affected samples...")

# Count total samples affected by missing/corrupt images
affected_samples = []

for idx, row in df.iterrows():
    img_id = str(row.get("img_id", row.get("image_id", ""))).strip()
    
    if img_id in missing_images:
        affected_samples.append({
            "img_id": img_id,
            "question": str(row["question"])[:80],
            "reason": "IMAGE_NOT_FOUND"
        })
    elif any(c["img_id"] == img_id for c in corrupt_images):
        affected_samples.append({
            "img_id": img_id,
            "question": str(row["question"])[:80],
            "reason": "IMAGE_CORRUPT"
        })

print(f"   Total samples affected by missing/corrupt images: {len(affected_samples)}")

# ===========================
# 6. GENERATE REPORT
# ===========================
print(f"\n{'='*70}")
print("SUMMARY REPORT")
print(f"{'='*70}")

report = {
    "csv_stats": {
        "total_rows": len(df),
        "unique_pairs": len(csv_pairs),
        "duplicate_pairs": len(duplicate_pairs),
        "unique_images": len(unique_img_ids)
    },
    "image_stats": {
        "valid_images": len(valid_images),
        "missing_images": len(missing_images),
        "corrupt_images": len(corrupt_images),
        "missing_image_ids": missing_images,
        "corrupt_image_details": corrupt_images
    },
    "teacher_stats": {
        "generated_pairs": len(teacher_pairs),
        "coverage_percentage": (len(teacher_pairs) / len(csv_pairs) * 100) if csv_pairs else 0
    },
    "missing_samples": {
        "total_missing": len(missing_pairs),
        "by_reason": {
            "image_not_found": len(missing_samples_by_reason["image_not_found"]),
            "image_corrupt": len(missing_samples_by_reason["image_corrupt"]),
            "valid_image_but_missing": len(missing_samples_by_reason["valid_image_but_missing"]),
            "unknown": len(missing_samples_by_reason["unknown"])
        },
        "details": missing_samples_by_reason
    },
    "affected_samples_count": len(affected_samples),
    "affected_samples_details": affected_samples[:100]  # First 100 for report
}

# Print summary
print(f"\n📊 CSV Dataset:")
print(f"   - Total rows: {report['csv_stats']['total_rows']}")
print(f"   - Unique (img_id, question) pairs: {report['csv_stats']['unique_pairs']}")
print(f"   - Unique images: {report['csv_stats']['unique_images']}")

print(f"\n🖼️  Image Files:")
print(f"   - Valid: {report['image_stats']['valid_images']}")
print(f"   - Missing: {report['image_stats']['missing_images']}")
print(f"   - Corrupt: {report['image_stats']['corrupt_images']}")

print(f"\n🎓 Teacher Generation:")
print(f"   - Generated: {report['teacher_stats']['generated_pairs']}")
print(f"   - Coverage: {report['teacher_stats']['coverage_percentage']:.2f}%")

print(f"\n❓ Missing Samples Analysis:")
print(f"   - Total missing: {report['missing_samples']['total_missing']}")
print(f"   - Due to missing image: {report['missing_samples']['by_reason']['image_not_found']}")
print(f"   - Due to corrupt image: {report['missing_samples']['by_reason']['image_corrupt']}")
print(f"   - Valid image but not generated: {report['missing_samples']['by_reason']['valid_image_but_missing']}")
print(f"   - Unknown reason: {report['missing_samples']['by_reason']['unknown']}")

print(f"\n💥 Total samples affected by image issues: {len(affected_samples)}")

# Conclusion
print(f"\n{'='*70}")
print("CONCLUSION")
print(f"{'='*70}")

if len(missing_pairs) == report['missing_samples']['by_reason']['image_not_found'] + report['missing_samples']['by_reason']['image_corrupt']:
    print(f"✅ ALL {len(missing_pairs)} missing samples are due to missing/corrupt images!")
    print(f"   This confirms the dataset has image file issues.")
elif report['missing_samples']['by_reason']['valid_image_but_missing'] > 0:
    print(f"⚠️  WARNING: {report['missing_samples']['by_reason']['valid_image_but_missing']} samples have valid images but were not generated!")
    print(f"   These might need re-generation.")
else:
    print(f"❓ Unknown issue - needs further investigation")

# Save report
with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"\n📄 Full report saved to: {OUTPUT_REPORT}")
print(f"{'='*70}")
