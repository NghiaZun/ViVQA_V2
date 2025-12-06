"""
Check teacher output quality for the 5 test samples
"""
import json
import pandas as pd

# Load test data
test_df = pd.read_csv("/kaggle/input/vivqa/ViVQA-main/ViVQA-main/test.csv")

# Load teacher outputs if available
teacher_file = "/kaggle/input/teacher-5-12/teacher_outputs_test.jsonl"  # Adjust path

print("="*70)
print("CHECKING TEACHER QUALITY FOR FAILED SAMPLES")
print("="*70)

# Check first 5 samples
failed_samples = [
    (0, "557067", "màu của miếng vá là gì", "màu xanh dương"),
    (1, "436394", "màu của áo là gì", "màu cam"),
    (2, "541050", "màu của áo là gì", "màu xanh dương"),
    (4, "314710", "màu của quả bóng là gì", "màu đỏ"),
]

try:
    teacher_outputs = {}
    with open(teacher_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            img_id = str(data.get('img_id', data.get('image_id')))
            teacher_outputs[img_id] = data
    
    for idx, img_id, question, gt in failed_samples:
        print(f"\n{'='*70}")
        print(f"Sample {idx+1} - Image ID: {img_id}")
        print(f"{'='*70}")
        print(f"Question: {question}")
        print(f"GT Answer: {gt}")
        
        if img_id in teacher_outputs:
            teacher = teacher_outputs[img_id]
            print(f"\n🧑‍🏫 TEACHER OUTPUT:")
            print(f"Answer: {teacher.get('teacher_answer', 'N/A')}")
            print(f"Reasoning: {teacher.get('teacher_reasoning', 'N/A')[:200]}...")
            
            # Check if teacher is correct
            if teacher.get('teacher_answer', '').strip().lower() == gt.strip().lower():
                print(f"✅ Teacher answer CORRECT")
            else:
                print(f"❌ Teacher answer WRONG")
        else:
            print(f"\n⚠️  No teacher output found for this sample")

except FileNotFoundError:
    print(f"\n❌ Teacher file not found: {teacher_file}")
    print("\nThis means:")
    print("1. Model was trained WITHOUT test set teacher outputs")
    print("2. Model may not generalize well to test set")
    print("3. Need to generate teacher outputs for test set first!")

print("\n" + "="*70)
