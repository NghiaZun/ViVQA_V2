"""
Debug script to check if model predictions match images
"""
import torch
from PIL import Image
from model_optimal import OptimalVQAModel
from transformers import CLIPProcessor
import os

def test_sample():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    print("Loading model...")
    checkpoint_dir = "/kaggle/input/base-model/transformers/default/1/checkpoints"
    model = OptimalVQAModel(
        vision_model_name="openai/clip-vit-large-patch14",
        phobert_dir=os.path.join(checkpoint_dir, "phobert_tokenizer"),
        vit5_dir=os.path.join(checkpoint_dir, "vit5_tokenizer"),
        hidden_dim=768,
        num_fusion_layers=4,
        num_heads=12,
        dropout=0.1,
        use_lora=True,
        use_type_routing=True
    ).to(device)
    
    # Load checkpoint
    checkpoint_path = "/kaggle/working/latest_checkpoint_optimal.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    vision_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model.eval()
    
    # Test with a specific image
    image_dir = "/kaggle/input/vivqa/drive-download-20220309T020508Z-001/train"
    test_img_id = "1"  # Change this to any image ID
    img_path = os.path.join(image_dir, f"{test_img_id}.jpg")
    
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return
    
    image = Image.open(img_path).convert("RGB")
    
    # Test questions
    questions = [
        "Có bao nhiêu người trong ảnh?",
        "Màu gì chiếm ưu thế trong ảnh?",
        "Họ đang làm gì?",
    ]
    
    print(f"\n{'='*70}")
    print(f"Testing Image: {test_img_id}")
    print(f"{'='*70}\n")
    
    for question in questions:
        # Process
        vision_inputs = vision_processor(images=image, return_tensors="pt")
        pixel_values = vision_inputs["pixel_values"].to(device)
        
        text_inputs = model.text_tokenizer(
            question, max_length=64, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        input_ids = text_inputs["input_ids"].to(device)
        attention_mask = text_inputs["attention_mask"].to(device)
        
        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=96,
                num_beams=4,
                length_penalty=1.2,
                early_stopping=True
            )
        
        # Decode
        prediction = model.decoder_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        print(f"Question: {question}")
        print(f"Full Output: {prediction}")
        
        # Extract answer
        if "Answer:" in prediction:
            answer = prediction.split("Answer:")[-1].split("\n")[0].strip()
            print(f"Extracted Answer: {answer}")
        else:
            print("WARNING: No 'Answer:' found in output!")
        
        print()

if __name__ == "__main__":
    test_sample()
