"""
Download Teacher Model (Qwen2-VL-7B-Instruct) to local directory
Author: Nghia-Duong

Usage:
1. Run locally (with internet): python download_teacher_model.py
2. Upload saved model to Kaggle Dataset
3. Use in training: MODEL_NAME = "/kaggle/input/your-dataset/qwen2-vl-7b-instruct"
"""

import os
import argparse
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch

def download_model(
    model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
    save_dir: str = "./qwen2-vl-7b-instruct",
    use_auth_token: str = None
):
    """
    Download teacher model and processor to local directory
    
    Args:
        model_name: HuggingFace model name
        save_dir: Local directory to save model
        use_auth_token: Optional HuggingFace token for gated models
    """
    print(f"{'='*70}")
    print(f"DOWNLOADING TEACHER MODEL: {model_name}")
    print(f"{'='*70}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Download processor
    print(f"\n[1/2] Downloading Processor...")
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=use_auth_token
    )
    processor.save_pretrained(save_dir)
    print(f"✅ Processor saved to: {save_dir}")
    
    # Download model
    print(f"\n[2/2] Downloading Model (this may take a while ~14GB)...")
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Save as fp16 to reduce size
        trust_remote_code=True,
        token=use_auth_token,
        low_cpu_mem_usage=True
    )
    model.save_pretrained(save_dir)
    print(f"✅ Model saved to: {save_dir}")
    
    # Verify download
    print(f"\n[VERIFY] Checking downloaded files...")
    required_files = [
        "config.json",
        "preprocessor_config.json", 
        "tokenizer_config.json"
    ]
    
    for file in required_files:
        path = os.path.join(save_dir, file)
        if os.path.exists(path):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} - MISSING!")
    
    # Check model weights
    weight_files = [f for f in os.listdir(save_dir) if f.endswith('.safetensors') or f.endswith('.bin')]
    if weight_files:
        print(f"  ✅ Model weights: {len(weight_files)} file(s)")
        total_size = sum(os.path.getsize(os.path.join(save_dir, f)) for f in weight_files) / (1024**3)
        print(f"  📦 Total size: {total_size:.2f} GB")
    else:
        print(f"  ❌ No model weights found!")
    
    print(f"\n{'='*70}")
    print(f"DOWNLOAD COMPLETE!")
    print(f"{'='*70}")
    print(f"\n📁 Model saved to: {os.path.abspath(save_dir)}")
    print(f"\n📝 Next steps:")
    print(f"  1. Compress folder: tar -czf qwen2-vl-7b-instruct.tar.gz {save_dir}")
    print(f"  2. Upload to Kaggle Dataset")
    print(f"  3. In training code, use:")
    print(f"     MODEL_NAME = '/kaggle/input/your-dataset-name/qwen2-vl-7b-instruct'")

def test_load_model(model_dir: str):
    """Test loading the downloaded model"""
    print(f"\n[TEST] Loading model from {model_dir}...")
    
    try:
        processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        print("✅ Processor loaded successfully")
        
        model = AutoModelForVision2Seq.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        print("✅ Model loaded successfully")
        
        print(f"✅ Model ready to use!")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Qwen2-VL teacher model")
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="Qwen/Qwen2-VL-7B-Instruct",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--save_dir", 
        type=str, 
        default="./qwen2-vl-7b-instruct",
        help="Directory to save model"
    )
    parser.add_argument(
        "--token", 
        type=str, 
        default=None,
        help="HuggingFace token for gated models (if needed)"
    )
    parser.add_argument(
        "--test", 
        action="store_true",
        help="Test loading the model after download"
    )
    
    args = parser.parse_args()
    
    # Download
    download_model(
        model_name=args.model_name,
        save_dir=args.save_dir,
        use_auth_token=args.token
    )
    
    # Test if requested
    if args.test:
        test_load_model(args.save_dir)
