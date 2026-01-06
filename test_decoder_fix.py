"""
Test decoder fix - Verify BOS token không gây ký tự rác
"""

import torch
from model_dinov2_bartpho_2 import DINOv2BARTphoVQA
from PIL import Image
import numpy as np

def test_decoder_output():
    print("=" * 80)
    print("Testing Decoder BOS Token Fix")
    print("=" * 80)
    
    # Initialize model
    print("\n[1] Initializing model...")
    model = DINOv2BARTphoVQA(
        dinov2_model_name='facebook/dinov2-base',
        bartpho_model_name='vinai/bartpho-syllable',
        num_cross_attn_layers=3,
        use_reasoning_quality_check=False,
        gradient_checkpointing=False
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded on {device}")
    
    # Test tokenization
    print("\n[2] Testing BARTpho tokenizer...")
    test_texts = [
        "màu xanh lá",
        "màu đỏ",
        "có 3 người",
        "không có"
    ]
    
    for text in test_texts:
        # Tokenize
        tokens = model.tokenizer.encode(text, add_special_tokens=True)
        
        # Decode with special tokens
        decoded_with_special = model.tokenizer.decode(tokens, skip_special_tokens=False)
        
        # Decode without special tokens
        decoded_clean = model.tokenizer.decode(tokens, skip_special_tokens=True)
        
        # Manual decode (remove BOS)
        tokens_no_bos = [t for t in tokens if t != model.tokenizer.bos_token_id]
        decoded_manual = model.tokenizer.decode(tokens_no_bos, skip_special_tokens=True)
        
        print(f"\n  Original: {text}")
        print(f"  Tokens: {tokens}")
        print(f"  With special: {decoded_with_special}")
        print(f"  Skip special: {decoded_clean}")
        print(f"  Manual (no BOS): {decoded_manual}")
    
    # Test generation
    print("\n[3] Testing generation with dummy data...")
    
    # Create dummy image
    dummy_image = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )
    
    # Process image
    pixel_values = model.vision_processor(
        images=dummy_image,
        return_tensors='pt'
    )['pixel_values'].to(device)
    
    # Tokenize question
    question = "màu của xe là gì"
    encoding = model.tokenizer(
        question,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Generate
    print(f"\n  Question: {question}")
    print("  Generating answer...")
    
    with torch.no_grad():
        visual_features = model.encode_image(pixel_values)
        text_features = model.encode_text(input_ids, attention_mask)
        fused_features, _ = model.fuse_multimodal(text_features, visual_features)
        
        # Generate answer (short, max 16 tokens)
        answer_ids = torch.full(
            (1, 1),
            model.tokenizer.bos_token_id,
            dtype=torch.long,
            device=device
        )
        
        for step in range(15):
            decoder_outputs = model.answer_decoder(
                input_ids=answer_ids,
                encoder_hidden_states=fused_features,
                return_dict=True,
                use_cache=False
            )
            
            hidden = decoder_outputs.last_hidden_state[:, -1, :]
            logits = model.lm_head(hidden)
            next_token = logits.argmax(dim=-1, keepdim=True)
            answer_ids = torch.cat([answer_ids, next_token], dim=1)
            
            if next_token.item() == model.tokenizer.eos_token_id:
                break
        
        # Method 1: Direct decode (skip_special_tokens=True)
        answer_direct = model.tokenizer.batch_decode(
            answer_ids, 
            skip_special_tokens=True
        )[0]
        
        # Method 2: Manual remove BOS
        tokens = answer_ids[0].tolist()
        if tokens and tokens[0] == model.tokenizer.bos_token_id:
            tokens = tokens[1:]
        if model.tokenizer.eos_token_id in tokens:
            eos_idx = tokens.index(model.tokenizer.eos_token_id)
            tokens = tokens[:eos_idx]
        answer_manual = model.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        
        print(f"\n  Generated tokens: {answer_ids[0].tolist()}")
        print(f"  Answer (direct decode): '{answer_direct}'")
        print(f"  Answer (manual BOS removal): '{answer_manual}'")
        print(f"  First char (direct): {answer_direct[0] if answer_direct else 'EMPTY'}")
        print(f"  First char (manual): {answer_manual[0] if answer_manual else 'EMPTY'}")
    
    print("\n" + "=" * 80)
    print("✓ Test completed!")
    print("=" * 80)

if __name__ == '__main__':
    test_decoder_output()
