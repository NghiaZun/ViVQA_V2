"""
TRUE SEQUENTIAL CHAIN-OF-THOUGHT VQA MODEL
===========================================
Documentation & Expected Behavior

Author: VQA Expert
Date: December 26, 2025
"""

# ============================================================================
# 1. ARCHITECTURE OVERVIEW
# ============================================================================

"""
BEFORE (Parallel CoT - WRONG):
==============================
Encoder (Image + Question)
    |
    ├──> Reasoning Decoder ──> reasoning_logits
    └──> Answer Decoder ──> answer_logits
    
Problem: Answer and Reasoning are INDEPENDENT!


AFTER (Sequential CoT - CORRECT):
==================================
Encoder (Image + Question)
    |
    v
Reasoning Decoder ──> reasoning_hidden_states
    |
    v
[Concat: Encoder + Reasoning Hidden]
    |
    v
Answer Decoder ──> answer_logits

✅ Answer DEPENDS on Reasoning!
"""


# ============================================================================
# 2. TRAINING MODE - Forward Pass
# ============================================================================

"""
Step 1: Encode Image + Question
---------------------------------
pixel_values: [B, 3, 224, 224]
input_ids: [B, L_q]  (question tokens)
    ↓
image_embeds: [B, 512] (CLIP)
text_embeds: [B, 768] (PhoBERT)
    ↓
fused_embeds: [B, 768] (fusion)
    ↓
encoder_hidden: [B, 1, 768] (projected)


Step 2: Decode Reasoning (Teacher Forcing)
-------------------------------------------
reasoning_labels: [B, L_r] (e.g., [B, 128])
    ↓
reasoning_outputs = decoder(
    decoder_input_ids=reasoning_labels,
    encoder_outputs=(encoder_hidden,),
    output_hidden_states=True  # KEY: Get hidden states
)
    ↓
reasoning_logits: [B, L_r, 40000]
reasoning_hidden_states: [B, L_r, 768]  # KEY: Used for answer conditioning


Step 3: Decode Answer (Conditioned on Reasoning)
-------------------------------------------------
# Combine encoder context with reasoning context
combined_encoder_hidden = concat([
    encoder_hidden,           # [B, 1, 768]
    reasoning_hidden_states   # [B, L_r, 768]
], dim=1)
# Result: [B, 1+L_r, 768]

answer_labels: [B, L_a] (e.g., [B, 32])
    ↓
answer_outputs = decoder(
    decoder_input_ids=answer_labels,
    encoder_outputs=(combined_encoder_hidden,),  # KEY: Uses reasoning!
)
    ↓
answer_logits: [B, L_a, 40000]


Step 4: Loss Calculation
-------------------------
# Reasoning loss (60% weight)
reasoning_loss = CrossEntropy(
    reasoning_logits[:, :-1, :],  # Shift for teacher forcing
    reasoning_labels[:, 1:]
)

# Answer loss (40% weight)
answer_loss = CrossEntropy(
    answer_logits[:, :-1, :],     # Shift for teacher forcing
    answer_labels[:, 1:]
)

# Total loss
total_loss = 0.6 * reasoning_loss + 0.4 * answer_loss
weighted_loss = total_loss * (confidence / 3.0)

# Backward
weighted_loss.backward()
optimizer.step()

✅ Gradients flow through: Encoder → Reasoning → Answer
"""


# ============================================================================
# 3. INFERENCE MODE - Generate Answer
# ============================================================================

"""
Step 1: Encode Image + Question
---------------------------------
Same as training


Step 2: Generate Reasoning Autoregressively
--------------------------------------------
reasoning_ids = decoder.generate(
    inputs_embeds=encoder_hidden,
    max_length=128,
    num_beams=1
)
# Result: [B, L_r] (e.g., [B, 85] actual length)

# Decode to text
reasoning_text = "Trong ảnh có một chiếc bình màu xanh lá cây"


Step 3: Encode Reasoning to Get Hidden States
----------------------------------------------
# Run decoder again to extract hidden states
reasoning_outputs = decoder(
    decoder_input_ids=reasoning_ids,
    encoder_outputs=(encoder_hidden,),
    output_hidden_states=True
)
reasoning_hidden_states: [B, 85, 768]


Step 4: Generate Answer Conditioned on Reasoning
-------------------------------------------------
combined_encoder_hidden = concat([
    encoder_hidden,           # [B, 1, 768]
    reasoning_hidden_states   # [B, 85, 768]
], dim=1)
# Result: [B, 86, 768]

answer_ids = decoder.generate(
    inputs_embeds=combined_encoder_hidden,  # KEY: Uses reasoning!
    max_length=32,
    num_beams=1
)
# Result: [B, L_a] (e.g., [B, 8] actual length)

# Decode to text
answer_text = "màu xanh lá"

✅ Answer is conditioned on reasoning!
"""


# ============================================================================
# 4. KEY DIFFERENCES vs OLD MODEL
# ============================================================================

"""
OLD MODEL (Parallel):
---------------------
1. Reasoning and Answer decoded independently
2. Both use SAME encoder_hidden
3. No information flow from reasoning to answer
4. Answer loss weight must be tuned carefully
5. Reasoning length doesn't affect answer quality much

❌ Problem: Not true chain-of-thought!


NEW MODEL (Sequential):
-----------------------
1. Reasoning decoded FIRST
2. Answer decoder sees reasoning hidden states
3. Clear information flow: reasoning → answer
4. Answer explicitly conditioned on reasoning
5. Longer reasoning = more context for answer

✅ True chain-of-thought reasoning!
"""


# ============================================================================
# 5. EXPECTED TRAINING BEHAVIOR
# ============================================================================

"""
Epoch 1-2: Model learns basic reasoning
----------------------------------------
[EPOCH 1] Train Loss: 2.5432
  reasoning_loss: 3.1245  ← High (still learning)
  answer_loss: 1.8234     ← Lower (simpler task)
  confidence_scale: 0.7828
  total_loss: 2.5432

[VALIDATION] Loss: 2.4156
  reasoning_loss: 2.9876  ← Improving
  answer_loss: 1.7345
  
✅ Reasoning loss should be higher initially


Epoch 5-10: Model learns reasoning-to-answer mapping
-----------------------------------------------------
[EPOCH 5] Train Loss: 1.2345
  reasoning_loss: 1.4567  ← Much better
  answer_loss: 0.9123     ← Also improving
  confidence_scale: 0.7828
  total_loss: 1.2345

[VALIDATION] Loss: 1.1234
  reasoning_loss: 1.3456
  answer_loss: 0.8234
  
✅ Both losses decrease together


Epoch 15+: Model converges
---------------------------
[EPOCH 15] Train Loss: 0.6789
  reasoning_loss: 0.8234  ← Stable
  answer_loss: 0.4567     ← Stable
  confidence_scale: 0.7828
  total_loss: 0.6789

[VALIDATION] Loss: 0.6543
  reasoning_loss: 0.7891
  answer_loss: 0.4321
  
✅ Both losses stable and low
"""


# ============================================================================
# 6. ADVANTAGES OF SEQUENTIAL CoT
# ============================================================================

"""
1. True Chain-of-Thought
   - Model learns to reason first, then answer
   - Mimics human cognitive process
   
2. Better Answer Quality
   - Answer conditioned on reasoning context
   - Longer reasoning = more context = better answer
   
3. Interpretability
   - Can inspect reasoning output
   - Understand why model gave specific answer
   
4. Training Stability
   - Clear information flow
   - Easier to debug (can check reasoning separately)
   
5. Scalability
   - Can extend to multi-step reasoning
   - Can add more intermediate steps
"""


# ============================================================================
# 7. POTENTIAL ISSUES & SOLUTIONS
# ============================================================================

"""
Issue 1: Slower Inference
--------------------------
Problem: Need to generate reasoning first (128 tokens) before answer (32 tokens)
Solution: 
  - Use shorter max_reasoning_length (e.g., 64 instead of 128)
  - Use beam_search only for answer, greedy for reasoning
  - Cache reasoning for similar questions


Issue 2: Memory Usage
----------------------
Problem: Storing reasoning_hidden_states [B, 128, 768] takes more memory
Solution:
  - Use gradient checkpointing
  - Reduce batch size if needed
  - Use mixed precision (FP16)


Issue 3: Longer Training Time
------------------------------
Problem: Extra forward pass for reasoning
Solution:
  - Use gradient accumulation
  - Train in stages (freeze encoder first)
  - Use distributed training if available


Issue 4: Reasoning Quality Affects Answer
-----------------------------------------
Problem: Bad reasoning → Bad answer
Solution:
  - Ensure high-quality reasoning data from teacher
  - Higher weight on reasoning loss (e.g., 0.7 vs 0.3)
  - Monitor reasoning perplexity separately
"""


# ============================================================================
# 8. TESTING CHECKLIST
# ============================================================================

"""
Before Training:
□ Test forward pass with dummy data
□ Check output shapes match expected
□ Verify backward pass (gradients computed)
□ Test that answer depends on reasoning (not independent)

During Training:
□ Monitor both reasoning_loss and answer_loss
□ Check that reasoning_loss > 0 (not zero!)
□ Verify answer_loss decreases with reasoning_loss
□ Sample reasoning outputs to check quality

After Training:
□ Generate reasoning for test samples
□ Check reasoning makes sense
□ Verify answer matches reasoning
□ Compare accuracy with old model
"""


# ============================================================================
# 9. EXAMPLE OUTPUT
# ============================================================================

"""
Question: "Chiếc bình màu gì?"
Image: [Picture of green vase]

Model Output:
-------------
Reasoning: "Trong ảnh, tôi thấy một chiếc bình. Chiếc bình này có màu xanh 
            lá cây. Nó được đặt trên bàn gỗ."
            
Answer: "màu xanh lá"

✅ Answer is supported by reasoning!


Question: "Có bao nhiêu người trong ảnh?"
Image: [Picture of 3 people]

Model Output:
-------------
Reasoning: "Tôi thấy ba người trong ảnh. Họ đang đứng cạnh nhau. Có hai 
            người đàn ông và một phụ nữ."
            
Answer: "ba người"

✅ Reasoning provides context for answer!
"""


if __name__ == '__main__':
    print(__doc__)
