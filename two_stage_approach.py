"""
TWO-STAGE GENERATION APPROACH
Stage 1: Generate Answer only (short, precise)
Stage 2: Generate Reasoning given Answer (long, detailed)

Ưu điểm:
- Mỗi model đơn giản, dễ train
- Answer accuracy cao hơn (không bị nhiễu từ reasoning)
- Có thể ensemble nhiều reasoning cho 1 answer
"""

import torch
from model import VQAGenModel

# ============================================================================
# STAGE 1: ANSWER GENERATION MODEL
# ============================================================================

class AnswerGenerator:
    """
    Specialized model for answer generation only
    - Max length: 20 tokens (short)
    - Focus: High accuracy
    - Training: Only answer data
    """
    def __init__(self, model_path):
        self.model = VQAGenModel(...)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
    
    def generate_answer(self, image, question):
        """Generate answer only"""
        with torch.no_grad():
            output = self.model.generate(
                pixel_values=image,
                input_ids=question,
                max_length=20,  # Short answer
                num_beams=5,    # Higher beam search for accuracy
                do_sample=False,
                early_stopping=True
            )
        return self.decode(output)


# ============================================================================
# STAGE 2: REASONING GENERATION MODEL
# ============================================================================

class ReasoningGenerator:
    """
    Specialized model for reasoning generation
    - Input: Image + Question + Generated Answer
    - Output: Detailed reasoning
    - Max length: 100 tokens (long)
    """
    def __init__(self, model_path):
        self.model = VQAGenModel(...)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
    
    def generate_reasoning(self, image, question, answer):
        """Generate reasoning given answer"""
        # Augment question with answer
        augmented_q = f"{question} Đáp án: {answer}. Giải thích:"
        
        with torch.no_grad():
            output = self.model.generate(
                pixel_values=image,
                input_ids=self.encode(augmented_q),
                max_length=100,  # Longer reasoning
                num_beams=3,
                do_sample=True,
                temperature=0.7,  # More diverse
                top_p=0.9
            )
        return self.decode(output)


# ============================================================================
# TWO-STAGE PIPELINE
# ============================================================================

class TwoStageVQA:
    """Complete two-stage VQA system"""
    
    def __init__(self, answer_model_path, reasoning_model_path):
        self.answer_gen = AnswerGenerator(answer_model_path)
        self.reasoning_gen = ReasoningGenerator(reasoning_model_path)
    
    def predict(self, image, question):
        # Stage 1: Generate answer
        answer = self.answer_gen.generate_answer(image, question)
        
        # Stage 2: Generate reasoning
        reasoning = self.reasoning_gen.generate_reasoning(
            image, question, answer
        )
        
        return {
            "answer": answer,
            "reasoning": reasoning
        }


# ============================================================================
# TRAINING DATA PREPARATION
# ============================================================================

def prepare_two_stage_data(teacher_outputs_jsonl):
    """
    Split data into 2 datasets:
    1. Answer dataset: (image, question) -> answer
    2. Reasoning dataset: (image, question, answer) -> reasoning
    """
    import json
    
    answer_data = []
    reasoning_data = []
    
    with open(teacher_outputs_jsonl) as f:
        for line in f:
            item = json.loads(line)
            
            # Dataset 1: Answer generation
            answer_data.append({
                "image_path": item["image_path"],
                "question": item["question"],
                "target": item["teacher_answer"]  # Only answer
            })
            
            # Dataset 2: Reasoning generation
            # Input includes answer as context
            reasoning_data.append({
                "image_path": item["image_path"],
                "question": f"{item['question']} Đáp án: {item['teacher_answer']}.",
                "target": item["teacher_reasoning"]  # Only reasoning
            })
    
    return answer_data, reasoning_data


# ============================================================================
# TRAINING SCRIPTS
# ============================================================================

# train_answer_model.py
"""
Simple single-task training for answer generation
- Loss: Cross-entropy only
- Max length: 20
- Focus: High accuracy
"""

# train_reasoning_model.py  
"""
Single-task training for reasoning generation
- Input: Question + Answer (as context)
- Loss: Cross-entropy only
- Max length: 100
- Focus: Coherent explanation
"""


# ============================================================================
# ADVANTAGES
# ============================================================================

"""
1. SIMPLE: Mỗi model chỉ có 1 task duy nhất
2. RELIABLE: Answer không bị ảnh hưởng bởi reasoning generation
3. FLEXIBLE: Có thể swap reasoning model mà không ảnh hưởng answer
4. ENSEMBLE: Có thể generate nhiều reasoning cho 1 answer
5. DEBUGGING: Dễ debug từng stage riêng biệt

Nhược điểm:
- Cần train 2 models (tốn thời gian x2)
- Inference chậm hơn (2 forward passes)
- Cần 2x storage cho models
"""
