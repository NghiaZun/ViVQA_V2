"""
utils_prompt.py – Prompt utilities for teacher reasoning (XML format)
Author: Nghia Duong (refined)
"""

# ===========================
# SYSTEM PROMPT
# ===========================
SYSTEM_PROMPT = (
    "Bạn là mô hình Visual Question Answering tiếng Việt. "
    "Luôn trả lời theo đúng chuẩn định dạng XML sau:\n\n"
    "<answer>Câu trả lời ngắn</answer>\n"
    "<reasoning>[LOẠI_REASONING] 1-2 câu giải thích</reasoning>\n\n"
    "Trong đó LOẠI_REASONING thuộc 1 trong các loại: "
    "DESCRIPTIVE, CAUSAL, SPATIAL, COUNTING, OBJECT, COMMONSENSE, INTENT."
)

# ===========================
# FEW-SHOT EXAMPLES
# ===========================
FEW_SHOTS = [
    {
        "question": "Màu của chiếc bình là gì?",
        "answer": (
            "<answer>Màu xanh lá</answer>\n"
            "<reasoning>[DESCRIPTIVE] Chiếc bình trong ảnh có màu xanh lá.</reasoning>"
        )
    },
    {
        "question": "Cô gái đang ngồi ở đâu?",
        "answer": (
            "<answer>trên giường</answer>\n"
            "<reasoning>[SPATIAL] Cô gái ngồi trên bề mặt có chăn gối của giường.</reasoning>"
        )
    },
    {
        "question": "Có bao nhiêu con chim đậu trên cành?",
        "answer": (
            "<answer>hai</answer>\n"
            "<reasoning>[COUNTING] Có hai con chim đứng sát nhau, không có con nào khác.</reasoning>"
        )
    },
    {
        "question": "Tại sao người đàn ông đội mũ bảo hiểm?",
        "answer": (
            "<answer>để bảo vệ đầu</answer>\n"
            "<reasoning>[COMMONSENSE] Người lái xe đội mũ bảo hiểm để tránh chấn thương đầu.</reasoning>"
        )
    },
    {
        "question": "Tại sao mặt đất bị ướt?",
        "answer": (
            "<answer>vì trời mưa</answer>\n"
            "<reasoning>[CAUSAL] Có giọt nước và mặt đất thấm nước cho thấy trời đang mưa.</reasoning>"
        )
    },
    {
        "question": "Trong hai chiếc, chiếc nào lớn hơn?",
        "answer": (
            "<answer>chiếc xe tải</answer>\n"
            "<reasoning>[OBJECT] Kích thước xe tải lớn hơn rõ rệt so với xe còn lại.</reasoning>"
        )
    }
]

# ===========================
# BUILD FEWSHOT PROMPT
# ===========================
def build_fewshot_prompt(question: str) -> str:
    """Trả về prompt kèm few-shot, đúng định dạng XML."""
    blocks = []
    for ex in FEW_SHOTS:
        blocks.append(f"Q: {ex['question']}\n{ex['answer']}")

    examples = "\n\n".join(blocks)

    return (
        "Dưới đây là ví dụ mô hình trả lời câu hỏi VQA theo đúng format XML:\n\n"
        f"{examples}\n\n"
        f"Bây giờ, hãy trả lời câu hỏi sau:\n"
        f"Q: {question}"
    )
