import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from PIL import Image
import pandas as pd
import os

class VQAGenDataset(Dataset):
    def __init__(self, csv_path, image_folder,
                 vision_processor,
                 phobert_tokenizer_name='vinai/phobert-base',
                 vit5_tokenizer_name='VietAI/vit5-base',
                 max_q_len=32, max_a_len=32):

        self.data = pd.read_csv(csv_path)
        self.image_folder = image_folder
        self.vision_processor = vision_processor
        self.q_tokenizer = AutoTokenizer.from_pretrained(phobert_tokenizer_name)
        self.a_tokenizer = AutoTokenizer.from_pretrained(vit5_tokenizer_name)
        self.max_q_len = max_q_len
        self.max_a_len = max_a_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        question, answer, img_id = row['question'], row['answer'], str(row['img_id'])

        # Load image
        img_path = os.path.join(self.image_folder, f"{img_id}.jpg")
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"[Warning] Failed to load image: {img_path} - {e}")
            image = Image.new('RGB', (224, 224), color='white')

        vision_inputs = self.vision_processor(images=image, return_tensors='pt')
        pixel_values = vision_inputs['pixel_values'].squeeze(0)  # (3, H, W)

        # Tokenize question
        q_enc = self.q_tokenizer(question,
                                 truncation=True,
                                 padding='max_length',
                                 max_length=self.max_q_len,
                                 return_tensors='pt')

        input_ids = q_enc['input_ids'].squeeze(0)
        attention_mask = q_enc['attention_mask'].squeeze(0)

        # Tokenize answer (for target)
        a_enc = self.a_tokenizer(answer,
                                 truncation=True,
                                 padding='max_length',
                                 max_length=self.max_a_len,
                                 return_tensors='pt')

        labels = a_enc['input_ids'].squeeze(0)
        labels[labels == self.a_tokenizer.pad_token_id] = -100  # important for loss masking

        return pixel_values, input_ids, attention_mask, labels