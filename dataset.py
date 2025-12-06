"""
Simple VQA Dataset for evaluation
"""
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class VQAGenDataset(Dataset):
    """
    Simple dataset for VQA evaluation
    Returns: (pixel_values, input_ids, attention_mask, labels)
    """
    def __init__(self, csv_path, image_dir, vision_processor):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.vision_processor = vision_processor
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['img_id']
        
        # Load image
        img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            # Fallback to white image
            image = Image.new("RGB", (224, 224), (255, 255, 255))
        
        # Process image
        pixel_values = self.vision_processor(image, return_tensors="pt").pixel_values[0]
        
        # Return dummy values for input_ids, attention_mask, labels
        # These will be filled by the evaluation script
        import torch
        input_ids = torch.zeros(64, dtype=torch.long)  # Will be replaced
        attention_mask = torch.ones(64, dtype=torch.long)
        labels = torch.zeros(128, dtype=torch.long)  # Will be replaced
        
        return pixel_values, input_ids, attention_mask, labels
