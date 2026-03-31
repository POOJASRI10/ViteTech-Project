"""
Step 4: Fine-Tuning - Simplified training without Trainer API

This script demonstrates fine-tuning with a simple training loop.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import argparse

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleFineTuner:
    """Simple fine-tuning without Trainer API"""
    
    def __init__(self, model_name: str = "microsoft/trocr-small-handwritten"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
        
        # Set pad token
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.model.config.decoder_start_token_id = self.processor.tokenizer.bos_token_id
        
        logger.info("Model loaded successfully")
    
    def load_dataset(self, data_file: str) -> List[Dict]:
        """Load dataset from JSON"""
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} samples from {data_file}")
        return data
    
    def finetune(self, train_file: str, val_file: str, output_dir: str, 
                 num_epochs: int = 3, lr: float = 5e-5):
        """Fine-tune model"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        train_data = self.load_dataset(train_file)
        val_data = self.load_dataset(val_file)
        
        # Optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Training
            self.model.train()
            train_loss = 0.0
            
            with tqdm(train_data, desc="Training") as pbar:
                for sample in pbar:
                    try:
                        # Load image
                        image_path = sample["image_path"]
                        image = Image.open(image_path).convert("RGB")
                        
                        # Get text
                        text = sample.get("text_normalized", "")
                        
                        # Process
                        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
                        labels = self.processor.tokenizer(text, return_tensors="pt", padding=True, 
                                                         truncation=True, max_length=128).input_ids.to(self.device)
                        
                        # Forward pass
                        outputs = self.model(pixel_values=pixel_values, decoder_input_ids=labels)
                        loss = outputs.loss
                        
                        # Backward pass
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        optimizer.step()
                        
                        train_loss += loss.item()
                        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                    
                    except Exception as e:
                        logger.debug(f"Error processing sample: {e}")
                        continue
            
            avg_train_loss = train_loss / len(train_data) if len(train_data) > 0 else 0
            logger.info(f"Avg training loss: {avg_train_loss:.4f}")
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                with tqdm(val_data, desc="Validation") as pbar:
                    for sample in pbar:
                        try:
                            image_path = sample["image_path"]
                            image = Image.open(image_path).convert("RGB")
                            text = sample.get("text_normalized", "")
                            
                            pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
                            labels = self.processor.tokenizer(text, return_tensors="pt", padding=True, 
                                                             truncation=True, max_length=128).input_ids.to(self.device)
                            
                            outputs = self.model(pixel_values=pixel_values, decoder_input_ids=labels)
                            loss = outputs.loss
                            val_loss += loss.item()
                            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                        
                        except Exception as e:
                            logger.debug(f"Error in validation: {e}")
                            continue
            
            avg_val_loss = val_loss / len(val_data) if len(val_data) > 0 else 0
            logger.info(f"Avg validation loss: {avg_val_loss:.4f}")
            
            # Save best model
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                self.model.save_pretrained(output_dir)
                self.processor.save_pretrained(output_dir)
                logger.info(f"✓ Saved best model to {output_dir}")
        
        logger.info("✓ Fine-tuning complete!")
        return output_dir


def main():
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR model")
    parser.add_argument("--train_data", type=str, default="data/processed/demo/train.json",
                       help="Path to training data JSON")
    parser.add_argument("--val_data", type=str, default="data/processed/demo/val.json",
                       help="Path to validation data JSON")
    parser.add_argument("--model_name", type=str, default="microsoft/trocr-small-handwritten",
                       help="Base model name")
    parser.add_argument("--output_dir", type=str, default="models/finetuned_trocr",
                       help="Output directory for trained model")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate")
    
    args = parser.parse_args()
    
    tuner = SimpleFineTuner(model_name=args.model_name)
    output_dir = tuner.finetune(
        train_file=args.train_data,
        val_file=args.val_data,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        lr=args.learning_rate
    )
    
    logger.info(f"\n✓ Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
