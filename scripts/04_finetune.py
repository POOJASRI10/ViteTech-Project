"""
Step 4: Fine-Tuning - Train TrOCR model on IAM dataset

This script:
- Sets up training configuration
- Fine-tunes the model on training data
- Validates on validation set
- Saves best checkpoint
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from transformers import (
    TrOCRProcessor, VisionEncoderDecoderModel,
    Seq2SeqTrainer, Seq2SeqTrainingArguments
)
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IAMDataset(Dataset):
    """Dataset for IAM Handwriting"""
    
    def __init__(self, data_file: str, processor, max_target_length: int = 128):
        """
        Initialize dataset.
        
        Args:
            data_file: Path to JSON file with dataset split
            processor: TrOCRProcessor for image processing
            max_target_length: Maximum length of target text
        """
        self.processor = processor
        self.max_target_length = max_target_length
        
        # Load data
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        logger.info(f"Loaded {len(self.data)} samples from {data_file}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get single sample"""
        sample = self.data[idx]
        
        # Load image
        image_path = sample["image_path"]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Cannot load {image_path}: {e}")
            # Return blank image on error
            image = Image.new("RGB", (280, 64))
        
        # Get text
        text = sample.get("text_normalized", sample.get("text", ""))
        
        # Process image
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        
        # Tokenize text
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        return {
            "pixel_values": pixel_values.squeeze(0),
            "labels": labels.squeeze(0),
            "text": text,
        }


class OCRModelTrainer:
    """Train OCR model"""
    
    def __init__(self, model_name: str = "microsoft/trocr-small-handwritten",
                 output_dir: str = "models/finetuned-trocr"):
        """
        Initialize trainer.
        
        Args:
            model_name: Base model name
            output_dir: Output directory for checkpoints
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Using device: {self.device}")
        
        # Load processor and model
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.model.to(self.device)
    
    def create_dataloaders(self, train_file: str, val_file: str,
                          batch_size: int = 8, num_workers: int = 0) -> tuple:
        """Create train and validation dataloaders"""
        logger.info("Creating dataloaders...")
        
        # Create datasets
        train_dataset = IAMDataset(train_file, self.processor)
        val_dataset = IAMDataset(val_file, self.processor)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        return train_loader, val_loader
    
    def train_simple(self, train_loader: DataLoader, val_loader: DataLoader,
                    num_epochs: int = 3, learning_rate: float = 5e-5,
                    warmup_steps: int = 100, weight_decay: float = 0.0005) -> Dict[str, Any]:
        """
        Simple training loop (alternative to Seq2SeqTrainer).
        
        Use this for more control and debugging.
        """
        logger.info("Starting simple training loop...")
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        training_history = {
            "loss": [],
            "val_loss": [],
            "best_val_loss": float('inf'),
            "best_checkpoint": None,
        }
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Training
            self.model.train()
            train_loss = 0.0
            
            with tqdm(train_loader, desc="Training") as pbar:
                for batch_idx, batch in enumerate(pbar):
                    # Move to device
                    pixel_values = batch["pixel_values"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    
                    # Forward pass
                    outputs = self.model(
                        pixel_values=pixel_values,
                        labels=labels
                    )
                    loss = outputs.loss
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    
                    train_loss += loss.item()
                    pbar.set_postfix({"loss": loss.item()})
            
            avg_train_loss = train_loss / len(train_loader)
            training_history["loss"].append(avg_train_loss)
            logger.info(f"Avg training loss: {avg_train_loss:.4f}")
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validating"):
                    pixel_values = batch["pixel_values"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    
                    outputs = self.model(
                        pixel_values=pixel_values,
                        labels=labels
                    )
                    val_loss += outputs.loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            training_history["val_loss"].append(avg_val_loss)
            logger.info(f"Avg validation loss: {avg_val_loss:.4f}")
            
            # Save best checkpoint
            if avg_val_loss < training_history["best_val_loss"]:
                training_history["best_val_loss"] = avg_val_loss
                checkpoint_path = Path(self.output_dir) / f"checkpoint-epoch-{epoch+1}"
                self._save_checkpoint(checkpoint_path)
                training_history["best_checkpoint"] = str(checkpoint_path)
                logger.info(f"✓ Saved best checkpoint to {checkpoint_path}")
        
        return training_history
    
    def _save_checkpoint(self, path: str):
        """Save model checkpoint"""
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)
        logger.info(f"Checkpoint saved to {path}")
    
    def save_model(self):
        """Save final model"""
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(output_path)
        self.processor.save_pretrained(output_path)
        
        logger.info(f"Model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR model")
    parser.add_argument("--train_data", type=str, default="data/processed/train.json",
                       help="Path to training data JSON")
    parser.add_argument("--val_data", type=str, default="data/processed/val.json",
                       help="Path to validation data JSON")
    parser.add_argument("--model_name", type=str, default="microsoft/trocr-small-handwritten",
                       help="Base model name")
    parser.add_argument("--output_dir", type=str, default="models/finetuned-trocr",
                       help="Output directory for trained model")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100,
                       help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.0005,
                       help="Weight decay")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = OCRModelTrainer(
        model_name=args.model_name,
        output_dir=args.output_dir
    )
    
    # Create dataloaders
    train_loader, val_loader = trainer.create_dataloaders(
        args.train_data,
        args.val_data,
        batch_size=args.batch_size
    )
    
    # Train
    history = trainer.train_simple(
        train_loader,
        val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay
    )
    
    # Save model
    trainer.save_model()
    
    # Save training history
    history_path = Path(args.output_dir) / "training_history.json"
    with open(history_path, 'w') as f:
        # Convert best_checkpoint to string
        history_to_save = {
            "loss": history["loss"],
            "val_loss": history["val_loss"],
            "best_val_loss": history["best_val_loss"],
            "best_checkpoint": history["best_checkpoint"],
        }
        json.dump(history_to_save, f, indent=2)
    
    logger.info(f"Training history saved to {history_path}")
    logger.info("✓ Fine-tuning complete!")
    
    # Print summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Best validation loss: {history['best_val_loss']:.4f}")
    print(f"Best checkpoint: {history['best_checkpoint']}")
    print(f"Final training loss: {history['loss'][-1]:.4f}")
    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
