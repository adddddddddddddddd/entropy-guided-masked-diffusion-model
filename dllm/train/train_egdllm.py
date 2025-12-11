"""
Training script for Entropy-Guided Masked Diffusion Model (EGMDM).
"""
import json
import math
import os
from pathlib import Path
from collections import Counter

import torch
import transformers
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from dllm.core.schedulers import LinearAlphaScheduler, CosineAlphaScheduler
from dllm.core.trainers.mdlm import MDLMTrainer
from dllm.pipelines.llada.models import LLaDAConfig, LLaDAModelLM


def create_frequency_dict(dataset, tokenizer, num_samples=10000, output_path="frequency_dict.json"):
    """
    Create a frequency dictionary from the dataset.
    
    Args:
        dataset: HuggingFace dataset
        tokenizer: Tokenizer to use
        num_samples: Number of samples to process
        output_path: Where to save the frequency dict
    
    Returns:
        dict: Token ID -> frequency (normalized to sum to 1)
    """
    print(f"Building frequency dictionary from {num_samples} samples...")
    
    token_counts = Counter()
    total_tokens = 0
    
    for i, example in enumerate(dataset):
        if i >= num_samples:
            break
        
        # Tokenize the text and count the occurences 
        # Probabilities are estimated with frequency
        tokens = tokenizer(example["text"], truncation=True, max_length=512)["input_ids"]
        token_counts.update(tokens)
        total_tokens += len(tokens)
        
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{num_samples} samples...")
    
    # Normalize to get frequencies (probabilities)
    frequency_dict = {
        token_id: count / total_tokens 
        for token_id, count in token_counts.items()
    }
    
    # Add small epsilon for unseen tokens
    vocab_size = tokenizer.vocab_size
    epsilon = 1e-10
    for token_id in range(vocab_size):
        if token_id not in frequency_dict:
            frequency_dict[token_id] = epsilon
    
    # Save to file
    with open(output_path, "w") as f:
        json.dump(frequency_dict, f)
    
    print(f"Frequency dict saved to {output_path}")
    print(f"Total unique tokens: {len(token_counts)}")
    print(f"Total tokens processed: {total_tokens}")
    
    return frequency_dict


def load_and_prepare_dataset(tokenizer, max_length=512, num_train_samples=50000, num_eval_samples=5000):
    """
    Load FineWeb dataset and prepare it for training.
    
    Args:
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        num_train_samples: Number of training samples
        num_eval_samples: Number of evaluation samples
    
    Returns:
        tuple: (train_dataset, eval_dataset)
    """
    print("Loading FineWeb dataset...")
    
    # Load a subset of FineWeb (sample version for faster loading)
    # You can use "HuggingFaceFW/fineweb" for the full dataset
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    
    # Take a subset for training
    dataset = dataset.take(num_train_samples + num_eval_samples)
    
    def tokenize_function(examples):
        # Tokenize texts
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized
    
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )
    
    # Split into train and eval
    train_dataset = tokenized_dataset.take(num_train_samples)
    eval_dataset = tokenized_dataset.skip(num_train_samples).take(num_eval_samples)
    
    return train_dataset, eval_dataset


def create_small_model(vocab_size, max_seq_length=512):
    """
    Create a small LLaDA model suitable for single GPU training.
    
    Args:
        vocab_size: Size of the vocabulary
        max_seq_length: Maximum sequence length
    
    Returns:
        LLaDAModelLM: Small model instance
    """
    config = LLaDAConfig(
        # Model architecture
        d_model=512,              # Hidden dimension (small for single GPU)
        n_heads=8,                # Number of attention heads
        n_layers=6,               # Number of transformer layers (small)
        mlp_ratio=4,              # MLP expansion ratio
        vocab_size=vocab_size,
        max_sequence_length=max_seq_length,
        
        # Training settings
        weight_tying=True,        # Share input/output embeddings
        rope=True,                # Use RoPE (required for MDLM)
        alibi=False,              # Don't use ALiBi
        flash_attention=False,    # Disable flash attention for compatibility
        
        # Regularization
        attention_dropout=0.1,
        embedding_dropout=0.1,
        residual_dropout=0.1,
        
        # Initialization
        init_device="cuda",
        init_fn="normal",
        init_std=0.02,
    )
    
    model = LLaDAModelLM(config, init_params=True)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {num_params:,} parameters")
    
    return model


def main():
    # ============================================
    # 1. Setup
    # ============================================
    output_dir = "./egdllm_output"
    os.makedirs(output_dir, exist_ok=True)
    
    max_length = 512
    batch_size = 8  # Adjust based on your GPU memory
    num_train_samples = 50000
    num_eval_samples = 5000
    num_epochs = 3
    
    # Temperature for entropy-guided masking
    T = 1.0
    
    # ============================================
    # 2. Load tokenizer
    # ============================================
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Add special tokens if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})
    
    vocab_size = len(tokenizer)
    print(f"Vocab size: {vocab_size}")
    
    # ============================================
    # 3. Create or load frequency dictionary
    # ============================================
    frequency_dict_path = os.path.join(output_dir, "frequency_dict.json")
    
    if os.path.exists(frequency_dict_path):
        print(f"Loading existing frequency dict from {frequency_dict_path}")
        with open(frequency_dict_path, "r") as f:
            frequency_dict = json.load(f)
        # Convert keys back to int
        frequency_dict = {int(k): v for k, v in frequency_dict.items()}
    else:
        # Load dataset for frequency analysis
        freq_dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            "sample-10BT",
            split="train",
            streaming=True
        )
        frequency_dict = create_frequency_dict(
            freq_dataset,
            tokenizer,
            num_samples=10000,
            output_path=frequency_dict_path
        )
    
    # ============================================
    # 4. Load and prepare dataset
    # ============================================
    train_dataset, eval_dataset = load_and_prepare_dataset(
        tokenizer,
        max_length=max_length,
        num_train_samples=num_train_samples,
        num_eval_samples=num_eval_samples
    )
    
    # ============================================
    # 5. Create model
    # ============================================
    print("Creating model...")
    model = create_small_model(vocab_size, max_seq_length=max_length)
    
    # ============================================
    # 6. Setup training arguments
    # ============================================
    training_args = TrainingArguments(
        output_dir=output_dir,
        
        # Training hyperparameters
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,  # Effective batch size = 32
        
        # Optimization
        learning_rate=5e-4,
        weight_decay=0.01,
        warmup_steps=500,
        max_grad_norm=1.0,
        
        # Logging and evaluation
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=3,
        
        # Performance
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        
        # Reproducibility
        seed=42,
        
        # Reporting
        report_to=["tensorboard"],
        logging_dir=os.path.join(output_dir, "logs"),
    )
    
    # ============================================
    # 7. Create scheduler
    # ============================================
    scheduler = CosineAlphaScheduler()  # or LinearAlphaScheduler()
    
    # ============================================
    # 8. Create trainer
    # ============================================
    trainer = MDLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        
        # MDLM-specific args
        scheduler=scheduler,
        time_epsilon=1e-3,
        loss_weight_type="scheduler",
        right_shift_logits=False,
        
        # Pass frequency dict and temperature to compute_loss
        # Note: You'll need to modify compute_loss to accept these properly
        # For now, we'll pass them via the trainer's attributes
    )
    
    # Store frequency_dict and T as attributes for compute_loss
    trainer.frequency_dict = frequency_dict
    trainer.T = T
    
    # ============================================
    # 9. Train
    # ============================================
    print("Starting training...")
    trainer.train()
    
    # ============================================
    # 10. Save final model
    # ============================================
    print("Saving final model...")
    trainer.save_model(os.path.join(output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))
    
    print("Training complete!")


if __name__ == "__main__":
    main()