#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Example training script for SAM 3D Objects.

This script demonstrates training dynamics without a real dataloader.
It uses dummy data to verify the training pipeline works correctly.

Usage:
    python sam3d_objects/train/train.py --config checkpoints/hf/pipeline.yaml
"""

import argparse
from loguru import logger

import torch


def main():
    parser = argparse.ArgumentParser(description="Train SAM 3D Objects")
    parser.add_argument(
        "--config",
        type=str,
        default="checkpoints/hf/pipeline.yaml",
        help="Path to pipeline config",
    )
    parser.add_argument(
        "--load-pretrained",
        action="store_true",
        help="Load pretrained weights",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum training steps (for demo)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--use-amp",
        action="store_true",
        default=True,
        help="Use automatic mixed precision",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/train",
        help="Directory to save checkpoints",
    )
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("SAM 3D Objects Training Script")
    logger.info("=" * 60)
    
    # Import here to avoid slow imports when just checking help
    from sam3d_objects.train import TrainPipeline, Trainer, TrainerConfig, create_dummy_batch
    
    # Initialize pipeline
    logger.info(f"Loading pipeline from: {args.config}")
    pipeline = TrainPipeline(
        args.config,
        load_pretrained=args.load_pretrained,
    )
    
    # Initialize trainer
    config = TrainerConfig(
        lr=args.lr,
        max_steps=args.max_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_amp=args.use_amp,
        checkpoint_dir=args.checkpoint_dir,
        log_every_n_steps=1,
        save_every_n_steps=args.max_steps,  # Save at end
        
        # Training configuration
        train_ss_generator=True,
        train_slat_generator=False,
        train_decoders=False,
        freeze_condition_embedders=True,
    )
    
    trainer = Trainer(pipeline, config)
    
    # Training loop with dummy data
    logger.info("Starting training loop with dummy data...")
    logger.info(f"This is a demonstration - replace create_dummy_batch with real dataloader")
    logger.info("-" * 60)
    
    total_steps = args.max_steps * args.gradient_accumulation_steps
    
    for step in range(total_steps):
        # Create dummy batch
        batch = create_dummy_batch(batch_size=args.batch_size, device=str(pipeline.device))
        
        # Training step
        try:
            metrics = trainer.train_step(
                batch,
                compute_ss_loss=True,
                compute_slat_loss=False,
            )
        except Exception as e:
            logger.error(f"Training step failed: {e}")
            logger.exception(e)
            break
        
        # Check if we've done enough optimizer steps
        if trainer.global_step >= args.max_steps:
            break
    
    logger.info("-" * 60)
    logger.info(f"Training completed at step {trainer.global_step}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
