# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Training dynamics for SAM 3D Objects.

This module provides the Trainer class that handles:
- Optimizer and scheduler configuration
- Mixed precision training (AMP)
- Gradient accumulation
- Training step logic
- Checkpointing
- Logging and metrics

Usage:
    from sam3d_objects.train.trainer import Trainer
    from sam3d_objects.train.train_pipeline import TrainPipeline
    
    pipeline = TrainPipeline("checkpoints/hf/pipeline.yaml", load_pretrained=True)
    trainer = Trainer(pipeline, lr=1e-4)
    
    # Training loop (dataloader to be implemented separately)
    for batch in dataloader:
        loss = trainer.train_step(batch)
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Union, Literal
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from loguru import logger


@dataclass
class TrainerConfig:
    """Configuration for training dynamics."""
    
    # Optimizer settings
    optimizer: Literal["adamw", "adam", "sgd"] = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    
    # Scheduler settings
    scheduler: Optional[Literal["cosine", "linear", "constant", "warmup_cosine"]] = "warmup_cosine"
    warmup_steps: int = 1000
    max_steps: int = 100000
    min_lr: float = 1e-6
    
    # Training settings
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    use_amp: bool = True
    amp_dtype: Literal["float16", "bfloat16"] = "bfloat16"
    
    # Which models to train
    train_ss_generator: bool = True
    train_slat_generator: bool = False
    train_decoders: bool = False
    freeze_condition_embedders: bool = True
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints/train"
    save_every_n_steps: int = 1000
    
    # Logging
    log_every_n_steps: int = 10


class Trainer:
    """
    Trainer class for SAM 3D Objects.
    
    Handles training dynamics including optimization, scheduling,
    mixed precision, and gradient accumulation.
    """
    
    def __init__(
        self,
        pipeline,  # TrainPipeline or TrainPipelinePyTorch
        config: Optional[TrainerConfig] = None,
        **kwargs,
    ):
        """
        Initialize trainer.
        
        Args:
            pipeline: The training pipeline containing models.
            config: Training configuration. If None, uses defaults.
            **kwargs: Override config values.
        """
        self.pipeline = pipeline
        self.config = config or TrainerConfig()
        
        # Override config with kwargs
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        self.device = pipeline.device
        self.global_step = 0
        self.accumulated_loss = 0.0
        self.accumulated_steps = 0
        
        # Setup training
        self._setup_trainable_params()
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_amp()
        
        logger.info("Trainer initialized")
        self._log_config()
    
    def _setup_trainable_params(self) -> None:
        """Configure which parameters are trainable."""
        config = self.config
        
        # Freeze condition embedders if specified
        if config.freeze_condition_embedders:
            self.pipeline.freeze([
                "ss_condition_embedder",
                "slat_condition_embedder",
            ])
        
        # Configure generator training
        if not config.train_ss_generator:
            self.pipeline.freeze(["ss_generator"])
        else:
            self.pipeline.unfreeze(["ss_generator"])
            
        if not config.train_slat_generator:
            self.pipeline.freeze(["slat_generator"])
        else:
            self.pipeline.unfreeze(["slat_generator"])
        
        # Configure decoder training
        decoder_names = ["ss_decoder", "slat_decoder_gs", "slat_decoder_gs_4", "slat_decoder_mesh"]
        if not config.train_decoders:
            self.pipeline.freeze(decoder_names)
        else:
            self.pipeline.unfreeze(decoder_names)
        
        # Collect trainable parameters
        self.trainable_params = []
        
        for name, model in self.pipeline.models.items():
            if model is not None:
                params = [p for p in model.parameters() if p.requires_grad]
                if params:
                    self.trainable_params.extend(params)
                    logger.info(f"  {name}: {sum(p.numel() for p in params):,} trainable params")
        
        for name, model in self.pipeline.condition_embedders.items():
            if model is not None:
                params = [p for p in model.parameters() if p.requires_grad]
                if params:
                    self.trainable_params.extend(params)
                    logger.info(f"  {name}: {sum(p.numel() for p in params):,} trainable params")
        
        total_trainable = sum(p.numel() for p in self.trainable_params)
        logger.info(f"Total trainable parameters: {total_trainable:,}")
    
    def _setup_optimizer(self) -> None:
        """Setup optimizer."""
        config = self.config
        
        if not self.trainable_params:
            logger.warning("No trainable parameters! Optimizer not created.")
            self.optimizer = None
            return
        
        if config.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.trainable_params,
                lr=config.lr,
                weight_decay=config.weight_decay,
                betas=config.betas,
                eps=config.eps,
            )
        elif config.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                self.trainable_params,
                lr=config.lr,
                betas=config.betas,
                eps=config.eps,
            )
        elif config.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                self.trainable_params,
                lr=config.lr,
                weight_decay=config.weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer}")
        
        logger.info(f"Optimizer: {config.optimizer} (lr={config.lr})")
    
    def _setup_scheduler(self) -> None:
        """Setup learning rate scheduler."""
        config = self.config
        
        if self.optimizer is None or config.scheduler is None:
            self.scheduler = None
            return
        
        if config.scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.max_steps,
                eta_min=config.min_lr,
            )
        elif config.scheduler == "linear":
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=config.min_lr / config.lr,
                total_iters=config.max_steps,
            )
        elif config.scheduler == "warmup_cosine":
            # Warmup + Cosine annealing
            def lr_lambda(step):
                if step < config.warmup_steps:
                    return step / max(1, config.warmup_steps)
                progress = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
                return max(config.min_lr / config.lr, 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item()))
            
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lr_lambda,
            )
        elif config.scheduler == "constant":
            self.scheduler = torch.optim.lr_scheduler.ConstantLR(
                self.optimizer,
                factor=1.0,
                total_iters=config.max_steps,
            )
        else:
            raise ValueError(f"Unknown scheduler: {config.scheduler}")
        
        logger.info(f"Scheduler: {config.scheduler}")
    
    def _setup_amp(self) -> None:
        """Setup automatic mixed precision."""
        config = self.config
        
        if config.use_amp:
            self.amp_dtype = torch.bfloat16 if config.amp_dtype == "bfloat16" else torch.float16
            self.scaler = GradScaler(enabled=(config.amp_dtype == "float16"))
            logger.info(f"AMP enabled with dtype: {config.amp_dtype}")
        else:
            self.amp_dtype = torch.float32
            self.scaler = None
            logger.info("AMP disabled")
    
    def _log_config(self) -> None:
        """Log training configuration."""
        logger.info("=" * 60)
        logger.info("Training Configuration:")
        logger.info("=" * 60)
        for key, value in vars(self.config).items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 60)
    
    def compute_ss_generator_loss(
        self,
        target_data: dict,
        condition_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute loss for SS Generator training.
        
        Args:
            target_data: Dictionary containing target tensors for each modality
                - "shape": [B, 8, 16, 16, 16] - 3D occupancy latent
                - "6drotation_normalized": [B, 6] - rotation
                - "translation": [B, 3] - translation
                - "scale": [B, 3] - scale
                - "translation_scale": [B, 1] - translation scale
            condition_embedding: Condition embedding from embedder [B, seq_len, dim]
        
        Returns:
            loss: Scalar loss tensor
            loss_dict: Dictionary of individual loss components
        """
        ss_generator = self.pipeline.models["ss_generator"]
        
        # The generator's loss method expects x1 (target) and conditionals
        # x1 is a dict of targets for each modality
        loss, loss_dict = ss_generator.loss(
            target_data,
            condition_embedding,
        )
        
        return loss, loss_dict
    
    def compute_slat_generator_loss(
        self,
        target_slat: torch.Tensor,
        condition_embedding: torch.Tensor,
        coords: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute loss for SLAT Generator training.
        
        Args:
            target_slat: Target structured latents [B, C, D, H, W]
            condition_embedding: Condition embedding from embedder
            coords: Sparse structure coordinates
        
        Returns:
            loss: Scalar loss tensor
            loss_dict: Dictionary of individual loss components
        """
        slat_generator = self.pipeline.models["slat_generator"]
        
        loss, loss_dict = slat_generator.loss(
            target_slat,
            condition_embedding,
            coords=coords,
        )
        
        return loss, loss_dict
    
    def train_step(
        self,
        batch: dict,
        compute_ss_loss: bool = True,
        compute_slat_loss: bool = False,
    ) -> dict:
        """
        Perform a single training step.
        
        Args:
            batch: Dictionary containing:
                - "condition_input": Input for condition embedder
                - "ss_target": Target for SS generator (if compute_ss_loss)
                - "slat_target": Target for SLAT generator (if compute_slat_loss)
                - "coords": Sparse structure coords (if compute_slat_loss)
            compute_ss_loss: Whether to compute SS generator loss
            compute_slat_loss: Whether to compute SLAT generator loss
        
        Returns:
            Dictionary of metrics from this step
        """
        if self.optimizer is None:
            raise RuntimeError("No optimizer configured (no trainable parameters)")
        
        self.pipeline.train()
        
        # Determine if we should do optimizer step
        do_optimizer_step = (self.accumulated_steps + 1) % self.config.gradient_accumulation_steps == 0
        
        metrics = {}
        total_loss = torch.tensor(0.0, device=self.device)
        
        # Forward pass with AMP
        with autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.config.use_amp):
            # Get condition embedding
            ss_condition_embedder = self.pipeline.condition_embedders.get("ss_condition_embedder")
            slat_condition_embedder = self.pipeline.condition_embedders.get("slat_condition_embedder")
            
            # Compute SS Generator loss
            if compute_ss_loss and self.config.train_ss_generator:
                condition_input = batch["condition_input"]
                
                # Get condition embedding (frozen)
                # condition_input is a dict of kwargs for the embedder
                with torch.no_grad():
                    ss_cond_emb = ss_condition_embedder(**condition_input)
                
                ss_loss, ss_loss_dict = self.compute_ss_generator_loss(
                    batch["ss_target"],
                    ss_cond_emb,
                )
                total_loss = total_loss + ss_loss
                metrics["ss_loss"] = ss_loss.item()
                metrics.update({f"ss_{k}": v.item() if torch.is_tensor(v) else v for k, v in ss_loss_dict.items()})
            
            # Compute SLAT Generator loss
            if compute_slat_loss and self.config.train_slat_generator:
                condition_input = batch["condition_input"]
                
                # Get condition embedding (frozen)
                with torch.no_grad():
                    slat_cond_emb = slat_condition_embedder(**condition_input)
                
                slat_loss, slat_loss_dict = self.compute_slat_generator_loss(
                    batch["slat_target"],
                    slat_cond_emb,
                    batch["coords"],
                )
                total_loss = total_loss + slat_loss
                metrics["slat_loss"] = slat_loss.item()
                metrics.update({f"slat_{k}": v.item() if torch.is_tensor(v) else v for k, v in slat_loss_dict.items()})
        
        # Scale loss for gradient accumulation
        scaled_loss = total_loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        
        self.accumulated_loss += total_loss.item()
        self.accumulated_steps += 1
        
        # Optimizer step
        if do_optimizer_step:
            # Gradient clipping
            if self.config.max_grad_norm > 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.trainable_params,
                    self.config.max_grad_norm,
                )
            
            # Optimizer step
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            
            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Update metrics
            metrics["loss"] = self.accumulated_loss / self.config.gradient_accumulation_steps
            metrics["lr"] = self.optimizer.param_groups[0]["lr"]
            metrics["step"] = self.global_step
            
            # Reset accumulation
            self.accumulated_loss = 0.0
            self.accumulated_steps = 0
            self.global_step += 1
            
            # Logging
            if self.global_step % self.config.log_every_n_steps == 0:
                self._log_metrics(metrics)
            
            # Checkpointing
            if self.global_step % self.config.save_every_n_steps == 0:
                self.save_checkpoint()
        
        return metrics
    
    def _log_metrics(self, metrics: dict) -> None:
        """Log training metrics."""
        log_str = f"Step {metrics.get('step', self.global_step)}"
        for key, value in metrics.items():
            if key != "step":
                if isinstance(value, float):
                    log_str += f" | {key}: {value:.4f}"
                else:
                    log_str += f" | {key}: {value}"
        logger.info(log_str)
    
    def save_checkpoint(self, path: Optional[str] = None) -> str:
        """
        Save training checkpoint.
        
        Args:
            path: Optional path to save checkpoint. If None, uses config.
        
        Returns:
            Path where checkpoint was saved.
        """
        if path is None:
            os.makedirs(self.config.checkpoint_dir, exist_ok=True)
            path = os.path.join(
                self.config.checkpoint_dir,
                f"checkpoint_step_{self.global_step}.pt"
            )
        
        checkpoint = {
            "global_step": self.global_step,
            "config": vars(self.config),
            "models": {},
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "scaler": self.scaler.state_dict() if self.scaler else None,
        }
        
        # Save trainable model weights
        for name, model in self.pipeline.models.items():
            if model is not None:
                has_trainable = any(p.requires_grad for p in model.parameters())
                if has_trainable:
                    checkpoint["models"][name] = model.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
        return path
    
    def load_checkpoint(self, path: str) -> None:
        """
        Load training checkpoint.
        
        Args:
            path: Path to checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.global_step = checkpoint["global_step"]
        
        # Load model weights
        for name, state_dict in checkpoint.get("models", {}).items():
            if name in self.pipeline.models and self.pipeline.models[name] is not None:
                self.pipeline.models[name].load_state_dict(state_dict)
        
        # Load optimizer state
        if self.optimizer and checkpoint.get("optimizer"):
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        
        # Load scheduler state
        if self.scheduler and checkpoint.get("scheduler"):
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        
        # Load scaler state
        if self.scaler and checkpoint.get("scaler"):
            self.scaler.load_state_dict(checkpoint["scaler"])
        
        logger.info(f"Checkpoint loaded: {path} (step {self.global_step})")
    
    def get_current_lr(self) -> float:
        """Get current learning rate."""
        if self.optimizer is None:
            return 0.0
        return self.optimizer.param_groups[0]["lr"]


def create_dummy_batch(
    batch_size: int = 2,
    device: str = "cuda",
) -> dict:
    """
    Create a dummy batch for testing training dynamics.
    
    This is a placeholder until actual dataloaders are implemented.
    The condition_input keys match what the SS condition embedder expects:
    - image: cropped image [B, 3, 518, 518]
    - mask: cropped mask [B, 3, 518, 518] (repeated to 3 channels)
    - pointmap: cropped pointmap [B, 3, 256, 256]
    - rgb_image: full image [B, 3, 518, 518]
    - rgb_image_mask: full mask [B, 3, 518, 518]
    - rgb_pointmap: full pointmap [B, 3, 256, 256]
    """
    # Dummy condition input matching SS condition embedder kwargs
    # From ss_generator.yaml embedder_list configuration
    condition_input = {
        # DINOv2 image embedder inputs
        "image": torch.randn(batch_size, 3, 518, 518, device=device),
        "rgb_image": torch.randn(batch_size, 3, 518, 518, device=device),
        # DINOv2 mask embedder inputs (mask repeated to 3 channels)
        "mask": torch.randn(batch_size, 3, 518, 518, device=device),
        "rgb_image_mask": torch.randn(batch_size, 3, 518, 518, device=device),
        # PointPatchEmbed inputs
        "pointmap": torch.randn(batch_size, 3, 256, 256, device=device),
        "rgb_pointmap": torch.randn(batch_size, 3, 256, 256, device=device),
    }
    
    # Dummy SS target - multi-modal output for ShortCut model
    # Shape matches latent_mapping in ss_generator config
    ss_target = {
        "shape": torch.randn(batch_size, 4096, 8, device=device),  # [B, 16*16*16, 8]
        "6drotation_normalized": torch.randn(batch_size, 1, 6, device=device),
        "translation": torch.randn(batch_size, 1, 3, device=device),
        "scale": torch.randn(batch_size, 1, 3, device=device),
        "translation_scale": torch.randn(batch_size, 1, 1, device=device),
    }
    
    # Dummy SLAT target
    slat_target = torch.randn(batch_size, 8, 64, 64, 64, device=device)
    
    # Dummy coords for sparse structure
    coords = torch.randint(0, 16, (batch_size, 100, 3), device=device)
    
    return {
        "condition_input": condition_input,
        "ss_target": ss_target,
        "slat_target": slat_target,
        "coords": coords,
    }
