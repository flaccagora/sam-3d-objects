# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Training pipeline for SAM 3D Objects.

This module provides functionality to instantiate and optionally load pretrained 
weights for the neural network components used in the SAM 3D Objects pipeline.

Usage:
    from sam3d_objects.train.train_pipeline import TrainPipeline
    
    # Instantiate models without pretrained weights
    pipeline = TrainPipeline("checkpoints/hf/pipeline.yaml", load_pretrained=False)
    
    # Instantiate models with pretrained weights
    pipeline = TrainPipeline("checkpoints/hf/pipeline.yaml", load_pretrained=True)
"""

import os
from typing import Optional, Callable, Any

import torch
from loguru import logger
from omegaconf import OmegaConf
from hydra.utils import instantiate
from safetensors.torch import load_file

from sam3d_objects.model.io import (
    load_model_from_checkpoint,
    filter_and_remove_prefix_state_dict_fn,
)


class TrainPipeline:
    """
    Training pipeline that instantiates neural network models for SAM 3D Objects.
    
    This class handles model instantiation and optional loading of pretrained weights
    for all components: SS Generator, SLAT Generator, decoders, and condition embedders.
    
    Args:
        config_path: Path to the pipeline YAML config file.
        load_pretrained: If True, load pretrained weights from checkpoint files.
        device: Device to place models on (default: "cuda").
        dtype: Data type for models (default: "float16").
    """
    
    def __init__(
        self,
        config_path: str,
        load_pretrained: bool = False,
        device: str = "cuda",
        dtype: str = "float16",
    ):
        self.device = torch.device(device)
        self.dtype = self._get_dtype(dtype)
        self.load_pretrained = load_pretrained
        self.workspace_dir = os.path.dirname(config_path)
        
        # Load main config
        config = OmegaConf.load(config_path)
        
        logger.info(f"Initializing TrainPipeline (load_pretrained={load_pretrained})")
        logger.info(f"Device: {self.device}, dtype: {self.dtype}")
        
        with self.device:
            self._init_models(config)
    
    def _get_dtype(self, dtype: str) -> torch.dtype:
        """Convert string dtype to torch dtype."""
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype: {dtype}. Choose from {list(dtype_map.keys())}")
        return dtype_map[dtype]
    
    def _init_models(self, config: OmegaConf) -> None:
        """Initialize all model components."""
        
        logger.info("Initializing models...")
        
        # Initialize generators
        ss_generator = self._init_ss_generator(
            config.ss_generator_config_path,
            config.ss_generator_ckpt_path if self.load_pretrained else None,
        )
        
        slat_generator = self._init_slat_generator(
            config.slat_generator_config_path,
            config.slat_generator_ckpt_path if self.load_pretrained else None,
        )
        
        # Initialize decoders
        ss_decoder = self._init_ss_decoder(
            config.ss_decoder_config_path,
            config.ss_decoder_ckpt_path if self.load_pretrained else None,
        )
        
        slat_decoder_gs = self._init_slat_decoder(
            config.slat_decoder_gs_config_path,
            config.slat_decoder_gs_ckpt_path if self.load_pretrained else None,
        )
        
        slat_decoder_gs_4 = self._init_slat_decoder(
            config.get("slat_decoder_gs_4_config_path"),
            config.get("slat_decoder_gs_4_ckpt_path") if self.load_pretrained else None,
        )
        
        slat_decoder_mesh = self._init_slat_decoder(
            config.slat_decoder_mesh_config_path,
            config.slat_decoder_mesh_ckpt_path if self.load_pretrained else None,
        )
        
        # Initialize condition embedders
        ss_condition_embedder = self._init_condition_embedder(
            config.ss_generator_config_path,
            config.ss_generator_ckpt_path if self.load_pretrained else None,
            embedder_type="ss",
        )
        
        slat_condition_embedder = self._init_condition_embedder(
            config.slat_generator_config_path,
            config.slat_generator_ckpt_path if self.load_pretrained else None,
            embedder_type="slat",
        )
        
        # Store models in ModuleDict for easy access
        self.models = torch.nn.ModuleDict({
            "ss_generator": ss_generator,
            "slat_generator": slat_generator,
            "ss_decoder": ss_decoder,
            "slat_decoder_gs": slat_decoder_gs,
            "slat_decoder_gs_4": slat_decoder_gs_4,
            "slat_decoder_mesh": slat_decoder_mesh,
        })
        
        self.condition_embedders = {
            "ss_condition_embedder": ss_condition_embedder,
            "slat_condition_embedder": slat_condition_embedder,
        }
        
        logger.info("Model initialization completed!")
        self._log_model_summary()
    
    def _instantiate_model(
        self,
        config: OmegaConf,
        ckpt_path: Optional[str] = None,
        state_dict_fn: Optional[Callable[[Any], Any]] = None,
        state_dict_key: Optional[str] = "state_dict",
    ) -> torch.nn.Module:
        """
        Instantiate a model from config and optionally load pretrained weights.
        
        Args:
            config: OmegaConf config for the model.
            ckpt_path: Path to checkpoint file (None to skip loading weights).
            state_dict_fn: Function to transform state dict before loading.
            state_dict_key: Key to extract state dict from checkpoint.
        
        Returns:
            Instantiated model (optionally with pretrained weights).
        """
        model = instantiate(config)
        
        if ckpt_path is not None:
            full_path = os.path.join(self.workspace_dir, ckpt_path)
            logger.info(f"Loading pretrained weights from: {full_path}")
            
            if full_path.endswith(".safetensors"):
                state_dict = load_file(full_path, device="cuda")
                if state_dict_fn is not None:
                    state_dict = state_dict_fn(state_dict)
                model.load_state_dict(state_dict, strict=False)
            else:
                model = load_model_from_checkpoint(
                    model,
                    full_path,
                    strict=False,  # Use strict=False for training flexibility
                    device="cpu",
                    freeze=False,  # Don't freeze for training
                    eval=False,    # Keep in train mode
                    state_dict_key=state_dict_key,
                    state_dict_fn=state_dict_fn,
                )
        
        model = model.to(self.device)
        return model
    
    def _init_ss_generator(
        self, 
        config_path: str, 
        ckpt_path: Optional[str],
    ) -> torch.nn.Module:
        """Initialize the Sparse Structure Generator."""
        config = OmegaConf.load(
            os.path.join(self.workspace_dir, config_path)
        )["module"]["generator"]["backbone"]
        
        state_dict_fn = filter_and_remove_prefix_state_dict_fn(
            "_base_models.generator."
        ) if ckpt_path else None
        
        model = self._instantiate_model(
            config,
            ckpt_path,
            state_dict_fn=state_dict_fn,
        )
        logger.info(f"Initialized ss_generator: {type(model).__name__}")
        return model
    
    def _init_slat_generator(
        self, 
        config_path: str, 
        ckpt_path: Optional[str],
    ) -> torch.nn.Module:
        """Initialize the Structured Latent Generator."""
        config = OmegaConf.load(
            os.path.join(self.workspace_dir, config_path)
        )["module"]["generator"]["backbone"]
        
        state_dict_fn = filter_and_remove_prefix_state_dict_fn(
            "_base_models.generator."
        ) if ckpt_path else None
        
        model = self._instantiate_model(
            config,
            ckpt_path,
            state_dict_fn=state_dict_fn,
        )
        logger.info(f"Initialized slat_generator: {type(model).__name__}")
        return model
    
    def _init_ss_decoder(
        self, 
        config_path: str, 
        ckpt_path: Optional[str],
    ) -> torch.nn.Module:
        """Initialize the Sparse Structure Decoder."""
        config = OmegaConf.load(
            os.path.join(self.workspace_dir, config_path)
        )
        # Remove pretrained_ckpt_path to avoid double loading
        if "pretrained_ckpt_path" in config:
            del config["pretrained_ckpt_path"]
        
        model = self._instantiate_model(
            config,
            ckpt_path,
            state_dict_key=None,
        )
        logger.info(f"Initialized ss_decoder: {type(model).__name__}")
        return model
    
    def _init_slat_decoder(
        self, 
        config_path: Optional[str], 
        ckpt_path: Optional[str],
    ) -> Optional[torch.nn.Module]:
        """Initialize a SLAT decoder (Gaussian or Mesh)."""
        if config_path is None:
            return None
        
        config = OmegaConf.load(
            os.path.join(self.workspace_dir, config_path)
        )
        
        model = self._instantiate_model(
            config,
            ckpt_path,
            state_dict_key=None,
        )
        logger.info(f"Initialized slat_decoder: {type(model).__name__}")
        return model
    
    def _init_condition_embedder(
        self,
        generator_config_path: str,
        generator_ckpt_path: Optional[str],
        embedder_type: str,
    ) -> Optional[torch.nn.Module]:
        """Initialize condition embedder (for image/mask encoding)."""
        config = OmegaConf.load(
            os.path.join(self.workspace_dir, generator_config_path)
        )
        
        if "condition_embedder" not in config["module"]:
            logger.warning(f"No condition_embedder found in {generator_config_path}")
            return None
        
        embedder_config = config["module"]["condition_embedder"]["backbone"]
        
        state_dict_fn = filter_and_remove_prefix_state_dict_fn(
            "_base_models.condition_embedder."
        ) if generator_ckpt_path else None
        
        model = self._instantiate_model(
            embedder_config,
            generator_ckpt_path,
            state_dict_fn=state_dict_fn,
        )
        logger.info(f"Initialized {embedder_type}_condition_embedder: {type(model).__name__}")
        return model
    
    def _log_model_summary(self) -> None:
        """Log a summary of all initialized models."""
        logger.info("=" * 60)
        logger.info("Model Summary:")
        logger.info("=" * 60)
        
        total_params = 0
        trainable_params = 0
        
        for name, model in self.models.items():
            if model is None:
                logger.info(f"  {name}: None")
                continue
            
            n_params = sum(p.numel() for p in model.parameters())
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params += n_params
            trainable_params += n_trainable
            
            logger.info(f"  {name}: {n_params:,} params ({n_trainable:,} trainable)")
        
        for name, model in self.condition_embedders.items():
            if model is None:
                logger.info(f"  {name}: None")
                continue
            
            n_params = sum(p.numel() for p in model.parameters())
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params += n_params
            trainable_params += n_trainable
            
            logger.info(f"  {name}: {n_params:,} params ({n_trainable:,} trainable)")
        
        logger.info("-" * 60)
        logger.info(f"Total: {total_params:,} params ({trainable_params:,} trainable)")
        logger.info("=" * 60)
    
    def get_model(self, name: str) -> torch.nn.Module:
        """Get a model by name."""
        if name in self.models:
            return self.models[name]
        if name in self.condition_embedders:
            return self.condition_embedders[name]
        raise KeyError(f"Model '{name}' not found. Available: {list(self.models.keys()) + list(self.condition_embedders.keys())}")
    
    def train(self) -> None:
        """Set all models to training mode."""
        for model in self.models.values():
            if model is not None:
                model.train()
        for model in self.condition_embedders.values():
            if model is not None:
                model.train()
        logger.info("All models set to training mode")
    
    def eval(self) -> None:
        """Set all models to evaluation mode."""
        for model in self.models.values():
            if model is not None:
                model.eval()
        for model in self.condition_embedders.values():
            if model is not None:
                model.eval()
        logger.info("All models set to evaluation mode")
    
    def freeze(self, model_names: Optional[list[str]] = None) -> None:
        """
        Freeze model parameters.
        
        Args:
            model_names: List of model names to freeze. If None, freeze all.
        """
        all_models = {**self.models, **{k: v for k, v in self.condition_embedders.items()}}
        
        if model_names is None:
            model_names = list(all_models.keys())
        
        for name in model_names:
            if name not in all_models:
                logger.warning(f"Model '{name}' not found, skipping freeze")
                continue
            model = all_models[name]
            if model is not None:
                for param in model.parameters():
                    param.requires_grad = False
                logger.info(f"Frozen: {name}")
    
    def unfreeze(self, model_names: Optional[list[str]] = None) -> None:
        """
        Unfreeze model parameters.
        
        Args:
            model_names: List of model names to unfreeze. If None, unfreeze all.
        """
        all_models = {**self.models, **{k: v for k, v in self.condition_embedders.items()}}
        
        if model_names is None:
            model_names = list(all_models.keys())
        
        for name in model_names:
            if name not in all_models:
                logger.warning(f"Model '{name}' not found, skipping unfreeze")
                continue
            model = all_models[name]
            if model is not None:
                for param in model.parameters():
                    param.requires_grad = True
                logger.info(f"Unfrozen: {name}")


if __name__ == "__main__":
    # from sam3d_objects.train import TrainPipeline

    # Without pretrained weights (fresh initialization)
    pipeline = TrainPipeline("checkpoints/hf/pipeline.yaml", load_pretrained=False)

    # With pretrained weights
    pipeline = TrainPipeline("checkpoints/hf/pipeline.yaml", load_pretrained=True)

    # Access individual models
    ss_gen = pipeline.get_model("ss_generator")

    # Freeze condition embedders, train generators
    pipeline.freeze(["ss_condition_embedder", "slat_condition_embedder"])
    pipeline.train()

    # dummy forward pass can be added here for testing purposes
    input = torch.randn(1, 3, 224, 224).to(pipeline.device)
    ss_condition_embedder = pipeline.get_model("ss_condition_embedder")
    if ss_condition_embedder is not None:
        with torch.no_grad():
            output = ss_condition_embedder(input)
            logger.info(f"ss_condition_embedder output shape: {output.shape}")
    