# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Pure PyTorch Training Pipeline for SAM 3D Objects.

This module instantiates models using direct Python class constructors instead of 
Hydra's instantiate. This provides clearer control over model architecture and 
makes the code more explicit and debuggable.

Usage:
    from sam3d_objects.train.train_pipeline_pytorch import TrainPipelinePyTorch
    
    # Instantiate models without pretrained weights
    pipeline = TrainPipelinePyTorch(load_pretrained=False)
    
    # Instantiate models with pretrained weights
    pipeline = TrainPipelinePyTorch(
        load_pretrained=True,
        checkpoint_dir="checkpoints/hf"
    )
"""

import os
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
from loguru import logger
from safetensors.torch import load_file

# Import model components directly
from sam3d_objects.model.backbone.dit.embedder.embedder_fuser import EmbedderFuser
from sam3d_objects.model.backbone.dit.embedder.dino import Dino
from sam3d_objects.model.backbone.dit.embedder.pointmap import PointPatchEmbed
from sam3d_objects.model.backbone.generator.shortcut.model import ShortCut
from sam3d_objects.model.backbone.generator.flow_matching.model import FlowMatching, lognorm_sampler
from sam3d_objects.model.backbone.generator.classifier_free_guidance import (
    ClassifierFreeGuidance,
    ClassifierFreeGuidanceWithExternalUnconditionalProbability,
)
from sam3d_objects.model.backbone.tdfy_dit.models.mot_sparse_structure_flow import (
    SparseStructureFlowTdfyWrapper,
)
from sam3d_objects.model.backbone.tdfy_dit.models.structured_latent_flow import (
    SLatFlowModelTdfyWrapper,
)
from sam3d_objects.model.backbone.tdfy_dit.models.sparse_structure_vae import (
    SparseStructureDecoderTdfyWrapper,
)
from sam3d_objects.model.backbone.tdfy_dit.models.structured_latent_vae.decoder_gs import (
    SLatGaussianDecoderTdfyWrapper,
)
from sam3d_objects.model.backbone.tdfy_dit.models.structured_latent_vae.decoder_mesh import (
    SLatMeshDecoderTdfyWrapper,
)
from sam3d_objects.model.backbone.tdfy_dit.models.mm_latent import (
    Latent,
    LearntPositionEmbedder,
    ShapePositionEmbedder,
)
from sam3d_objects.model.io import (
    load_model_from_checkpoint,
    filter_and_remove_prefix_state_dict_fn,
)


class TrainPipelinePyTorch:
    """
    Pure PyTorch training pipeline with explicit model instantiation.
    
    All models are instantiated using direct Python constructors,
    providing full transparency and control over model architecture.
    """

    def __init__(
        self,
        load_pretrained: bool = False,
        checkpoint_dir: str = "checkpoints/hf",
        device: str = "cuda",
        dtype: str = "float16",
    ):
        self.device = torch.device(device)
        self.dtype = self._get_dtype(dtype)
        self.load_pretrained = load_pretrained
        self.checkpoint_dir = checkpoint_dir

        logger.info(f"Initializing TrainPipelinePyTorch (load_pretrained={load_pretrained})")
        logger.info(f"Device: {self.device}, dtype: {self.dtype}")

        with self.device:
            self._init_models()

    def _get_dtype(self, dtype: str) -> torch.dtype:
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype: {dtype}")
        return dtype_map[dtype]

    def _init_models(self) -> None:
        """Initialize all models using pure PyTorch constructors."""
        logger.info("Initializing models with pure PyTorch...")

        # Build condition embedders
        ss_condition_embedder = self._build_ss_condition_embedder()
        slat_condition_embedder = self._build_slat_condition_embedder()

        # Build generators
        ss_generator = self._build_ss_generator()
        slat_generator = self._build_slat_generator()

        # Build decoders
        ss_decoder = self._build_ss_decoder()
        slat_decoder_gs = self._build_slat_decoder_gs()
        slat_decoder_gs_4 = self._build_slat_decoder_gs_4()
        slat_decoder_mesh = self._build_slat_decoder_mesh()

        # Load pretrained weights if requested
        if self.load_pretrained:
            self._load_pretrained_weights(
                ss_generator=ss_generator,
                slat_generator=slat_generator,
                ss_decoder=ss_decoder,
                slat_decoder_gs=slat_decoder_gs,
                slat_decoder_gs_4=slat_decoder_gs_4,
                slat_decoder_mesh=slat_decoder_mesh,
                ss_condition_embedder=ss_condition_embedder,
                slat_condition_embedder=slat_condition_embedder,
            )

        # Store in ModuleDict
        self.models = nn.ModuleDict({
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

    # =========================================================================
    # SS Condition Embedder (DINOv2 for image + mask, PointPatchEmbed for pointmap)
    # =========================================================================
    def _build_ss_condition_embedder(self) -> EmbedderFuser:
        """Build SS condition embedder with DINOv2 and PointPatchEmbed."""
        logger.info("Building ss_condition_embedder...")

        # DINOv2 embedder for image
        dino_image = Dino(
            dino_model="dinov2_vitl14_reg",
            input_size=518,
            normalize_images=True,
        )

        # DINOv2 embedder for mask
        dino_mask = Dino(
            dino_model="dinov2_vitl14_reg",
            input_size=518,
            normalize_images=True,
        )

        # PointPatchEmbed for pointmap
        pointmap_embedder = PointPatchEmbed(
            embed_dim=512,
            input_size=256,
            patch_size=8,
            remap_output="linear",
        )

        # Build embedder list as expected by EmbedderFuser
        embedder_list = [
            (dino_image, [[["image", "cropped"], ["rgb_image", "full"]]]),
            (dino_mask, [[["mask", "cropped"], ["rgb_image_mask", "full"]]]),
            (pointmap_embedder, [[["pointmap", "cropped"], ["rgb_pointmap", "full"]]]),
        ]

        embedder = EmbedderFuser(
            embedder_list=embedder_list,
            drop_modalities_weight=[[[["pointmap", "rgb_pointmap"]], 1.0]],
            dropout_prob=0.1,
            freeze=True,
            projection_net_hidden_dim_multiplier=4.0,
            use_pos_embedding="learned",
        )

        return embedder.to(self.device)

    # =========================================================================
    # SLAT Condition Embedder (DINOv2 only)
    # =========================================================================
    def _build_slat_condition_embedder(self) -> EmbedderFuser:
        """Build SLAT condition embedder with DINOv2."""
        logger.info("Building slat_condition_embedder...")

        dino_image = Dino(
            dino_model="dinov2_vitl14_reg",
            input_size=518,
            normalize_images=True,
            prenorm_features=True,
        )

        dino_mask = Dino(
            dino_model="dinov2_vitl14_reg",
            input_size=518,
            normalize_images=True,
            prenorm_features=True,
        )

        embedder_list = [
            (dino_image, [[["image", "cropped"], ["rgb_image", "full"]]]),
            (dino_mask, [[["mask", "cropped"], ["rgb_image_mask", "full"]]]),
        ]

        embedder = EmbedderFuser(
            embedder_list=embedder_list,
            projection_net_hidden_dim_multiplier=4.0,
            use_pos_embedding="learned",
        )

        return embedder.to(self.device)

    # =========================================================================
    # SS Generator (ShortCut with SparseStructureFlowTdfyWrapper)
    # =========================================================================
    def _build_ss_generator(self) -> ShortCut:
        """Build Sparse Structure Generator."""
        logger.info("Building ss_generator...")

        # Build latent mappings for pose/scale/translation
        latent_mapping = {
            "6drotation_normalized": Latent(
                in_channels=6,
                model_channels=1024,
                pos_embedder=LearntPositionEmbedder(model_channels=1024, token_len=1),
            ),
            "scale": Latent(
                in_channels=3,
                model_channels=1024,
                pos_embedder=LearntPositionEmbedder(model_channels=1024, token_len=1),
            ),
            "shape": Latent(
                in_channels=8,
                model_channels=1024,
                pos_embedder=ShapePositionEmbedder(
                    model_channels=1024,
                    patch_size=1,
                    resolution=16,
                ),
            ),
            "translation": Latent(
                in_channels=3,
                model_channels=1024,
                pos_embedder=LearntPositionEmbedder(model_channels=1024, token_len=1),
            ),
            "translation_scale": Latent(
                in_channels=1,
                model_channels=1024,
                pos_embedder=LearntPositionEmbedder(model_channels=1024, token_len=1),
            ),
        }

        # Backbone transformer
        backbone = SparseStructureFlowTdfyWrapper(
            cond_channels=1024,
            condition_embedder=None,
            force_zeros_cond=True,
            freeze_d_time_embedder=True,
            freeze_shared_parameters=True,
            in_channels=8,
            is_shortcut_model=True,
            latent_mapping=latent_mapping,
            latent_share_transformer={
                "6drotation_normalized": [
                    "6drotation_normalized",
                    "translation",
                    "scale",
                    "translation_scale",
                ],
            },
            mlp_ratio=4,
            model_channels=1024,
            num_blocks=24,
            num_heads=16,
            out_channels=8,
            patch_size=1,
            pe_mode="ape",
            qk_rms_norm=True,
            resolution=16,
            use_checkpoint=False,
            use_fp16=False,
        )

        # Wrap in classifier-free guidance
        cfg_wrapper = ClassifierFreeGuidanceWithExternalUnconditionalProbability(
            backbone=backbone,
            interval=[0, 500],
            p_unconditional=0.1,
            strength=2.0,
            unconditional_handling="add_flag",
        )

        # ShortCut generator
        generator = ShortCut(
            reverse_fn=cfg_wrapper,
            batch_mode=True,
            cfg_modalities=["shape"],
            inference_steps=2,
            loss_weights={
                "6drotation_normalized": 0.1,
                "scale": 0.1,
                "shape": 0,
                "translation": 1.0,
                "translation_scale": 0.0,
            },
            ratio_cfg_samples_in_self_consistency_target=0.25,
            rescale_t=1,
            self_consistency_cfg_strength=2.0,
            self_consistency_prob=0.25,
            shortcut_loss_weight=1.0,
            sigma_min=0.0,
            time_scale=1000.0,
            training_time_sampler_fn=partial(lognorm_sampler, mean=-1.0, std=1.0),
        )

        return generator.to(self.device)

    # =========================================================================
    # SLAT Generator (FlowMatching with SLatFlowModelTdfyWrapper)
    # =========================================================================
    def _build_slat_generator(self) -> FlowMatching:
        """Build Structured Latent Generator."""
        logger.info("Building slat_generator...")

        backbone = SLatFlowModelTdfyWrapper(
            cond_channels=1024,
            condition_embedder=None,
            force_zeros_cond=True,
            in_channels=8,
            io_block_channels=[128],
            mlp_ratio=4,
            model_channels=1024,
            num_blocks=24,
            num_heads=16,
            num_io_res_blocks=2,
            out_channels=8,
            patch_size=2,
            pe_mode="ape",
            qk_rms_norm=True,
            resolution=64,
            use_fp16=True,
        )

        cfg_wrapper = ClassifierFreeGuidance(
            backbone=backbone,
            p_unconditional=0.0,
            strength=0.0,
            unconditional_handling="add_flag",
        )

        generator = FlowMatching(
            reverse_fn=cfg_wrapper,
            inference_steps=12,
            sigma_min=0.0,
            time_scale=1000.0,
            training_time_sampler_fn=partial(lognorm_sampler, mean=-1.0, std=1.0),
        )

        return generator.to(self.device)

    # =========================================================================
    # Decoders
    # =========================================================================
    def _build_ss_decoder(self) -> SparseStructureDecoderTdfyWrapper:
        """Build Sparse Structure Decoder."""
        logger.info("Building ss_decoder...")

        decoder = SparseStructureDecoderTdfyWrapper(
            out_channels=1,
            latent_channels=8,
            num_res_blocks=2,
            num_res_blocks_middle=2,
            channels=[512, 128, 32],
            reshape_input_to_cube=False,
        )

        return decoder.to(self.device)

    def _build_slat_decoder_gs(self) -> SLatGaussianDecoderTdfyWrapper:
        """Build SLAT Gaussian Decoder."""
        logger.info("Building slat_decoder_gs...")

        decoder = SLatGaussianDecoderTdfyWrapper(
            resolution=64,
            model_channels=768,
            latent_channels=8,
            num_blocks=12,
            num_heads=12,
            mlp_ratio=4,
            attn_mode="swin",
            window_size=8,
            representation_config={
                "lr": {
                    "_xyz": 1.0,
                    "_features_dc": 1.0,
                    "_opacity": 1.0,
                    "_scaling": 1.0,
                    "_rotation": 0.1,
                },
                "perturb_offset": True,
                "voxel_size": 1.5,
                "num_gaussians": 32,
                "2d_filter_kernel_size": 0.1,
                "3d_filter_kernel_size": 0.0009,
                "scaling_bias": 0.004,
                "opacity_bias": 0.1,
                "scaling_activation": "softplus",
            },
            use_fp16=True,
        )

        return decoder.to(self.device)

    def _build_slat_decoder_gs_4(self) -> Optional[SLatGaussianDecoderTdfyWrapper]:
        """Build SLAT Gaussian Decoder variant (4)."""
        logger.info("Building slat_decoder_gs_4...")

        # Based on slat_decoder_gs_4.yaml config
        decoder = SLatGaussianDecoderTdfyWrapper(
            resolution=64,
            model_channels=768,
            latent_channels=8,
            num_blocks=12,
            num_heads=12,
            mlp_ratio=4,
            attn_mode="swin",
            window_size=8,
            representation_config={
                "lr": {
                    "_xyz": 1.0,
                    "_features_dc": 1.0,
                    "_opacity": 1.0,
                    "_scaling": 1.0,
                    "_rotation": 0.1,
                },
                "perturb_offset": True,
                "voxel_size": 1.5,
                "num_gaussians": 4,  # Different from slat_decoder_gs
                "2d_filter_kernel_size": 0.1,
                "3d_filter_kernel_size": 0.0009,
                "scaling_bias": 0.004,
                "opacity_bias": 0.1,
                "scaling_activation": "softplus",
            },
            use_fp16=True,
        )

        return decoder.to(self.device)

    def _build_slat_decoder_mesh(self) -> SLatMeshDecoderTdfyWrapper:
        """Build SLAT Mesh Decoder."""
        logger.info("Building slat_decoder_mesh...")

        decoder = SLatMeshDecoderTdfyWrapper(
            resolution=64,
            model_channels=768,
            latent_channels=8,
            num_blocks=12,
            num_heads=12,
            mlp_ratio=4,
            attn_mode="swin",
            window_size=8,
            representation_config={"use_color": True},
            use_fp16=True,
        )

        return decoder.to(self.device)

    # =========================================================================
    # Pretrained weights loading
    # =========================================================================
    def _load_pretrained_weights(
        self,
        ss_generator,
        slat_generator,
        ss_decoder,
        slat_decoder_gs,
        slat_decoder_gs_4,
        slat_decoder_mesh,
        ss_condition_embedder,
        slat_condition_embedder,
    ) -> None:
        """Load pretrained weights for all models."""
        logger.info("Loading pretrained weights...")

        # SS Generator
        self._load_weights(
            ss_generator,
            os.path.join(self.checkpoint_dir, "ss_generator.ckpt"),
            state_dict_fn=filter_and_remove_prefix_state_dict_fn("_base_models.generator."),
        )

        # SLAT Generator
        self._load_weights(
            slat_generator,
            os.path.join(self.checkpoint_dir, "slat_generator.ckpt"),
            state_dict_fn=filter_and_remove_prefix_state_dict_fn("_base_models.generator."),
        )

        # SS Decoder
        self._load_weights(
            ss_decoder,
            os.path.join(self.checkpoint_dir, "ss_decoder.ckpt"),
            state_dict_key=None,
        )

        # SLAT Decoders
        self._load_weights(
            slat_decoder_gs,
            os.path.join(self.checkpoint_dir, "slat_decoder_gs.ckpt"),
            state_dict_key=None,
        )
        self._load_weights(
            slat_decoder_gs_4,
            os.path.join(self.checkpoint_dir, "slat_decoder_gs_4.ckpt"),
            state_dict_key=None,
        )
        self._load_weights(
            slat_decoder_mesh,
            os.path.join(self.checkpoint_dir, "slat_decoder_mesh.ckpt"),
            state_dict_key=None,
        )

        # Condition Embedders (loaded from generator checkpoints)
        self._load_weights(
            ss_condition_embedder,
            os.path.join(self.checkpoint_dir, "ss_generator.ckpt"),
            state_dict_fn=filter_and_remove_prefix_state_dict_fn("_base_models.condition_embedder."),
        )
        self._load_weights(
            slat_condition_embedder,
            os.path.join(self.checkpoint_dir, "slat_generator.ckpt"),
            state_dict_fn=filter_and_remove_prefix_state_dict_fn("_base_models.condition_embedder."),
        )

        logger.info("Pretrained weights loaded successfully!")

    def _load_weights(
        self,
        model: nn.Module,
        ckpt_path: str,
        state_dict_fn=None,
        state_dict_key: str = "state_dict",
    ) -> None:
        """Load weights into a model."""
        if not os.path.exists(ckpt_path):
            logger.warning(f"Checkpoint not found: {ckpt_path}")
            return

        logger.info(f"Loading weights from: {ckpt_path}")

        if ckpt_path.endswith(".safetensors"):
            state_dict = load_file(ckpt_path, device="cuda")
            if state_dict_fn is not None:
                state_dict = state_dict_fn(state_dict)
            model.load_state_dict(state_dict, strict=False)
        else:
            load_model_from_checkpoint(
                model,
                ckpt_path,
                strict=False,
                device="cpu",
                freeze=False,
                eval=False,
                state_dict_key=state_dict_key,
                state_dict_fn=state_dict_fn,
            )

    # =========================================================================
    # Utility methods
    # =========================================================================
    def _log_model_summary(self) -> None:
        """Log a summary of all initialized models."""
        logger.info("=" * 60)
        logger.info("Model Summary (Pure PyTorch):")
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

    def get_model(self, name: str) -> nn.Module:
        """Get a model by name."""
        if name in self.models:
            return self.models[name]
        if name in self.condition_embedders:
            return self.condition_embedders[name]
        raise KeyError(f"Model '{name}' not found")

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
        """Freeze model parameters."""
        all_models = {**self.models, **self.condition_embedders}

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
        """Unfreeze model parameters."""
        all_models = {**self.models, **self.condition_embedders}

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
    # Test instantiation
    pipeline = TrainPipelinePyTorch(load_pretrained=True, checkpoint_dir="checkpoints/hf")
    print("\nModels instantiated successfully!")

    # Test with pretrained weights
    # pipeline = TrainPipelinePyTorch(load_pretrained=True, checkpoint_dir="checkpoints/hf")
