# Sparse Structure Generator (ss_generator) Data Flow Documentation

This document describes the complete data flow through the `ss_generator` pipeline, including all neural network components, their input/output shapes, and the path data takes from input image to generated sparse structure.

## Logging

Detailed logging has been added to all major components to trace data flow during inference. The logging uses `loguru.logger` and outputs shapes, dtypes, and statistics at each stage.

### Files with Added Logging

1. **`sam3d_objects/pipeline/inference_pipeline.py`**
   - `preprocess_image()`: Logs input image and preprocessed tensor shapes
   - `sample_sparse_structure()`: Logs complete pipeline flow with phases
   - `embed_condition()`: Logs condition embedding output

2. **`sam3d_objects/model/backbone/dit/embedder/embedder_fuser.py`**
   - `forward()`: Logs each embedder's input/output, projection net outputs, and final concatenation

3. **`sam3d_objects/model/backbone/dit/embedder/dino.py`**
   - `forward()`: Logs input, preprocessed, and output token shapes

4. **`sam3d_objects/model/backbone/dit/embedder/pointmap.py`**
   - `forward()`: Logs pointmap input and window embedding outputs

5. **`sam3d_objects/model/backbone/generator/shortcut/model.py`**
   - `generate_iter()`: Logs ODE solver steps, noise initialization, and final outputs

6. **`sam3d_objects/model/backbone/generator/flow_matching/model.py`**
   - `generate_iter()`: Logs flow matching solver progress

7. **`sam3d_objects/model/backbone/generator/classifier_free_guidance.py`**
   - `inner_forward()`: Logs CFG conditional/unconditional forward passes

8. **`sam3d_objects/model/backbone/tdfy_dit/models/mot_sparse_structure_flow.py`**
   - `forward()` (both classes): Logs transformer block processing
   - `project_input()`, `project_output()`: Logs latent mapping transformations

9. **`sam3d_objects/model/backbone/tdfy_dit/models/mm_latent.py`**
   - `__init__()`: Logs Latent module initialization with dimensions

10. **`sam3d_objects/model/backbone/tdfy_dit/models/sparse_structure_vae.py`**
    - `forward()`: Logs decoder layer-by-layer processing

## Overview

The `ss_generator` is responsible for generating a sparse 3D structure (occupancy grid) from a 2D input image. It uses a flow matching / shortcut model architecture with classifier-free guidance.

## Architecture Diagram

```
Input Image (512x512x4 RGBA)
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                   PREPROCESSING                              │
│  Image: (B, 3, H, W) → resized to (B, 3, 518, 518)          │
│  Mask:  (B, 1, H, W) → resized to (B, 1, 518, 518)          │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                   CONDITION EMBEDDER                         │
│                   (EmbedderFuser)                            │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Dino Embedder (Image)                                 │   │
│  │ Input:  (B, 3, 518, 518)                             │   │
│  │ Output: (B, 1370, 1024) [cls + 37x37 patches]        │   │
│  │ dinov2_vitl14_reg, embed_dim=1024                    │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Dino Embedder (Mask)                                  │   │
│  │ Input:  (B, 1, 518, 518) → repeated to (B, 3, 518, 518)│  │
│  │ Output: (B, 1370, 1024)                              │   │
│  │ dinov2_vitl14_reg, embed_dim=1024                    │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ PointPatchEmbed (optional, for pointmap)             │   │
│  │ Input:  (B, 3, 256, 256) pointmap                    │   │
│  │ Output: (B, 1024, 512) [32x32 windows]               │   │
│  │ patch_size=8, embed_dim=512                          │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Projection Networks (FeedForward per modality)       │   │
│  │ Input:  each modality embedding                      │   │
│  │ Output: projected to common dim (1024)               │   │
│  │ LayerNorm → FeedForward (dim*4 hidden)              │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Positional Embedding Addition                         │   │
│  │ Learned embeddings for each modality group           │   │
│  │ Shape: (num_groups, 1024)                            │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                   │
│  Concatenate all tokens along sequence dimension             │
│  Final Output: (B, ~2740+, 1024) condition tokens            │
│  (1370 image + 1370 mask tokens, optionally + pointmap)     │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                   SS_GENERATOR                               │
│            (ShortCut / FlowMatching)                         │
│                                                              │
│  Initial latent shape dict (multi-modal DiT):               │
│  {                                                           │
│    'shape':               (B, 4096, 8)  [16x16x16 grid]     │
│    'translation':         (B, 1, 3)                          │
│    '6drotation_normalized': (B, 1, 6)                        │
│    'scale':               (B, 1, 3)                          │
│    'translation_scale':   (B, 1, 1)                          │
│  }                                                           │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Generate Initial Noise x_0                            │   │
│  │ x_0 ~ N(0, I) with same shapes as latent_shape_dict  │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ ODE Solver (Euler/Midpoint/RK4)                       │   │
│  │ Steps: 25 (configurable via ss_inference_steps)       │   │
│  │ t: 0 → 1 (or rescaled)                                │   │
│  │                                                        │   │
│  │ For each timestep t:                                  │   │
│  │   velocity = reverse_fn(x_t, t, cond)                 │   │
│  │   x_{t+dt} = x_t + velocity * dt                      │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         REVERSE_FN (CFG Wrapper)                      │   │
│  │ ClassifierFreeGuidanceWithExternalUnconditionalProb   │   │
│  │                                                        │   │
│  │ inference mode with cfg_strength > 0:                 │   │
│  │   y_cond = backbone(x, t, cond)                       │   │
│  │   y_uncond = backbone(x, t, zeros)                    │   │
│  │   y = (1+strength)*y_cond - strength*y_uncond         │   │
│  │                                                        │   │
│  │ cfg_strength: 7 (default)                             │   │
│  │ cfg_interval: [0, 500]                                │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         BACKBONE                                       │   │
│  │    (SparseStructureFlowTdfyWrapper)                   │   │
│  │                                                        │   │
│  │  ┌────────────────────────────────────────────────┐   │   │
│  │  │ Latent Mapping (project_input)                  │   │   │
│  │  │                                                  │   │   │
│  │  │ For each modality in latent_mapping:            │   │   │
│  │  │   input_layer: Linear(in_ch → 1024)             │   │   │
│  │  │   + positional embedding                        │   │   │
│  │  │                                                  │   │   │
│  │  │ 'shape':          (B, 4096, 8) → (B, 4096, 1024)│   │   │
│  │  │ 'translation':    (B, 1, 3)    → (B, 1, 1024)   │   │   │
│  │  │ '6drotation_normalized': (B,1,6) → (B,1,1024)   │   │   │
│  │  │ 'scale':          (B, 1, 3)    → (B, 1, 1024)   │   │   │
│  │  │ 'translation_scale': (B,1,1)   → (B,1,1024)     │   │   │
│  │  │                                                  │   │   │
│  │  │ Merge shared transformer groups:                │   │   │
│  │  │ '6drotation_normalized' group merged:           │   │   │
│  │  │   [6drot, trans, scale, trans_scale]            │   │   │
│  │  │   → (B, 4, 1024)                                │   │   │
│  │  │                                                  │   │   │
│  │  │ Final dict: {'shape': (B,4096,1024),            │   │   │
│  │  │              '6drotation_normalized': (B,4,1024)}│   │   │
│  │  └────────────────────────────────────────────────┘   │   │
│  │                       │                                │   │
│  │  ┌────────────────────────────────────────────────┐   │   │
│  │  │ Condition Embedder (if attached)               │   │   │
│  │  │ Input: condition tokens from EmbedderFuser     │   │   │
│  │  │ cond shape: (B, ~2740, 1024)                   │   │   │
│  │  │ (or zeros if cfg_activate)                     │   │   │
│  │  └────────────────────────────────────────────────┘   │   │
│  │                       │                                │   │
│  │  ┌────────────────────────────────────────────────┐   │   │
│  │  │ Timestep Embedder                               │   │   │
│  │  │ t_emb = TimestepEmbedder(t * time_scale)       │   │   │
│  │  │ Input:  scalar t (scaled to [0, 1000])          │   │   │
│  │  │ Output: (B, 1024)                               │   │   │
│  │  │                                                  │   │   │
│  │  │ if is_shortcut_model:                           │   │   │
│  │  │   d_emb = TimestepEmbedder(d * time_scale)     │   │   │
│  │  │   t_emb = t_emb + d_emb                        │   │   │
│  │  └────────────────────────────────────────────────┘   │   │
│  │                       │                                │   │
│  │  ┌────────────────────────────────────────────────┐   │   │
│  │  │ Transformer Blocks (24 blocks)                  │   │   │
│  │  │ MOTModulatedTransformerCrossBlock              │   │   │
│  │  │                                                  │   │   │
│  │  │ For each block:                                 │   │   │
│  │  │   Input h: dict of latent tensors              │   │   │
│  │  │   mod: t_emb (B, 1024) → adaLN (B, 6*1024)     │   │   │
│  │  │   context: cond tokens (B, ~2740, 1024)        │   │   │
│  │  │                                                  │   │   │
│  │  │   For each modality in h:                       │   │   │
│  │  │     h = LayerNorm(h)                            │   │   │
│  │  │     h = h * (1+scale) + shift  # adaLN         │   │   │
│  │  │     h = MultiHeadSelfAttention(h)  # 16 heads  │   │   │
│  │  │     h = h * gate                                │   │   │
│  │  │     x = x + h                                   │   │   │
│  │  │                                                  │   │   │
│  │  │     h = LayerNorm(x)                            │   │   │
│  │  │     h = CrossAttention(h, context) # 16 heads  │   │   │
│  │  │     x = x + h                                   │   │   │
│  │  │                                                  │   │   │
│  │  │     h = LayerNorm(x)                            │   │   │
│  │  │     h = h * (1+scale) + shift                   │   │   │
│  │  │     h = FeedForward(h)  # mlp_ratio=4          │   │   │
│  │  │     h = h * gate                                │   │   │
│  │  │     x = x + h                                   │   │   │
│  │  │                                                  │   │   │
│  │  │ Output: same dict structure                     │   │   │
│  │  └────────────────────────────────────────────────┘   │   │
│  │                       │                                │   │
│  │  ┌────────────────────────────────────────────────┐   │   │
│  │  │ Latent Mapping (project_output)                 │   │   │
│  │  │                                                  │   │   │
│  │  │ Split merged groups back:                       │   │   │
│  │  │ '6drotation_normalized' → 4 separate tensors   │   │   │
│  │  │                                                  │   │   │
│  │  │ For each modality:                              │   │   │
│  │  │   h = LayerNorm(h)                              │   │   │
│  │  │   output = out_layer(h): Linear(1024 → in_ch)  │   │   │
│  │  │                                                  │   │   │
│  │  │ 'shape':          (B, 4096, 1024) → (B,4096,8) │   │   │
│  │  │ 'translation':    (B, 1, 1024)    → (B, 1, 3)  │   │   │
│  │  │ '6drotation_normalized': → (B, 1, 6)           │   │   │
│  │  │ 'scale':                  → (B, 1, 3)          │   │   │
│  │  │ 'translation_scale':      → (B, 1, 1)          │   │   │
│  │  └────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  Output: velocity dict (same shapes as input latents)       │
│  After all ODE steps: final latent dict                     │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                   SS_DECODER                                 │
│            (SparseStructureDecoder)                          │
│                                                              │
│  Input: shape latent (B, 4096, 8)                           │
│         → reshape to (B, 8, 16, 16, 16)                     │
│                                                              │
│  Architecture:                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Input Layer: Conv3d(8 → 64, 3x3x3)                   │   │
│  │ Output: (B, 64, 16, 16, 16)                          │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Middle Block: 2x ResBlock3d(64)                       │   │
│  │ Output: (B, 64, 16, 16, 16)                          │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Decoder Blocks (upsample path):                       │   │
│  │   ResBlock3d(64) x num_res_blocks                    │   │
│  │   Upsample3d(64 → 32): (B, 32, 32, 32, 32)          │   │
│  │   ResBlock3d(32) x num_res_blocks                    │   │
│  │   Upsample3d(32 → 16): (B, 16, 64, 64, 64)          │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Output Layer:                                         │   │
│  │   LayerNorm + SiLU + Conv3d(16 → 1, 3x3x3)          │   │
│  │   Output: (B, 1, 64, 64, 64) occupancy logits        │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  Final Output: Occupancy grid (B, 64, 64, 64)               │
│  coords = argwhere(ss > 0)  # sparse voxel coordinates     │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                   POSE DECODER                               │
│  Decode pose parameters from latent predictions:            │
│                                                              │
│  Input: return_dict containing predicted latents            │
│    'translation': (B, 1, 3)                                 │
│    '6drotation_normalized': (B, 1, 6)                       │
│    'scale': (B, 1, 3)                                       │
│    'translation_scale': (B, 1, 1)                           │
│                                                              │
│  Output: Decoded 3D transformation parameters               │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
     Final Output:
     {
       'coords': (N, 4) sparse voxel coordinates [batch_idx, x, y, z]
       'shape': (B, 4096, 8) shape latent
       'translation': (B, 1, 3) 
       '6drotation_normalized': (B, 1, 6)
       'scale': (B, 1, 3)
       'translation_scale': (B, 1, 1)
       ... decoded pose parameters
     }
```

## Key Classes and Their Roles

### 1. EmbedderFuser (`sam3d_objects/model/backbone/dit/embedder/embedder_fuser.py`)
- **Purpose**: Fuses multiple condition embeddings (image, mask, pointmap)
- **Components**:
  - Multiple Dino embedders for image/mask
  - PointPatchEmbed for pointmap (optional)
  - Projection networks (FeedForward)
  - Positional embeddings

### 2. Dino (`sam3d_objects/model/backbone/dit/embedder/dino.py`)
- **Purpose**: Extracts visual features using DINOv2
- **Model**: `dinov2_vitl14_reg` (ViT-Large with registers)
- **Input**: (B, 3, 518, 518) normalized images
- **Output**: (B, 1370, 1024) patch tokens with CLS token

### 3. PointPatchEmbed (`sam3d_objects/model/backbone/dit/embedder/pointmap.py`)
- **Purpose**: Embeds 3D pointmaps into tokens
- **Input**: (B, 3, H, W) pointmap
- **Output**: (B, num_windows, embed_dim) window tokens

### 4. ShortCut (`sam3d_objects/model/backbone/generator/shortcut/model.py`)
- **Purpose**: Flow matching with shortcut/self-consistency training
- **Parent**: FlowMatching
- **Key methods**:
  - `generate_iter()`: ODE solver iteration
  - `_generate_dynamics()`: Velocity prediction

### 5. ClassifierFreeGuidanceWithExternalUnconditionalProbability
- **Purpose**: CFG wrapper for conditional generation
- **Formula**: `y = (1 + strength) * y_cond - strength * y_uncond`

### 6. SparseStructureFlowTdfyWrapper (`sam3d_objects/model/backbone/tdfy_dit/models/mot_sparse_structure_flow.py`)
- **Purpose**: Multi-modal DiT backbone
- **Components**:
  - Latent mapping (project input/output)
  - Timestep embedder
  - 24 transformer blocks

### 7. Latent (`sam3d_objects/model/backbone/tdfy_dit/models/mm_latent.py`)
- **Purpose**: Maps between raw latent space and model hidden dim
- **Components**:
  - input_layer: Linear(in_channels → model_channels)
  - out_layer: Linear(model_channels → in_channels)
  - pos_emb: Positional embedding

### 8. MOTModulatedTransformerCrossBlock
- **Purpose**: Transformer block with AdaLN modulation
- **Components**:
  - Self-attention (multi-modal)
  - Cross-attention to condition tokens
  - FeedForward MLP
  - AdaLN modulation from timestep

### 9. SparseStructureDecoder (`sam3d_objects/model/backbone/tdfy_dit/models/sparse_structure_vae.py`)
- **Purpose**: Decode latent to occupancy grid
- **Architecture**: 3D CNN with upsampling
- **Input**: (B, 8, 16, 16, 16) latent cube
- **Output**: (B, 1, 64, 64, 64) occupancy

## Shape Summary Table

| Stage | Tensor | Shape | Description |
|-------|--------|-------|-------------|
| Input | RGBA Image | (B, 4, 512, 512) | Input image with alpha |
| Preprocess | RGB Image | (B, 3, 518, 518) | Resized for DINO |
| Preprocess | Mask | (B, 1, 518, 518) | Binary mask |
| DINO | Image Tokens | (B, 1370, 1024) | 1 CLS + 37x37 patches |
| DINO | Mask Tokens | (B, 1370, 1024) | Same structure |
| Fuser | Condition | (B, 2740+, 1024) | Concatenated tokens |
| Latent | shape | (B, 4096, 8) | 16x16x16 grid, 8 channels |
| Latent | translation | (B, 1, 3) | XYZ translation |
| Latent | 6drot | (B, 1, 6) | 6D rotation repr |
| Latent | scale | (B, 1, 3) | XYZ scale |
| Backbone | hidden | (B, 4096, 1024) | Transformed latent |
| Decoder | Cube | (B, 8, 16, 16, 16) | Reshaped latent |
| Decoder | Occupancy | (B, 1, 64, 64, 64) | Final grid |
| Output | Coords | (N, 4) | Sparse coordinates |

## Configuration Reference (ss_generator.yaml)

```yaml
# Key configuration parameters:
model_channels: 1024      # Hidden dimension
num_blocks: 24            # Number of transformer blocks
num_heads: 16             # Attention heads
mlp_ratio: 4              # FFN hidden multiplier
patch_size: 1             # For shape position embedder
resolution: 16            # Latent grid resolution
inference_steps: 25       # ODE solver steps
cfg_strength: 7           # Classifier-free guidance
rescale_t: 3              # Time rescaling factor
```
