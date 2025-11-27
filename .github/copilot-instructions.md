# SAM 3D Objects - AI Coding Agent Instructions

## Project Overview

SAM 3D Objects is a foundation model that reconstructs 3D shape geometry, texture, and layout from a single image. The system converts masked objects in images into 3D Gaussian Splat or mesh outputs with pose, shape, and texture.

## Architecture

### Pipeline Flow
```
Image + Mask → Preprocessing → Depth/Pointmap (MoGe) → SS Generator → SLAT Generator → Decoder → Gaussian/Mesh
```

**Key Components:**
- **`InferencePipelinePointMap`** (`sam3d_objects/pipeline/inference_pipeline_pointmap.py`): Main inference entry point, extends `InferencePipeline`
- **`Inference`** (`notebook/inference.py`): Public-facing API wrapper that handles RGBA images with embedded masks
- **SS (Sparse Structure) Generator**: Flow-matching model that generates 3D occupancy from image embeddings
- **SLAT Generator**: Generates structured latents for detailed geometry/texture
- **Decoders**: Convert latents to Gaussian Splats (`slat_decoder_gs`) or Meshes (`slat_decoder_mesh`)

### Configuration System
- Uses **Hydra** with YAML configs in `checkpoints/hf/`
- The `_target_` key specifies the Python class to instantiate
- `pipeline.yaml` orchestrates model configs and preprocessing pipelines
- Custom Hydra patch required: run `./patching/hydra` after install

### Data Flow
- Inputs: RGBA image where alpha channel contains the object mask
- Preprocessing uses torchvision transforms defined in `sam3d_objects/data/dataset/tdfy/`
- DINOv2 (ViT-L) embeddings extracted for both image and mask
- Pointmap normalization via `ObjectCentricSSI` for scale-shift invariant processing

## Development Setup

```bash
mamba env create -f environments/default.yml
mamba activate sam3d-objects
export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
pip install -e '.[dev]' && pip install -e '.[p3d]'
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
pip install -e '.[inference]'
./patching/hydra  # Required patch
```

**Requirements:** Linux 64-bit, NVIDIA GPU with ≥32GB VRAM, CUDA 12.1

## Key Patterns

### Loading and Running Inference
```python
sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask

inference = Inference("checkpoints/hf/pipeline.yaml", compile=False)
output = inference(image, mask, seed=42)
output["gs"].save_ply("output.ply")  # Gaussian Splat output
```

### Mask Loading Convention
Masks are stored as numbered PNG files (`0.png`, `1.png`, ...) in image directories. Use `load_single_mask(folder, index=N)` or `load_masks(folder)` for multiple.

### Output Structure
- `output["gs"]`: `Gaussian3DGS` object with `save_ply()` method
- `output["rotation"]`, `output["translation"]`, `output["scale"]`: Pose parameters
- Multi-object scenes use `make_scene(*outputs)` to merge Gaussians

## Directory Structure

- `sam3d_objects/pipeline/`: Inference pipelines and preprocessing
- `sam3d_objects/model/backbone/`: Neural network architectures (DIT, generators, decoders)
- `sam3d_objects/model/backbone/tdfy_dit/representations/gaussian/`: Gaussian Splat implementation
- `sam3d_objects/data/dataset/tdfy/`: Image/mask/pointmap transforms
- `notebook/`: Example notebooks and public `inference.py` API

## Coding Conventions

- Use `loguru.logger` for logging, not `print()`
- Prefer `torch.inference_mode()` context for inference code
- Type hints follow Python 3.10+ style (`list[X]` not `List[X]`)
- Imports: `sam3d_objects` must be imported before using its submodules (initializes LIDRA)
- Set `LIDRA_SKIP_INIT=true` env var for lightweight tooling that doesn't need full init

## Testing

No test files exist yet. The project uses `pytest` (in `requirements.dev.txt`). Run `python demo.py` for quick validation.
