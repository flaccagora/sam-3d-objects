import sys

# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask

# import training pipeline
from sam3d_objects.train import TrainPipeline

# load model
tag = "hf"
config_path = f"checkpoints/{tag}/pipeline.yaml"
inference = Inference(config_path, compile=False)

# load image (RGBA only, mask is embedded in the alpha channel)
image = load_image("notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png")
mask = load_single_mask("notebook/images/shutterstock_stylish_kidsroom_1640806567", index=14)

# run model
output = inference(image, mask, seed=42)

# export gaussian splat
output["gs"].save_ply(f"splat.ply")
print("Your reconstruction has been saved to splat.ply")


# ============================================================================
# Verify TrainPipeline produces equivalent model weights to InferencePipeline
# ============================================================================
print("\n" + "=" * 60)
print("Comparing TrainPipeline with InferencePipeline...")
print("=" * 60)

# Instantiate training pipeline with pretrained weights
train_pipeline = TrainPipeline(config_path, load_pretrained=True)
train_pipeline.eval()  # Set to eval mode for comparison

# Get the internal inference pipeline for comparison
inference_pipeline = inference._pipeline


def compare_model_weights(model1, model2, name: str) -> bool:
    """Compare weights between two models."""
    if model1 is None and model2 is None:
        print(f"  {name}: Both None ✓")
        return True
    if model1 is None or model2 is None:
        print(f"  {name}: One is None, other is not ✗")
        return False

    state1 = model1.state_dict()
    state2 = model2.state_dict()

    if set(state1.keys()) != set(state2.keys()):
        print(f"  {name}: Different keys ✗")
        print(f"    Only in model1: {set(state1.keys()) - set(state2.keys())}")
        print(f"    Only in model2: {set(state2.keys()) - set(state1.keys())}")
        return False

    all_close = True
    max_diff = 0.0
    for key in state1.keys():
        if not state1[key].shape == state2[key].shape:
            print(f"  {name}.{key}: Shape mismatch {state1[key].shape} vs {state2[key].shape} ✗")
            all_close = False
            continue

        diff = (state1[key].float() - state2[key].float()).abs().max().item()
        max_diff = max(max_diff, diff)
        if diff > 1e-5:
            print(f"  {name}.{key}: Max diff = {diff:.6e} ✗")
            all_close = False

    if all_close:
        print(f"  {name}: Weights match (max diff: {max_diff:.6e}) ✓")
    return all_close


# Compare all models
print("\nComparing model weights:")
print("-" * 60)

all_match = True

# Compare generators
all_match &= compare_model_weights(
    train_pipeline.models["ss_generator"],
    inference_pipeline.models["ss_generator"],
    "ss_generator",
)
all_match &= compare_model_weights(
    train_pipeline.models["slat_generator"],
    inference_pipeline.models["slat_generator"],
    "slat_generator",
)

# Compare decoders
all_match &= compare_model_weights(
    train_pipeline.models["ss_decoder"],
    inference_pipeline.models["ss_decoder"],
    "ss_decoder",
)
all_match &= compare_model_weights(
    train_pipeline.models["slat_decoder_gs"],
    inference_pipeline.models["slat_decoder_gs"],
    "slat_decoder_gs",
)
all_match &= compare_model_weights(
    train_pipeline.models["slat_decoder_gs_4"],
    inference_pipeline.models["slat_decoder_gs_4"],
    "slat_decoder_gs_4",
)
all_match &= compare_model_weights(
    train_pipeline.models["slat_decoder_mesh"],
    inference_pipeline.models["slat_decoder_mesh"],
    "slat_decoder_mesh",
)

# Compare condition embedders
all_match &= compare_model_weights(
    train_pipeline.condition_embedders["ss_condition_embedder"],
    inference_pipeline.condition_embedders["ss_condition_embedder"],
    "ss_condition_embedder",
)
all_match &= compare_model_weights(
    train_pipeline.condition_embedders["slat_condition_embedder"],
    inference_pipeline.condition_embedders["slat_condition_embedder"],
    "slat_condition_embedder",
)

print("-" * 60)
if all_match:
    print("✓ All model weights match between TrainPipeline and InferencePipeline!")
else:
    print("✗ Some model weights differ. Check the output above for details.")
print("=" * 60)
