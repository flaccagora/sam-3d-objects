# Copyright (c) Meta Platforms, Inc. and affiliates.
import os

from tqdm import tqdm
import torch
from loguru import logger
from functools import wraps
from torch.utils._pytree import tree_map_only


def set_attention_backend():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)

    logger.info(f"GPU name is {gpu_name}")
    if "A100" in gpu_name or "H100" in gpu_name or "H200" in gpu_name:
        # logger.info("Use flash_attn")
        os.environ["ATTN_BACKEND"] = "flash_attn"
        os.environ["SPARSE_ATTN_BACKEND"] = "flash_attn"

set_attention_backend()

from typing import List, Union
from hydra.utils import instantiate
from omegaconf import OmegaConf
import numpy as np

from PIL import Image
from sam3d_objects.pipeline import preprocess_utils
from sam3d_objects.data.dataset.tdfy.img_and_mask_transforms import (
    get_mask,
)
from sam3d_objects.pipeline.inference_utils import (
    get_pose_decoder,
    SLAT_MEAN,
    SLAT_STD,
    downsample_sparse_structure,
    prune_sparse_structure,
)

from sam3d_objects.model.io import (
    load_model_from_checkpoint,
    filter_and_remove_prefix_state_dict_fn,
)

from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp
from sam3d_objects.model.backbone.tdfy_dit.utils import postprocessing_utils
from safetensors.torch import load_file


class InferencePipeline:
    def __init__(
        self,
        ss_generator_config_path,
        ss_generator_ckpt_path,
        slat_generator_config_path,
        slat_generator_ckpt_path,
        ss_decoder_config_path,
        ss_decoder_ckpt_path,
        slat_decoder_gs_config_path,
        slat_decoder_gs_ckpt_path,
        slat_decoder_mesh_config_path,
        slat_decoder_mesh_ckpt_path,
        slat_decoder_gs_4_config_path=None,
        slat_decoder_gs_4_ckpt_path=None,
        ss_encoder_config_path=None,
        ss_encoder_ckpt_path=None,
        decode_formats=["gaussian", "mesh"],
        dtype="bfloat16",
        pad_size=1.0,
        version="v0",
        device="cuda",
        ss_preprocessor=preprocess_utils.get_default_preprocessor(),
        slat_preprocessor=preprocess_utils.get_default_preprocessor(),
        ss_condition_input_mapping=["image"],
        slat_condition_input_mapping=["image"],
        pose_decoder_name="default",
        workspace_dir="",
        downsample_ss_dist=0,  # the distance we use to downsample
        ss_inference_steps=25,
        ss_rescale_t=3,
        ss_cfg_strength=7,
        ss_cfg_interval=[0, 500],
        ss_cfg_strength_pm=0.0,
        slat_inference_steps=25,
        slat_rescale_t=3,
        slat_cfg_strength=5,
        slat_cfg_interval=[0, 500],
        rendering_engine: str = "nvdiffrast",  # nvdiffrast OR pytorch3d,
        shape_model_dtype=None,
        compile_model=False,
        slat_mean=SLAT_MEAN,
        slat_std=SLAT_STD,
    ):
        self.rendering_engine = rendering_engine
        self.device = torch.device(device)
        self.compile_model = compile_model
        logger.info(f"self.device: {self.device}")
        logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', None)}")
        logger.info(f"Actually using GPU: {torch.cuda.current_device()}")
        with self.device:
            self.decode_formats = decode_formats
            self.pad_size = pad_size
            self.version = version
            self.ss_condition_input_mapping = ss_condition_input_mapping
            self.slat_condition_input_mapping = slat_condition_input_mapping
            self.workspace_dir = workspace_dir
            self.downsample_ss_dist = downsample_ss_dist
            self.ss_inference_steps = ss_inference_steps
            self.ss_rescale_t = ss_rescale_t
            self.ss_cfg_strength = ss_cfg_strength
            self.ss_cfg_interval = ss_cfg_interval
            self.ss_cfg_strength_pm = ss_cfg_strength_pm
            self.slat_inference_steps = slat_inference_steps
            self.slat_rescale_t = slat_rescale_t
            self.slat_cfg_strength = slat_cfg_strength
            self.slat_cfg_interval = slat_cfg_interval

            self.dtype = self._get_dtype(dtype)
            if shape_model_dtype is None:
                self.shape_model_dtype = self.dtype
            else:
                self.shape_model_dtype = self._get_dtype(shape_model_dtype) 


            # Setup preprocessors
            self.pose_decoder = self.init_pose_decoder(ss_generator_config_path, pose_decoder_name)
            self.ss_preprocessor = self.init_ss_preprocessor(ss_preprocessor, ss_generator_config_path)
            self.slat_preprocessor = slat_preprocessor
    
            logger.info("Loading model weights...")

            ss_generator = self.init_ss_generator(
                ss_generator_config_path, ss_generator_ckpt_path
            )
            slat_generator = self.init_slat_generator(
                slat_generator_config_path, slat_generator_ckpt_path
            )
            ss_decoder = self.init_ss_decoder(
                ss_decoder_config_path, ss_decoder_ckpt_path
            )
            ss_encoder = self.init_ss_encoder(
                ss_encoder_config_path, ss_encoder_ckpt_path
            )
            slat_decoder_gs = self.init_slat_decoder_gs(
                slat_decoder_gs_config_path, slat_decoder_gs_ckpt_path
            )
            slat_decoder_gs_4 = self.init_slat_decoder_gs(
                slat_decoder_gs_4_config_path, slat_decoder_gs_4_ckpt_path
            )
            slat_decoder_mesh = self.init_slat_decoder_mesh(
                slat_decoder_mesh_config_path, slat_decoder_mesh_ckpt_path
            )

            # Load conditioner embedder so that we only load it once
            ss_condition_embedder = self.init_ss_condition_embedder(
                ss_generator_config_path, ss_generator_ckpt_path
            )
            slat_condition_embedder = self.init_slat_condition_embedder(
                slat_generator_config_path, slat_generator_ckpt_path
            )

            self.condition_embedders = {
                "ss_condition_embedder": ss_condition_embedder,
                "slat_condition_embedder": slat_condition_embedder,
            }

            # override generator and condition embedder setting
            self.override_ss_generator_cfg_config(
                ss_generator,
                cfg_strength=ss_cfg_strength,
                inference_steps=ss_inference_steps,
                rescale_t=ss_rescale_t,
                cfg_interval=ss_cfg_interval,
                cfg_strength_pm=ss_cfg_strength_pm,
            )
            self.override_slat_generator_cfg_config(
                slat_generator,
                cfg_strength=slat_cfg_strength,
                inference_steps=slat_inference_steps,
                rescale_t=slat_rescale_t,
                cfg_interval=slat_cfg_interval,
            )

            self.models = torch.nn.ModuleDict(
                {
                    "ss_generator": ss_generator,
                    "slat_generator": slat_generator,
                    "ss_encoder": ss_encoder,
                    "ss_decoder": ss_decoder,
                    "slat_decoder_gs": slat_decoder_gs,
                    "slat_decoder_gs_4": slat_decoder_gs_4,
                    "slat_decoder_mesh": slat_decoder_mesh,
                }
            )
            logger.info("Loading model weights completed!")

            if self.compile_model:
                logger.info("Compiling model...")
                self._compile()
                logger.info("Model compilation completed!")
            self.slat_mean = torch.tensor(slat_mean)
            self.slat_std = torch.tensor(slat_std)

    def _compile(self):
        torch._dynamo.config.cache_size_limit = 64
        torch._dynamo.config.accumulated_cache_size_limit = 2048
        torch._dynamo.config.capture_scalar_outputs = True
        compile_mode = "max-autotune"
        logger.info(f"Compile mode {compile_mode}")

        def clone_output_wrapper(f):
            @wraps(f)
            def wrapped(*args, **kwargs):
                outputs = f(*args, **kwargs)
                return tree_map_only(
                    torch.Tensor, lambda t: t.clone() if t.is_cuda else t, outputs
                )

            return wrapped

        self.embed_condition = clone_output_wrapper(
            torch.compile(
                self.embed_condition,
                mode=compile_mode,
                fullgraph=True,  # _preprocess_input in dino is not compatible with fullgraph
            )
        )
        self.models["ss_generator"].reverse_fn.inner_forward = clone_output_wrapper(
            torch.compile(
                self.models["ss_generator"].reverse_fn.inner_forward,
                mode=compile_mode,
                fullgraph=True,
            )
        )

        self.models["ss_decoder"].forward = clone_output_wrapper(
            torch.compile(
                self.models["ss_decoder"].forward,
                mode=compile_mode,
                fullgraph=True,
            )
        )

        self._warmup()

    def _warmup(self, num_warmup_iters=3):
        test_image = np.ones((512, 512, 4), dtype=np.uint8) * 255
        test_image[:, :, :3] = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        image = Image.fromarray(test_image)
        mask = None
        image = self.merge_image_and_mask(image, mask)

        for _ in tqdm(range(num_warmup_iters)):
            ss_input_dict = self.preprocess_image(image, self.ss_preprocessor)
            slat_input_dict = self.preprocess_image(image, self.slat_preprocessor)
            ss_return_dict = self.sample_sparse_structure(ss_input_dict)
            coords = ss_return_dict["coords"]
            slat = self.sample_slat(slat_input_dict, coords)

    def instantiate_and_load_from_pretrained(
        self,
        config,
        ckpt_path,
        state_dict_fn=None,
        state_dict_key="state_dict",
        device="cuda", 
    ):
        model = instantiate(config)

        if ckpt_path.endswith(".safetensors"):
            state_dict = load_file(ckpt_path, device="cuda")
            if state_dict_fn is not None:
                state_dict = state_dict_fn(state_dict)
            model.load_state_dict(state_dict, strict=False)
            model.eval()
        else:
            model = load_model_from_checkpoint(
                model,
                ckpt_path,
                strict=True,
                device="cpu",
                freeze=True,
                eval=True,
                state_dict_key=state_dict_key,
                state_dict_fn=state_dict_fn,
            )
        model = model.to(device)

        return model

    def init_pose_decoder(self, ss_generator_config_path, pose_decoder_name):
        if pose_decoder_name is None:
            pose_decoder_name = OmegaConf.load(os.path.join(self.workspace_dir, ss_generator_config_path))["module"]["pose_target_convention"]
        logger.info(f"Using pose decoder: {pose_decoder_name}")
        return get_pose_decoder(pose_decoder_name)

    def init_ss_preprocessor(self, ss_preprocessor, ss_generator_config_path):
        if ss_preprocessor is not None:
            return ss_preprocessor
        config = OmegaConf.load(os.path.join(self.workspace_dir, ss_generator_config_path))["tdfy"]["val_preprocessor"]
        return instantiate(config)

    def init_ss_generator(self, ss_generator_config_path, ss_generator_ckpt_path):
        config = OmegaConf.load(
            os.path.join(self.workspace_dir, ss_generator_config_path)
        )["module"]["generator"]["backbone"]

        state_dict_prefix_func = filter_and_remove_prefix_state_dict_fn(
            "_base_models.generator."
        )

        model = self.instantiate_and_load_from_pretrained(
            config,
            os.path.join(self.workspace_dir, ss_generator_ckpt_path),
            state_dict_fn=state_dict_prefix_func,
            device=self.device,
        )

        # Log model input/output related info
        try:
            self._log_model_io("ss_generator", model, config)
        except Exception:
            logger.exception("Failed to log ss_generator IO info")

        return model

    def init_slat_generator(self, slat_generator_config_path, slat_generator_ckpt_path):
        config = OmegaConf.load(
            os.path.join(self.workspace_dir, slat_generator_config_path)
        )["module"]["generator"]["backbone"]
        state_dict_prefix_func = filter_and_remove_prefix_state_dict_fn(
            "_base_models.generator."
        )

        model = self.instantiate_and_load_from_pretrained(
            config,
            os.path.join(self.workspace_dir, slat_generator_ckpt_path),
            state_dict_fn=state_dict_prefix_func,
            device=self.device,
        )

        # Log model input/output related info
        try:
            self._log_model_io("slat_generator", model, config)
        except Exception:
            logger.exception("Failed to log slat_generator IO info")

        return model

    def init_ss_encoder(self, ss_encoder_config_path, ss_encoder_ckpt_path):
        if ss_encoder_ckpt_path is not None:
            # override to avoid problem loading
            config = OmegaConf.load(
                os.path.join(self.workspace_dir, ss_encoder_config_path)
            )
            if "pretrained_ckpt_path" in config:
                del config["pretrained_ckpt_path"]
            model = self.instantiate_and_load_from_pretrained(
                config,
                os.path.join(self.workspace_dir, ss_encoder_ckpt_path),
                device=self.device,
                state_dict_key=None,
            )
            try:
                self._log_model_io("ss_encoder", model, config)
            except Exception:
                logger.exception("Failed to log ss_encoder IO info")
            return model
        else:
            return None

    def init_ss_decoder(self, ss_decoder_config_path, ss_decoder_ckpt_path):
        # override to avoid problem loading
        config = OmegaConf.load(
            os.path.join(self.workspace_dir, ss_decoder_config_path)
        )
        if "pretrained_ckpt_path" in config:
            del config["pretrained_ckpt_path"]
        model = self.instantiate_and_load_from_pretrained(
            config,
            os.path.join(self.workspace_dir, ss_decoder_ckpt_path),
            device=self.device,
            state_dict_key=None,
        )
        try:
            self._log_model_io("ss_decoder", model, config)
        except Exception:
            logger.exception("Failed to log ss_decoder IO info")
        return model

    def init_slat_decoder_gs(
        self, slat_decoder_gs_config_path, slat_decoder_gs_ckpt_path
    ):
        if slat_decoder_gs_config_path is None:
            return None
        else:
            config = OmegaConf.load(
                os.path.join(self.workspace_dir, slat_decoder_gs_config_path)
            )
            model = self.instantiate_and_load_from_pretrained(
                config,
                os.path.join(self.workspace_dir, slat_decoder_gs_ckpt_path),
                device=self.device,
                state_dict_key=None,
            )
            try:
                self._log_model_io("slat_decoder_gs", model, config)
            except Exception:
                logger.exception("Failed to log slat_decoder_gs IO info")
            return model

    def init_slat_decoder_mesh(
        self, slat_decoder_mesh_config_path, slat_decoder_mesh_ckpt_path
    ):
        config = OmegaConf.load(
            os.path.join(self.workspace_dir, slat_decoder_mesh_config_path)
        )
        model = self.instantiate_and_load_from_pretrained(
            config,
            os.path.join(self.workspace_dir, slat_decoder_mesh_ckpt_path),
            device=self.device,
            state_dict_key=None,
        )
        try:
            self._log_model_io("slat_decoder_mesh", model, config)
        except Exception:
            logger.exception("Failed to log slat_decoder_mesh IO info")
        return model

    def init_ss_condition_embedder(
        self, ss_generator_config_path, ss_generator_ckpt_path
    ):
        conf = OmegaConf.load(
            os.path.join(self.workspace_dir, ss_generator_config_path)
        )
        if "condition_embedder" in conf["module"]:
            config = conf["module"]["condition_embedder"]["backbone"]
            model = self.instantiate_and_load_from_pretrained(
                config,
                os.path.join(self.workspace_dir, ss_generator_ckpt_path),
                state_dict_fn=filter_and_remove_prefix_state_dict_fn(
                    "_base_models.condition_embedder."
                ),
                device=self.device,
            )
            try:
                self._log_model_io("ss_condition_embedder", model, config)
            except Exception:
                logger.exception("Failed to log ss_condition_embedder IO info")
            return model
        else:
            return None

    def init_slat_condition_embedder(
        self, slat_generator_config_path, slat_generator_ckpt_path
    ):
        return self.init_ss_condition_embedder(
            slat_generator_config_path, slat_generator_ckpt_path
        )


    def override_ss_generator_cfg_config(
        self,
        ss_generator,
        cfg_strength=7,
        inference_steps=25,
        rescale_t=3,
        cfg_interval=[0, 500],
        cfg_strength_pm=0.0,
    ):
        # override generator setting
        ss_generator.inference_steps = inference_steps
        ss_generator.reverse_fn.strength = cfg_strength
        ss_generator.reverse_fn.interval = cfg_interval
        ss_generator.rescale_t = rescale_t
        ss_generator.reverse_fn.backbone.condition_embedder.normalize_images = True
        ss_generator.reverse_fn.unconditional_handling = "add_flag"
        ss_generator.reverse_fn.strength_pm = cfg_strength_pm

        logger.info(
            "ss_generator parameters: inference_steps={}, cfg_strength={}, cfg_interval={}, rescale_t={}, cfg_strength_pm={}",
            inference_steps,
            cfg_strength,
            cfg_interval,
            rescale_t,
            cfg_strength_pm,
        )

    def override_slat_generator_cfg_config(
        self,
        slat_generator,
        cfg_strength=5,
        inference_steps=25,
        rescale_t=3,
        cfg_interval=[0, 500],
    ):
        slat_generator.inference_steps = inference_steps
        slat_generator.reverse_fn.strength = cfg_strength
        slat_generator.reverse_fn.interval = cfg_interval
        slat_generator.rescale_t = rescale_t

        logger.info(
            "slat_generator parameters: inference_steps={}, cfg_strength={}, cfg_interval={}, rescale_t={}",
            inference_steps,
            cfg_strength,
            cfg_interval,
            rescale_t,
        )


    def run(
        self,
        image: Union[None, Image.Image, np.ndarray],
        mask: Union[None, Image.Image, np.ndarray] = None,
        seed=42,
        stage1_only=False,
        with_mesh_postprocess=True,
        with_texture_baking=True,
        use_vertex_color=False,
        stage1_inference_steps=None,
        stage2_inference_steps=None,
        use_stage1_distillation=False,
        use_stage2_distillation=False,
        decode_formats=None,
    ) -> dict:
        """
        Parameters:
        - image (Image): The input image to be processed.
        - seed (int, optional): The random seed for reproducibility. Default is 42.
        - stage1_only (bool, optional): If True, only the sparse structure is sampled and returned. Default is False.
        - with_mesh_postprocess (bool, optional): If True, performs mesh post-processing. Default is True.
        - with_texture_baking (bool, optional): If True, applies texture baking to the 3D model. Default is True.
        Returns:
        - dict: A dictionary containing the GLB file and additional data from the sparse structure sampling.
        """
        # This should only happen if called from demo
        image = self.merge_image_and_mask(image, mask)
        with self.device:
            ss_input_dict = self.preprocess_image(image, self.ss_preprocessor)
            slat_input_dict = self.preprocess_image(image, self.slat_preprocessor)
            torch.manual_seed(seed)
            ss_return_dict = self.sample_sparse_structure(
                ss_input_dict,
                inference_steps=stage1_inference_steps,
                use_distillation=use_stage1_distillation,
            )

            ss_return_dict.update(self.pose_decoder(ss_return_dict))

            if "scale" in ss_return_dict:
                logger.info(f"Rescaling scale by {ss_return_dict['downsample_factor']}")
                ss_return_dict["scale"] = ss_return_dict["scale"] * ss_return_dict["downsample_factor"]
            if stage1_only:
                logger.info("Finished!")
                ss_return_dict["voxel"] = ss_return_dict["coords"][:, 1:] / 64 - 0.5
                return ss_return_dict

            coords = ss_return_dict["coords"]
            slat = self.sample_slat(
                slat_input_dict,
                coords,
                inference_steps=stage2_inference_steps,
                use_distillation=use_stage2_distillation,
            )
            outputs = self.decode_slat(
                slat, self.decode_formats if decode_formats is None else decode_formats
            )
            outputs = self.postprocess_slat_output(
                outputs, with_mesh_postprocess, with_texture_baking, use_vertex_color
            )
            logger.info("Finished!")

            return {
                **ss_return_dict,
                **outputs,
            }

    def postprocess_slat_output(
        self, outputs, with_mesh_postprocess, with_texture_baking, use_vertex_color
    ):
        # GLB files can be extracted from the outputs
        logger.info(
            f"Postprocessing mesh with option with_mesh_postprocess {with_mesh_postprocess}, with_texture_baking {with_texture_baking}..."
        )
        if "mesh" in outputs:
            glb = postprocessing_utils.to_glb(
                outputs["gaussian"][0],
                outputs["mesh"][0],
                # Optional parameters
                simplify=0.95,  # Ratio of triangles to remove in the simplification process
                texture_size=1024,  # Size of the texture used for the GLB
                verbose=False,
                with_mesh_postprocess=with_mesh_postprocess,
                with_texture_baking=with_texture_baking,
                use_vertex_color=use_vertex_color,
                rendering_engine=self.rendering_engine,
            )

        # glb.export("sample.glb")
        else:
            glb = None

        outputs["glb"] = glb

        if "gaussian" in outputs:
            outputs["gs"] = outputs["gaussian"][0]

        if "gaussian_4" in outputs:
            outputs["gs_4"] = outputs["gaussian_4"][0]

        return outputs

    def merge_image_and_mask(
        self,
        image: Union[np.ndarray, Image.Image],
        mask: Union[None, np.ndarray, Image.Image],
    ):
        if mask is not None:
            if isinstance(image, Image.Image):
                image = np.array(image)

            mask = np.array(mask)
            if mask.ndim == 2:
                mask = mask[..., None]

            logger.info(f"Replacing alpha channel with the provided mask")
            assert mask.shape[:2] == image.shape[:2]
            image = np.concatenate([image[..., :3], mask], axis=-1)

        image = np.array(image)
        return image

    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ["mesh", "gaussian"],
    ) -> dict:
        """
        Decode the structured latent.

        Args:
            slat (sp.SparseTensor): The structured latent.
            formats (List[str]): The formats to decode the structured latent to.

        Returns:
            dict: The decoded structured latent.
        """
        logger.info("Decoding sparse latent...")
        ret = {}
        with torch.no_grad():
            if "mesh" in formats:
                ret["mesh"] = self._call_model(
                    "slat_decoder_mesh", self.models["slat_decoder_mesh"], slat
                )
            if "gaussian" in formats:
                ret["gaussian"] = self._call_model(
                    "slat_decoder_gs", self.models["slat_decoder_gs"], slat
                )
            if "gaussian_4" in formats:
                ret["gaussian_4"] = self._call_model(
                    "slat_decoder_gs_4", self.models.get("slat_decoder_gs_4"), slat
                )
        # if "radiance_field" in formats:
        #     ret["radiance_field"] = self.models["slat_decoder_rf"](slat)
        return ret

    def is_mm_dit(self, model_name="ss_generator"):
        return hasattr(self.models[model_name].reverse_fn.backbone, "latent_mapping")

    def embed_condition(self, condition_embedder, *args, **kwargs):
        if condition_embedder is not None:
            tokens = self._call_model("condition_embedder", condition_embedder, *args, **kwargs)
            # try to log token shape if tensor-like
            try:
                if isinstance(tokens, torch.Tensor):
                    logger.info("Condition embedder output tokens shape: {}", tokens.shape)
                elif isinstance(tokens, (list, tuple)) and len(tokens) > 0 and isinstance(tokens[0], torch.Tensor):
                    logger.info("Condition embedder output tokens shape: {}", tokens[0].shape)
            except Exception:
                pass
            return tokens, None, None
        return None, args, kwargs

    def get_condition_input(self, condition_embedder, input_dict, input_mapping):
        condition_args = self.map_input_keys(input_dict, input_mapping)
        condition_kwargs = {
            k: v for k, v in input_dict.items() if k not in input_mapping
        }
        logger.info("Running condition embedder ...")
        embedded_cond, condition_args, condition_kwargs = self.embed_condition(
            condition_embedder, *condition_args, **condition_kwargs
        )
        logger.info("Condition embedder finishes!")
        if embedded_cond is not None:
            condition_args = (embedded_cond,)
            condition_kwargs = {}

        return condition_args, condition_kwargs

    def sample_sparse_structure(
        self, ss_input_dict: dict, inference_steps=None, use_distillation=False
    ):
        logger.info("=" * 80)
        logger.info("SAMPLE_SPARSE_STRUCTURE STARTED")
        logger.info("=" * 80)
        
        ss_generator = self.models["ss_generator"]
        ss_decoder = self.models["ss_decoder"]
        if use_distillation:
            ss_generator.no_shortcut = False
            ss_generator.reverse_fn.strength = 0
            ss_generator.reverse_fn.strength_pm = 0
        else:
            ss_generator.no_shortcut = True
            ss_generator.reverse_fn.strength = self.ss_cfg_strength
            ss_generator.reverse_fn.strength_pm = self.ss_cfg_strength_pm

        prev_inference_steps = ss_generator.inference_steps
        if inference_steps:
            ss_generator.inference_steps = inference_steps

        image = ss_input_dict["image"]
        bs = image.shape[0]
        
        logger.info("Input ss_input_dict:")
        for k, v in ss_input_dict.items():
            if hasattr(v, 'shape'):
                logger.info(f"  '{k}': shape {v.shape}, dtype {v.dtype}")
            else:
                logger.info(f"  '{k}': {type(v)}")
        
        logger.info(
            "Sampling sparse structure: inference_steps={}, strength={}, interval={}, rescale_t={}, cfg_strength_pm={}",
            ss_generator.inference_steps,
            ss_generator.reverse_fn.strength,
            ss_generator.reverse_fn.interval,
            ss_generator.rescale_t,
            ss_generator.reverse_fn.strength_pm,
        )

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=self.shape_model_dtype):
                if self.is_mm_dit():
                    latent_shape_dict = {
                        k: (bs,) + (v.pos_emb.shape[0], v.input_layer.in_features)
                        for k, v in ss_generator.reverse_fn.backbone.latent_mapping.items()
                    }
                    logger.info("Multi-modal DiT latent_shape_dict:")
                    for k, v in latent_shape_dict.items():
                        logger.info(f"  '{k}': {v}")
                else:
                    latent_shape_dict = (bs,) + (4096, 8)
                    logger.info(f"Single-modal latent_shape: {latent_shape_dict}")

                logger.info("-" * 40)
                logger.info("CONDITION EMBEDDING PHASE")
                logger.info("-" * 40)
                condition_args, condition_kwargs = self.get_condition_input(
                    self.condition_embedders["ss_condition_embedder"],
                    ss_input_dict,
                    self.ss_condition_input_mapping,
                )
                
                logger.info("-" * 40)
                logger.info("SS_GENERATOR PHASE (Flow Matching / ODE Solve)")
                logger.info("-" * 40)
                return_dict = self._call_model(
                    "ss_generator",
                    ss_generator,
                    latent_shape_dict,
                    image.device,
                    *condition_args,
                    **condition_kwargs,
                )
                if not self.is_mm_dit():
                    return_dict = {"shape": return_dict}

                logger.info("ss_generator output (return_dict):")
                for k, v in return_dict.items():
                    if hasattr(v, 'shape'):
                        logger.info(f"  '{k}': shape {v.shape}")
                    else:
                        logger.info(f"  '{k}': {type(v)}")

                logger.info("-" * 40)
                logger.info("SS_DECODER PHASE (Latent -> Occupancy Grid)")
                logger.info("-" * 40)
                shape_latent = return_dict["shape"]
                decoder_input = shape_latent.permute(0, 2, 1).contiguous().view(shape_latent.shape[0], 8, 16, 16, 16)
                logger.info(f"ss_decoder input shape: {decoder_input.shape}")
                
                ss = self._call_model(
                    "ss_decoder",
                    ss_decoder,
                    decoder_input,
                )
                logger.info(f"ss_decoder output (occupancy) shape: {ss.shape}")
                logger.info(f"Occupancy stats: min={ss.min().item():.4f}, max={ss.max().item():.4f}, "
                           f"mean={ss.mean().item():.4f}, num_positive={(ss > 0).sum().item()}")
                
                coords = torch.argwhere(ss > 0)[:, [0, 2, 3, 4]].int()
                logger.info(f"Extracted coords shape: {coords.shape}")

                # downsample output
                return_dict["coords_original"] = coords
                original_shape = coords.shape
                if self.downsample_ss_dist > 0:
                    coords = prune_sparse_structure(
                        coords,
                        max_neighbor_axes_dist=self.downsample_ss_dist,
                    )
                coords, downsample_factor = downsample_sparse_structure(coords)
                logger.info(
                    f"Downsampled coords from {original_shape[0]} to {coords.shape[0]}"
                )
                return_dict["coords"] = coords
                return_dict["downsample_factor"] = downsample_factor

        ss_generator.inference_steps = prev_inference_steps
        
        logger.info("=" * 80)
        logger.info("SAMPLE_SPARSE_STRUCTURE COMPLETED")
        logger.info("=" * 80)
        return return_dict

    def sample_slat(
        self,
        slat_input: dict,
        coords: torch.Tensor,
        inference_steps=25,
        use_distillation=False,
    ) -> sp.SparseTensor:
        image = slat_input["image"]
        DEVICE = image.device
        slat_generator = self.models["slat_generator"]
        latent_shape = (image.shape[0],) + (coords.shape[0], 8)
        prev_inference_steps = slat_generator.inference_steps
        if inference_steps:
            slat_generator.inference_steps = inference_steps
        if use_distillation:
            slat_generator.no_shortcut = False
            slat_generator.reverse_fn.strength = 0
        else:
            slat_generator.no_shortcut = True
            slat_generator.reverse_fn.strength = self.slat_cfg_strength

        logger.info(
            "Sampling sparse latent: inference_steps={}, strength={}, interval={}, rescale_t={}",
            slat_generator.inference_steps,
            slat_generator.reverse_fn.strength,
            slat_generator.reverse_fn.interval,
            slat_generator.rescale_t,
        )

        with torch.autocast(device_type="cuda", dtype=self.dtype):
            with torch.no_grad():
                condition_args, condition_kwargs = self.get_condition_input(
                    self.condition_embedders["slat_condition_embedder"],
                    slat_input,
                    self.slat_condition_input_mapping,
                )
                condition_args += (coords.cpu().numpy(),)
                slat = self._call_model(
                    "slat_generator",
                    slat_generator,
                    latent_shape,
                    DEVICE,
                    *condition_args,
                    **condition_kwargs,
                )
                slat = sp.SparseTensor(
                    coords=coords,
                    feats=slat[0],
                ).to(DEVICE)
                slat = slat * self.slat_std.to(DEVICE) + self.slat_mean.to(DEVICE)

        slat_generator.inference_steps = prev_inference_steps
        return slat

    def _apply_transform(self, input: torch.Tensor, transform):
        if input is not None:
            input = transform(input)

        return input

    def _preprocess_image_and_mask(
        self, rgb_image, mask_image, img_mask_joint_transform
    ):
        for trans in img_mask_joint_transform:
            rgb_image, mask_image = trans(rgb_image, mask_image)
        return rgb_image, mask_image

    def map_input_keys(self, item, condition_input_mapping):
        output = [item[k] for k in condition_input_mapping]

        return output

    def image_to_float(self, image):
        image = np.array(image)
        image = image / 255
        image = image.astype(np.float32)
        return image

    def preprocess_image(
        self, image: Union[Image.Image, np.ndarray], preprocessor
    ) -> torch.Tensor:
        logger.info("-" * 40)
        logger.info("preprocess_image called")
        
        # canonical type is numpy
        if not isinstance(input, np.ndarray):
            image = np.array(image)

        assert image.ndim == 3  # no batch dimension as of now
        assert image.shape[-1] == 4  # rgba format
        assert image.dtype == np.uint8  # [0,255] range
        
        logger.info(f"Input image: shape {image.shape}, dtype {image.dtype}")

        rgba_image = torch.from_numpy(self.image_to_float(image))
        rgba_image = rgba_image.permute(2, 0, 1).contiguous()
        logger.info(f"After to_float and permute: {rgba_image.shape}")
        
        rgb_image = rgba_image[:3]
        rgb_image_mask = (get_mask(rgba_image, None, "ALPHA_CHANNEL") > 0).float()
        logger.info(f"rgb_image shape: {rgb_image.shape}")
        logger.info(f"rgb_image_mask shape: {rgb_image_mask.shape}")
        
        processed_rgb_image, processed_mask = self._preprocess_image_and_mask(
            rgb_image, rgb_image_mask, preprocessor.img_mask_joint_transform
        )
        logger.info(f"After img_mask_joint_transform: rgb {processed_rgb_image.shape}, mask {processed_mask.shape}")

        # transform tensor to model input
        processed_rgb_image = self._apply_transform(
            processed_rgb_image, preprocessor.img_transform
        )
        processed_mask = self._apply_transform(
            processed_mask, preprocessor.mask_transform
        )
        logger.info(f"After individual transforms: rgb {processed_rgb_image.shape}, mask {processed_mask.shape}")

        # full image, with only processing from the image
        rgb_image = self._apply_transform(rgb_image, preprocessor.img_transform)
        rgb_image_mask = self._apply_transform(
            rgb_image_mask, preprocessor.mask_transform
        )
        item = {
            "mask": processed_mask[None].to(self.device),
            "image": processed_rgb_image[None].to(self.device),
            "rgb_image": rgb_image[None].to(self.device),
            "rgb_image_mask": rgb_image_mask[None].to(self.device),
        }
        
        logger.info("Preprocessed output dict:")
        for k, v in item.items():
            logger.info(f"  '{k}': shape {v.shape}, dtype {v.dtype}")
        logger.info("-" * 40)

        return item

    @staticmethod
    def _get_dtype(dtype):
        if dtype == "bfloat16":
            return torch.bfloat16
        elif dtype == "float16":
            return torch.float16
        elif dtype == "float32":
            return torch.float32
        else:
            raise NotImplementedError

    def _log_model_io(self, name, model, config=None):
        """Collect and log basic model IO-related info: parameter count, common attribute dims,
        and any latent mapping shapes if present in the model/backbone.

        This is a best-effort helper — it inspects common attributes and the provided config
        to produce useful logging for debugging. It must not raise on inspection failures.
        """
        try:
            params = sum(p.numel() for p in model.parameters())
        except Exception:
            params = None

        dims = {}
        # common attribute names that may indicate input/output dims
        try:
            candidates = [
                "in_features",
                "out_features",
                "input_dim",
                "output_dim",
                "latent_dim",
                "num_features",
                "embedding_dim",
            ]
            for attr in candidates:
                if hasattr(model, attr):
                    dims[attr] = getattr(model, attr)

            # mm-dit style backbone latent mapping
            if hasattr(model, "reverse_fn") and hasattr(model.reverse_fn, "backbone"):
                backbone = model.reverse_fn.backbone
                if hasattr(backbone, "latent_mapping"):
                    for k, v in backbone.latent_mapping.items():
                        try:
                            if hasattr(v, "pos_emb"):
                                dims[f"latent_mapping.{k}.pos_emb_shape"] = tuple(v.pos_emb.shape)
                        except Exception:
                            dims[f"latent_mapping.{k}.pos_emb_shape"] = "<error>"
                        try:
                            if hasattr(v, "input_layer") and hasattr(v.input_layer, "in_features"):
                                dims[f"latent_mapping.{k}.in_features"] = v.input_layer.in_features
                        except Exception:
                            dims[f"latent_mapping.{k}.in_features"] = "<error>"

            # Also look for encoder/decoder-specific accessible attributes
            if hasattr(model, "input_shape"):
                dims["input_shape"] = getattr(model, "input_shape")
            if hasattr(model, "output_shape"):
                dims["output_shape"] = getattr(model, "output_shape")
        except Exception:
            logger.exception("Error while extracting dims for %s", name)

        # config keys may carry hints about shapes
        cfg_keys = None
        try:
            if config is not None:
                if isinstance(config, dict):
                    cfg_keys = list(config.keys())
                else:
                    try:
                        cfg_keys = list(config.keys())
                    except Exception:
                        cfg_keys = str(config)
        except Exception:
            cfg_keys = "<error>"

        logger.info(
            "Initialized model {}: params={}, dims={}, config_keys={}",
            name,
            params,
            dims,
            cfg_keys,
        )

    def _summarize_obj(self, obj):
        """Return a compact summary (type and shape/length) for tensors, numpy arrays,
        sparse tensors and common Python containers."""
        try:
            import numpy as _np
        except Exception:
            _np = None

        try:
            # torch tensor
            if isinstance(obj, torch.Tensor):
                return f"Tensor{tuple(obj.shape)}"
            # numpy array
            if _np is not None and isinstance(obj, _np.ndarray):
                return f"ndarray{obj.shape}"
            # sparse tensor from sp module
            try:
                if hasattr(obj, "coords") and hasattr(obj, "feats"):
                    c = getattr(obj, "coords")
                    f = getattr(obj, "feats")
                    return f"SparseTensor(coords={getattr(c, 'shape', None)}, feats={getattr(f, 'shape', None)})"
            except Exception:
                pass
            # dict
            if isinstance(obj, dict):
                return {k: self._summarize_obj(v) for k, v in obj.items()}
            # list/tuple
            if isinstance(obj, (list, tuple)):
                return [self._summarize_obj(v) for v in obj]
            # fallback to str
            return str(type(obj))
        except Exception:
            return "<error summarizing>"

    def _call_model(self, name, model, *args, **kwargs):
        """Call a model while logging input and output summaries and performing
        a simple batch-dimension consistency check.

        - Logs summaries for inputs (shapes/types).
        - Calls the model and logs summaries for outputs.
        - If a batch-size can be inferred from inputs, verifies that tensor outputs
          share that batch size on their first dimension.
        """
        try:
            inp_summary = {}
            # summarize positional args
            for i, a in enumerate(args):
                inp_summary[f"arg{i}"] = self._summarize_obj(a)
            # summarize kwargs
            for k, v in kwargs.items():
                inp_summary[f"kw:{k}"] = self._summarize_obj(v)

            logger.info("Calling model {} with inputs={}", name, inp_summary)

            out = model(*args, **kwargs)

            out_summary = self._summarize_obj(out)
            logger.info("Model {} returned {}", name, out_summary)

            # simple batch-size consistency check
            try:
                batch = None
                # find batch size from inputs
                for a in list(args) + list(kwargs.values()):
                    if isinstance(a, torch.Tensor) and a.dim() >= 1:
                        batch = a.shape[0]
                        break
                    # numpy arrays
                    try:
                        import numpy as _np

                        if _np is not None and isinstance(a, _np.ndarray) and a.ndim >= 1:
                            batch = a.shape[0]
                            break
                    except Exception:
                        pass

                # validate outputs if batch known
                if batch is not None:
                    def _check(o):
                        if isinstance(o, torch.Tensor) and o.dim() >= 1:
                            if o.shape[0] != batch:
                                logger.warning(
                                    "Model {} output tensor first-dim {} != inferred batch {}",
                                    name,
                                    o.shape[0],
                                    batch,
                                )
                        elif isinstance(o, dict):
                            for v in o.values():
                                _check(v)
                        elif isinstance(o, (list, tuple)):
                            for v in o:
                                _check(v)

                    _check(out)
            except Exception:
                logger.exception("Error during batch-size check for {}", name)

            return out
        except Exception:
            logger.exception("Error while calling model %s", name)
            raise
