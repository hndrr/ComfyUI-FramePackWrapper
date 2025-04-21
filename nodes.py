import os
import torch

# import torch.nn.functional as F # Unused
# import gc # Unused
# import numpy as np # Unused
import math
from tqdm import tqdm

from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

import folder_paths
import comfy.model_management as mm
from comfy.utils import load_torch_file  # ProgressBar, common_upscale # Unused
import comfy.model_base
import comfy.latent_formats

# from comfy.cli_args import args, LatentPreviewMethod # Unused

# LoRA specific imports
from diffusers.loaders.lora_pipeline import _fetch_state_dict
from diffusers.loaders.lora_conversion_utils import (
    _convert_hunyuan_video_lora_to_diffusers,
)

# Moved helper imports here for better organization
from .diffusers_helper.models.hunyuan_video_packed import (
    HunyuanVideoTransformer3DModelPacked,
)
from .diffusers_helper.memory import (
    DynamicSwapInstaller,
    move_model_to_device_with_memory_preservation,
    # offload_model_from_device_for_memory_preservation # Unused in this file
)
from .diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from .diffusers_helper.utils import crop_or_pad_yield_mask
from .diffusers_helper.bucket_tools import find_nearest_bucket

script_directory = os.path.dirname(os.path.abspath(__file__))
vae_scaling_factor = 0.476986


class HyVideoModel(comfy.model_base.BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipeline = {}
        self.load_device = mm.get_torch_device()

    def __getitem__(self, k):
        return self.pipeline[k]

    def __setitem__(self, k, v):
        self.pipeline[k] = v


class HyVideoModelConfig:
    def __init__(self, dtype):
        self.unet_config = {}
        self.unet_extra_config = {}
        self.latent_format = comfy.latent_formats.HunyuanVideo()
        self.latent_format.latent_channels = 16
        self.manual_cast_dtype = dtype
        self.sampling_settings = {"multiplier": 1.0}
        self.memory_usage_factor = 2.0
        self.unet_config["disable_unet_model_creation"] = True


class FramePackTorchCompileSettings:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "backend": (["inductor", "cudagraphs"], {"default": "inductor"}),
                "fullgraph": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Enable full graph mode"},
                ),
                "mode": (
                    [
                        "default",
                        "max-autotune",
                        "max-autotune-no-cudagraphs",
                        "reduce-overhead",
                    ],
                    {"default": "default"},
                ),
                "dynamic": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Enable dynamic mode"},
                ),
                "dynamo_cache_size_limit": (
                    "INT",
                    {
                        "default": 64,
                        "min": 0,
                        "max": 1024,
                        "step": 1,
                        "tooltip": "torch._dynamo.config.cache_size_limit",
                    },
                ),
                "compile_single_blocks": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Enable single block compilation"},
                ),
                "compile_double_blocks": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Enable double block compilation"},
                ),
            },
        }

    RETURN_TYPES = ("FRAMEPACKCOMPILEARGS",)
    RETURN_NAMES = ("torch_compile_args",)
    FUNCTION = "loadmodel"
    CATEGORY = "FramePackWrapper"  # Changed category for consistency
    DESCRIPTION = (
        "torch.compile settings, when connected to the model loader, "
        "torch.compile of the selected layers is attempted. "
        "Requires Triton and torch 2.5.0 is recommended"
    )

    def loadmodel(
        self,
        backend,
        fullgraph,
        mode,
        dynamic,
        dynamo_cache_size_limit,
        compile_single_blocks,
        compile_double_blocks,
    ):

        compile_args = {
            "backend": backend,
            "fullgraph": fullgraph,
            "mode": mode,
            "dynamic": dynamic,
            "dynamo_cache_size_limit": dynamo_cache_size_limit,
            "compile_single_blocks": compile_single_blocks,
            "compile_double_blocks": compile_double_blocks,
        }

        return (compile_args,)


# region Model loading
class DownloadAndLoadFramePackModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (["lllyasviel/FramePackI2V_HY"],),
                "base_precision": (["fp32", "bf16", "fp16"], {"default": "bf16"}),
                "quantization": (
                    ["disabled", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],
                    {"default": "disabled", "tooltip": "optional quantization method"},
                ),
            },
            "optional": {
                "attention_mode": (
                    [
                        "sdpa",
                        "flash_attn",
                        "sageattn",
                    ],
                    {"default": "sdpa"},
                ),
                "compile_args": ("FRAMEPACKCOMPILEARGS",),
            },
        }

    RETURN_TYPES = ("FramePackMODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "loadmodel"
    CATEGORY = "FramePackWrapper"

    def loadmodel(
        self,
        model,
        base_precision,
        quantization,
        compile_args=None,
        attention_mode="sdpa",
    ):

        base_dtype = {
            "fp8_e4m3fn": torch.float8_e4m3fn,
            "fp8_e4m3fn_fast": torch.float8_e4m3fn,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp16_fast": torch.float16,
            "fp32": torch.float32,
        }[base_precision]

        device = mm.get_torch_device()

        model_path = os.path.join(
            folder_paths.models_dir, "diffusers", "lllyasviel", "FramePackI2V_HY"
        )
        if not os.path.exists(model_path):
            print(f"Downloading clip model to: {model_path}")
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=model,
                local_dir=model_path,
                local_dir_use_symlinks=False,
            )

        transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(
            model_path, torch_dtype=base_dtype, attention_mode=attention_mode
        ).cpu()
        params_to_keep = {
            "norm",
            "bias",
            "time_in",
            "vector_in",
            "guidance_in",
            "txt_in",
            "img_in",
        }
        if quantization == "fp8_e4m3fn" or quantization == "fp8_e4m3fn_fast":
            transformer = transformer.to(torch.float8_e4m3fn)
            if quantization == "fp8_e4m3fn_fast":
                from .fp8_optimization import convert_fp8_linear

                convert_fp8_linear(
                    transformer, base_dtype, params_to_keep=params_to_keep
                )
        elif quantization == "fp8_e5m2":
            transformer = transformer.to(torch.float8_e5m2)
        else:
            transformer = transformer.to(base_dtype)

        DynamicSwapInstaller.install_model(transformer, device=device)

        if compile_args is not None:
            if compile_args["compile_single_blocks"]:
                for i, block in enumerate(transformer.single_transformer_blocks):
                    transformer.single_transformer_blocks[i] = torch.compile(
                        block,
                        fullgraph=compile_args["fullgraph"],
                        dynamic=compile_args["dynamic"],
                        backend=compile_args["backend"],
                        mode=compile_args["mode"],
                    )
            if compile_args["compile_double_blocks"]:
                for i, block in enumerate(transformer.transformer_blocks):
                    transformer.transformer_blocks[i] = torch.compile(
                        block,
                        fullgraph=compile_args["fullgraph"],
                        dynamic=compile_args["dynamic"],
                        backend=compile_args["backend"],
                        mode=compile_args["mode"],
                    )

            # transformer = torch.compile(
            #     transformer, fullgraph=compile_args["fullgraph"],
            #     dynamic=compile_args["dynamic"],
            #     backend=compile_args["backend"], mode=compile_args["mode"]
            # )

        pipe = {
            "transformer": transformer.eval(),
            "dtype": base_dtype,
        }
        return (pipe,)


class LoadFramePackModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    folder_paths.get_filename_list("diffusion_models"),
                    {"tooltip": "Models from 'ComfyUI/models/diffusion_models'"},
                ),
                "base_precision": (["fp32", "bf16", "fp16"], {"default": "bf16"}),
                "quantization": (
                    ["disabled", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],
                    {"default": "disabled", "tooltip": "optional quantization method"},
                ),
            },
            "optional": {
                "attention_mode": (
                    [
                        "sdpa",
                        "flash_attn",
                        "sageattn",
                    ],
                    {"default": "sdpa"},
                ),
                "compile_args": ("FRAMEPACKCOMPILEARGS",),
            },
        }

    RETURN_TYPES = ("FramePackMODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "loadmodel"
    CATEGORY = "FramePackWrapper"

    def loadmodel(
        self,
        model,
        base_precision,
        quantization,
        compile_args=None,
        attention_mode="sdpa",
    ):

        base_dtype = {
            "fp8_e4m3fn": torch.float8_e4m3fn,
            "fp8_e4m3fn_fast": torch.float8_e4m3fn,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp16_fast": torch.float16,
            "fp32": torch.float32,
        }[base_precision]

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model)
        model_config_path = os.path.join(script_directory, "transformer_config.json")
        import json

        with open(model_config_path, "r") as f:
            config = json.load(f)
        sd = load_torch_file(model_path, device=offload_device, safe_load=True)

        with init_empty_weights():
            transformer = HunyuanVideoTransformer3DModelPacked(**config)

        params_to_keep = {
            "norm",
            "bias",
            "time_in",
            "vector_in",
            "guidance_in",
            "txt_in",
            "img_in",
        }
        if (
            quantization == "fp8_e4m3fn"
            or quantization == "fp8_e4m3fn_fast"
            or quantization == "fp8_scaled"
        ):
            dtype = torch.float8_e4m3fn
        elif quantization == "fp8_e5m2":
            dtype = torch.float8_e5m2
        else:
            dtype = base_dtype
        print("Using accelerate to load and assign model weights to device...")
        param_count = sum(1 for _ in transformer.named_parameters())
        for name, _ in tqdm(  # Use _ for unused param
            transformer.named_parameters(),
            desc=f"Loading transformer parameters to {offload_device}",
            total=param_count,
            leave=True,
        ):
            dtype_to_use = (
                base_dtype
                if any(keyword in name for keyword in params_to_keep)
                else dtype
            )

            set_module_tensor_to_device(
                transformer,
                name,
                device=offload_device,
                dtype=dtype_to_use,
                value=sd[name],
            )

        if quantization == "fp8_e4m3fn_fast":
            from .fp8_optimization import convert_fp8_linear

            convert_fp8_linear(transformer, base_dtype, params_to_keep=params_to_keep)

        DynamicSwapInstaller.install_model(transformer, device=device)

        if compile_args is not None:
            if compile_args["compile_single_blocks"]:
                for i, block in enumerate(transformer.single_transformer_blocks):
                    transformer.single_transformer_blocks[i] = torch.compile(
                        block,
                        fullgraph=compile_args["fullgraph"],
                        dynamic=compile_args["dynamic"],
                        backend=compile_args["backend"],
                        mode=compile_args["mode"],
                    )
            if compile_args["compile_double_blocks"]:
                for i, block in enumerate(transformer.transformer_blocks):
                    transformer.transformer_blocks[i] = torch.compile(
                        block,
                        fullgraph=compile_args["fullgraph"],
                        dynamic=compile_args["dynamic"],
                        backend=compile_args["backend"],
                        mode=compile_args["mode"],
                    )

            # transformer = torch.compile(
            #     transformer, fullgraph=compile_args["fullgraph"],
            #     dynamic=compile_args["dynamic"],
            #     backend=compile_args["backend"], mode=compile_args["mode"]
            # )

        pipe = {
            "transformer": transformer.eval(),
            "dtype": base_dtype,
        }
        return (pipe,)


class LoadFramePackLora:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("FramePackMODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("FramePackMODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_lora"
    CATEGORY = "FramePackWrapper"

    def load_lora(self, model, lora_name, strength):
        if strength == 0:
            return (model,)

        lora_path = folder_paths.get_full_path("loras", lora_name)
        transformer = model["transformer"]
        base_dtype = model["dtype"]
        device = mm.get_torch_device()

        print(f"Loading LoRA: {lora_name} with strength {strength}")

        # Use _fetch_state_dict similar to diffusers_helper/load_lora.py
        state_dict = _fetch_state_dict(
            lora_path,
            weight_name=None,  # Auto-detect based on file extension
            use_safetensors=True,  # Assume safetensors by default
            local_files_only=True,  # Load from local files
            cache_dir=None,
            force_download=False,
            proxies=None,
            revision=None,
            token=None,
            user_agent=None,
            subfolder=None,
            mirror=None,
        )

        # Try converting assuming Hunyuan format first, similar to diffusers_helper
        original_keys = set(state_dict.keys())
        try:
            print(
                "Attempting conversion assuming non-diffusers LoRA format (e.g., Hunyuan)..."
            )
            converted_state_dict = _convert_hunyuan_video_lora_to_diffusers(state_dict)
            # Check if conversion actually changed keys or shapes (simple check)
            conversion_successful = False
            if set(converted_state_dict.keys()) != original_keys:
                conversion_successful = True
            else:
                # Check shapes if keys are the same
                for k in original_keys:
                    if (
                        k in converted_state_dict
                        and state_dict[k].shape != converted_state_dict[k].shape
                    ):
                        conversion_successful = True
                        break

            if conversion_successful:
                print("Conversion successful.")
                state_dict = converted_state_dict
            else:
                print(
                    "Conversion did not change keys/shapes, assuming Diffusers format or already converted."
                )
        except Exception as e:
            print(
                f"WARN: Failed to convert LoRA keys assuming Hunyuan format: {e}. "
                "Proceeding with original keys, assuming Diffusers format."
            )

        # Apply strength scaling
        scaled_state_dict = {}
        for k, v in state_dict.items():
            # Ensure tensor is on the correct device and dtype before scaling
            try:
                scaled_state_dict[k] = v.to(device=device, dtype=base_dtype) * strength
            except Exception as e:
                print(f"WARN: Could not scale key {k}: {e}. Skipping this key.")

        # Load the scaled LoRA adapter
        try:
            # Ensure the transformer is on the correct device before loading
            transformer.to(device)
            transformer.load_lora_adapter(
                scaled_state_dict, network_alphas=None
            )  # Use scaled_state_dict
            print("LoRA weights loaded and applied successfully.")
        except Exception as e:
            print(f"ERROR: Failed to load LoRA adapter: {e}")
            # Return original model on failure
            return (model,)

        # Return the modified model dictionary
        pipe = {
            "transformer": transformer,  # Return the modified transformer
            "dtype": base_dtype,
        }
        return (pipe,)


class FramePackFindNearestBucket:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image to resize"}),
                "base_resolution": (
                    "INT",
                    {
                        "default": 640,
                        "min": 64,
                        "max": 2048,
                        "step": 8,
                        "tooltip": "Target resolution for bucket search",
                    },
                ),
            },
        }

    RETURN_TYPES = (
        "INT",
        "INT",
    )
    RETURN_NAMES = (
        "width",
        "height",
    )

    FUNCTION = "process"
    CATEGORY = "FramePackWrapper"  # Assuming this was intended

    def process(self, image, base_resolution):
        if image.shape[0] == 0:
            # Handle case with no image input gracefully
            print("WARN: No image provided to FramePackFindNearestBucket.")
            # Return a default or common bucket size? Or raise error?
            # Returning default for now.
            return (
                base_resolution,
                base_resolution,
            )

        H, W = image.shape[1], image.shape[2]

        new_height, new_width = find_nearest_bucket(H, W, resolution=base_resolution)

        return (
            new_width,
            new_height,
        )


class FramePackSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("FramePackMODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "image_embeds": ("CLIP_VISION_OUTPUT",),
                "steps": ("INT", {"default": 30, "min": 1}),
                "use_teacache": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Use teacache for faster sampling."},
                ),
                "teacache_rel_l1_thresh": (
                    "FLOAT",
                    {
                        "default": 0.15,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "The threshold for the relative L1 loss.",
                    },
                ),
                "cfg": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.01},
                ),
                "guidance_scale": (
                    "FLOAT",
                    {"default": 10.0, "min": 0.0, "max": 32.0, "step": 0.01},
                ),
                "shift": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.01},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "latent_window_size": (
                    "INT",
                    {
                        "default": 9,
                        "min": 1,
                        "max": 33,
                        "step": 1,
                        "tooltip": "Size of the latent window for sampling.",
                    },
                ),
                "total_second_length": (
                    "FLOAT",
                    {
                        "default": 5,
                        "min": 1,
                        "max": 120,
                        "step": 0.1,
                        "tooltip": "Total length of the video in seconds.",
                    },
                ),
                "gpu_memory_preservation": (
                    "FLOAT",
                    {
                        "default": 6.0,
                        "min": 0.0,
                        "max": 128.0,
                        "step": 0.1,
                        "tooltip": "Amount of GPU memory (GB) to preserve.",
                    },
                ),
                "sampler": (["unipc_bh1", "unipc_bh2"], {"default": "unipc_bh1"}),
            },
            "optional": {
                "start_latent": ("LATENT", {"tooltip": "Init Latents for image2video"}),
                "initial_samples": (
                    "LATENT",
                    {"tooltip": "Init Latents for video2video"},
                ),
                "denoise_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "process"
    CATEGORY = "FramePackWrapper"

    def process(
        self,
        model,
        shift,
        positive,
        negative,
        latent_window_size,
        use_teacache,
        total_second_length,
        teacache_rel_l1_thresh,
        image_embeds,
        steps,
        cfg,
        guidance_scale,
        seed,
        sampler,
        gpu_memory_preservation,
        start_latent=None,
        initial_samples=None,
        denoise_strength=1.0,
    ):

        if start_latent is None:
            raise ValueError("start_latent is required for FramePackSampler")

        total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
        total_latent_sections = int(max(round(total_latent_sections), 1))
        print(f"Total latent sections: {total_latent_sections}")

        transformer = model["transformer"]
        base_dtype = model["dtype"]

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        mm.unload_all_models()
        mm.cleanup_models()
        mm.soft_empty_cache()

        # Ensure start_latent is processed correctly
        processed_start_latent = start_latent["samples"] * vae_scaling_factor
        if initial_samples is not None:
            initial_samples = initial_samples["samples"] * vae_scaling_factor

        print(f"Start latent shape: {processed_start_latent.shape}")
        B, C, T, H, W = processed_start_latent.shape

        image_encoder_last_hidden_state = (
            image_embeds["last_hidden_state"].to(base_dtype).to(device)
        )

        llama_vec = positive[0][0].to(base_dtype).to(device)
        clip_l_pooler = positive[0][1]["pooled_output"].to(base_dtype).to(device)

        if not math.isclose(cfg, 1.0):
            llama_vec_n = negative[0][0].to(base_dtype)
            clip_l_pooler_n = negative[0][1]["pooled_output"].to(base_dtype).to(device)
        else:
            llama_vec_n = torch.zeros_like(llama_vec, device=device)
            clip_l_pooler_n = torch.zeros_like(clip_l_pooler, device=device)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(
            llama_vec_n, length=512
        )

        # Sampling

        rnd = torch.Generator("cpu").manual_seed(seed)

        num_frames = latent_window_size * 4 - 3

        # Initialize history_latents based on start_latent shape
        history_latents = torch.zeros(
            size=(B, C, 1 + 2 + 16, H, W), dtype=torch.float32
        ).cpu()

        total_generated_latent_frames = 0

        latent_paddings_list = list(reversed(range(total_latent_sections)))
        latent_paddings = latent_paddings_list.copy()  # Create a copy for iteration

        comfy_model = HyVideoModel(
            HyVideoModelConfig(base_dtype),
            model_type=comfy.model_base.ModelType.FLOW,
            device=device,
        )

        patcher = comfy.model_patcher.ModelPatcher(
            comfy_model, device, torch.device("cpu")
        )
        # Import locally to avoid potential top-level import issues if optional
        try:
            from latent_preview import prepare_callback

            callback = prepare_callback(patcher, steps)
        except ImportError:
            print("WARN: latent_preview module not found. Preview disabled.")
            callback = None

        move_model_to_device_with_memory_preservation(
            transformer,
            target_device=device,
            preserved_memory_gb=gpu_memory_preservation,
        )

        if total_latent_sections > 4:
            # Hunyuan original padding trick
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]
            latent_paddings_list = latent_paddings.copy()

        for latent_padding in latent_paddings:
            print(f"Latent padding: {latent_padding}")
            is_last_section = latent_padding == 0
            latent_padding_size = latent_padding * latent_window_size

            print(
                f"Padding size = {latent_padding_size}, "
                f"Is last section = {is_last_section}"
            )

            indices = torch.arange(
                0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])
            ).unsqueeze(0)
            (
                clean_latent_indices_pre,
                blank_indices,
                latent_indices,
                clean_latent_indices_post,
                clean_latent_2x_indices,
                clean_latent_4x_indices,
            ) = indices.split(
                [1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1
            )
            clean_latent_indices = torch.cat(
                [clean_latent_indices_pre, clean_latent_indices_post], dim=1
            )

            clean_latents_pre = processed_start_latent.to(history_latents)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[
                :, :, : 1 + 2 + 16, :, :
            ].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

            # vid2vid logic
            input_init_latents = None
            if initial_samples is not None:
                total_length = initial_samples.shape[2]
                max_padding = max(latent_paddings_list) if latent_paddings_list else 0

                if is_last_section:
                    start_idx = max(0, total_length - latent_window_size)
                else:
                    if max_padding > 0:
                        progress = (max_padding - latent_padding) / max_padding
                        start_idx = int(
                            progress * max(0, total_length - latent_window_size)
                        )
                    else:
                        start_idx = 0

                end_idx = min(start_idx + latent_window_size, total_length)
                print(
                    f"Vid2Vid indices: start={start_idx}, end={end_idx}, "
                    f"total_length={total_length}"
                )
                input_init_latents = initial_samples[:, :, start_idx:end_idx, :, :].to(
                    device
                )

            if use_teacache:
                transformer.initialize_teacache(
                    enable_teacache=True,
                    num_steps=steps,
                    rel_l1_thresh=teacache_rel_l1_thresh,
                )
            else:
                transformer.initialize_teacache(enable_teacache=False)

            with torch.autocast(
                device_type=mm.get_autocast_device(device),
                dtype=base_dtype,
                enabled=True,
            ):
                generated_latents = sample_hunyuan(
                    transformer=transformer,
                    sampler=sampler,
                    initial_latent=input_init_latents,  # Pass processed init latents
                    strength=denoise_strength,
                    width=W * 8,
                    height=H * 8,
                    frames=num_frames,
                    real_guidance_scale=cfg,
                    distilled_guidance_scale=guidance_scale,
                    guidance_rescale=0,
                    shift=shift if shift != 0 else None,
                    num_inference_steps=steps,
                    generator=rnd,
                    prompt_embeds=llama_vec,
                    prompt_embeds_mask=llama_attention_mask,
                    prompt_poolers=clip_l_pooler,
                    negative_prompt_embeds=llama_vec_n,
                    negative_prompt_embeds_mask=llama_attention_mask_n,
                    negative_prompt_poolers=clip_l_pooler_n,
                    device=device,
                    dtype=base_dtype,
                    image_embeddings=image_encoder_last_hidden_state,
                    latent_indices=latent_indices,
                    clean_latents=clean_latents,
                    clean_latent_indices=clean_latent_indices,
                    clean_latents_2x=clean_latents_2x,
                    clean_latent_2x_indices=clean_latent_2x_indices,
                    clean_latents_4x=clean_latents_4x,
                    clean_latent_4x_indices=clean_latent_4x_indices,
                    callback=callback,
                )

            if is_last_section:
                # Prepend the original start_latent frame
                generated_latents = torch.cat(
                    [processed_start_latent.to(generated_latents), generated_latents],
                    dim=2,
                )

            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat(
                [generated_latents.to(history_latents), history_latents], dim=2
            )

            # Slice the history to keep only the actually generated frames
            real_history_latents = history_latents[
                :, :, :total_generated_latent_frames, :, :
            ]

            if is_last_section:
                break

        transformer.to(offload_device)
        mm.soft_empty_cache()

        # Ensure real_history_latents is defined before returning
        if "real_history_latents" not in locals():
            raise RuntimeError("Sampling loop finished without generating latents.")

        return ({"samples": real_history_latents / vae_scaling_factor},)


NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadFramePackModel": DownloadAndLoadFramePackModel,
    "FramePackSampler": FramePackSampler,
    "FramePackTorchCompileSettings": FramePackTorchCompileSettings,
    "FramePackFindNearestBucket": FramePackFindNearestBucket,
    "LoadFramePackModel": LoadFramePackModel,
    "LoadFramePackLora": LoadFramePackLora,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadFramePackModel": "(Down)Load FramePackModel",
    "FramePackSampler": "FramePackSampler",
    "FramePackTorchCompileSettings": "Torch Compile Settings",
    "FramePackFindNearestBucket": "Find Nearest Bucket",
    "LoadFramePackModel": "Load FramePackModel",
    "LoadFramePackLora": "Load FramePack LoRA",
}
