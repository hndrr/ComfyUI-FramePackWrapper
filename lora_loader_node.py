from typing import Dict, Any, Tuple
import folder_paths
from .diffusers_helper.load_lora import (
    _fetch_state_dict,
    _convert_hunyuan_video_lora_to_diffusers,
)


class LoadFramePackLora:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "model": ("FramePackMODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
                ),
                "diffuser_lora": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Is this a diffuser format LoRA?",
                    },
                ),
            }
        }

    RETURN_TYPES = ("FramePackMODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_lora"
    CATEGORY = "FramePackWrapper"

    def load_lora(
        self,
        model: Dict[str, Any],
        lora_name: str,
        strength: float,
        diffuser_lora: bool,
    ) -> Tuple[Dict[str, Any]]:
        if strength == 0:
            return (model,)

        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not lora_path:
            print(f"LoRA path not found: {lora_name}")
            return (model,)

        transformer = model["transformer"]
        dtype = model["dtype"]  # Get dtype from the original model

        try:
            # Fetch the state dict
            # weight_name is handled inside _fetch_state_dict if None
            # Assuming safetensors is preferred, let _fetch_state_dict handle
            # logic. Other params default to None or False as in original
            # load_lora
            state_dict = _fetch_state_dict(
                lora_path,
                None,  # weight_name
                lora_path.endswith(".safetensors"),  # is_safetensors
                False,  # use_remote_ckpt
                None,  # variant
                None,  # image_size
                None,  # cache_dir
                True,  # local_files_only
                None,  # token
                None,  # revision
                None,  # subfolder
                None,  # mirror
            )

            # Convert if necessary (assuming Hunyuan if not diffuser format)
            if not diffuser_lora:
                print("Not a diffusers lora, assuming Hunyuan.")
                state_dict = _convert_hunyuan_video_lora_to_diffusers(state_dict)

            # Scale the state dict
            scaled_state_dict = {}
            for k, v in state_dict.items():
                # Ensure the tensor is loaded to the correct device and
                # dtype before scaling. Use transformer's device and
                # model's dtype.
                scaled_state_dict[k] = (
                    v.to(device=transformer.device, dtype=dtype) * strength
                )

            # Load the scaled state dict
            # network_alphas=None is consistent with diffusers_helper/load_lora.py
            transformer.load_lora_adapter(scaled_state_dict, network_alphas=None)
            print(
                f"LoRA '{lora_name}' loaded successfully " f"with strength {strength}."
            )

            # --- Ensure LoRA parameters stay on the correct device ---
            try:
                import peft
                import comfy.model_management as mm
                lora_device = mm.get_torch_device() # Explicitly set target device to GPU
                count = 0
                for name, module in transformer.named_modules():
                    if isinstance(module, peft.tuners.lora.layer.Linear) or \
                       isinstance(module, peft.tuners.lora.layer.Conv2d): # Add other LoRA layer types if needed
                        if hasattr(module, 'lora_A'):
                            for layer in module.lora_A.values():
                                if layer.weight.device != lora_device:
                                    layer.to(lora_device)
                                    count += 1
                        if hasattr(module, 'lora_B'):
                            for layer in module.lora_B.values():
                                if layer.weight.device != lora_device:
                                    layer.to(lora_device)
                                    count += 1
                if count > 0:
                    print(f"Moved {count} LoRA parameter tensors back to device: {lora_device}")
            except ImportError:
                print("PEFT library not found, skipping LoRA device check.")
            except Exception as peft_e:
                print(f"Error ensuring LoRA parameters device: {peft_e}")
            # --- End LoRA device check ---

        except Exception as e:
            print(f"Error loading LoRA {lora_name}: {e}")
            # Return the original model if loading fails
            return (model,)

        # Return the modified model dictionary
        new_model = model.copy()
        new_model["transformer"] = transformer
        return (new_model,)


# This dictionary should be imported and merged in nodes.py
NODE_CLASS_MAPPINGS = {"LoadFramePackLora": LoadFramePackLora}

NODE_DISPLAY_NAME_MAPPINGS = {"LoadFramePackLora": "Load FramePack LoRA"}
