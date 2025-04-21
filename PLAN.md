# ComfyUI FramePackWrapper LoRA Integration Plan

## Goal

Enable the use of LoRA weights with the FramePack model within ComfyUI.

## Approach

Create a new custom node, `LoadFramePackLora`, to handle LoRA loading and application. This minimizes modifications to existing nodes.

## New Node: `LoadFramePackLora`

*   **Category:** `FramePackWrapper`
*   **Display Name:** `Load FramePack LoRA`
*   **Inputs:**
    *   `model`: `FramePackMODEL` (Input model before LoRA application)
    *   `lora_name`: `STRING` (LoRA filename from the ComfyUI `loras` folder)
    *   `strength`: `FLOAT` (Strength of the LoRA application, default 1.0)
*   **Outputs:**
    *   `model`: `FramePackMODEL` (Output model after LoRA application)
*   **Functionality:**
    1.  Get the full path to the selected LoRA file using `folder_paths.get_full_path("loras", lora_name)`.
    2.  Retrieve the `transformer` object from the input `model`.
    3.  Load the LoRA `state_dict` using `diffusers_helper.load_lora._fetch_state_dict`.
    4.  (Apply `_convert_hunyuan_video_lora_to_diffusers` if necessary, based on `diffusers_helper/load_lora.py` logic).
    5.  Scale the values in the loaded `state_dict` by the `strength` factor.
    6.  Apply the scaled `state_dict` to the model using `transformer.load_lora_adapter(scaled_state_dict, network_alphas=None)`.
    7.  Create and return a new `model` dictionary containing the modified `transformer`.

## Workflow Integration (Mermaid)

```mermaid
graph TD
    A[LoadFramePackModel] -->|FramePackMODEL| C(LoadFramePackLora);
    B[LoRA File Selector] -->|lora_name| C;
    D[Strength Slider] -->|strength| C;
    C -->|FramePackMODEL (LoRA applied)| E[FramePackSampler];
    F[Positive Prompt] --> E;
    G[Negative Prompt] --> E;
    H[Image Embeds] --> E;
    I[...] --> E;
    E -->|LATENT| J[VAEDecode or other nodes];
```

## Implementation Steps

1.  Add the `LoadFramePackLora` class definition to `nodes.py`.
2.  Update `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` in `nodes.py` to include the new node.