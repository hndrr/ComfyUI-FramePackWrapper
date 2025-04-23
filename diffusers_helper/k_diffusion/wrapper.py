import torch


def append_dims(x, target_dims):
    return x[(...,) + (None,) * (target_dims - x.ndim)]


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=1.0):
    if guidance_rescale == 0:
        return noise_cfg

    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1.0 - guidance_rescale) * noise_cfg
    return noise_cfg


def fm_wrapper(transformer, t_scale=1000.0):
    def k_model(x, sigma, **extra_args):
        dtype = extra_args['dtype']
        cfg_scale = extra_args['cfg_scale']
        cfg_rescale = extra_args['cfg_rescale']
        concat_latent = extra_args['concat_latent']

        original_dtype = x.dtype
        sigma = sigma.float()

        # --- Ensure tensors are on the correct device before transformer call ---
        transformer_device = transformer.device # Get the device the transformer is on

        x = x.to(device=transformer_device, dtype=dtype)
        timestep = (sigma * t_scale).to(device=transformer_device, dtype=dtype)

        if concat_latent is None:
            hidden_states = x
        else:
            # Ensure concat_latent is also on the correct device
            concat_latent = concat_latent.to(device=transformer_device, dtype=x.dtype)
            hidden_states = torch.cat([x, concat_latent], dim=1)

        # Explicitly move required conditioning tensors to the transformer's device
        positive_cond = extra_args['positive'].copy() # Create a copy to avoid modifying original dict
        negative_cond = extra_args['negative'].copy() # Create a copy to avoid modifying original dict
        # Define keys that are expected to hold tensors for conditioning
        tensor_keys = ['prompt_embeds', 'prompt_embeds_mask', 'prompt_poolers', 'image_embeddings']

        for key in tensor_keys:
            if key in positive_cond and isinstance(positive_cond[key], torch.Tensor):
                # Ensure dtype consistency, consider if specific dtypes are needed per key
                positive_cond[key] = positive_cond[key].to(device=transformer_device, dtype=dtype)
            if key in negative_cond and isinstance(negative_cond[key], torch.Tensor):
                # Assuming negative conditions use the same dtype for simplicity
                negative_cond[key] = negative_cond[key].to(device=transformer_device, dtype=dtype)

        # Ensure hidden_states and timestep are on the correct device right before the call
        hidden_states = hidden_states.to(device=transformer_device)
        timestep = timestep.to(device=transformer_device)
        # --- End device check ---

        pred_positive = transformer(hidden_states=hidden_states, timestep=timestep, return_dict=False, **positive_cond)[0].float()

        if cfg_scale == 1.0:
            pred_negative = torch.zeros_like(pred_positive)
        else:
            # Ensure hidden_states and timestep are on the correct device right before the call
            hidden_states = hidden_states.to(device=transformer_device)
            timestep = timestep.to(device=transformer_device)
            # --- End device check ---
            # Note: negative_cond dictionary was already prepared above
            pred_negative = transformer(hidden_states=hidden_states, timestep=timestep, return_dict=False, **negative_cond)[0].float()

        pred_cfg = pred_negative + cfg_scale * (pred_positive - pred_negative)
        pred = rescale_noise_cfg(pred_cfg, pred_positive, guidance_rescale=cfg_rescale)

        x0 = x.float() - pred.float() * append_dims(sigma, x.ndim)

        return x0.to(dtype=original_dtype)

    return k_model
