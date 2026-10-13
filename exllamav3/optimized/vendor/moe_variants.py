"""Folded reconstruction for hot prefill experts."""
# Copyright 2026 Peisong Xiao (MIT).

from .. import hooks


def folded_hot(module, x, params, out_dtype, original):
    if module.cpu_offload or module.cpu_split_first is not None or module.latent_in is not None:
        return original(module, x, params, out_dtype)
    recon = module._batch_recon_layer(x.reshape(-1, module.hidden_size))
    if recon is None or not recon.folded:
        return original(module, x, params, out_dtype)
    cap, max_rows = recon.cap, recon.max_rows
    try:
        recon.max_rows = hooks.CONFIG["moe_max_rows"]
        if module.fused_mode_buffers is not None:
            recon.cap = min(cap, hooks.CONFIG["moe_batch_cap"])
        return original(module, x, params, out_dtype)
    finally:
        recon.cap, recon.max_rows = cap, max_rows
