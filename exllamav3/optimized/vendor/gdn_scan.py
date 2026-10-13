"""FlashInfer adapter for native-dtype gated-delta prefill."""
# Copyright 2026 Peisong Xiao (MIT).

import logging
import torch
from .. import hooks

UNSUPPORTED_DEVICES = set()


def flashinfer_scan(fn, q, k, v, g, beta, state, save_state):
    fallback = hooks.ORIGINALS["gdn_scan"]
    if (fn.__name__ != "chunk_gated_delta_rule" or q.shape[0] != 1 or g.ndim != 3
            or q.device in UNSUPPORTED_DEVICES):
        return fallback(fn, q, k, v, g, beta, state, save_state)
    from exllamav3.vendor.fla.l2norm import l2norm_fwd
    from flashinfer.gdn_prefill import chunk_gated_delta_rule

    initial_state = state
    if initial_state is None:
        initial_state = torch.zeros((1, v.shape[2], q.shape[-1], v.shape[-1]),
                                    device=q.device, dtype=torch.float32)
    nq = l2norm_fwd(q.contiguous())[0].squeeze(0).contiguous()
    nk = l2norm_fwd(k.contiguous())[0].squeeze(0).contiguous()
    try:
        result = chunk_gated_delta_rule(
            q=nq, k=nk, v=v.squeeze(0).contiguous(),
            g=g.squeeze(0).float().exp().contiguous(),
            beta=beta.squeeze(0).float().contiguous(),
            initial_state=initial_state.transpose(-1, -2).contiguous().float(),
            output_final_state=save_state,
            cu_seqlens=torch.arange(2, device=q.device, dtype=torch.int64) * q.shape[1])
    except (NotImplementedError, AssertionError):
        UNSUPPORTED_DEVICES.add(q.device)
        logging.getLogger(__name__).warning(
            "FlashInfer GDN is unsupported on %s; using the native gated-delta scan", q.device)
        return fallback(fn, q, k, v, g, beta, state, save_state)
    if save_state:
        output, final_state = result
        return output.to(q.dtype).unsqueeze(0), final_state.transpose(-1, -2).contiguous()
    return result.to(q.dtype).unsqueeze(0), None
