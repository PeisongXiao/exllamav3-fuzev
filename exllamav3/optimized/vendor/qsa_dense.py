"""Use native dense prefill for the still-dense prefix of a sparse QSA chunk."""
# Copyright 2026 Peisong Xiao (MIT).

import torch
from .. import hooks

ZEROS = {}


def attend(q, k, v, indices, sm_scale, block_table=None, page_size=0,
           qc=None, n_kv_heads=None, *, fallback):
    context = hooks.STAGE_CONTEXT.get()
    rows = q.shape[0]
    if (context is None or qc is None or rows < hooks.CONFIG["prefill_min_rows"]
            or context["length"] != rows or indices.shape[1] != context["k_pad"]):
        return fallback(q, k, v, indices, sm_scale, block_table, page_size, qc, n_kv_heads)
    from exllamav3.modules.attention_fn import triton_paged as paged
    if q.device not in ZEROS:
        ZEROS[q.device] = torch.zeros((1,), dtype=torch.int32, device=q.device)
    count = min(context["budget"], rows)
    # Stay in the rotated quantized-cache domain without changing other calls.
    prefix = paged.paged_attn_triton_prefill(
        q[:count].unsqueeze(0), None, None, k, v, block_table[:1], ZEROS[q.device],
        causal=True, softmax_scale=sm_scale, qc=qc, pre_appended_len=count,
        max_kv_len=0, n_kv_heads_override=n_kv_heads, qc_staging=False)
    if rows == count:
        return prefix.squeeze(0)
    rest = fallback(q[count:], k, v, indices[count:], sm_scale, block_table[count:],
                    page_size, qc, n_kv_heads)
    return torch.cat((prefix.squeeze(0), rest), dim=0)
