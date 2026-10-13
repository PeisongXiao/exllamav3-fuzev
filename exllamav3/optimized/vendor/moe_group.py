"""Prefill-only routed expert groups, retaining original reconstruction and slots."""
# Derived from ExLlamaV3 by turboderp (MIT). Modifications Copyright 2026 Peisong Xiao.

import numpy as np
import torch
from .. import hooks
from exllamav3.ext import exllamav3_ext as ext
from .moe_fusion import gate_up


def run_group(layer, y_ext, out_ext, tok_ext, w_ext, ids, starts, counts,
              ptrs=None, out_slab=None, *, fallback):
    supported = (y_ext.shape[0] >= hooks.CONFIG["prefill_min_rows"] and layer.folded
                 and layer.gated and not layer.interm_fp32 and layer.act_limit == 0
                 and layer.act_call == ext.silu_mul and ptrs is None and out_slab is not None
                 and layer.dims_g[:2] == layer.dims_u[:2])
    if not supported:
        return fallback(layer, y_ext, out_ext, tok_ext, w_ext, ids, starts, counts,
                               ptrs=ptrs, out_slab=out_slab)
    b, m = len(counts), max(counts)
    k, n, ku = layer.dims_u
    kd, nd, kdown = layer.dims_d
    if b > layer.cap or out_slab.shape != (b * m, nd) or kd != n:
        return fallback(layer, y_ext, out_ext, tok_ext, w_ext, ids, starts, counts,
                               ptrs=ptrs, out_slab=out_slab)
    meta = torch.from_numpy(np.array([ids, starts, counts], dtype=np.int64)).to(
        y_ext.device, non_blocking=True)
    ids_d = meta[0]
    pg, pu, pd = (table.index_select(0, ids_d) for table in layer.ptr_tables)
    ar = torch.arange(m, device=y_ext.device)
    source = torch.where(ar[None, :] < meta[2, :, None],
                         meta[1, :, None] + ar[None, :], tok_ext.shape[0] - 1)
    tokens = tok_ext.index_select(0, source.reshape(-1))
    # Same peak weight scratch as upstream: two half projections are live.
    scratch = torch.empty(b * max(k * n, kd * nd), dtype=torch.float16, device=y_ext.device)
    wu = torch.empty((b, k, n), dtype=torch.float16, device=y_ext.device)
    wg = scratch[:b * k * n].view(b, k, n)
    layer._recon(wu, pu, "u", ids_d, ku, layer.cb_u)
    layer._recon(wg, pg, "g", ids_d, layer.dims_g[2], layer.cb_g)
    u = torch.empty((b, m, n), dtype=torch.float16, device=y_ext.device)
    gate_up(y_ext, tokens, wg, wu, u)
    del wg, wu
    wd = scratch[:b * kd * nd].view(b, kd, nd)
    layer._recon(wd, pd, "d", ids_d, kdown, layer.cb_d)
    layer._linear(u, wd, "d", ids_d, nd, out_dtype=torch.float32,
                  out=out_slab.view(b, m, nd))
