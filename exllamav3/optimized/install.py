"""Prefill kernels using the model's existing cache and scheduler."""

from contextvars import ContextVar
import functools
import logging
import re
from threading import RLock
import torch

from . import hooks

INSTALLED = False
LOCK = RLock()
GDN_LAYER = ContextVar("exllamav3_optimized_gdn_layer", default=False)


def install():
    global INSTALLED
    with LOCK:
        if INSTALLED:
            return
        patches, originals = _prepare()
        hooks.ORIGINALS.update(originals)
        for target, name, replacement in patches:
            setattr(target, name, replacement)
        INSTALLED = True
        logging.getLogger(__name__).info("Prefill optimizations enabled")


def _prepare():
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.modules.attention_fn import qsa_triton
    from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
    from exllamav3.modules.moe_batch_recon import BatchReconLayer
    from exllamav3.modules.gated_delta_net import GatedDeltaNet
    from exllamav3.modules.gated_delta_net_fn import gated_delta_rule as gdn
    from exllamav3.modules.qsa_indexer import QSAIndexer
    from .vendor.bmm_kernels import gemm
    from .vendor.gdn_scan import flashinfer_scan
    from .vendor.kernels import qsa_attend
    from .vendor.moe_group import run_group
    from .vendor.moe_variants import folded_hot
    from .vendor.qsa_dense import attend as dense_prefix
    from .vendor import checkpoint_transfer

    originals = {"gdn_scan": gdn._chunked_scan, "qsa_attend": qsa_triton.qsa_sparse_attend_rows}
    patches = checkpoint_transfer.patches()
    try:
        import flashinfer.gdn_prefill
    except ImportError:
        logging.getLogger(__name__).warning(
            "FlashInfer GDN is unavailable; using the native gated-delta scan")
    else:
        original_gdn_init, original_gdn_forward = GatedDeltaNet.__init__, GatedDeltaNet.forward

        @functools.wraps(original_gdn_init)
        def gdn_init(self, *args, **kwargs):
            original_gdn_init(self, *args, **kwargs)
            match = re.search(r"(?:^|\.)layers\.(\d+)\.", self.key)
            self._optimized_gdn_eligible = (
                match is not None and int(match[1]) not in hooks.CONFIG["gdn_native_layers"])

        @functools.wraps(original_gdn_forward)
        def gdn_forward(self, x, *args, **kwargs):
            if (not getattr(self, "_optimized_gdn_eligible", False)
                    or x.shape[1] < hooks.CONFIG["prefill_min_rows"]):
                return original_gdn_forward(self, x, *args, **kwargs)
            token = GDN_LAYER.set(True)
            try:
                return original_gdn_forward(self, x, *args, **kwargs)
            finally:
                GDN_LAYER.reset(token)

        def selective_scan(*args, **kwargs):
            function = flashinfer_scan if GDN_LAYER.get() else originals["gdn_scan"]
            return function(*args, **kwargs)

        patches += [(GatedDeltaNet, "__init__", gdn_init), (GatedDeltaNet, "forward", gdn_forward),
                    (gdn, "_chunked_scan", selective_scan)]

    original_sparse = QSAIndexer.sparse_attend

    @functools.wraps(original_sparse)
    def sparse(self, layer, attn, q, q_idx, block_table, cache_seqlens_cpu):
        context = None
        if q.shape[0] == 1 and q.shape[1] >= hooks.CONFIG["prefill_min_rows"]:
            context = {"block_table": block_table[0],
                       "length": int(cache_seqlens_cpu[0]) + q.shape[1],
                       "budget": self.token_budget, "k_pad": self.k_pad(),
                       "max_length": self.config.max_position_embeddings}
        token = hooks.STAGE_CONTEXT.set(context)
        try:
            return original_sparse(self, layer, attn, q, q_idx, block_table, cache_seqlens_cpu)
        finally:
            hooks.STAGE_CONTEXT.reset(token)

    def attend(*args, **kwargs):
        context = hooks.STAGE_CONTEXT.get()
        if context is not None and context["length"] > context["max_length"]:
            return originals["qsa_attend"](*args, **kwargs)
        return dense_prefix(*args, **kwargs, fallback=qsa_attend)

    original_group, original_bmm = BatchReconLayer.run_group, ext.hgemm_batched

    @functools.wraps(original_group)
    def group(self, *args, **kwargs):
        return run_group(self, *args, **kwargs, fallback=original_group)

    original_moe = BlockSparseMLP.forward

    @functools.wraps(original_moe)
    def moe(self, x, params, out_dtype=None):
        if (hooks.CONFIG["moe_impl"] != "folded-hot"
                or x.numel() // self.hidden_size < hooks.CONFIG["prefill_min_rows"]):
            return original_moe(self, x, params, out_dtype)
        return folded_hot(self, x, params, out_dtype, original_moe)

    def bmm(x, weights, out):
        if not (x.is_cuda and x.dtype == weights.dtype == torch.float16
                and x.is_contiguous() and weights.is_contiguous() and out.is_contiguous()):
            return original_bmm(x, weights, out)
        return gemm(x, weights, out, tile=hooks.CONFIG["bmm_tile"])

    patches += [(QSAIndexer, "sparse_attend", sparse), (qsa_triton, "qsa_sparse_attend_rows", attend),
                (BatchReconLayer, "run_group", group), (BlockSparseMLP, "forward", moe),
                (ext, "hgemm_batched", bmm)]
    return patches, originals
