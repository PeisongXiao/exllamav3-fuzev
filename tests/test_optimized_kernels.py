"""Small real-CUDA numerical comparisons; no model weights or serving engine."""

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.cuda


@pytest.fixture(autouse=True)
def cuda_only(request):
    if request.config.getoption("--cpu") or not torch.cuda.is_available():
        pytest.skip("requires CUDA")


@pytest.mark.parametrize("budget", [64, 96])
@pytest.mark.parametrize("quantized", [False, True])
def test_sparse_prefill_matches_native_with_model_budget(monkeypatch, budget, quantized):
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.modules.attention_fn.qsa_triton import qsa_sparse_attend_rows
    from exllamav3.optimized import hooks
    from exllamav3.optimized.vendor.kernels import qsa_attend

    torch.manual_seed(37)
    q = torch.randn(256, 4, 64, device="cuda", dtype=torch.float16)
    k, v = torch.randn(2, 256, 2, 64, device="cuda", dtype=torch.float16)
    kp = budget + 32
    indices = torch.arange(kp, device="cuda", dtype=torch.int32).repeat(256, 1)
    indices.masked_fill_(indices > torch.arange(256, device="cuda")[:, None], -1)
    table = torch.zeros(256, 1, device="cuda", dtype=torch.int32)
    qc = None
    if quantized:
        packed = [torch.empty(1, 256, 32, device="cuda", dtype=torch.int32) for _ in range(2)]
        scales = [torch.empty(1, 256, 4, device="cuda", dtype=torch.float16) for _ in range(2)]
        ext.quant_cache_paged(k.unsqueeze(0), packed[0], scales[0],
                              v.unsqueeze(0), packed[1], scales[1],
                              torch.zeros(1, device="cuda", dtype=torch.int32), table[:1],
                              256, 256, 0.0, False)
        k, v = packed
        qc = (*scales, 8, 8)
    monkeypatch.setitem(hooks.ORIGINALS, "qsa_attend", qsa_sparse_attend_rows)
    context = {"budget": budget, "k_pad": kp, "length": 256, "block_table": table[0]}
    token = hooks.STAGE_CONTEXT.set(context)
    try:
        actual = qsa_attend(q, k, v, indices, 0.125, table, 256, qc, 2)
    finally:
        hooks.STAGE_CONTEXT.reset(token)
    expected = qsa_sparse_attend_rows(q, k, v, indices, 0.125, table, 256, qc, 2)
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.003)


@pytest.mark.parametrize("budget", [96, 256])
def test_dense_qsa_prefix_matches_native_sparse_attention(budget):
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.modules.attention_fn.qsa_triton import qsa_sparse_attend_rows
    from exllamav3.optimized import hooks
    from exllamav3.optimized.vendor.qsa_dense import attend

    torch.manual_seed(37)
    rows, heads, kv_heads, width = 256, 4, 2, 64
    q = torch.randn(rows, heads, width, device="cuda", dtype=torch.float16)
    k, v = torch.randn(2, 1, rows, kv_heads, width, device="cuda", dtype=torch.float16)
    packed = [torch.empty(1, rows, kv_heads * width // 4, device="cuda", dtype=torch.int32)
              for _ in range(2)]
    scales = [torch.empty(1, rows, kv_heads * width // 32, device="cuda", dtype=torch.float16)
              for _ in range(2)]
    table = torch.zeros(rows, 1, device="cuda", dtype=torch.int32)
    ext.quant_cache_paged(k, packed[0], scales[0], v, packed[1], scales[1],
                          torch.zeros(1, device="cuda", dtype=torch.int32), table[:1],
                          rows, rows, 0.0, False)
    kp = budget + 32
    indices = torch.arange(kp, device="cuda", dtype=torch.int32).repeat(rows, 1)
    indices.masked_fill_(indices > torch.arange(rows, device="cuda")[:, None], -1)
    qc = (*scales, 8, 8)
    token = hooks.STAGE_CONTEXT.set({"length": rows, "budget": budget, "k_pad": kp})
    try:
        actual = attend(q, *packed, indices, width ** -0.5, table, rows, qc, kv_heads,
                        fallback=qsa_sparse_attend_rows)
    finally:
        hooks.STAGE_CONTEXT.reset(token)
    expected = qsa_sparse_attend_rows(q, *packed, indices, width ** -0.5, table, rows, qc, kv_heads)
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.003)


def test_gather_fused_moe_matches_native_group():
    from exllamav3.modules.moe_batch_recon import BatchReconLayer
    from exllamav3.optimized.vendor.moe_group import run_group

    torch.manual_seed(37)
    experts, rows, width, bits = 3, 256, 128, 3
    trellises = {p: [torch.randint(0, 65536, (width // 16, width // 16, 16 * bits),
                                   device="cuda", dtype=torch.int32).short() for _ in range(experts)]
                for p in ("g", "u", "d")}
    scales = {p: (torch.ones(experts, width, device="cuda", dtype=torch.float16),
                  torch.full((experts, width), 0.125, device="cuda", dtype=torch.float16))
              for p in trellises}
    dims = (width, width, bits)
    layer = BatchReconLayer(dims, dims, dims, (False, False), (False, False), (False, False),
                           "silu", 0, "cuda", scales, folded=True)
    layer.set_static_pointers(*(torch.tensor([t.data_ptr() for t in trellises[p]],
                                            device="cuda", dtype=torch.int64)
                                for p in ("g", "u", "d")))
    ids, starts, counts = [2, 0], [0, 129], [129, 97]
    x = torch.randn(rows + 1, width, device="cuda", dtype=torch.float16) * 0.1
    x[-1].zero_()
    tokens = torch.cat((torch.randperm(rows, device="cuda")[:sum(counts)],
                        torch.tensor([rows], device="cuda")))
    weights = torch.ones_like(tokens, dtype=torch.float16)
    weights[-1] = 0
    output = torch.zeros(rows + 1, width, device="cuda", dtype=torch.float32)
    expected = torch.empty(len(ids) * max(counts), width, device="cuda", dtype=torch.float32)
    actual = torch.empty_like(expected)
    layer.run_group(x, output, tokens, weights, ids, starts, counts, out_slab=expected)
    run_group(layer, x, output, tokens, weights, ids, starts, counts,
              out_slab=actual, fallback=BatchReconLayer.run_group)
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.003)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gdn_native_dtype_matches_native_scan(monkeypatch, dtype):
    pytest.importorskip("flashinfer.gdn_prefill")
    from exllamav3.vendor.fla import chunk_gated_delta_rule
    from exllamav3.modules.gated_delta_net_fn.gated_delta_rule import _chunked_scan
    from exllamav3.optimized import hooks
    from exllamav3.optimized.vendor.gdn_scan import flashinfer_scan

    torch.manual_seed(37)
    q, k, v = torch.randn(3, 1, 256, 2, 128, device="cuda", dtype=dtype)
    g = -torch.rand(1, 256, 2, device="cuda")
    beta = torch.rand_like(g)
    monkeypatch.setitem(hooks.ORIGINALS, "gdn_scan", _chunked_scan)
    expected, expected_state = _chunked_scan(chunk_gated_delta_rule, q, k, v, g, beta, None, True)
    actual, state = flashinfer_scan(chunk_gated_delta_rule, q, k, v, g, beta, None, True)
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.005)
    torch.testing.assert_close(state, expected_state, rtol=0.03, atol=0.005)


def test_bmm_matches_native():
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.optimized.vendor.bmm_kernels import gemm

    torch.manual_seed(37)
    x = torch.randn(2, 256, 64, device="cuda", dtype=torch.float16)
    w = torch.randn(2, 64, 128, device="cuda", dtype=torch.float16)
    expected = torch.empty(2, 256, 128, device="cuda", dtype=torch.float16)
    actual = torch.empty_like(expected)
    ext.hgemm_batched(x, w, expected)
    gemm(x, w, actual)
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.02)


def test_coalesced_checkpoint_roundtrip(monkeypatch):
    from exllamav3.modules.gated_delta_net import GDNLayerState, GDNState
    from exllamav3.optimized.vendor import checkpoint_transfer as transfer

    module = SimpleNamespace(fdim_qkv=8, conv_kernel_size=4,
                             num_v_heads=2, k_head_dim=4, v_head_dim=4)
    layer = GDNLayerState(module, max_batch_size=2, max_history=2, cache_id=0)
    layer.alloc("cuda")
    layer.conv_state.normal_()
    layer.recurrent_state.normal_()
    state = GDNState.__new__(GDNState)
    state.slot, state.position, state.checkpoint_size = 1, 256, layer.get_checkpoint_size()
    state.cache = SimpleNamespace(model=SimpleNamespace(loaded_tp=False),
                                 get_all_recurrent_layers=lambda: {"layer": layer})
    expected = state.stash()
    for target, name, replacement in transfer.patches():
        monkeypatch.setattr(target, name, replacement)
    saved = state.stash()
    layer.clear(1)
    state.unstash(saved)
    restored = layer.stash(1)
    for actual, original in zip(restored, expected["layer"]):
        assert torch.equal(actual, original)
