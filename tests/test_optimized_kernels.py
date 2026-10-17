"""Small real-CUDA numerical comparisons; no model weights or serving engine."""

from types import SimpleNamespace

import pytest
import torch

@pytest.mark.parametrize("budget", [64, 96])
@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("rows", [1, 256])
def test_sparse_prefill_matches_native_with_model_budget(monkeypatch, budget, quantized, rows):
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.modules.attention_fn.qsa_triton import qsa_sparse_attend_rows
    from exllamav3.optimized import settings
    from exllamav3.modules.attention_fn.qsa_prefill import qsa_attend

    torch.manual_seed(37)
    q = torch.randn(rows, 4, 64, device="cuda", dtype=torch.float16)
    k, v = torch.randn(2, 256, 2, 64, device="cuda", dtype=torch.float16)
    kp = budget + 32
    indices = torch.arange(kp, device="cuda", dtype=torch.int32).repeat(rows, 1)
    indices.masked_fill_(indices > torch.arange(256 - rows, 256, device="cuda")[:, None], -1)
    table = torch.zeros(rows, 1, device="cuda", dtype=torch.int32)
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
    monkeypatch.setitem(settings.CONFIG, "prefill_min_rows", 1)
    context = {"budget": budget, "k_pad": kp, "length": 256, "block_table": table[0]}
    actual = qsa_attend(q, k, v, indices, 0.125, table, 256, qc, 2, context=context)
    expected = qsa_sparse_attend_rows(q, k, v, indices, 0.125, table, 256, qc, 2)
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.003)


@pytest.mark.parametrize("budget", [96, 256])
def test_dense_qsa_prefix_matches_native_sparse_attention(budget):
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.modules.attention_fn.qsa_triton import qsa_sparse_attend_rows
    from exllamav3.modules.attention_fn.qsa_prefill import attend

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
    actual = attend(q, *packed, indices, width ** -0.5, table, rows, qc, kv_heads,
                    context={"length": rows, "budget": budget, "k_pad": kp, "max_length": rows})
    expected = qsa_sparse_attend_rows(q, *packed, indices, width ** -0.5, table, rows, qc, kv_heads)
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.003)


@pytest.mark.parametrize("slot_mode", [False, True])
@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_gather_fused_moe_matches_native_group(monkeypatch, slot_mode, streamed, failure):
    from exllamav3.modules.moe_batch_recon import BatchReconLayer
    from exllamav3.modules import moe_batch_recon_triton as kernels

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
    weights = torch.linspace(0.1, 1.0, tokens.numel(), device="cuda", dtype=torch.float16)
    weights[-1] = 0
    expected_sum = torch.zeros(rows + 1, width, device="cuda", dtype=torch.float32)
    actual_sum = torch.zeros_like(expected_sum)
    expected = torch.empty(len(ids) * max(counts), width, device="cuda", dtype=torch.float32) if slot_mode else None
    actual = torch.empty_like(expected) if slot_mode else None
    ptrs = tuple([trellises[p][i].data_ptr() for i in ids] for p in ("g", "u", "d")) if streamed else None
    layer.run_group(x, expected_sum, tokens, weights, ids, starts, counts, ptrs=ptrs, out_slab=expected)
    layer.prefill_optimized = True
    if failure:
        class Unavailable:
            def __getitem__(self, grid):
                def launch(*args, **kwargs):
                    raise kernels.OutOfResources(2, 1, "shared memory")
                return launch
        monkeypatch.setattr(kernels, "_unsupported_devices", set())
        monkeypatch.setattr(kernels, "_gate_up", Unavailable())
        monkeypatch.setattr(kernels, "_mm", Unavailable())
    layer.run_group(x, actual_sum, tokens, weights, ids, starts, counts, ptrs=ptrs, out_slab=actual)
    torch.testing.assert_close(actual if slot_mode else actual_sum,
                               expected if slot_mode else expected_sum, rtol=0.01, atol=0.003)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("warm_state", [False, True])
def test_gdn_native_dtype_matches_native_scan(monkeypatch, dtype, warm_state):
    pytest.importorskip("flashinfer.gdn_prefill")
    from exllamav3.vendor.fla import chunk_gated_delta_rule
    from exllamav3.modules.gated_delta_net_fn.gated_delta_rule import _chunked_scan, _flashinfer_scan

    torch.manual_seed(37)
    q, k = torch.randn(2, 1, 256, 2, 128, device="cuda", dtype=dtype)
    v = torch.randn(1, 256, 2, 128, device="cuda", dtype=dtype)
    g = -torch.rand(1, 256, 2, device="cuda")
    beta = torch.rand_like(g)
    initial_state = torch.randn(1, 2, 128, 128, device="cuda") if warm_state else None
    expected, expected_state = _chunked_scan(chunk_gated_delta_rule, q, k, v, g, beta, initial_state, True)
    result = _flashinfer_scan(q, k, v, g, beta, initial_state, True)
    if result is None:
        pytest.skip("FlashInfer GDN is unavailable on this device")
    actual, state = result
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.005)
    torch.testing.assert_close(state, expected_state, rtol=0.03, atol=0.005)


def test_bmm_matches_native():
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.modules.moe_batch_recon_triton import gemm
    from exllamav3.optimized import settings

    torch.manual_seed(37)
    x = torch.randn(2, 256, 64, device="cuda", dtype=torch.float16)
    w = torch.randn(2, 64, 128, device="cuda", dtype=torch.float16)
    expected = torch.empty(2, 256, 128, device="cuda", dtype=torch.float16)
    actual = torch.empty_like(expected)
    ext.hgemm_batched(x, w, expected)
    gemm(x, w, actual, tile=settings.CONFIG["bmm_tile"])
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.02)


@pytest.fixture
def checkpoint_state(monkeypatch):
    from exllamav3.cache import recurrent
    from exllamav3.modules.gated_delta_net import GDNLayerState, GDNState
    from exllamav3.modules.ple import PLELayerState

    pool = recurrent.HostPool()
    monkeypatch.setattr(recurrent, "host_pool", pool)

    def make(parity):
        module = SimpleNamespace(fdim_qkv=8, conv_kernel_size=4,
                                 num_v_heads=2, k_head_dim=4, v_head_dim=4)
        layer = GDNLayerState(module, max_batch_size=2, max_history=2, cache_id=0)
        layer.alloc("cuda")
        layer.conv_state.normal_()
        layer.recurrent_state.normal_()
        ple_module = SimpleNamespace(conv_state_len=3, hc_mult=2, hidden_size=4,
                                     ple_embedding=SimpleNamespace(context_len=2, eos_token_id=0))
        ple = PLELayerState(ple_module, max_batch_size=2, max_history=2, cache_id=0)
        ple.alloc("cuda")
        ple.conv_state.normal_()
        ple.id_state.copy_(torch.arange(ple.id_state.numel()).reshape_as(ple.id_state))
        layers = {"layer": layer, "ple": ple}
        from exllamav3.model.config import InferParams
        config = SimpleNamespace(infer_params=InferParams())
        cache = SimpleNamespace(model=SimpleNamespace(loaded_tp=False, config=config),
                                get_all_recurrent_layers=lambda: layers)
        state = GDNState(cache, slot=1, position=256, clear=False, test_state=True)
        state.parity = parity
        return state

    return make


def checkpoint_values(state):
    from exllamav3.modules.gated_delta_net import GDNLayerState
    return {key: item.stash(state.slot, state.position, state.parity)
            if isinstance(item, GDNLayerState) else item.stash(state.slot, state.position)
            for key, item in state.cache.get_all_recurrent_layers().items()}


def assert_checkpoint_equal(actual, expected):
    for key, originals in expected.items():
        for value, original in zip(actual[key], originals):
            assert torch.equal(value.contiguous().view(torch.uint8),
                               original.contiguous().view(torch.uint8))


@pytest.mark.parametrize("parity", [0, 1])
@pytest.mark.parametrize("cpu_restore", [False, True])
def test_coalesced_checkpoint_roundtrip(monkeypatch, checkpoint_state, parity, cpu_restore):
    state = checkpoint_state(parity)
    layers = state.cache.get_all_recurrent_layers()
    expected_before = checkpoint_values(state)
    saved_before = state.stash()
    for item in layers.values():
        item.clear(state.slot)
    expected_after = checkpoint_values(state)
    saved_after = state.stash()
    state.parity = 1 - parity
    if cpu_restore:
        for item in layers.values():
            item.alloc("cpu")
    for saved, expected in ((saved_before, expected_before), (saved_after, expected_after)):
        state.unstash(saved)
        assert_checkpoint_equal(checkpoint_values(state), expected)


def test_paired_coalesced_checkpoint_pool_ownership(monkeypatch, checkpoint_state):
    from exllamav3.cache import recurrent
    from exllamav3.cache.recurrent import PairedState

    pair = PairedState(checkpoint_state(0), checkpoint_state(1))
    first = pair.stash()
    for index, component in enumerate(pair.components):
        for layer in component.cache.get_all_recurrent_layers().values():
            layer.conv_state.fill_(index + 1)
        layers = component.cache.get_all_recurrent_layers()
        layers["layer"].recurrent_state.fill_(index + 2)
        layers["ple"].id_state.fill_(index + 3)
    retained = pair.stash()
    recurrent.host_pool.give(first)
    # Native checkpoints must not borrow sub-views of a slab the pool can reuse independently.
    for component in pair.components:
        component.cache.model.config.infer_params.prefill_optimizations = False
    held = [component.stash() for component in pair.components]
    for component in pair.components:
        component.cache.model.config.infer_params.prefill_optimizations = True
    expected = [checkpoint_values(component) for component in pair.components]
    for index, component in enumerate(pair.components):
        for layer in component.cache.get_all_recurrent_layers().values():
            layer.conv_state.fill_(index + 5)
        layers = component.cache.get_all_recurrent_layers()
        layers["layer"].recurrent_state.fill_(index + 6)
        layers["ple"].id_state.fill_(index + 7)
    recycled = pair.stash()
    for component, saved, golden in zip(pair.components, held, expected):
        component.unstash(saved)
        assert_checkpoint_equal(checkpoint_values(component), golden)
    for name, component in (("target", pair.target), ("draft", pair.draft)):
        component.unstash(retained[name])
    for component, golden in zip(pair.components, expected):
        assert_checkpoint_equal(checkpoint_values(component), golden)
    recurrent.host_pool.give(recycled)
