"""Prefill optimization activation and native fallbacks, without loading weights or CUDA kernels."""

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from exllamav3.optimized import enable_prefill_optimizations, hooks

installer = importlib.import_module("exllamav3.optimized.install")


@pytest.fixture
def activation(monkeypatch):
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.modules.attention_fn import qsa_triton
    from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
    from exllamav3.modules.gated_delta_net import GatedDeltaNet, GDNState
    from exllamav3.modules.gated_delta_net_fn import gated_delta_rule as gdn
    from exllamav3.modules.moe_batch_recon import BatchReconLayer
    from exllamav3.modules.qsa_indexer import QSAIndexer

    for target, name in [(qsa_triton, "qsa_sparse_attend_rows"), (gdn, "_chunked_scan"),
                         (GatedDeltaNet, "__init__"), (GatedDeltaNet, "forward"),
                         (GDNState, "stash"), (GDNState, "unstash"),
                         (QSAIndexer, "sparse_attend"), (BatchReconLayer, "run_group"),
                         (BlockSparseMLP, "forward"), (ext, "hgemm_batched")]:
        monkeypatch.setattr(target, name, getattr(target, name))
    monkeypatch.setattr(hooks, "ORIGINALS", {})
    monkeypatch.setattr(hooks, "CONFIG", hooks.CONFIG.copy())
    monkeypatch.setattr(installer, "INSTALLED", False)


@pytest.fixture
def flashinfer(monkeypatch):
    package = ModuleType("flashinfer")
    module = ModuleType("flashinfer.gdn_prefill")
    package.gdn_prefill = module
    monkeypatch.setitem(sys.modules, "flashinfer", package)
    monkeypatch.setitem(sys.modules, "flashinfer.gdn_prefill", module)
    return module


def test_activation_is_idempotent(activation, flashinfer):
    from exllamav3.modules.attention_fn import qsa_triton
    from exllamav3.modules.gated_delta_net_fn import gated_delta_rule as gdn

    native_scan = gdn._chunked_scan
    enable_prefill_optimizations()
    attend = qsa_triton.qsa_sparse_attend_rows
    enable_prefill_optimizations()
    assert qsa_triton.qsa_sparse_attend_rows is attend
    assert hooks.ORIGINALS["gdn_scan"] is native_scan


def test_unknown_options_do_not_apply_partial_configuration(activation):
    original = hooks.CONFIG["prefill_min_rows"]
    with pytest.raises(ValueError, match="unknown_kernel"):
        enable_prefill_optimizations({"prefill_min_rows": original * 2, "unknown_kernel": True})
    assert hooks.CONFIG["prefill_min_rows"] == original


def make_gdn():
    from exllamav3.modules.gated_delta_net import GatedDeltaNet
    return GatedDeltaNet(None, "model.layers.2.linear_attn", 2, 8, 4, 4, 0, 0, 1e-6, 4)


@pytest.mark.parametrize("disabled", [False, True])
def test_model_construction_enables_prefill_unless_opted_out(monkeypatch, activation, flashinfer, disabled):
    from exllamav3.model.model import Model
    from exllamav3.modules.gated_delta_net import GatedDeltaNet

    monkeypatch.setattr(GatedDeltaNet, "forward", lambda self, x: installer.GDN_LAYER.get())
    if disabled:
        monkeypatch.setenv("EXLLAMAV3_NO_PREFILL_OPTIMIZATIONS", "1")
    else:
        monkeypatch.delenv("EXLLAMAV3_NO_PREFILL_OPTIMIZATIONS", raising=False)

    class Component:
        def __init__(self, config):
            self.layer = make_gdn()

        def prepare_layer_map(self):
            pass

    model = Model.from_config(SimpleNamespace(model_classes={"text": Component}))
    assert model.layer.forward(torch.zeros(1, hooks.CONFIG["prefill_min_rows"], 8)) is not disabled


def test_options_apply_before_layer_construction(monkeypatch, activation, flashinfer):
    from exllamav3.modules.gated_delta_net import GatedDeltaNet
    monkeypatch.setattr(GatedDeltaNet, "forward", lambda self, x: installer.GDN_LAYER.get())
    enable_prefill_optimizations({"gdn_native_layers": [2], "prefill_min_rows": 1})
    assert make_gdn().forward(torch.zeros(1, 1, 8)) is False
    enable_prefill_optimizations({"gdn_native_layers": []})
    assert make_gdn().forward(torch.zeros(1, 1, 8)) is True


@pytest.mark.parametrize("rows", [1, 256])
def test_layers_created_before_activation_keep_native_forward(monkeypatch, activation, flashinfer, rows):
    from exllamav3.modules.gated_delta_net import GatedDeltaNet
    monkeypatch.setattr(GatedDeltaNet, "forward", lambda self, x: x + 1)
    layer = make_gdn()
    enable_prefill_optimizations()
    x = torch.zeros(1, rows, 8)
    torch.testing.assert_close(layer.forward(x), x + 1)


def test_unsupported_flashinfer_device_keeps_native_scan(monkeypatch, flashinfer, caplog):
    from exllamav3.optimized.vendor import gdn_scan
    from exllamav3.vendor.fla import l2norm

    def unsupported(**kwargs):
        raise NotImplementedError("unsupported device")

    def chunk_gated_delta_rule():
        pass

    q = torch.ones(1, 4, 2, 8)
    g = torch.zeros(1, 4, 2)
    expected = (q + 1, torch.ones(1, 2, 8, 8))
    monkeypatch.setattr(flashinfer, "chunk_gated_delta_rule", unsupported, raising=False)
    monkeypatch.setattr(l2norm, "l2norm_fwd", lambda x: (x, None))
    monkeypatch.setattr(gdn_scan, "UNSUPPORTED_DEVICES", set())
    monkeypatch.setitem(hooks.ORIGINALS, "gdn_scan", lambda *args: expected)
    assert gdn_scan.flashinfer_scan(chunk_gated_delta_rule, q, q, q, g, g, None, True) is expected
    # Subsequent calls must use the native scan without trying FlashInfer again.
    def must_not_retry(**kwargs):
        raise AssertionError("retried an unsupported device")
    monkeypatch.setattr(flashinfer, "chunk_gated_delta_rule", must_not_retry)
    assert gdn_scan.flashinfer_scan(chunk_gated_delta_rule, q, q, q, g, g, None, True) is expected
    assert len(caplog.records) == 1


@pytest.mark.parametrize("save_state", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("warm_state", [False, True])
def test_flashinfer_preserves_native_operand_dtype_and_state(
        monkeypatch, activation, flashinfer, save_state, dtype, warm_state):
    from exllamav3.optimized.vendor import gdn_scan
    from exllamav3.vendor.fla import l2norm

    def chunk_gated_delta_rule():
        pass

    def scan(**kwargs):
        assert kwargs["q"].dtype == kwargs["k"].dtype == kwargs["v"].dtype == dtype
        result = kwargs["v"] * 3
        return (result, kwargs["initial_state"]) if kwargs["output_final_state"] else result

    monkeypatch.setattr(flashinfer, "chunk_gated_delta_rule", scan, raising=False)
    monkeypatch.setattr(l2norm, "l2norm_fwd", lambda x: (x, None))
    monkeypatch.setattr(gdn_scan, "UNSUPPORTED_DEVICES", set())
    enable_prefill_optimizations()
    q = torch.ones(1, 4, 2, 8, dtype=dtype)
    v = torch.ones(1, 4, 2, 4, dtype=dtype)
    g = torch.zeros(1, 4, 2)
    initial_state = torch.arange(64).reshape(1, 2, 8, 4).float() if warm_state else None
    actual, state = gdn_scan.flashinfer_scan(
        chunk_gated_delta_rule, q, q, v, g, g, initial_state, save_state)
    torch.testing.assert_close(actual, v * 3)
    if save_state:
        expected = initial_state if warm_state else torch.zeros(1, 2, 8, 4)
        torch.testing.assert_close(state, expected)
    else:
        assert state is None


def test_qsa_non_power_of_two_geometry_uses_native(monkeypatch):
    from exllamav3.optimized.vendor.kernels import qsa_attend
    expected = torch.ones(256, 1, 80)
    monkeypatch.setitem(hooks.ORIGINALS, "qsa_attend", lambda *args: expected)
    actual = qsa_attend(torch.zeros_like(expected), torch.zeros_like(expected),
                        torch.zeros_like(expected), torch.zeros(256, 32, dtype=torch.int32), 0.1)
    assert actual is expected


def test_offloaded_moe_keeps_native_path():
    from exllamav3.optimized.vendor.moe_group import run_group
    layer = SimpleNamespace(folded=False)
    result = run_group(layer, torch.empty(256, 8), None, None, None, [], [], [],
                       fallback=lambda owner, *args, **kwargs: owner)
    assert result is layer


def test_cpu_checkpoint_uses_native_stash_and_restore(monkeypatch):
    from exllamav3.modules.gated_delta_net import GDNState
    from exllamav3.optimized.vendor import checkpoint_transfer as transfer

    class Layer:
        value = torch.arange(8)

        def stash(self, slot):
            return self.value.clone()

        def unstash(self, slot, saved):
            self.value.copy_(saved)

    layer = Layer()
    state = GDNState.__new__(GDNState)
    state.slot, state.position, state.checkpoint_size = 0, 256, 64
    state.cache = SimpleNamespace(model=SimpleNamespace(loaded_tp=False),
                                 get_all_recurrent_layers=lambda: {"layer": layer})
    for target, name, replacement in transfer.patches():
        monkeypatch.setattr(target, name, replacement)
    saved = state.stash()
    layer.value.zero_()
    state.unstash(saved)
    assert torch.equal(layer.value, torch.arange(8))
