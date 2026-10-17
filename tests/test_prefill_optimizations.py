"""Prefill dispatch and checkpoint behavior without weights or CUDA kernels."""

import sys
from types import ModuleType
from contextlib import nullcontext

import pytest
import torch

from exllamav3.optimized import enable_prefill_optimizations, settings

pytestmark = pytest.mark.nogpu


@pytest.fixture(autouse=True)
def configuration(monkeypatch):
    monkeypatch.setattr(settings, "CONFIG", settings.CONFIG.copy())


@pytest.fixture
def flashinfer(monkeypatch):
    from exllamav3.modules.gated_delta_net_fn import gated_delta_rule as gdn
    from exllamav3.vendor.fla import l2norm

    package = ModuleType("flashinfer")
    module = ModuleType("flashinfer.gdn_prefill")
    package.gdn_prefill = module
    monkeypatch.setitem(sys.modules, "flashinfer", package)
    monkeypatch.setitem(sys.modules, "flashinfer.gdn_prefill", module)
    # Exercise the adapter's tensor transforms on CPU with a simulated CUDA scan.
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(l2norm, "l2norm_fwd", lambda x: (x, None))
    monkeypatch.setattr(gdn, "_flashinfer_unsupported", set())
    monkeypatch.setattr(gdn, "_flashinfer_warned", set())
    return module


def test_unknown_options_do_not_apply_partial_configuration():
    original = settings.CONFIG.copy()
    with pytest.raises(ValueError):
        enable_prefill_optimizations({"prefill_min_rows": 1, "unknown_kernel": True})
    assert settings.CONFIG == original


@pytest.mark.parametrize("failure", [NotImplementedError, RuntimeError, AssertionError])
def test_unsupported_flashinfer_device_keeps_native_scan(monkeypatch, flashinfer, failure):
    from exllamav3.modules.gated_delta_net_fn import gated_delta_rule as gdn

    def unsupported(**kwargs):
        raise failure("unsupported device")

    def native(q, k, v, **kwargs):
        return v + 1, kwargs["initial_state"]

    monkeypatch.setattr(flashinfer, "chunk_gated_delta_rule", unsupported, raising=False)
    q = torch.ones(1, 4, 2, 8, dtype=torch.bfloat16)
    g = torch.zeros(1, 4, 2)
    state = torch.arange(128).reshape(1, 2, 8, 8).float()
    for _ in range(2):
        actual, saved = gdn._chunked_scan(native, q, q, q, g, g, state, True,
                                         use_flashinfer=True)
        torch.testing.assert_close(actual, q + 1)
        torch.testing.assert_close(saved, state)


@pytest.mark.parametrize("save_state", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("warm_state", [False, True])
def test_flashinfer_preserves_native_operand_dtype_and_state(
        monkeypatch, flashinfer, save_state, dtype, warm_state):
    from exllamav3.modules.gated_delta_net_fn import gated_delta_rule as gdn

    def scan(**kwargs):
        assert kwargs["q"].dtype == kwargs["k"].dtype == kwargs["v"].dtype == dtype
        output = kwargs["v"] * 3
        return (output, kwargs["initial_state"]) if kwargs["output_final_state"] else output

    monkeypatch.setattr(flashinfer, "chunk_gated_delta_rule", scan, raising=False)
    q = torch.ones(1, 4, 2, 8, dtype=dtype)
    v = torch.ones(1, 4, 2, 4, dtype=dtype)
    g = torch.zeros(1, 4, 2)
    initial_state = torch.arange(64).reshape(1, 2, 8, 4).float() if warm_state else None
    actual, state = gdn._flashinfer_scan(q, q, v, g, g, initial_state, save_state)
    torch.testing.assert_close(actual, v * 3)
    if save_state:
        expected = initial_state if warm_state else torch.zeros(1, 2, 8, 4)
        torch.testing.assert_close(state, expected)
    else:
        assert state is None


def test_flashinfer_import_failure_keeps_native_scan(monkeypatch, flashinfer):
    from exllamav3.modules.gated_delta_net_fn import gated_delta_rule as gdn
    monkeypatch.setitem(sys.modules, "flashinfer.gdn_prefill", None)
    q = torch.ones(1, 4, 2, 8, dtype=torch.bfloat16)
    g = torch.zeros(1, 4, 2)
    output, _ = gdn._chunked_scan(lambda q, k, v, **kw: (v + 1, None),
                                  q, q, q, g, g, None, False, use_flashinfer=True)
    torch.testing.assert_close(output, q + 1)


def test_flashinfer_out_of_memory_propagates(monkeypatch, flashinfer):
    from exllamav3.modules.gated_delta_net_fn import gated_delta_rule as gdn
    def exhausted(**kwargs):
        raise torch.OutOfMemoryError()
    monkeypatch.setattr(flashinfer, "chunk_gated_delta_rule", exhausted, raising=False)
    q = torch.ones(1, 4, 2, 8, dtype=torch.bfloat16)
    g = torch.zeros(1, 4, 2)
    with pytest.raises(torch.OutOfMemoryError):
        gdn._flashinfer_scan(q, q, q, g, g, None, False)


def test_cache_close_releases_checkpoint_staging_storage(monkeypatch):
    import weakref
    from exllamav3.cache import recurrent
    from cpu_generator_fixture import make_generator
    arena = torch.empty(128, dtype=torch.uint8)
    released = weakref.ref(arena)
    monkeypatch.setattr(recurrent, "_checkpoint_arena", arena)
    del arena
    make_generator().recurrent_cache.close()
    assert released() is None
