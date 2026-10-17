"""Native expert fallback behavior for CPU and unfolded reconstruction."""

import pytest
import torch

pytestmark = pytest.mark.nogpu


@pytest.mark.parametrize("folded", [False, True])
def test_offloaded_moe_keeps_native_reconstruction(monkeypatch, folded):
    from exllamav3.modules import moe_batch_recon as recon

    width = 4
    scales = {p: (torch.ones(2, width), torch.ones(2, width)) for p in ("g", "u", "d")}
    dims = (width, width, 3)
    layer = recon.BatchReconLayer(dims, dims, dims, (False, False), (False, False),
                                 (False, False), "silu", 0, "cpu", scales, folded=folded,
                                 prefill_optimized=True)
    layer.set_static_pointers(*(torch.arange(2) for _ in range(3)))
    monkeypatch.setitem(recon.settings.CONFIG, "prefill_min_rows", 1)
    monkeypatch.setattr(recon.ext, "hgemm_batched",
                        lambda x, w, out: out.copy_(torch.bmm(x, w).to(out.dtype)))
    monkeypatch.setattr(recon.ext, "had_r_128_batch",
                        lambda x, out, *args: out.copy_(x))
    monkeypatch.setattr(layer, "_recon",
                        lambda w, *args: w.copy_(torch.eye(width).expand_as(w)))
    layer.act_call = lambda g, u, out, limit: out.copy_(torch.nn.functional.silu(g) * u)
    y = torch.arange(20).reshape(5, width).half() / 20
    y[-1].zero_()
    tokens = torch.tensor([2, 0, 3, 1, 4])
    weights = torch.tensor([0.5, 0.25, 0.75, 1, 0], dtype=torch.half)
    actual = torch.zeros(5, width)
    layer.run_group(y, actual, tokens, weights, [1, 0], [0, 2], [2, 2])
    expected = torch.zeros_like(actual)
    for token, weight in zip(tokens[:-1], weights[:-1]):
        value = (torch.nn.functional.silu(y[token]) * y[token]).half().float()
        expected[token] += value * weight.float()
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("slot_mode", [False, True])
def test_resident_profile_reserves_dispatched_group_scratch(monkeypatch, slot_mode):
    from exllamav3.modules import moe_batch_recon as recon
    monkeypatch.setattr(recon.settings, "CONFIG", dict(recon.settings.CONFIG, moe_impl="folded-hot"))
    hidden, intermediate = 4096, 8192
    up, down = (hidden, intermediate, 3), (intermediate, hidden, 3)
    scales = {p: (torch.ones(1, dims[0]), torch.ones(1, dims[1]))
              for p, dims in (("g", up), ("u", up), ("d", down))}
    layer = recon.BatchReconLayer(up, up, down, (False, False), (False, False), (False, False),
                                  "silu", 0, "cpu", scales, folded=True,
                                  prefill_optimized=True, resident=True)
    assignments = intermediate
    batch = layer.cap
    rows = min(batch * min(layer.max_rows, assignments), recon.RECON_ROWS)
    shapes = [(batch, max(hidden * intermediate, intermediate * hidden)),
              (batch, hidden, intermediate), (rows, hidden),
              (rows, intermediate), (rows, intermediate)]
    temporaries = [torch.empty(shape, device="meta", dtype=torch.half) for shape in shapes]
    if not slot_mode:
        temporaries.append(torch.empty((rows, hidden), device="meta", dtype=torch.float32))
    allocated_bytes = sum(t.numel() * t.element_size() for t in temporaries)
    assert layer.worst_case_bytes(assignments, slot_mode) >= allocated_bytes
