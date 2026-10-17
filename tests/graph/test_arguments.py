"""Replay the real native graph implementation with patched kernel pointers."""

from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.cuda_only


@pytest.fixture(scope = "module")
def graph_extension():
    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parents[2]
    native = root / "exllamav3/exllamav3_ext"
    return load(
        name="exl3_graph_arguments_test",
        sources=[str(Path(__file__).parent / "fixtures/graph_arguments.cu"),
                 str(native / "graph.cu"), str(native / "cuda_drv.cpp"),
                 str(native / "quant/exl3_devctx.cu")],
        extra_include_paths=[str(native)], extra_ldflags=["-lcublas", "-ldl"],
    )


def test_kernel_arguments_survive_repeated_pointer_updates(graph_extension, device):
    original = torch.arange(257, dtype=torch.float32, device=device)
    output = torch.empty_like(original)
    graph = graph_extension.AffineGraph(original, output, 1.25, -3.5)
    # Keep distinct allocations alive, so every replay really changes both pointers.
    inputs = [original + i for i in range(4)]
    outputs = [torch.empty_like(original) for _ in inputs]
    for value, result in zip(inputs, outputs):
        graph.replay(value, result)
        torch.testing.assert_close(result, value * 1.25 - 3.5, rtol=0, atol=0)


def test_scalar_updates_preserve_unpatched_trailing_arguments(graph_extension, device):
    value = torch.arange(257, dtype=torch.float32, device=device)
    output = torch.empty_like(value)
    graph = graph_extension.AffineGraph(value, output, 1.25, -3.5)
    for count in (129, 257, 0, 1):
        output.fill_(-999)
        graph.replay(value, output, count)
        expected = torch.full_like(output, -999)
        expected[:count] = value[:count] * 1.25 - 3.5
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize("abort", [False, True])
def test_recapture_replaces_owned_arguments(graph_extension, device, abort):
    value = torch.arange(257, dtype=torch.float32, device=device)
    output = torch.empty_like(value)
    graph = graph_extension.AffineGraph(value, output, 1.25, -3.5)
    for scale, bias in ((-2., 9.), (0.5, -7.)):
        graph.replay(value, output)
        torch.cuda.current_stream(device).synchronize()
        if abort:
            graph.abort()
        graph.recapture(value, output, scale, bias)
        graph.replay(value + 1, output)
        torch.testing.assert_close(output, (value + 1) * scale + bias, rtol=0, atol=0)


def test_invalid_argument_capture_can_be_retried(graph_extension, device):
    value = torch.arange(257, dtype=torch.float32, device=device)
    output = torch.empty_like(value)
    graph = graph_extension.AffineGraph(value, output, 1.25, -3.5)
    with pytest.raises(RuntimeError):
        graph.recapture(value, output, 2., 4., -1, 8)
    graph.recapture(value, output, -2., 9.)
    graph.replay(value, output)
    torch.testing.assert_close(output, value * -2. + 9., rtol=0, atol=0)
