"""Replay the real native graph implementation with patched kernel pointers."""

from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.cuda


def test_kernel_arguments_survive_repeated_pointer_updates(request):
    if request.config.getoption("--cpu") or not torch.cuda.is_available():
        pytest.skip("requires CUDA and the CUDA build toolchain")
    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parents[1]
    native = root / "exllamav3/exllamav3_ext"
    extension = load(
        name="exl3_graph_arguments_test",
        sources=[str(Path(__file__).parent / "fixtures/graph_arguments.cu"),
                 str(native / "graph.cu"), str(native / "cuda_drv.cpp"),
                 str(native / "quant/exl3_devctx.cu")],
        extra_include_paths=[str(native)], extra_ldflags=["-lcublas", "-ldl"],
    )
    original = torch.arange(257, dtype=torch.float32, device="cuda")
    output = torch.empty_like(original)
    graph = extension.AffineGraph(original, output, 1.25, -3.5)
    # Keep distinct allocations alive, so every replay really changes both pointers.
    inputs = [original + i for i in range(4)]
    outputs = [torch.empty_like(original) for _ in inputs]
    for value, result in zip(inputs, outputs):
        graph.replay(value, result)
        torch.testing.assert_close(result, value * 1.25 - 3.5, rtol=0, atol=0)
