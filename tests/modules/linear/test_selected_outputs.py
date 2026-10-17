"""Native CUDA selected-output projections match the full EXL3 label logits."""

from types import SimpleNamespace

import pytest
import torch
import numpy as np

from exllamav3.modules import Linear
from testlib.exl3 import checkpoint_tensors, generator
from testlib.checkpoint import module_config


def layer(tmp_path, device, bits = 6, codebook = "mul1", dtype = torch.float16, bias = True):
    module = Linear(None, "lm_head", 251, 507, out_dtype = dtype,
                    pre_scale = 0.5, softcap = 4.0, post_scale = 1.5, trim_padded_out = True)
    tensors = checkpoint_tensors("lm_head", module.in_features, module.out_features, bits,
                                 generator(19), codebook = codebook, bias = bias)
    module.config = module_config(tensors, tmp_path, no_reconstruct=True)
    module.load(device)
    return module


def inputs(device):
    return (torch.randn((2, 1, 251), generator = generator(23)) * 0.05).half().to(device)


@pytest.mark.parametrize("bits, codebook", [(2, "3inst"), (4, "mcg"), (6, "mul1"), (3.5, "mul1")])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_selected_label_logits_match_full_projection(tmp_path, device, bits, codebook, dtype):
    source, x = layer(tmp_path, device, bits, codebook, dtype), inputs(device)
    ids = [385, np.int64(4), 130, 506, 127, 128, 385]
    expected = source.forward(x, {})[..., ids]
    selected, gather = source.select_outputs(ids)
    actual = selected.forward(x, {}).index_select(-1, gather)
    torch.testing.assert_close(actual, expected, rtol = 0.02, atol = 0.02)
    torch.testing.assert_close(actual.float().softmax(-1), expected.float().softmax(-1), rtol = 0.02, atol = 0.02)


def test_bias_free_selection_and_output_dtype_override(tmp_path, device):
    source, x = layer(tmp_path, device, bias = False), inputs(device)
    selected, gather = source.select_outputs(torch.tensor([130, 2]))
    actual = selected.forward(x, {}, out_dtype = torch.float32).index_select(-1, gather)
    torch.testing.assert_close(actual, source.forward(x, {}, out_dtype = torch.float32)[..., [130, 2]],
                               rtol = 0.02, atol = 0.02)


def test_selection_owns_output_blocks_and_leaves_source_unchanged(tmp_path, device):
    source, x = layer(tmp_path, device), inputs(device)
    original = {key: value.clone() for key, value in source.inner.get_tensors("lm_head").items()}
    expected = source.forward(x, {}).clone()
    selected, gather = source.select_outputs([385, 4])
    torch.testing.assert_close(selected.forward(x, {}).index_select(-1, gather), expected[..., [385, 4]],
                               rtol = 0.02, atol = 0.02)
    with torch.inference_mode():
        selected.inner.trellis.zero_()
        selected.inner.svh.zero_()
        selected.inner.bias.zero_()
    for key, value in source.inner.get_tensors("lm_head").items():
        torch.testing.assert_close(value, original[key], rtol = 0, atol = 0)
    torch.testing.assert_close(source.forward(x, {}), expected, rtol = 0, atol = 0)


@pytest.mark.parametrize("condition", ["sliced", "adapter", "backend"])
def test_unsupported_selection_keeps_native_fallback_outputs(tmp_path, device, condition):
    source = layer(tmp_path, device)
    if condition == "sliced":
        source.is_sliced = True
    elif condition == "adapter":
        source.lora_a_tensors["test"] = torch.full((256, 1), 0.01, dtype = torch.float16)
        source.lora_b_tensors["test"] = torch.full((1, 507), 0.02, dtype = torch.float16)
    elif condition == "backend":
        source.inner = SimpleNamespace(forward = source.inner.forward)
    ids = [130, 2]
    assert source.select_outputs(ids) is None


@pytest.mark.parametrize("ids", [[], [-1], [507]])
def test_invalid_output_ids_raise_before_selection(tmp_path, device, ids):
    source = layer(tmp_path, device)
    with pytest.raises(ValueError):
        source.select_outputs(ids)
