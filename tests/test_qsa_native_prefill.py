"""QSA autosplit measurement with an explicit sparse-prefill context on CPU tensors."""

import pytest
import torch

pytestmark = pytest.mark.nogpu


@pytest.mark.parametrize("pool_tokens, chunk, expected_past, expected_length", [
    (3072, 256, 2816, 3072),
    (8192, 256, 3840, 4096),
    (8192, 8192, 0, 4096),
])
def test_qsa_autosplit_measures_largest_supported_context(
        monkeypatch, pool_tokens, chunk, expected_past, expected_length):
    from types import SimpleNamespace
    from exllamav3.cache.qsa import QSAPlanes
    from exllamav3.constants import PAGE_SIZE
    from exllamav3.modules.attn import Attention
    from exllamav3.modules.attention_fn import qsa_prefill, qsa_triton

    monkeypatch.setattr(qsa_prefill, "qsa_attend", lambda q, *args, **kwargs: q + 2)
    monkeypatch.setattr(qsa_triton, "qsa_sparse_attend_rows", lambda q, *args, **kwargs: q + 3)
    layer = QSAPlanes()
    # The hook uses only plane capacity and first-page zeroing before its forward.
    for name in ("qk", "qv", "sk", "sv", "raw_k", "pooled"):
        setattr(layer, name, torch.ones((pool_tokens // PAGE_SIZE, 1)))
    observed = {}

    def forward(x, params):
        observed["past"] = int(params["cache_seqlens"][0])
        length = observed["past"] + x.shape[1]
        q = x.reshape(params["batch_shape"][1], 1, 8)
        indices = torch.zeros((q.shape[0], 32), dtype=torch.int32)
        observed["length"] = length
        observed["output"] = qsa_prefill.attend(
            q, q, q, indices, 0.1, context={"length": length, "max_length": 4096})

    attn = SimpleNamespace(
        config=SimpleNamespace(max_position_embeddings=4096),
        device=torch.device("cpu"), hidden_size=8,
        qsa_indexer=SimpleNamespace(sparse_threshold=lambda: 2051),
        _autosplit_layer=lambda params: (layer, True), forward=forward)
    Attention.autosplit_extra_measure(attn, {"batch_shape": (1, chunk)})
    assert observed["past"] == expected_past
    assert observed["length"] == expected_length
    torch.testing.assert_close(
        observed["output"], torch.full((expected_length - expected_past, 1, 8), 2.0, dtype=torch.half))
