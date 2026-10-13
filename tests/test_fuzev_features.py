from types import SimpleNamespace

import pytest
import torch

from exllamav3.model.model import Model
from exllamav3.model.model_ls import Model_LSMixin


class Module:
    def __init__(self, function, logits=False):
        self.function = function
        self.caps = {"logits_output": logits}

    def prepare_for_device(self, x, params):
        return x

    def forward(self, x, params):
        return self.function(x)


def test_features_execute_final_mixer_but_never_vocabulary():
    called = []
    modules = [Module(lambda x: x + 1), Module(lambda x: x.mean(-2)),
               Module(lambda x: called.append("vocabulary"), logits=True)]
    model = SimpleNamespace(config=SimpleNamespace(hidden_size=2), loaded_tp=False,
                            fwd_modules=[(m, 0, i) for i, m in enumerate(modules)])
    model.forward = lambda ids, params: Model_LSMixin.forward_ls(
        model, torch.arange(24).reshape(1, 3, 4, 2).float(), params)
    params = {}
    result = Model.forward_features(model, torch.zeros((1, 3), dtype=torch.long), params)
    assert result.shape == (1, 1, 2)
    assert result.dtype == torch.float32
    assert not called
    torch.testing.assert_close(result, torch.tensor([[[20., 21.]]]))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_features_survive_reuse_of_forward_storage(dtype):
    scratch = torch.arange(12, dtype=dtype).reshape(2, 3, 2)
    expected = scratch[:, -1:].float().clone()
    model = SimpleNamespace(config=SimpleNamespace(hidden_size=2), loaded_tp=False,
                            forward=lambda ids, params: scratch)
    params = {}
    actual = Model.forward_features(model, torch.zeros((2, 3), dtype=torch.long), params)
    scratch.zero_()
    torch.testing.assert_close(actual, expected)
    assert params == {}
