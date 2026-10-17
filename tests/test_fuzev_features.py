from types import SimpleNamespace

import pytest
import torch

from exllamav3.model.model import Model
from exllamav3.model.model_ls import Model_LSMixin

pytestmark = pytest.mark.nogpu


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
    model.prepare_inputs = lambda ids, params: torch.arange(24).reshape(1, 3, 4, 2).float()
    model._get_prefetch_layers = []
    model.forward_ls = lambda x, params: Model_LSMixin.forward_ls(model, x, params)
    model.forward = lambda ids, params: Model.forward(model, ids, params)
    state = SimpleNamespace(position=0, last_history=0, post_advance=lambda: None)
    params = {"recurrent_states": [state]}
    result = Model.forward_features(model, torch.zeros((1, 3), dtype=torch.long), params)
    assert result.shape == (1, 1, 2)
    assert result.dtype == torch.float32
    assert not called
    assert state.position == 3 and state.last_history == 0
    assert "_fuzev_features" not in params
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


def test_features_clear_capture_flag_after_failure():
    def fail(ids, params):
        raise RuntimeError("forward failed")
    model = SimpleNamespace(loaded_tp=False, forward=fail)
    params = {}
    with pytest.raises(RuntimeError, match="forward failed"):
        Model.forward_features(model, torch.ones((1, 1), dtype=torch.long), params)
    assert params == {}
