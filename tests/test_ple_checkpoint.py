from types import SimpleNamespace

import pytest
import torch

from exllamav3.modules.ple import PLELayerState

pytestmark = pytest.mark.nogpu


def test_ple_checkpoint_ids_survive_slot_clear_and_reuse():
    module = SimpleNamespace(
        conv_state_len=2,
        hc_mult=1,
        hidden_size=2,
        ple_embedding=SimpleNamespace(context_len=3, eos_token_id=0),
    )
    expected = torch.tensor([11, 22, 33], dtype=torch.long)
    reused = torch.tensor([44, 55, 66], dtype=torch.long)

    for restore_slot in (0, 1):
        state = PLELayerState(module, max_batch_size=2, max_history=1, cache_id=0)
        state.alloc("cpu")
        state.id_state[0, :state.ctx].copy_(expected)
        saved = state.stash(0)

        state.clear(0)
        torch.testing.assert_close(saved[1], expected)
        state.id_state[0, :state.ctx].copy_(reused)
        torch.testing.assert_close(saved[1], expected)

        state.unstash(restore_slot, saved)
        torch.testing.assert_close(state.id_state[restore_slot, :state.ctx], expected)
        state.clear(restore_slot)
        torch.testing.assert_close(saved[1], expected)
        state.unstash(restore_slot, saved)
        torch.testing.assert_close(state.id_state[restore_slot, :state.ctx], expected)
