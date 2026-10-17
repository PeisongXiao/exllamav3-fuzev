"""Recurrent readout features, shared roots, and the final-norm full-forward golden."""

import pytest
import torch

from exllamav3 import Generator, Job
from exllamav3.constants import PAGE_SIZE
from testlib.e2e import REFERENCE_TEXT, assert_logits_agree
from testlib.graph import load_with_caches, unload_model


def readouts(generator, jobs):
    generator.enqueue(jobs)
    results = {}
    for _ in range(generator.max_total_tokens + len(jobs)):
        for event in generator.iterate():
            if event["stage"] == "error":
                raise event["error"]
            if event["stage"] == "readout":
                results[event["job"]] = event
        if not generator.num_remaining_jobs():
            return [results[job] for job in jobs]
    pytest.fail("Readout jobs did not drain")


@pytest.mark.model("recurrent")
@torch.inference_mode()
def test_roots_match_serial_and_features_match_final_norm(model_dir, device, monkeypatch):
    _, model, (cache,), tokenizer = load_with_caches(model_dir, device, [{"max_batch_size": 4}])
    generator = None
    try:
        ids = tokenizer.encode(REFERENCE_TEXT, add_bos=True)
        prefix = ids.repeat(1, (2 * PAGE_SIZE + ids.shape[-1]) // ids.shape[-1])[:, :2 * PAGE_SIZE]
        queries = [torch.cat((prefix, tokenizer.encode(text)), dim=-1)
                   for text in (" Which answer is correct?", " What should happen next?")]
        tails = [tokenizer.encode(text) for text in (" Yes", " No")]
        lm = next(m for m, _, _ in model.fwd_modules if m.caps.get("logits_output"))
        forward = lm.forward
        final_norm = []
        def capture(x, params, *args, **kwargs):
            final_norm.append(x[:, -1:].float().clone())
            return forward(x, params, *args, **kwargs)
        monkeypatch.setattr(lm, "forward", capture)
        model.forward(queries[0], {"attn_mode": "flash_attn_nc", "last_tokens_only": 1})
        expected = final_norm[-1]
        actual = model.forward_features(queries[0], {"attn_mode": "flash_attn_nc"})
        torch.testing.assert_close(actual, expected)
        monkeypatch.setattr(lm, "forward", forward)

        for warm in (False, True):
            generator = Generator(model=model, cache=cache, tokenizer=tokenizer,
                                  max_batch_size=4, max_chunk_size=PAGE_SIZE,
                                  recurrent_checkpoint_interval=PAGE_SIZE)
            if warm:
                readouts(generator, [Job(prefix, readout=True)])
            root_jobs = [Job(query, readout=True, readout_root_tokens=max(t.shape[-1] for t in tails))
                         for query in queries]
            roots = [event["readout_root"] for event in readouts(generator, root_jobs)]
            assert all(root is not None for root in roots)
            try:
                children = [Job(torch.cat((query, tail), dim=-1), readout=True, readout_root=root)
                            for query, root in zip(queries, roots) for tail in tails]
                actual = readouts(generator, children)
            finally:
                for root in roots:
                    root.close()
            serial = [readouts(generator, [Job(torch.cat((query, tail), dim=-1), readout=True)])[0]
                      for query in queries for tail in tails]
            for reference, result in zip(serial, actual):
                # The full projection lets the existing e2e numerical policy cover
                # quantized GEMM/GEMV geometry changes without a new feature threshold.
                ref_logits = lm.forward(reference["features"].reshape(1, 1, -1).half(), {})[0].float()
                got_logits = lm.forward(result["features"].reshape(1, 1, -1).half(), {})[0].float()
                assert_logits_agree(ref_logits, got_logits)
                assert result["forwarded_tokens"] == result["job"].sequences[0].sequence_ids.torch().shape[-1] \
                    - result["job"].readout_root.position
            generator.pagetable.validate_pagetable([])
            assert not generator.readout_roots and not generator.recurrent_cache.pinned
            generator.close()
            generator = None
    finally:
        if generator is not None:
            generator.close()
        unload_model(model)
