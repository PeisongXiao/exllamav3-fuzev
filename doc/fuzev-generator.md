# Generator Feature Readout

Submit `Job(input_ids, readout=True)` to a running Generator to consume input
without sampling. The terminal `stage="readout"` event includes:

- `features`: an FP32 vector from the final hidden state before the vocabulary head.
- `cached_tokens` and `forwarded_tokens`: reused and performed work.

Readout uses the Generator's page table, recurrent slots and prefix cache.
`readout=True` sets `max_new_tokens=0`.

- Without multi-token prediction (MTP), nonterminal chunks use ordinary prefill.
- With MTP, every chunk also feeds the draft. Target and draft cache updates
  complete before the last page is published for reuse.

```python
from exllamav3.generator import AsyncJob, Job

async def read_features(generator, input_ids):
    job = Job(input_ids, readout=True)
    async for event in AsyncJob.from_job(generator, job):
        if event.get("stage") == "readout":
            return event["features"]
```

For multimodal input, pass the `MMEmbedding` objects used to construct the
token IDs: `Job(input_ids, readout=True, embeddings=embeddings)`. Prefill and
the final feature forward then receive the image embeddings. The application
manages the vision model and embedding lifetime.

To read out after generation:

1. Set `retain_decision_prefix=True` (default false) on the generation job. Its
   terminal event then includes the accepted `sequence_ids`, including EOS or
   stop-string tokens.
2. Append the decision input and submit a readout job through the same Generator.

Retained IDs reserve no cache pages; a later readout reuses the prefix while
its pages remain cached. Cancelling an `AsyncJob` cancels only that job. The
[decision plugin](https://github.com/PeisongXiao/fuzev-exllamav3-plugin/tree/v1.1) loads
the head and provides the API.

Features, and therefore decisions, depend on batch geometry. A readout without
a readout root, or with an image-bearing tail, runs as its own batch-1 job;
children of active readout roots, including other requests' children, share
one batched forward
([archive geometry note](https://github.com/PeisongXiao/fuzev-research-archive/blob/v1.1/research-snapshot/README.md#feature-geometry)).
`tests/test_fuzev_generator.py` covers readout, retained prefixes and image
embeddings; `tests/test_readout_branches.py` covers readout roots.

## Request-Scoped Readout Roots

To read several continuations of one prefix, submit the prefix as
`Job(prefix_ids, readout=True, readout_root_tokens=max_tail_tokens)`. Its
terminal event adds `readout_root`, which keeps the consumed prefix and its
recurrent state on the Generator. Submit each continuation as
`Job(ids, readout=True, readout_root=root)`, where `ids` holds the full prefix
plus a nonempty tail of at most `max_tail_tokens` tokens; omitted `embeddings`
are inherited from the root. Close the root after its children finish, with
`await generator.run_on_owner(root.close)` on an `AsyncGenerator`.

If the root does not fit the cache, model context or checkpoint budget, or the
prefix ends with an image, the event has `readout_root=None` and a
`readout_fallback` reason; read the continuations as ordinary readout jobs.
Children of closed or foreign roots, and children with mismatched prefixes or
out-of-bound tails, raise `ValueError` at submission.

## Limitations

- Readout jobs currently support targets without tensor parallelism.
- Readout with a DFlash or classic draft model, and multi-GPU layer splits, are
  untested.
- Token healing changes the input, so it is incompatible with readout.
- Probabilities can differ between a decision's first run and later repeats
  with cached prefix pages, and between process starts. They can also differ
  with the number of root children in the same batched forward
  ([fork limitations](fuzev-fork.md#limitations)).
- Option counts, experimental image scope and the tested platform are in the
  [plugin limitations](https://github.com/PeisongXiao/fuzev-exllamav3-plugin/blob/v1.1/README.md#limitations).
