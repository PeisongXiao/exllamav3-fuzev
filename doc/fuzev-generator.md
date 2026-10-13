# Generator Feature Readout

Submit `Job(input_ids, readout=True)` to a running Generator to consume input
without sampling. The terminal `stage="readout"` event includes:

- `features`: an FP32 vector from the final hidden state before the vocabulary head.
- `sequence_ids`: consumed token IDs on CPU.
- `cached_tokens` and `forwarded_tokens`: reused and performed work.

Readout uses the Generator's page table, recurrent slots and prefix cache.
`readout=True` sets `max_new_tokens=0`.

- Without multi-token prediction (MTP), nonterminal chunks use ordinary prefill.
- With MTP, every chunk also feeds the draft. Target and draft cache updates
  complete before the last page is published for reuse.

```python
from exllamav3.generator import AsyncJob, Job

job = Job(input_ids, readout=True)
async for event in AsyncJob.from_job(generator, job):
    if event.get("stage") == "readout":
        features = event["features"]
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
[decision plugin](https://github.com/PeisongXiao/fuzev-exllamav3-plugin) loads
the head and provides the API.

Features, and therefore decisions, depend on batch geometry. Readout computes
each candidate as its own batch-1 job
([archive geometry note](https://github.com/PeisongXiao/fuzev-research-archive/blob/main/research-snapshot/README.md#feature-geometry)).
`tests/test_fuzev_generator.py` covers readout, retained prefixes and image
embeddings.

## Limitations

- Readout jobs currently support targets without tensor parallelism.
- Readout with a DFlash or classic draft model, and multi-GPU layer splits, are
  untested.
- Token healing changes the input, so it is incompatible with readout.
- Probabilities can differ slightly between a decision's first run and later
  repeats with cached prefix pages, and between process starts
  ([fork limitations](fuzev-fork.md#limitations)).
- Option counts, experimental image scope and the tested platform are in the
  [plugin limitations](https://github.com/PeisongXiao/fuzev-exllamav3-plugin/blob/main/README.md#limitations).
