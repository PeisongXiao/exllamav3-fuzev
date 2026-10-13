# FuZev Engine Changes

This fork adds Generator feature readout and prefill optimizations to
ExLlamaV3. The [FuZev integration](https://github.com/PeisongXiao/fuzev-exllamav3-plugin)
loads the head and provides the decision API. Version `1.5.2+fuzev.v1.0`,
based on upstream ExLlamaV3 `1.5.2`.

## Install

Build the fork with the install guide's
[ExLlamaV3 engine steps](https://github.com/PeisongXiao/fuzev/blob/main/docs/install.md#exllamav3).
The source build installs `exllamav3_ext` against CUDA Torch in the active
Python environment. The PyPI `exllamav3` package is upstream.

## Change Map

| Paths | Purpose |
| --- | --- |
| `exllamav3/model/model.py` | Final-token features from the running model; prefill optimizations enabled at model construction |
| `exllamav3/model/model_ls.py` | Final hidden state before the vocabulary projection, for feature readout |
| `exllamav3/generator/job.py` | Readout jobs, optional retention of accepted generation token IDs, and `checkpoint=False` for readouts whose recurrent state is not reused |
| `exllamav3/generator/generator.py` | Readouts through the Generator's existing cache and page tables |
| `exllamav3/generator/async_generator.py` | Prepared-job submission and cancellation through the existing asynchronous Generator |
| `exllamav3/exllamav3_ext/graph.cu`, `graph.cuh`, `cuda_drv.cpp`, `cuda_drv.h` | CUDA graph kernel arguments copied into extension-managed storage before replay updates |
| `exllamav3/optimized/__init__.py`, `install.py` | Configurable prefill optimizations enabled at model construction |
| `exllamav3/modules/attention_fn/triton_paged.py` | `qc_staging` override used by prefill optimizations |
| `LICENSE` | Peisong Xiao's copyright for FuZev additions; upstream MIT copyright retained |
| `exllamav3/optimized/hooks.py`, `profile.json` | Kernel configuration and per-request context |
| `exllamav3/optimized/vendor/` | Attention, gated-delta-network, expert-computation and cache-transfer kernels |
| `exllamav3/optimized/LICENSE` | License terms for the packaged optimization code |
| `exllamav3/version.py`, `VERSION` | FuZev fork release version |
| `pyproject.toml`, `setup.py` | Engine and optimization packaging, CUDA extension build and FlashInfer dependency |
| `tests/` | Regression fixtures for feature readout, Generator behavior, graph kernel arguments and prefill optimization activation |

## Generator Integration

A readout job consumes its input through the running Generator and returns the
final hidden state before the vocabulary projection. The integration evaluates
the supplied options from those features and aligned language model scores.

- **Retained prefixes:** with `retain_decision_prefix=True`, generation keeps
  its token IDs, including accepted stop tokens. The integration can append
  decision context to that prefix and submit one scoring job per option through
  the same Generator. Ordinary generation uses the default, `False`.
- **Images:** image readouts use embeddings from the vision encoder supplied to
  the integration. The [API guide](https://github.com/PeisongXiao/fuzev/blob/main/docs/api.md)
  documents image input, prefix handling and response fields.
- **Ownership:** the application manages its Generator and model lifecycle; the
  FuZev integration manages its decision head and request jobs.

See [Generator readout](fuzev-generator.md) for the Python interfaces and the
[plugin guide](https://github.com/PeisongXiao/fuzev-exllamav3-plugin) for the
head bundle and server setup.

## Prefill Optimizations

`exllamav3.optimized` holds the activation code, configuration and kernels.
Models enable the optimizations by default.

- **Plain path:** set `EXLLAMAV3_NO_PREFILL_OPTIMIZATIONS=1` before the process
  builds its first model. The setting applies to the whole process.
- **Fallback:** when FlashInfer is missing or its gated-delta-network (GDN)
  prefill does not support the GPU, GDN layers use the native scan on that
  device and log a warning once.
- **Chunk size:** pass `max_chunk_size=8192` to `model.load()` and the
  Generator to reach the measured throughput. Lower it if prefill runs out of
  memory. ExLlamaV3's default chunk is 2,048 tokens; the FuZev launcher uses
  8,192 by default.
- **First prefill:** it takes a few seconds longer in a new process while the
  kernels compile.

Each kernel applies to part of the model:

- Sparse-attention kernels: QSA layers, the sparse attention of Qwen3.8-Flash-Next.
- GDN scan: gated-delta-network layers.
- Expert options and `bmm_tile`: every mixture-of-experts model, replacing
  ExLlamaV3's per-device grouped matrix multiplication.

On other models or GPUs, compare throughput with
`EXLLAMAV3_NO_PREFILL_OPTIMIZATIONS=1`.

Tune options before constructing a model:

```python
from exllamav3.optimized import enable_prefill_optimizations

enable_prefill_optimizations({"prefill_min_rows": 512})
```

The defaults are tuned for Qwen3.8-Flash-Next EXL3 on GB10.

| Option | Default | Meaning |
| --- | --- | --- |
| `prefill_min_rows` | `256` | Smallest prefill (tokens) for the optimized attention, GDN and expert kernels; shorter inputs use the native path |
| `gdn_native_layers` | `[0]` | GDN layers kept on the native scan. On Qwen3.8-Flash-Next, layer 0's optimized scan differs numerically from the native scan; for other models, compare outputs with `[]` |
| `qsa_stage` | `true` | Dequantizes cached keys and values into contiguous FP16 buffers for sparse attention; `false` saves that memory |
| `moe_impl` | `"folded-hot"` | Expert implementation; `"native"` selects ExLlamaV3's expert path |
| `moe_batch_cap` | `8` | Maximum experts reconstructed per folded-hot batch |
| `moe_max_rows` | `16384` | Largest expert row count that joins a folded-hot reconstruction group; experts with more rows run individually |
| `bmm_tile` | `[64, 128, 64, 3]` | Triton tile for batched expert matrix multiplication: block M, N, K and pipeline stages |

The FlashInfer GDN prefill runs in the model's activation dtype (FP16 or BF16)
with native gating and normalization.

The optimizations are written by Peisong Xiao; derived files retain
turboderp's MIT copyright ([LICENSE](../LICENSE)).

## Results

- [Integration results](https://github.com/PeisongXiao/fuzev/blob/main/docs/integration.md):
  serving behavior and overhead with the measured configurations.
- [Benchmarks](https://github.com/PeisongXiao/fuzev/blob/main/docs/benchmarks.md): decision quality.

### Prefill Measurements

- Single-stream prefill, Qwen3.8-Flash-Next EXL3 4.05 bpw on GB10.
- 8-bit KV cache, 8,192-token chunks, one output token.
- The plain column sets `EXLLAMAV3_NO_PREFILL_OPTIMIZATIONS=1` in the same build.
- After a 512-token warm-up, each prompt length runs three timed jobs. Each
  value is the mean of the second and third.

The [results file](fuzev-prefill-results.json) holds every run.

| Prompt Tokens | Plain (tok/s) | Prefill Optimizations (tok/s) |
| ---: | ---: | ---: |
| 2,048 | 810 | 833 |
| 8,192 | 1,195 | 1,401 |
| 16,384 | 1,344 | 1,598 |
| 32,768 | 1,392 | 1,649 |

### Numerical Comparison

With the same settings, the next-token distribution differs from the plain path by:

- KL 1.3e-3 after a 2,048-token prompt.
- KL 1.2e-4 after a 32,768-token prompt.

In both cases the most likely token is unchanged and no token's probability
moves by more than 0.002. The [results file](fuzev-prefill-results.json) holds
the exact values.

CUDA-host numerical checks:

```bash
python -m pytest tests/test_optimized_kernels.py tests/test_graph_arguments.py
```

CPU behavior checks:

```bash
python -m pytest --cpu tests/test_fuzev_features.py tests/test_fuzev_generator.py tests/test_prefill_optimizations.py
```

The [install record](https://github.com/PeisongXiao/fuzev/blob/main/docs/install.md#verified-source-installations)
holds source-installation settings and HTTP results. The recorded checkpoint is
[turboderp/Qwen3.8-Flash-Next-exl3, revision 55a732e0](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/tree/55a732e0c4c3d4614bc42b68493bb930d9b02c0a).

## Limitations

- ExLlamaV3 readout currently supports Generators without tensor parallelism.
  Multi-GPU layer splits are untested.
- The head's dimensions must match the loaded model. Option counts and image
  scope are in the [decision integration limitations](https://github.com/PeisongXiao/fuzev-exllamav3-plugin/blob/main/README.md#limitations).
- Retained prefixes from generation requests with an image keep the image
  embeddings with the accepted tokens. Image decisions are experimental and
  have CPU-only coverage.
- The archived ExLlamaV3 quality and workload measurements come from a
  pre-release fork build. Its extension lacks the released CUDA-graph
  kernel-argument fix, and its optimized prefill casts GDN operands to FP16.
  Those measurements were not repeated on the released build.
- Decision probabilities can differ slightly between a request's first run and
  later repeats with cached prefix pages, and between process starts, because
  Triton selects kernel configurations by timing when it first compiles them.
- CUDA graph argument patching needs the CUDA 12.4 parameter-info APIs
  (`cuFuncGetParamInfo` and `cudaFuncGetParamInfo`). The matching toolkit
  driver is 550.54.14 or newer on Linux and 551.61 or newer on Windows
  ([NVIDIA CUDA 12.4 release notes](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-toolkit-release-notes/index.html)).
  Choose a driver compatible with the selected Torch/CUDA build.
- Tested on NVIDIA GB10 (Python 3.12, PyTorch 2.13.0+cu130).
