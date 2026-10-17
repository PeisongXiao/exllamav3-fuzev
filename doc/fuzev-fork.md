# FuZev Engine Changes

This fork adds Generator feature readout and prefill optimizations to
ExLlamaV3. The [FuZev integration](https://github.com/PeisongXiao/fuzev-exllamav3-plugin/tree/v1.1)
loads the head and provides the decision API. Version `1.6.1+fuzev.v1.1`,
based on upstream ExLlamaV3 `1.6.1`.

## Install

Build the fork with the install guide's
[ExLlamaV3 engine steps](https://github.com/PeisongXiao/fuzev/blob/v1.1/docs/install.md#exllamav3).
The source build installs `exllamav3_ext` against CUDA Torch in the active
Python environment. The PyPI `exllamav3` package is upstream.

## Change Map

| Paths | Purpose |
| --- | --- |
| `exllamav3/model/model.py` | Final-token features from the running model |
| `exllamav3/model/model_ls.py` | Final hidden state before the vocabulary projection, for feature readout |
| `exllamav3/model/config.py` | `EXLLAMAV3_NO_PREFILL_OPTIMIZATIONS` read into `infer_params.prefill_optimizations` |
| `exllamav3/generator/job.py` | Readout jobs and roots, optional retention of accepted generation token IDs, `checkpoint=False` for readouts whose recurrent state is not reused, and `max_rq_tokens` segments bounded by the remaining response and the cache, with MTP drafting carried across segments |
| `exllamav3/generator/generator.py` | Readouts through the Generator's existing cache and page tables; batched readout-root children |
| `exllamav3/generator/async_generator.py` | Prepared-job submission and cancellation through the existing asynchronous Generator; a dedicated owner thread for inference and `run_on_owner` |
| `exllamav3/generator/pagetable.py` | Readout-root pages counted as referenced; defragmentation paused while roots are open; reused prefixes capped at the last restorable recurrent checkpoint |
| `exllamav3/exllamav3_ext/graph.cu`, `graph.cuh`, `cuda_drv.cpp`, `cuda_drv.h` | CUDA graph kernel arguments copied into extension-managed storage before replay updates; eager execution when the driver lacks the parameter-info APIs |
| `exllamav3/exllamav3_ext/libtorch/attention.cpp`, `blocksparse_mlp.cpp`, `dsv4_attn.cpp`, `gated_delta_net.cpp`, `linear.cpp`, `mla_attention.cpp`, `mlp.cpp` | Eager execution when a CUDA graph capture is unavailable |
| `exllamav3/optimized/__init__.py`, `settings.py`, `profile.json` | Prefill optimization options and defaults |
| `exllamav3/modules/attention_fn/triton_paged.py` | `qc_staging` override used by prefill optimizations |
| `LICENSE` | Peisong Xiao's copyright for FuZev additions; upstream MIT copyright retained |
| `exllamav3/modules/attention_fn/qsa_prefill.py`, `exllamav3/modules/qsa_indexer.py`, `exllamav3/modules/attention_fn/qsa_triton.py` | Sparse-attention prefill kernel with staged keys and values, and its routing |
| `exllamav3/modules/gated_delta_net.py`, `exllamav3/modules/gated_delta_net_fn/gated_delta_rule.py` | FlashInfer GDN prefill per layer, with the native scan as fallback; coalesced recurrent checkpoints |
| `exllamav3/modules/moe_batch_recon.py`, `exllamav3/modules/moe_batch_recon_triton.py`, `exllamav3/modules/block_sparse_mlp.py`, `exllamav3/model/moe_cpu_host.py` | Expert grouping, fused gather and batched matrix multiplication |
| `exllamav3/modules/hyperconnections.py` | Optional cached int8 tables for tiled gated-residual mixes (`hc_cache_tables`) |
| `exllamav3/modules/linear.py` | `select_outputs`, a compact projection of requested label rows for readout |
| `exllamav3/modules/attn.py` | Loader probe length bounded by model context |
| `exllamav3/cache/recurrent.py` | Pooled coalesced checkpoint copies and checkpoint pin/unpin for readout roots |
| `exllamav3/version.py`, `VERSION` | FuZev fork release version |
| `pyproject.toml` | Engine and optimization packaging and FlashInfer dependency |
| `tests/` | Regression fixtures for feature readout, Generator behavior, graph kernel arguments and prefill optimizations |

## Generator Integration

A readout job consumes its input through the running Generator and returns the
final hidden state before the vocabulary projection. The integration evaluates
the supplied options from those features and aligned language model scores;
above 26 options it uses the features alone.

- **Retained prefixes:** with `retain_decision_prefix=True`, generation keeps
  its token IDs, including accepted stop tokens. The integration can append
  decision context to that prefix and submit one scoring job per option through
  the same Generator. Ordinary generation uses the default, `False`.
- **Images:** image readouts use embeddings from the vision encoder supplied to
  the integration. The [API guide](https://github.com/PeisongXiao/fuzev/blob/v1.1/docs/api.md)
  documents image input, prefix handling and response fields.
- **Ownership:** the application manages its Generator and model lifecycle; the
  FuZev integration manages its decision head and request jobs.

See [Generator readout](fuzev-generator.md) for the Python interfaces and the
[plugin guide](https://github.com/PeisongXiao/fuzev-exllamav3-plugin/tree/v1.1) for the
head bundle and server setup.

## Prefill Optimizations

`exllamav3.optimized` holds the configuration; the kernels live in the modules
they apply to. Models enable the optimizations by default.

- **Plain path:** set `EXLLAMAV3_NO_PREFILL_OPTIMIZATIONS=1` before creating
  the model config, or set `config.infer_params.prefill_optimizations=False`
  before `Model.from_config`.
- **Fallback:** when FlashInfer is missing or its gated-delta-network (GDN)
  prefill does not support the GPU, GDN layers use the native scan on that
  device and log a warning once. The sparse-attention and expert kernels
  likewise use ExLlamaV3's native path on a device or input shape they do not
  support.
- **Chunk size:** pass `max_chunk_size=32768` to `model.load()` and the
  Generator to reach the measured throughput. Lower it if prefill runs out of
  memory. ExLlamaV3's default chunk is 2,048 tokens; the FuZev launcher uses
  the profile's `prefill_chunk`, capped at the cache tokens, and
  `--max-chunk-size` overrides it.
- **Server options:** `fuzev-native --prefill-profile` takes a JSON object or
  a path to a JSON file and applies any option below before the model loads.
- **First prefills:** they take longer in a new process while the kernels
  compile.

Each kernel applies to part of the model:

- Sparse-attention kernels: QSA layers, the sparse attention of Qwen3.8-Flash-Next.
- GDN scan: gated-delta-network layers.
- Expert options and `bmm_tile`: mixture-of-experts layers with GPU-resident
  experts, replacing ExLlamaV3's batched expert matrix multiplication.

On other models or GPUs, compare throughput with
`EXLLAMAV3_NO_PREFILL_OPTIMIZATIONS=1`.

Tune options before constructing a model:

```python
from exllamav3.optimized import enable_prefill_optimizations

enable_prefill_optimizations({"prefill_min_rows": 512})
```

The defaults, and the fixed expert gate/up tile `(64, 64, 64, 3)`, are tuned
for Qwen3.8-Flash-Next EXL3 on GB10.

| Option | Default | Meaning |
| --- | --- | --- |
| `prefill_chunk` | `32768` | Prefill chunk for `fuzev-native`, capped at the cache tokens |
| `prefill_min_rows` | `256` | Smallest prefill (tokens) for the optimized attention, GDN and expert kernels; shorter inputs use the native path |
| `gdn_native_layers` | `[0]` | GDN layers kept on the native scan |
| `qsa_stage` | `true` | Dequantizes cached keys and values into contiguous FP16 buffers for sparse attention; `false` saves that memory |
| `moe_impl` | `"folded-hot"` | Expert implementation: `"folded-hot"` groups eligible experts for fused gather and batched Triton kernels; `"native"` selects ExLlamaV3's expert path |
| `moe_batch_cap` | `4` | Maximum experts reconstructed per folded-hot batch |
| `moe_max_rows` | `16384` | Largest expert row count that joins a folded-hot reconstruction group; experts with more rows run individually |
| `hc_cache_tables` | `true` | Keeps int8 tables and FP32 row scales for tiled gated-residual mixes; `false` derives them per call |
| `bmm_tile` | `[64, 128, 64, 3]` | Triton tile for batched expert matrix multiplication: block M, N, K and pipeline stages |

The FlashInfer GDN prefill takes BF16 q, k and v from the convolution, with
FP32 gates, beta and state; gating and q/k normalization run before the call.

The optimizations are written by Peisong Xiao; derived files retain
turboderp's MIT copyright ([LICENSE](../LICENSE)).

## Results

- [Integration results](https://github.com/PeisongXiao/fuzev/blob/v1.1/docs/integration.md):
  serving behavior with the measured configurations.
- [Benchmarks](https://github.com/PeisongXiao/fuzev/blob/v1.1/docs/benchmarks.md): decision quality.
- [Performance](https://github.com/PeisongXiao/fuzev/blob/v1.1/docs/performance.md):
  v1.1 prefill throughput against upstream ExLlamaV3, measured on one NVIDIA
  GB10 (128 GB unified memory).

CUDA-host checks:

```bash
python -m pip install pytest
python -m pytest \
  tests/test_fuzev_features.py tests/test_fuzev_generator.py \
  tests/test_readout_branches.py tests/test_async_owner.py \
  tests/test_prefill_optimizations.py tests/test_qsa_native_prefill.py \
  tests/test_moe_native_prefill.py tests/test_ple_checkpoint.py \
  tests/test_optimized_kernels.py tests/graph/test_arguments.py \
  tests/modules/linear/test_selected_outputs.py \
  tests/modules/hyperconnections/test_gated_residual.py \
  tests/cache/test_recurrent_cache.py \
  tests/generator/test_failure_containment.py
```

The [install record](https://github.com/PeisongXiao/fuzev/blob/v1.1/docs/install.md#verified-source-installations)
holds source-installation settings and HTTP results. The recorded checkpoint is
[turboderp/Qwen3.8-Flash-Next-exl3, revision 55a732e0](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/tree/55a732e0c4c3d4614bc42b68493bb930d9b02c0a).

### Decision-Root Measurements

Both measurements use one NVIDIA GB10 (128 GB unified memory), Python 3.12.3,
PyTorch 2.13.0+cu130 (CUDA 13.0), Triton 3.7.1, FlashInfer 0.7.2,
exllamav3-fuzev 1.6.1+fuzev.v1.1, head 1.1.0 and the recorded checkpoint, with
batch 4, an 8-bit KV cache, FP32 recurrent state, 2,048-token chunks and a
16,384-token cache.

- Four independent two-option Choice questions over a shared 2,048-token
  context, sent as four concurrent HTTP requests with MTP 3, take 2.02 s on
  the v1.1 release candidate before the decision-root scheduling fix and
  1.23 s after it. Each arm has one sample, taken after one grouped request
  with the same questions, without warmup or cache reset
  ([record](https://github.com/PeisongXiao/fuzev-research-archive/blob/v1.1/docs/results/integration/v1.1/decision-root-timing.json),
  [tool](https://github.com/PeisongXiao/fuzev-research-archive/blob/v1.1/tools/shared_prefix_timing.py)).
- In a native probe through the decision integration (harness unpublished),
  four three-option questions over one 1,883-token shared state, with MTP off,
  differ by up to 0.048 raw and 0.042 calibrated probability between
  concurrent and one-at-a-time runs. Repeated runs in each order match
  exactly, and no selected option changes
  ([record](https://github.com/PeisongXiao/fuzev-research-archive/blob/v1.1/docs/results/integration/v1.1/decision-root-concurrency.json)).

## Limitations

- ExLlamaV3 readout currently supports Generators without tensor parallelism.
  Multi-GPU layer splits are untested.
- The head's dimensions must match the loaded model. Option counts and image
  scope are in the [decision integration limitations](https://github.com/PeisongXiao/fuzev-exllamav3-plugin/blob/v1.1/README.md#limitations).
- Retained prefixes from generation requests with an image keep the image
  embeddings with the accepted tokens. Image decisions are experimental (run at
  your own risk); the [benchmarks](https://github.com/PeisongXiao/fuzev/blob/v1.1/docs/benchmarks.md#experimental-image-decisions)
  report their measured quality.
- The FuZev v1.1 decision results in the [benchmarks](https://github.com/PeisongXiao/fuzev/blob/v1.1/docs/benchmarks.md)
  were measured on the v1.1 release candidate before the decision-root
  scheduling fix; prefill and decode do not use that path.
- Decision probabilities can differ between a request's first run and later
  repeats with cached prefix pages, and between process starts. Concurrent
  decisions whose root children share a batched forward can differ from
  one-at-a-time runs ([decision-root measurements](#decision-root-measurements)).
- CUDA graph argument patching needs the CUDA 12.4 parameter-info APIs
  (`cuFuncGetParamInfo` and `cudaFuncGetParamInfo`). The matching toolkit
  driver is 550.54.14 or newer on Linux and 551.61 or newer on Windows
  ([NVIDIA CUDA 12.4 release notes](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-toolkit-release-notes/index.html)).
  Older drivers run without CUDA graphs.
- Tested on NVIDIA GB10 (Python 3.12, PyTorch 2.13.0+cu130).
