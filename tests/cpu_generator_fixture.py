"""Execute real scheduling/page/token code with CPU stand-ins for native kernels."""

import asyncio
from collections import deque
from types import SimpleNamespace

import torch
from exllamav3.generator import Job, Generator, AsyncGenerator, AsyncJob
from exllamav3.generator import job as jobs, pagetable as pages
from exllamav3.model.model import Model as EngineModel
from exllamav3.tokenizer import MMEmbedding


class State:
    def __init__(self, cache, slot, position=0, checksum=0):
        self.cache, self.slot = cache, slot
        self.position, self.checksum = position, checksum
        self.history = []

    def stash(self):
        return {"position": self.position, "checksum": self.checksum}

    def free(self):
        self.cache.free_list.append(self.slot)

    def rewind(self, count):
        if count:
            self.position, self.checksum = self.history[-count]
        self.history = []


class Cache:
    def __init__(self):
        self.max_num_tokens, self.num_slots, self.initialized = 4096, 4, True
        self.free_list = deque(range(4))
        self.recurrent_state_cls = SimpleNamespace(guaranteed_rollback=0)
        self.owner_serial = 1
        self.kv, self.qsa, self.hidden = {}, {}, {}

    def get_new_state(self):
        return State(self, self.free_list.popleft())

    def new_from_stashed(self, value, position):
        assert position == value["position"]
        return State(self, self.free_list.popleft(), position, value["checksum"])


class Recurrent(dict):
    get_stashed = dict.get

    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def put(self, phash, state):
        gen = self.generator
        page = gen.pagetable.referenced_pages[phash]
        if gen.draft_cache is not None:
            assert all(gen.cache.kv[page.page_index, i] == gen.draft_cache.kv[page.page_index, i]
                       for i in range(page.kv_position))
        self[phash] = state.stash()


class Model:
    loaded_tp = False
    output_device = torch.device("cpu")
    caps = {"recurrent_states": True}
    config = SimpleNamespace(hidden_size=2, vocab_size=100, max_position_embeddings=4096,
                             eos_token_id_list=[99])

    def __init__(self):
        self.vocabulary = 0
        self.features = 0
        self.fail = False
        self.embeddings = []
        self.fwd_modules = [(SimpleNamespace(caps={"logits_output": True}), 0, 0)]

    def forward(self, input_ids, params):
        if self.fail:
            raise RuntimeError("target forward failed")
        cache = params["cache"]
        self.embeddings.append(params.get("indexed_embeddings"))
        hidden = []
        for row, state in enumerate(params["recurrent_states"]):
            values = []
            for i, token in enumerate(input_ids[row].tolist()):
                state.history.append((state.position, state.checksum))
                pos = int(params["cache_seqlens"][row]) + i
                physical = int(params["block_table"][row, pos // 256])
                cache.kv[physical, pos % 256] = token
                cache.qsa[physical, pos % 256] = token * 2
                state.position += 1
                state.checksum += token
                values.append([state.checksum, state.checksum + 2, state.checksum + 4, state.checksum + 6])
            hidden.append(values)
        exported = torch.tensor(hidden, dtype=torch.float32)
        params["export_states"] = [exported]
        if params.get("_fuzev_features"):
            return exported.reshape(input_ids.shape[0], input_ids.shape[1], 2, 2).mean(-2)
        self.vocabulary += 1
        return torch.zeros((*input_ids.shape, 100))

    def forward_features(self, input_ids, params):
        self.features += 1
        return EngineModel.forward_features(self, input_ids, params)

    prefill = forward


class Draft:
    draft_verifier_params = {"export_state_norm_keys": {"final_mixer"}}

    def __init__(self):
        self.fail = False
        self.drafts = 0

    def prefill(self, input_ids, params):
        if self.fail:
            raise RuntimeError("draft prefill failed")
        for row in range(len(input_ids)):
            for i, token in enumerate(input_ids[row].tolist()):
                pos = int(params["cache_seqlens"][row]) + i
                physical = int(params["block_table"][row, pos // 256])
                params["cache"].kv[physical, pos % 256] = token
                params["cache"].hidden[physical, pos % 256] = params["target_hidden"][row, i].clone()

    def forward(self, input_ids, params):
        self.drafts += 1
        self.prefill(input_ids, params)
        return params["target_hidden"] + 1

    def sample_from_state(self, state, params):
        return torch.full((len(state), 1), 7 + params["draft_step"], dtype=torch.long)


class Tokenizer:
    def get_id_to_piece_list(self, special):
        return [str(i) for i in range(100)]


def make_generator(mtp=True):
    gen = Generator.__new__(Generator)
    gen.model, gen.cache, gen.draft_cache, gen.draft_model = Model(), Cache(), Cache(), Draft()
    gen.tokenizer = Tokenizer()
    gen.pagetable = pages.PageTable(gen, gen.cache)
    gen.recurrent_cache = Recurrent(gen)
    gen.max_batch_size, gen.max_chunk_size, gen.num_draft_tokens = gen.cache.num_slots, 256, 2 if mtp else 0
    gen.mtp_draft, gen.dflash_draft, gen.ngram_match_min = mtp, False, 0
    if not mtp:
        gen.draft_model = gen.draft_cache = None
    gen.max_total_tokens = gen.cache.max_num_tokens
    gen.active_jobs, gen.pending_jobs, gen.job_serial = [], [], 0
    gen.cache_owner_serial = gen.cache.owner_serial
    gen.recurrent_checkpoint_interval, gen.recurrent_checkpoint_interval_pp = 256, 512
    gen.padded_vocab_size, gen.visualizer = 128, None
    gen.draft_calibrator, gen.record_draft_stats = None, False
    gen.sample_pinned = torch.empty((gen.max_batch_size, 1), dtype=torch.long)
    gen.draft_input_ids_pinned = torch.empty((gen.max_batch_size, 1), dtype=torch.long)
    gen.draft_ids_pinned = torch.empty((gen.max_batch_size, gen.num_draft_tokens), dtype=torch.long)
    gen._staging = lambda name, rows, width=None, dtype=torch.int32: torch.zeros(
        (rows,) if width is None else (rows, width), dtype=dtype)
    gen.drains = 0
    def drain():
        assert not gen.pagetable.referenced_pages
        gen.drains += 1
    gen.on_queue_drained = drain
    gen.model.modules = [SimpleNamespace(prepare_for_device=lambda x, params: x)]
    gen.model.logit_layer_idx = 0
    return gen


def run_readout(gen, ids, embeddings=None, checkpoint=True):
    job = Job(torch.tensor([ids], dtype=torch.long), readout=True, embeddings=embeddings, checkpoint=checkpoint)
    gen.enqueue(job)
    events = []
    while not job.is_finished:
        events.extend(gen.iterate())
        error = next((event["error"] for event in events if event["stage"] == "error"), None)
        if error:
            raise error
    return next(event for event in events if event.get("stage") == "readout")
