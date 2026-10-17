import asyncio

import pytest
import torch

from cpu_generator_fixture import AsyncJob, Job, MMEmbedding, jobs, make_async_generator, make_generator, run_readout

pytestmark = pytest.mark.nogpu


@pytest.mark.parametrize("count", [1, 256, 257])
@pytest.mark.parametrize("mtp", [False, True])
def test_readout_exact_boundary_and_post_collapse(count, mtp):
    gen = make_generator(mtp=mtp)
    ids = [1] * count
    result = run_readout(gen, ids)
    assert result["cached_tokens"] + result["forwarded_tokens"] == count
    torch.testing.assert_close(result["features"], torch.tensor([count + 2., count + 4.]))
    if mtp:
        assert gen.model.vocabulary == gen.draft_model.drafts == 0
        assert gen.cache.kv == gen.draft_cache.kv
    assert sorted(gen.cache.free_list) == list(range(gen.cache.num_slots))
    gen.pagetable.validate_pagetable([])


def test_warm_prefix_preserves_paired_pages_and_returned_features():
    gen = make_generator()
    first = run_readout(gen, [1] * 256)
    result = run_readout(gen, [1] * 256 + [2, 3])
    assert result["cached_tokens"] == 256 and result["forwarded_tokens"] == 2
    torch.testing.assert_close(first["features"], torch.tensor([258., 260.]))
    torch.testing.assert_close(result["features"], torch.tensor([263., 265.]))
    assert gen.cache.kv == gen.draft_cache.kv
    gen.pagetable.validate_pagetable([])


def test_readout_restores_both_recurrent_sides():
    gen = make_generator(dflash=True)
    run_readout(gen, [1] * 256)
    actual = run_readout(gen, [1] * 256 + [2, 3])
    expected = run_readout(make_generator(dflash=True), [1] * 256 + [2, 3])
    assert actual["cached_tokens"] == 256 and actual["forwarded_tokens"] == 2
    torch.testing.assert_close(actual["features"], expected["features"])
    assert gen.cache.kv == gen.draft_cache.kv
    assert all(saved["target"]["checksum"] == saved["draft"]["checksum"]
               for saved in gen.recurrent_cache.values())
    for cache in (gen.cache, gen.draft_cache):
        assert sorted(cache.free_list) == list(range(cache.num_slots))
    gen.pagetable.validate_pagetable([])


def test_readout_can_fill_cache_without_generation_headroom():
    gen = make_generator()
    gen.max_draft_tokens = 16
    result = run_readout(gen, [1] * gen.max_total_tokens)
    assert result["forwarded_tokens"] == gen.max_total_tokens
    assert result["cached_tokens"] == 0
    torch.testing.assert_close(result["features"], torch.tensor([gen.max_total_tokens + 2., gen.max_total_tokens + 4.]))
    gen.pagetable.validate_pagetable([])


@pytest.mark.parametrize("count", [256, 257, 513])
@pytest.mark.parametrize("mtp", [False, True])
def test_candidate_without_checkpoints_matches_query_readout(count, mtp):
    gen = make_generator(mtp=mtp)
    gen.max_chunk_size = 1024
    run_readout(gen, [1] * 256)
    ids = [1] * 256 + [2] * (count - 256)
    actual = run_readout(gen, ids, checkpoint=False)
    reference = make_generator(mtp=mtp)
    reference.max_chunk_size = gen.max_chunk_size
    run_readout(reference, [1] * 256)
    expected = run_readout(reference, ids)
    torch.testing.assert_close(actual["features"], expected["features"])
    assert actual["cached_tokens"] == expected["cached_tokens"]
    assert actual["forwarded_tokens"] == expected["forwarded_tokens"]


def test_concurrent_readout_jobs_release_all_slots():
    gen = make_generator()
    queued = [Job(torch.tensor([[i] * 257]), readout=True) for i in range(1, 5)]
    gen.enqueue(queued)
    events = []
    while gen.num_remaining_jobs():
        events.extend(gen.iterate())
    results = [event for event in events if event["stage"] == "readout"]
    assert len(results) == len(queued)
    for result in results:
        count = int(result["job"].sequences[0].sequence_ids.torch().sum())
        torch.testing.assert_close(result["features"], torch.tensor([count + 2., count + 4.]))
    assert len(gen.cache.free_list) == gen.cache.num_slots
    gen.pagetable.validate_pagetable([])


@pytest.mark.parametrize("count", [256, 513])
def test_concurrent_identical_readouts_match_serial_features(count):
    gen = make_generator()
    gen.enqueue([Job(torch.ones((1, count), dtype=torch.long), readout=True) for _ in range(2)])
    events = []
    while gen.num_remaining_jobs():
        events.extend(gen.iterate())
    assert not any(event["stage"] == "error" for event in events)
    results = [event for event in events if event["stage"] == "readout"]
    assert len(results) == 2
    for result in results:
        torch.testing.assert_close(result["features"], torch.tensor([count + 2., count + 4.]))
    assert len(gen.cache.free_list) == gen.cache.num_slots
    gen.pagetable.validate_pagetable([])


@pytest.mark.parametrize("failure", ["target", "draft"])
def test_failed_terminal_readout_does_not_publish_unpaired_page(failure):
    gen = make_generator()
    job = Job(torch.ones((1, 256), dtype=torch.long), readout=True)
    gen.enqueue(job)
    gen.iterate_start_jobs([])
    while not job.is_prefill_done():
        job.prefill([])
    setattr(gen.model if failure == "target" else gen.draft_model, "fail", True)
    events = gen.iterate()
    assert [event["stage"] for event in events] == ["error"]
    assert not any(page.kv_position == 256 for page in gen.pagetable.all_pages)
    assert len(gen.cache.free_list) == gen.cache.num_slots and not gen.recurrent_cache
    gen.pagetable.validate_pagetable([])
    gen.model.fail = gen.draft_model.fail = False
    torch.testing.assert_close(run_readout(gen, [2])["features"], torch.tensor([4., 6.]))


def test_requeued_job_keeps_retained_prefix_opt_in():
    ids = torch.tensor([[1, 2]])
    gen = make_generator()
    job = Job(ids, max_new_tokens=5, retain_decision_prefix=True)
    gen.enqueue(job)
    gen.iterate_start_jobs([])
    job.prefill([])
    job.sequences[0].sequence_ids.append(torch.tensor([[7]]))
    job.deallocate_pages()
    requeued = job.prepare_for_requeue()
    assert requeued.retain_decision_prefix is True
    assert requeued.sequences[0].sequence_ids.torch().tolist() == [[1, 2, 7]]


@pytest.mark.parametrize("mtp", [False, True])
def test_generation_without_decisions_keeps_default_output(monkeypatch, mtp):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    gen = make_generator(mtp=mtp)
    job = Job(torch.tensor([[1, 2]]), max_new_tokens=2)
    outputs = iter([7, 8])
    job.receive_logits = lambda logits: (torch.tensor([[next(outputs)]]), None, None, None)
    gen.enqueue(job)
    events = []
    while gen.num_remaining_jobs():
        events.extend(gen.iterate())
    terminal = next(event for event in events if event.get("eos"))
    assert terminal["full_completion"] == "78"
    assert "sequence_ids" not in terminal
    assert all(event["stage"] != "readout" for event in events)
    assert gen.model.features == 0


def test_short_generation_finishes_while_segmented_long_job_reaches_cache_end(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    from exllamav3.generator.sampler.custom import CustomSampler
    monkeypatch.setattr(CustomSampler, "forward", lambda self, logits, *a, **kw: logits.argmax(-1))
    gen = make_generator()
    prompt = torch.ones((1, 256), dtype=torch.long)
    long = Job(prompt, max_new_tokens=gen.max_total_tokens - prompt.shape[-1] - gen.max_draft_tokens - 1,
               max_rq_tokens=gen.recurrent_checkpoint_interval, retain_decision_prefix=True)
    short = Job(torch.tensor([[2, 3]]), max_new_tokens=2,
                max_rq_tokens=gen.recurrent_checkpoint_interval)
    gen.enqueue([long, short])
    terminal = {}
    for _ in range(gen.max_total_tokens):
        for event in gen.iterate():
            if event["stage"] == "error":
                raise event["error"]
            if event.get("eos"):
                terminal[event["job"]] = event
                if event["job"] is short:
                    assert long not in terminal
        if not gen.num_remaining_jobs():
            break
    assert set(terminal) == {long, short}
    assert terminal[long]["sequence_ids"].shape[-1] == gen.max_total_tokens - gen.max_draft_tokens - 1
    assert len(gen.cache.free_list) == gen.cache.num_slots
    gen.pagetable.validate_pagetable([])


def test_short_segmented_generation_leaves_cache_for_other_jobs(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    from exllamav3.generator.sampler.custom import CustomSampler
    monkeypatch.setattr(CustomSampler, "forward", lambda self, logits, *a, **kw: logits.argmax(-1))
    gen = make_generator()
    short = Job(torch.ones((1, 100), dtype=torch.long), max_new_tokens=3, max_rq_tokens=gen.max_total_tokens)
    reader = Job(torch.full((1, 3000), 5, dtype=torch.long), readout=True)
    gen.enqueue([short, reader])
    gen.iterate()
    assert reader in gen.active_jobs


def test_segmented_mtp_generation_forwards_each_token_once(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    from exllamav3.generator.sampler.custom import CustomSampler
    monkeypatch.setattr(CustomSampler, "forward", lambda self, logits, *a, **kw: logits.argmax(-1))
    forwarded = []
    for segmented in (False, True):
        gen = make_generator()
        max_rq_tokens = gen.recurrent_checkpoint_interval if segmented else None
        gen.enqueue(Job(torch.ones((1, 300), dtype=torch.long), max_new_tokens=1000, max_rq_tokens=max_rq_tokens))
        while gen.num_remaining_jobs():
            gen.iterate()
        forwarded.append(sum(ids.numel() for ids in gen.model.inputs))
        gen.pagetable.validate_pagetable([])
    assert forwarded[0] == forwarded[1]


@pytest.mark.parametrize("stop", [False, True])
def test_retained_prefix_keeps_exact_tokens_after_length_or_string_stop(monkeypatch, stop):
    gen = make_generator()
    if stop:
        pieces = gen.tokenizer.get_id_to_piece_list(True)
        pieces[7] = "preSTOPafter"
        gen.tokenizer.get_id_to_piece_list = lambda special: pieces
        monkeypatch.setattr(jobs.ext, "partial_strings_match",
                            lambda text, offsets, buffer: text.tobytes().decode("utf-32-le").find("STOP"))
    job = Job(torch.tensor([[1, 2]]), max_new_tokens=5 if stop else 2,
              stop_conditions=["STOP"] if stop else None, retain_decision_prefix=True)
    gen.enqueue(job)
    gen.iterate_start_jobs([])
    job.prefill([])
    outputs = iter([7, 8])
    job.receive_logits = lambda logits: (torch.tensor([[next(outputs)]]), None, None, None)
    events = []
    gen.iterate_gen(events, gen.iterate_draftmodel_mtp_gen(events))
    terminal = next(event for event in events if event.get("eos"))
    assert terminal["sequence_ids"].tolist() == [[1, 2, 7] if stop else [1, 2, 7, 8]]
    assert terminal["full_completion"] == ("pre" if stop else "78")
    terminal["job"].sequences[0].sequence_ids.clear()
    assert terminal["sequence_ids"][0, -1].item() == (7 if stop else 8)


def test_readout_defaults_and_image_cache_identity():
    gen = make_generator()
    image = MMEmbedding(torch.ones(1, 2), torch.tensor([[-1]]))
    ids = image.token_list + [1] * 255
    result = run_readout(gen, ids, embeddings=[image])
    assert {"features", "cached_tokens", "forwarded_tokens", "stage"} <= result.keys()
    assert result["cached_tokens"] + result["forwarded_tokens"] == len(ids)
    assert gen.model.embeddings[-1] == [image]
    assert run_readout(gen, ids + [2, 3], embeddings=[image])["cached_tokens"] == 256
    other = MMEmbedding(torch.zeros(1, 2), torch.tensor([[-1]]))
    result = run_readout(gen, other.token_list + [1] * 255 + [2, 3], embeddings=[other])
    assert result["cached_tokens"] == 0
    assert gen.model.embeddings[-1] == [other]


def test_async_prepared_job_routing_and_cancellation(monkeypatch):
    async def run():
        owner = make_async_generator(monkeypatch)
        gen = owner.generator
        try:
            handle = AsyncJob.from_job(owner, Job(torch.tensor([[1]]), readout=True))
            results = [event async for event in handle]
            torch.testing.assert_close(results[-1]["features"], torch.tensor([3., 5.]))
            waiting = AsyncJob.from_job(owner, Job(torch.tensor([[1]]), readout=True))
            await waiting.cancel()
            assert not owner.jobs and not gen.pending_jobs
            with pytest.raises(AssertionError):
                AsyncJob.from_job(owner, Job(torch.ones((1, gen.max_total_tokens + 1), dtype=torch.long),
                                             readout=True))
            assert not owner.jobs and not gen.pending_jobs
            handle = AsyncJob.from_job(owner, Job(torch.tensor([[2]]), readout=True))
            results = [event async for event in handle]
            torch.testing.assert_close(results[-1]["features"], torch.tensor([4., 6.]))
        finally:
            await owner.close()
    asyncio.run(run())
