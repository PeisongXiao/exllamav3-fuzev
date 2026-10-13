import asyncio

import pytest
import torch

from cpu_generator_fixture import AsyncGenerator, AsyncJob, Job, MMEmbedding, jobs, make_generator, run_readout


@pytest.mark.parametrize("count", [1, 256, 257])
@pytest.mark.parametrize("mtp", [False, True])
def test_readout_exact_boundary_and_post_collapse(count, mtp):
    gen = make_generator(mtp=mtp)
    ids = [1] * count
    result = run_readout(gen, ids)
    assert result["sequence_ids"].tolist() == [ids]
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
    assert actual["sequence_ids"].tolist() == expected["sequence_ids"].tolist()
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
        count = int(result["sequence_ids"].sum())
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
    assert run_readout(gen, [2])["sequence_ids"].tolist() == [[2]]


def test_retained_prefix_includes_stop_token():
    gen = make_generator()
    job = Job(torch.tensor([[1, 2]]), max_new_tokens=5, stop_conditions=[99], retain_decision_prefix=True)
    gen.enqueue(job)
    gen.iterate_start_jobs([])
    job.prefill([])
    job.receive_logits = lambda logits: (torch.tensor([[99]]), None, None, None)
    events = []
    gen.iterate_gen(events, gen.iterate_draftmodel_mtp_gen(events))
    terminal = next(event for event in events if event.get("eos"))
    assert terminal["sequence_ids"].tolist() == [[1, 2, 99]]
    result = run_readout(gen, terminal["sequence_ids"][0].tolist() + [4])
    assert result["sequence_ids"].tolist() == [[1, 2, 99, 4]]
    gen.pagetable.validate_pagetable([])


def test_requeued_job_keeps_retained_prefix_opt_in():
    ids = torch.tensor([[1, 2]])
    assert Job(ids).retain_decision_prefix is False
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
    assert {"features", "sequence_ids", "stage"} <= result.keys()
    assert gen.model.embeddings[-1] == [image]
    assert run_readout(gen, ids + [2, 3], embeddings=[image])["cached_tokens"] == 256
    other = MMEmbedding(torch.zeros(1, 2), torch.tensor([[-1]]))
    result = run_readout(gen, other.token_list + [1] * 255 + [2, 3], embeddings=[other])
    assert result["cached_tokens"] == 0
    assert gen.model.embeddings[-1] == [other]


def test_async_prepared_job_routing_and_cancellation():
    async def run():
        gen = make_generator()
        owner = AsyncGenerator.__new__(AsyncGenerator)
        owner.generator, owner.jobs, owner.error = gen, {}, None
        owner.condition = asyncio.Condition()
        owner.iteration_task = asyncio.create_task(owner._run_iteration())
        try:
            handle = AsyncJob.from_job(owner, Job(torch.tensor([[1]]), readout=True))
            results = [event async for event in handle]
            torch.testing.assert_close(results[-1]["features"], torch.tensor([3., 5.]))
            waiting = AsyncJob.from_job(owner, Job(torch.tensor([[1]]), readout=True))
            await waiting.cancel()
            assert not owner.jobs and not gen.pending_jobs
        finally:
            await owner.close()
    asyncio.run(run())
