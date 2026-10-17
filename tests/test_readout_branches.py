import pytest
import torch
import asyncio

from cpu_generator_fixture import AsyncJob, Job, MMEmbedding, State, make_async_generator, make_generator, run_readout
from exllamav3.constants import PAGE_SIZE

pytestmark = pytest.mark.nogpu


def finish(gen, jobs):
    events = []
    for _ in range(gen.max_total_tokens + len(jobs)):
        events.extend(gen.iterate())
        errors = [e["error"] for e in events if e["stage"] == "error"]
        if errors:
            raise errors[0]
        if all(job.is_finished for job in jobs):
            return [e for e in events if e["stage"] == "readout"]
    pytest.fail("readout jobs made no progress")


def root_readout(gen, ids, bound, embeddings=None):
    job = Job(torch.tensor([ids]), readout=True, readout_root_tokens=bound, embeddings=embeddings)
    gen.enqueue(job)
    return finish(gen, [job])[0]


@pytest.mark.parametrize("length", [PAGE_SIZE - 1, PAGE_SIZE, PAGE_SIZE + 1, 2 * PAGE_SIZE + 7])
@pytest.mark.parametrize("mode", ["plain", "mtp", "dflash"])
def test_exact_root_unequal_tails_match_serial_without_replay(length, mode):
    options = {"mtp": mode == "mtp", "dflash": mode == "dflash"}
    gen = make_generator(**options)
    query = [1] * length
    result = root_readout(gen, query, PAGE_SIZE + 2)
    root = result["readout_root"]
    assert root is not None and len(gen.cache.free_list) == gen.cache.num_slots
    saved = {(p.page_index, i): gen.cache.qsa[p.page_index, i]
             for p in root.pages for i in range(p.kv_position)}
    gen.model.inputs.clear()
    tails = [[2] * n for n in (2, 5, 3, PAGE_SIZE + 2)]
    children = [Job(torch.tensor([query + tail]), readout=True, readout_root=root,
                    identifier=i) for i, tail in enumerate(tails)]
    gen.enqueue(children)
    actual = sorted(finish(gen, children), key=lambda e: e["identifier"])
    assert sum(t.numel() for t in gen.model.inputs) == sum(map(len, tails))
    assert any(t.shape[0] > 1 and t.shape[1] > 1 for t in gen.model.inputs)
    for event, tail in zip(actual, tails):
        expected = run_readout(make_generator(**options), query + tail)
        torch.testing.assert_close(event["features"], expected["features"])
        assert event["cached_tokens"] == length and event["forwarded_tokens"] == len(tail)
    assert all(gen.cache.qsa[key] == value for key, value in saved.items())
    gen.pagetable.validate_pagetable([])
    root.close()
    root.close()
    assert not gen.readout_roots and not gen.recurrent_cache.pinned
    assert len(gen.cache.free_list) == gen.cache.num_slots
    gen.pagetable.validate_pagetable([])


def test_mtp_branch_first_token_uses_pre_collapse_root_carry():
    gen = make_generator()
    query = [1] * (PAGE_SIZE + 1)
    result = root_readout(gen, query, 4)
    root = result["readout_root"]
    child = Job(torch.tensor([query + [2, 3]]), readout=True, readout_root=root)
    gen.enqueue(child)
    gen.iterate_start_jobs([])
    page = child.sequences[0].allocated_pages[len(query) // PAGE_SIZE].page_index
    finish(gen, [child])
    torch.testing.assert_close(gen.draft_cache.hidden[page, len(query) % PAGE_SIZE], root.mtp_carry[0, 0])
    assert root.mtp_carry.shape[-1] != result["features"].shape[-1]
    root.close()


def test_255_children_release_slots_and_forward_only_tails():
    gen = make_generator()
    query = [1] * (PAGE_SIZE + 5)
    root = root_readout(gen, query, 5)["readout_root"]
    gen.model.inputs.clear()
    features = []
    for start in range(0, 255, gen.max_batch_size):
        children = [Job(torch.tensor([query + [2] * (i % 5 + 1)]), readout=True, readout_root=root)
                    for i in range(start, min(start + gen.max_batch_size, 255))]
        gen.enqueue(children)
        features.extend(e["features"] for e in finish(gen, children))
        assert len(gen.cache.free_list) == gen.cache.num_slots
    assert sum(t.numel() for t in gen.model.inputs) == sum(i % 5 + 1 for i in range(255))
    assert len(features) == 255
    torch.testing.assert_close(features[0], torch.tensor([len(query) + 4., len(query) + 6.]))
    root.close()


@pytest.mark.parametrize("failure", ["target", "draft"])
def test_batch_failure_leaves_root_usable_and_pages_unpublished(failure):
    gen = make_generator()
    query = [1] * (PAGE_SIZE - 1)
    root = root_readout(gen, query, 3)["readout_root"]
    children = [Job(torch.tensor([query + [i, i]]), readout=True, readout_root=root) for i in (2, 3)]
    gen.enqueue(children)
    setattr(gen.model if failure == "target" else gen.draft_model, "fail", True)
    events = gen.iterate()
    assert len([e for e in events if e["stage"] == "error"]) == 2
    assert not gen.active_jobs and not gen.pending_jobs
    assert root.pages[-1].kv_position == len(query)
    assert not any(p.kv_position == PAGE_SIZE for p in gen.pagetable.all_pages)
    gen.model.fail = gen.draft_model.fail = False
    child = Job(torch.tensor([query + [4]]), readout=True, readout_root=root)
    gen.enqueue(child)
    assert finish(gen, [child])[0]["forwarded_tokens"] == 1
    root.close()
    gen.pagetable.validate_pagetable([])


def test_closing_root_waits_for_pending_children_and_cancellation_releases_once():
    gen = make_generator()
    query = [1] * (PAGE_SIZE + 1)
    root = root_readout(gen, query, 3)["readout_root"]
    child = Job(torch.tensor([query + [2]]), readout=True, readout_root=root)
    gen.enqueue(child)
    root.close()
    assert gen.recurrent_cache.pinned and gen.readout_roots
    gen.cancel(child)
    root.close()
    assert not gen.readout_roots and not gen.recurrent_cache.pinned
    gen.pagetable.validate_pagetable([])


@pytest.mark.parametrize("invalid", ["prefix", "bound", "owner", "closed"])
def test_child_boundary_validation_precedes_allocation(invalid):
    gen = make_generator()
    image = MMEmbedding(torch.ones(1, 2), torch.tensor([[-1]]))
    query = image.token_list + [1] * PAGE_SIZE
    root = root_readout(gen, query, 2, [image])["readout_root"]
    ids, embeddings, owner = query + [2], [image], gen
    if invalid == "prefix":
        ids = query[:-1] + [3, 2]
    elif invalid == "bound":
        ids += [2, 2]
    elif invalid == "owner":
        owner = make_generator()
    elif invalid == "closed":
        root.close()
    with pytest.raises(ValueError):
        owner.enqueue(Job(torch.tensor([ids]), readout=True, readout_root=root, embeddings=embeddings))
    assert not owner.active_jobs and not owner.pending_jobs
    assert len(owner.cache.free_list) == owner.cache.num_slots
    root.close()


def test_snapshot_and_page_capacity_fallback_preserves_serial_readout():
    for bound, budget in ((2, 1), (4096, 4096)):
        gen = make_generator()
        gen.recurrent_cache.max_size = budget
        result = root_readout(gen, [1] * (PAGE_SIZE + 1), bound)
        assert result["readout_root"] is None and result["readout_fallback"]
        torch.testing.assert_close(result["features"], torch.tensor([259., 261.]))
        assert not gen.readout_roots and not gen.pagetable.referenced_pages


def test_clear_queue_releases_root_and_active_and_pending_children():
    gen = make_generator()
    query = [1] * (PAGE_SIZE + 1)
    root = root_readout(gen, query, 3)["readout_root"]
    gen.enqueue([Job(torch.tensor([query + [2, 3]]), readout=True, readout_root=root) for _ in range(5)])
    gen.iterate_start_jobs([])
    gen.clear_queue()
    root.close()
    assert not gen.readout_roots and not gen.recurrent_cache.pinned
    assert len(gen.cache.free_list) == gen.cache.num_slots
    gen.pagetable.validate_pagetable([])


def test_root_children_progress_past_request_blocked_by_pinned_pages():
    gen = make_generator(mtp=False)
    query = [1] * (2 * PAGE_SIZE + 1)
    root = root_readout(gen, query, 2)["readout_root"]
    blocked = Job(torch.full((1, gen.max_total_tokens), 3), readout=True)
    gen.enqueue(blocked)
    blocked.skips = blocked.max_skips
    fitting = Job(torch.tensor([[4] * 60]), readout=True)
    gen.enqueue(fitting)
    child = Job(torch.tensor([query + [2, 3]]), readout=True, readout_root=root)
    gen.enqueue(child)
    assert finish(gen, [child])[0]["forwarded_tokens"] == 2
    root.close()
    assert finish(gen, [blocked])[0]["forwarded_tokens"] == gen.max_total_tokens
    finish(gen, [fitting])
    gen.pagetable.validate_pagetable([])


def test_multiple_roots_share_admission_without_double_counting_child_pages():
    gen = make_generator()
    queries = [[i] * (4 * PAGE_SIZE + 1) for i in (1, 2)]
    roots = [root_readout(gen, ids, 2 * PAGE_SIZE)["readout_root"] for ids in queries]
    children = [Job(torch.tensor([ids + [3] * (2 * PAGE_SIZE)]), readout=True, readout_root=root)
                for ids, root in zip(queries, roots)]
    gen.enqueue(children)
    gen.iterate_start_jobs([])
    assert len(gen.active_jobs) == 2
    assert gen.pagetable.num_unreferenced_pages() == 0
    assert sum(root.reserved_pages for root in roots) == 0
    assert len(finish(gen, children)) == 2
    for root in roots:
        root.close()
    gen.pagetable.validate_pagetable([])


@pytest.mark.parametrize("pending", [True, False])
def test_generator_close_releases_branch_leases(pending):
    gen = make_generator()
    query = [1] * (PAGE_SIZE + 1)
    root = root_readout(gen, query, 2)["readout_root"]
    gen.enqueue(Job(torch.tensor([query + [2, 3]]), readout=True, readout_root=root))
    if not pending:
        gen.iterate_start_jobs([])
    gen.close()
    gen.close()
    assert not gen.readout_roots and not gen.recurrent_cache.pinned
    assert len(gen.cache.free_list) == gen.cache.num_slots
    gen.pagetable.validate_pagetable([])


def test_superseded_owner_cannot_restore_root_or_release_replacement_slots():
    gen = make_generator()
    query = [1] * (PAGE_SIZE + 1)
    root = root_readout(gen, query, 2)["readout_root"]
    gen.enqueue(Job(torch.tensor([query + [2, 3]]), readout=True, readout_root=root))
    gen.iterate_start_jobs([])
    gen.cache.owner_serial += 1
    slots = list(gen.cache.free_list)
    with pytest.raises(ValueError):
        gen.enqueue(Job(torch.tensor([query + [4]]), readout=True, readout_root=root))
    with pytest.raises(RuntimeError):
        gen.iterate()
    root.close()
    assert list(gen.cache.free_list) == slots
    assert not gen.readout_roots and not gen.recurrent_cache.pinned


def test_nonrecurrent_root_uses_same_page_allocator(monkeypatch):
    def count_match(page, ids, count):
        n = min(count, ids.shape[-1])
        return next((i for i in range(n) if page[0, i] != ids[0, i]), n)
    from cpu_generator_fixture import jobs as job_module
    monkeypatch.setattr(job_module.ext, "count_match_tensor", count_match)
    gen = make_generator(mtp=False, recurrent=False)
    query = [1] * (PAGE_SIZE + 7)
    root = root_readout(gen, query, 4)["readout_root"]
    assert root.snapshot is None
    child = Job(torch.tensor([query + [2, 3]]), readout=True, readout_root=root)
    gen.enqueue(child)
    actual = finish(gen, [child])[0]
    expected = run_readout(make_generator(mtp=False, recurrent=False), query + [2, 3])
    torch.testing.assert_close(actual["features"], expected["features"])
    assert actual["forwarded_tokens"] == 2
    root.close()
    gen.pagetable.validate_pagetable([])


def test_retained_root_consumes_accepted_stop_token_not_provisional_draft_rows():
    gen = make_generator()
    job = Job(torch.tensor([[1, 2]]), max_new_tokens=5, stop_conditions=[99], retain_decision_prefix=True)
    job.receive_logits = lambda logits: (torch.tensor([[99]]), None, None, None)
    gen.enqueue(job)
    events = []
    while not job.is_finished:
        events.extend(gen.iterate())
    accepted = next(e["sequence_ids"] for e in events if e.get("eos"))[0].tolist()
    assert accepted == [1, 2, 99]
    root = root_readout(gen, accepted, 1)["readout_root"]
    child = Job(torch.tensor([accepted + [4]]), readout=True, readout_root=root)
    gen.enqueue(child)
    actual = finish(gen, [child])[0]
    torch.testing.assert_close(actual["features"], run_readout(make_generator(), [1, 2, 99, 4])["features"])
    root.close()


def test_visual_prefix_batches_keep_per_root_text_position_offsets():
    class PositionTable:
        def get_mrope_freqs(self, ids, embeddings, max_length):
            delta = embeddings[0].metadata["position_delta"]
            return (torch.arange(max_length) + delta).reshape(1, max_length, 1), ids.shape[-1] + delta

    def visual_generator():
        gen = make_generator()
        gen.model.caps["mrope"] = True
        gen.model.g_rope = PositionTable()
        gen.model.use_positions = True
        return gen

    gen = visual_generator()
    roots, children, expected = [], [], []
    for delta, tail in ((-7, [2, 3]), (-11, [4, 5, 6])):
        image = MMEmbedding(torch.ones(1, 2), torch.tensor([[-1]]))
        image.metadata["position_delta"] = delta
        query = image.token_list + [1] * PAGE_SIZE
        root = root_readout(gen, query, 3, [image])["readout_root"]
        roots.append(root)
        child = Job(torch.tensor([query + tail]), readout=True, readout_root=root, identifier=len(children))
        children.append(child)
        expected.append(run_readout(visual_generator(), query + tail, embeddings=[image])["features"])
    gen.enqueue(children)
    actual = sorted(finish(gen, children), key=lambda e: e["identifier"])
    for event, feature in zip(actual, expected):
        torch.testing.assert_close(event["features"], feature)
    for root in roots:
        root.close()


@pytest.mark.parametrize("length", [PAGE_SIZE, PAGE_SIZE + 1])
def test_image_tail_uses_seeded_serial_readout_without_retry_or_root_replay(length):
    class ImagePositions:
        def get_mrope_freqs(self, ids, embeddings, max_length):
            positions = torch.arange(max_length)
            count = 0
            for image in embeddings:
                for pos in torch.nonzero(ids[0] == image.first_index).flatten().tolist():
                    positions[pos:] -= 7
                    count += 1
            return positions.reshape(1, max_length, 1), ids.shape[-1] - 7 * count

    def visual_generator():
        gen = make_generator()
        gen.model.caps["mrope"] = True
        gen.model.g_rope = ImagePositions()
        gen.model.use_positions = True
        return gen

    gen = visual_generator()
    image = MMEmbedding(torch.ones(1, 2), torch.tensor([[-1]]))
    query = [1] * length
    root = root_readout(gen, query, 3, [image])["readout_root"]
    saved = {key: value for key, value in gen.cache.kv.items() if key[0] in {p.page_index for p in root.pages}}
    gen.model.inputs.clear()
    children = [Job(torch.tensor([query + image.token_list + tail]), readout=True, readout_root=root)
                for tail in ([], [2, 3])]
    gen.enqueue(children)
    actual = finish(gen, children)
    assert sum(t.numel() for t in gen.model.inputs) == 4
    for event in actual:
        assert event["readout_fallback"] and event.get("readout_root") is None
        expected = run_readout(visual_generator(), event["job"].sequences[0].sequence_ids.torch()[0].tolist(), embeddings=[image])
        torch.testing.assert_close(event["features"], expected["features"])
    assert all(gen.cache.kv[key] == value for key, value in saved.items())
    root.close()


@pytest.mark.parametrize("chunk", [PAGE_SIZE, 8 * PAGE_SIZE])
@pytest.mark.parametrize("budget", [64, 4096])
def test_concurrent_same_prefix_roots_match_isolated_and_release_pages(chunk, budget):
    gen = make_generator()
    gen.max_chunk_size = gen.recurrent_checkpoint_interval = chunk
    gen.recurrent_cache.max_size = budget
    queries = [[1] * (2 * PAGE_SIZE) + [i] * 3 for i in range(1, 5)]
    jobs = [Job(torch.tensor([ids]), readout=True, readout_root_tokens=2) for ids in queries]
    forward = gen.model.forward
    def private_forward(input_ids, params):
        for row, start in enumerate(params["cache_seqlens"].tolist()):
            for pos in range(start, start + input_ids.shape[-1]):
                page = params["block_table"][row, pos // PAGE_SIZE]
                assert gen.pagetable.all_pages[int(page)].ref_count == 1
        return forward(input_ids, params)
    gen.model.prefill = gen.model.forward = private_forward
    gen.enqueue(jobs)
    events = finish(gen, jobs)
    roots = [e["readout_root"] for e in events]
    assert all(root is not None for root in roots)
    for event in events:
        ids = event["job"].sequences[0].sequence_ids.torch()[0].tolist()
        torch.testing.assert_close(event["features"], run_readout(make_generator(), ids)["features"])
    children = [Job(torch.cat((root.ids, torch.tensor([[2, 3]])), dim=-1),
                    readout=True, readout_root=root) for root in roots]
    gen.enqueue(children)
    for event in finish(gen, children):
        ids = event["job"].sequences[0].sequence_ids.torch()[0].tolist()
        torch.testing.assert_close(event["features"], run_readout(make_generator(), ids)["features"])
    for root in roots:
        root.close()
    assert not gen.readout_roots and not gen.recurrent_cache.pinned
    assert len(gen.cache.free_list) == gen.cache.num_slots
    gen.pagetable.validate_pagetable([])


@pytest.mark.parametrize("length", [PAGE_SIZE, 2 * PAGE_SIZE, 2 * PAGE_SIZE + 1])
@pytest.mark.parametrize("mode", ["replay", "rewind", "dense"])
def test_banned_string_replay_preserves_live_root_and_rejected_continuation(monkeypatch, length, mode):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    gen = make_generator(mtp=False, recurrent=mode != "dense")
    query = [1] * length
    root = root_readout(gen, query, 2)["readout_root"]
    saved = [(p, p.phash, p.kv_position, p.sequence.clone()) for p in root.pages]
    saved_kv = dict(gen.cache.kv)
    if mode == "replay":
        monkeypatch.setattr(State, "rollback_capacity", lambda state: 0)
    # The fixture's vocabulary logits favor 7, then 8 after the banned continuation is masked.
    from exllamav3.generator import job as module
    monkeypatch.setattr(module.ext, "partial_strings_match",
                        lambda text, offsets, buffer: text.tobytes().decode("utf-32-le").find("7"))
    job = Job(torch.tensor([query]), max_new_tokens=2, banned_strings=["7"])
    job.pinned_logit_mask = torch.empty((1, gen.padded_vocab_size), dtype=torch.half)
    def sample(logits):
        logits.fill_(-float("inf"))
        logits[..., 7], logits[..., 8] = (2. if job.new_tokens == 0 else 0.), 1.
        if job.device_logit_mask is not None:
            logits += job.device_logit_mask[:, None, :logits.shape[-1]]
        return logits.argmax(-1), None, None, None
    job.receive_logits = sample
    gen.enqueue(job)
    gen.iterate_start_jobs([])
    if gen.recurrent_cache is not None:
        gen.recurrent_cache.clear()
    original = gen.model.forward
    def private_forward(input_ids, params):
        start = int(params["cache_seqlens"][0])
        pages = params["block_table"][0]
        for pos in range(start, start + input_ids.shape[-1]):
            assert gen.pagetable.all_pages[int(pages[pos // PAGE_SIZE])].ref_count == 1
        return original(input_ids, params)
    gen.model.prefill = private_forward
    gen.model.forward = private_forward
    events = []
    for _ in range(gen.max_total_tokens):
        events.extend(gen.iterate())
        assert not any(e["stage"] == "error" for e in events)
        if job.is_finished:
            break
    assert job.is_finished
    assert next(e for e in events if e.get("eos"))["full_completion"] == "88"
    for page, phash, position, ids in saved:
        assert page.phash == phash and page.kv_position == position
        torch.testing.assert_close(page.sequence, ids)
        for offset in range(position):
            assert gen.cache.kv[page.page_index, offset] == saved_kv[page.page_index, offset]
    root.close()
    gen.pagetable.validate_pagetable([])
    assert len(gen.cache.free_list) == gen.cache.num_slots


def test_staging_failure_is_contained_to_readout_jobs():
    gen = make_generator()
    query = [1] * (PAGE_SIZE + 1)
    root = root_readout(gen, query, 2)["readout_root"]
    gen.enqueue([Job(torch.tensor([query + [2]]), readout=True, readout_root=root) for _ in range(2)])
    def fail_staging(*args, **kwargs):
        raise MemoryError("staging failed")
    gen._staging = fail_staging
    events = gen.iterate()
    assert len([e for e in events if e["stage"] == "error"]) == 2
    root.close()
    gen.pagetable.validate_pagetable([])


def test_publication_failure_does_not_invalidate_another_jobs_shared_page(monkeypatch):
    gen = make_generator()
    query, tail = [1] * (PAGE_SIZE - 1), [2] * (PAGE_SIZE + 2)
    holder = root_readout(gen, query + tail, 1)["readout_root"]
    root = root_readout(gen, query, len(tail))["readout_root"]
    shared = holder.pages[0]
    child = Job(torch.tensor([query + tail]), readout=True, readout_root=root)
    gen.enqueue(child)
    publish = Job.hash_completed_page
    def fail_second_page(job, seq, pi):
        if job is child and pi == 1:
            raise RuntimeError("publication failed")
        return publish(job, seq, pi)
    monkeypatch.setattr(Job, "hash_completed_page", fail_second_page)
    events = []
    while child in gen.pending_jobs or child in gen.active_jobs:
        events.extend(gen.iterate())
    assert any(e["stage"] == "error" and e["job"] is child for e in events)
    assert shared.kv_position == PAGE_SIZE
    gen.pagetable.validate_pagetable([])
    holder.close()
    root.close()


def test_root_constructor_failure_unpins_snapshot(monkeypatch):
    from exllamav3.generator import job as module
    gen = make_generator()
    def fail_root(*args, **kwargs):
        raise MemoryError("root constructor failed")
    monkeypatch.setattr(module, "ReadoutRoot", fail_root)
    with pytest.raises(MemoryError):
        root_readout(gen, [1] * (PAGE_SIZE + 1), 2)
    assert not gen.recurrent_cache.pinned and not gen.readout_roots
    gen.pagetable.validate_pagetable([])


@pytest.mark.parametrize("delivery", ["unread", "dropped", "yielded"])
def test_async_root_queue_disposal_respects_receiver_ownership(delivery, monkeypatch):
    async def run():
        owner = make_async_generator(monkeypatch)
        gen, roots, terminal = owner.generator, [], asyncio.Event()
        deliver = owner.deliver_results
        def receive(events):
            roots.extend(e["readout_root"] for e in events if e.get("readout_root") is not None)
            if roots and delivery == "dropped":
                owner.jobs.clear()
            deliver(events)
            if roots:
                terminal.set()
        owner.deliver_results = receive
        try:
            handle = AsyncJob.from_job(owner, Job(torch.ones((1, PAGE_SIZE + 1), dtype=torch.long),
                                                 readout=True, readout_root_tokens=2))
            await asyncio.wait_for(terminal.wait(), 5)
            root, = roots
            if delivery == "yielded":
                delivered = [e async for e in handle]
                assert delivered[-1]["readout_root"] is root
            await handle.cancel()
            if delivery == "yielded":
                assert gen.readout_roots and gen.recurrent_cache.pinned
                root.close()
                await owner.run_on_owner(root.close)
            await owner.run_on_owner(lambda: None)
            assert not gen.readout_roots and not gen.recurrent_cache.pinned
            gen.pagetable.validate_pagetable([])
        finally:
            await owner.close()
    asyncio.run(run())
