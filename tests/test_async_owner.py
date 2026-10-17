import asyncio
from threading import Event

import pytest
import torch

from cpu_generator_fixture import AsyncJob, Job, jobs, make_async_generator

pytestmark = pytest.mark.nogpu


async def wait_entered(event):
    async def wait():
        while not event.is_set():
            await asyncio.sleep(0.001)
    await asyncio.wait_for(wait(), 5)


@pytest.mark.parametrize("cancel", [False, True])
def test_awaitable_admission_keeps_loop_responsive_and_cleans_cancelled_jobs(monkeypatch, cancel):
    async def run():
        owner = make_async_generator(monkeypatch)
        entered, release = Event(), Event()
        def occupied():
            entered.set()
            assert release.wait(5)
        busy = asyncio.create_task(owner.run_on_owner(occupied))
        admission = None
        try:
            await wait_entered(entered)
            handle = AsyncJob.from_job(owner, Job(torch.tensor([[1]]), readout=True), enqueue=False)
            admission = asyncio.create_task(owner.enqueue_async(handle))
            await asyncio.sleep(0.01)
            assert not admission.done()
            if cancel:
                admission.cancel()
            release.set()
            await busy
            if cancel:
                with pytest.raises(asyncio.CancelledError):
                    await admission
                assert not owner.jobs
                assert not owner.generator.num_remaining_jobs()
            else:
                await admission
                result = [e async for e in handle][-1]
                torch.testing.assert_close(result["features"], torch.tensor([3., 5.]))
        finally:
            release.set()
            await busy
            if admission is not None:
                await asyncio.gather(admission, return_exceptions=True)
            await owner.close()
    asyncio.run(run())


def test_iteration_leaves_loop_responsive_and_serializes_shared_model_calls(monkeypatch):
    async def run():
        owner = make_async_generator(monkeypatch)
        entered, release = Event(), Event()
        workspace = torch.empty(2)
        forward = owner.generator.model.forward_features
        def blocked(*args, **kwargs):
            workspace.fill_(float("nan"))
            entered.set()
            assert release.wait(5)
            features = forward(*args, **kwargs)
            workspace.copy_(features.reshape(-1, 2)[-1])
            return features
        monkeypatch.setattr(owner.generator.model, "forward_features", blocked)
        try:
            handle = AsyncJob(owner, torch.tensor([[1]]), readout=True)
            await wait_entered(entered)
            shared_call = asyncio.create_task(owner.run_on_owner(lambda: workspace.clone() * 2))
            await asyncio.sleep(0.01)
            assert not shared_call.done()
            release.set()
            torch.testing.assert_close(await shared_call, torch.tensor([6., 10.]))
            result = [e async for e in handle][-1]
            torch.testing.assert_close(result["features"], torch.tensor([3., 5.]))
            def head_error():
                raise ValueError("request head failed")
            with pytest.raises(ValueError, match="request head failed"):
                await owner.run_on_owner(head_error)
            again = AsyncJob(owner, torch.tensor([[2]]), readout=True)
            torch.testing.assert_close([e async for e in again][-1]["features"], torch.tensor([4., 6.]))
        finally:
            release.set()
            await owner.close()
    asyncio.run(run())


def test_sync_job_mutations_keep_their_errors_and_readout_features(monkeypatch):
    async def run():
        owner = make_async_generator(monkeypatch)
        try:
            handle = AsyncJob(owner, torch.tensor([[1]]), readout=True)
            handle.set_sampler(None)
            handle.set_filters([])
            handle.set_banned_strings([])
            handle.constrain_output_now(torch.tensor([[3]]))
            with pytest.raises(ValueError):
                handle.set_filters([])
            with pytest.raises(AssertionError):
                handle.constrain_output_now(torch.tensor([3]))
            torch.testing.assert_close([e async for e in handle][-1]["features"], torch.tensor([3., 5.]))
            await handle.cancel()
        finally:
            await owner.close()
    asyncio.run(run())


def test_inflight_iteration_failure_reaches_concurrent_and_later_jobs(monkeypatch):
    async def run():
        owner = make_async_generator(monkeypatch)
        entered, release = Event(), Event()
        def fail():
            entered.set()
            assert release.wait(5)
            raise RuntimeError("fatal iterate")
        monkeypatch.setattr(owner.generator, "iterate", fail)
        try:
            first = AsyncJob(owner, torch.tensor([[1]]), readout=True)
            await wait_entered(entered)
            release.set()
            second = AsyncJob(owner, torch.tensor([[2]]), readout=True)
            for handle in (first, second):
                with pytest.raises(RuntimeError, match="fatal iterate"):
                    _ = [e async for e in handle]
            later = AsyncJob(owner, torch.tensor([[3]]), readout=True)
            with pytest.raises(RuntimeError, match="fatal iterate"):
                _ = [e async for e in later]
            await first.cancel()
        finally:
            release.set()
            await owner.close()
    asyncio.run(run())


def test_cancelled_model_call_drains_before_releasing_inputs(monkeypatch):
    async def run():
        owner = make_async_generator(monkeypatch)
        entered, release, finished = Event(), Event(), Event()
        closed = []
        def use_input():
            entered.set()
            assert release.wait(5)
            assert not closed
            finished.set()
        async def request():
            try:
                await owner.run_on_owner(use_input)
            finally:
                closed.append(True)
        task = asyncio.create_task(request())
        try:
            await wait_entered(entered)
            task.cancel()
            await asyncio.sleep(0.01)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done() and not closed
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert finished.is_set() and closed
            handle = AsyncJob(owner, torch.tensor([[1]]), readout=True)
            torch.testing.assert_close([e async for e in handle][-1]["features"], torch.tensor([3., 5.]))
        finally:
            release.set()
            await owner.close()
    asyncio.run(run())


@pytest.mark.parametrize("retire", ["close", "cancel_iteration"])
def test_retirement_drains_inflight_readout_and_closes_late_roots(monkeypatch, retire):
    async def run():
        owner = make_async_generator(monkeypatch)
        entered, release, roots = Event(), Event(), []
        forward, init = owner.generator.model.forward_features, jobs.ReadoutRoot.__init__
        def blocked(*args, **kwargs):
            entered.set()
            assert release.wait(5)
            return forward(*args, **kwargs)
        def keep(root, *args, **kwargs):
            init(root, *args, **kwargs)
            roots.append(root)
        monkeypatch.setattr(owner.generator.model, "forward_features", blocked)
        monkeypatch.setattr(jobs.ReadoutRoot, "__init__", keep)
        try:
            handle = AsyncJob(owner, torch.ones((1, 257), dtype=torch.long),
                              readout=True, readout_root_tokens=2)
            await wait_entered(entered)
            if retire == "close":
                pending = asyncio.create_task(owner.close())
            else:
                owner.iteration_task.cancel()
                pending = owner.iteration_task
            await asyncio.sleep(0.01)
            assert not pending.done()
            if retire == "close":
                with pytest.raises(RuntimeError):
                    AsyncJob(owner, torch.tensor([[2]]), readout=True)
                pending.cancel()
                cancelling = asyncio.create_task(handle.cancel())
                await asyncio.sleep(0.01)
                assert not cancelling.done()
            release.set()
            if retire == "close":
                with pytest.raises(asyncio.CancelledError):
                    await pending
            else:
                await pending
                await owner.run_on_owner(lambda: None)
            assert roots and all(root.closed for root in roots)
            assert not owner.generator.readout_roots and not owner.generator.recurrent_cache.pinned
            if retire == "close":
                await cancelling
            else:
                await handle.cancel()
        finally:
            release.set()
            await owner.close()
        for root in roots:
            root.close()
    asyncio.run(run())
