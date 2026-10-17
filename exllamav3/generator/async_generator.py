from __future__ import annotations
from .generator import Generator
from .job import Job
import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import get_ident
import torch

# Sentinel pushed to an AsyncJob's queue on cancellation so a consumer parked in queue.get() wakes up and
# exits, rather than waiting forever for results that will no longer be produced
_CANCELLED_SENTINEL = object()


def _discard_result(result):
    if isinstance(result, dict) and (root := result.get("readout_root")) is not None:
        if owner := getattr(root.generator, "_async_submit", None):
            owner(root._close)
        else:
            root.close()


class AsyncGenerator:
    """
    Async wrapper for dynamic generator. See definition of Generator.
    """
    def __init__(self, *args, **kwargs):
        self.jobs = {}
        self.error = None
        self._owner_error = None
        self._closing = False
        self._close_task = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="exllamav3-owner")
        self._owner_thread = self._executor.submit(get_ident).result()
        try:
            self.generator = self._call_on_owner(Generator, *args, **kwargs)
            self._call_on_owner(setattr, self.generator, "_async_owner", self._call_on_owner)
            self._call_on_owner(setattr, self.generator, "_async_submit", self._executor.submit)
        except BaseException:
            self._executor.shutdown(wait=True)
            raise
        self.condition = asyncio.Condition()
        self.iteration_task = asyncio.create_task(self._run_iteration())

    def _check_open(self):
        if self._closing:
            raise RuntimeError("AsyncGenerator is closing")

    def _call_on_owner(self, function, *args, **kwargs):
        if get_ident() == self._owner_thread:
            return function(*args, **kwargs)
        return self._executor.submit(function, *args, **kwargs).result()

    async def _wait(self, future):
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            # Cancelling an await does not stop its native callable or release its inputs.
            while not future.done():
                try:
                    await asyncio.shield(future)
                except (asyncio.CancelledError, Exception):
                    pass
            if not future.cancelled():
                future.exception()
            raise

    async def run_on_owner(self, function, *args, **kwargs):
        """Serialize a synchronous model callable with inference; drain it before propagating cancellation."""
        self._check_open()
        return await self._wait(asyncio.wrap_future(self._executor.submit(function, *args, **kwargs)))

    def _iterate(self):
        try:
            return self.generator.iterate()
        except Exception as error:
            # Following synchronous enqueues can run before the loop observes this failed future.
            self._owner_error = error
            raise

    async def _run_iteration(self):
        try:
            while True:
                # Sleep while there is no async work registered. The condition releases its lock while waiting and
                # is notified by enqueue() or close(), so this background task does not spin between requests.
                async with self.condition:
                    await self.condition.wait_for(lambda: len(self.jobs) > 0 or self._closing)
                if self._closing:
                    return

                # Drive exactly one synchronous generator step and fan out any returned events to the owning
                # AsyncJob queues. Missing jobs can happen if a job was cancelled after iterate() started.
                # Delivery must never block: this single task serves every job, so waiting on one stalled
                # consumer (e.g. a disconnected client that stopped draining its queue) would wedge the whole
                # generator (issue #227).
                future = asyncio.wrap_future(self._executor.submit(self._iterate))
                try:
                    results = await self._wait(future)
                except asyncio.CancelledError:
                    if not future.cancelled() and future.exception() is None:
                        for result in future.result():
                            _discard_result(result)
                    raise
                if self._closing:
                    for result in results:
                        _discard_result(result)
                    return
                self.deliver_results(results)

                # Yield back to the event loop so result consumers and cancellation requests can run between
                # generator iterations, even if the synchronous generator immediately has more work available.
                await asyncio.sleep(0)

        except asyncio.CancelledError:
            # Silently return on cancel
            return

        except Exception as e:
            # If the generator throws an exception it won't pertain to any one ongoing job, so push it to all of
            # them. The iteration task ends here and the generator is unusable, so record the error and deliver
            # it to any job enqueued later as well, instead of letting those jobs wait forever on a dead loop.
            self.error = e
            for async_job in self.jobs.values():
                async_job.put_result(e)
            self.jobs.clear()

    def deliver_results(self, results):
        """
        Fan generator results out to the owning AsyncJob queues. Missing jobs can happen if a job
        was cancelled after iterate() started. Delivery must never block: this single task serves
        every job, so waiting on one stalled consumer would wedge the whole generator (issue #227).
        """
        for result in results:
            job = result["job"]
            async_job = self.jobs.get(job)
            if not async_job:
                _discard_result(result)
                continue
            if result.get("stage") == "error":
                # Per-job failure contained by Generator.reap_failed_job. The raw exception
                # must go into the queue as an exception object, not as a result dict:
                # __aiter__ re-raises queued exceptions, which the server surfaces as an
                # error for this request. A dict here would instead stream as a clean
                # (empty) completion.
                async_job.put_result(result["error"])
                del self.jobs[job]
                continue
            async_job.put_result(result)
            if result["eos"]:
                del self.jobs[job]

    def enqueue(self, job: AsyncJob):
        self._check_open()
        # The iteration task died on a generator exception; surface it to the new job's consumer immediately
        if self.error is not None:
            job.put_result(self.error)
            return

        # Track the AsyncJob before enqueueing the underlying Job so any immediate generator result has a queue to
        # land in. The sync generator still owns scheduling and serial assignment.
        assert job.job not in self.jobs
        self.jobs[job.job] = job
        try:
            # Preserve synchronous admission errors, even when this waits for the current iteration.
            error = self._call_on_owner(self._enqueue, job.job)
        except Exception:
            # A rejected job (e.g. too long for the cache) must not stay tracked: it would never produce a
            # result, and a non-empty job map keeps the iteration loop spinning instead of idling
            del self.jobs[job.job]
            raise
        if error is not None:
            self.error = error
            del self.jobs[job.job]
            job.put_result(error)
            return

        # Condition.notify_all() must run while holding the condition lock, so schedule a tiny coroutine instead of
        # trying to notify directly from this synchronous method.
        asyncio.create_task(self._notify_condition())

    def _enqueue(self, job):
        if self._owner_error is not None:
            return self._owner_error
        self.generator.enqueue(job)

    async def enqueue_async(self, job: AsyncJob):
        """Admit a prepared async job without blocking the caller's event loop."""
        self._check_open()
        if self.error is not None:
            job.put_result(self.error)
            return
        assert job.job not in self.jobs
        self.jobs[job.job] = job
        try:
            error = await self.run_on_owner(self._enqueue, job.job)
        except BaseException:
            self.jobs.pop(job.job, None)
            await self.cancel(job)
            raise
        if error is not None:
            self.error = error
            self.jobs.pop(job.job, None)
            job.put_result(error)
            return
        await self._notify_condition()

    async def _notify_condition(self):
        async with self.condition:
            self.condition.notify_all()

    async def close(self):
        if self._close_task is None:
            self._closing = True
            self._close_task = asyncio.create_task(self._close())
        await self._wait(self._close_task)

    async def _close(self):
        await self._notify_condition()
        try:
            await self.iteration_task
        except asyncio.CancelledError:
            pass

        # Wake any consumers still parked on job queues; no more results will be produced
        for async_job in self.jobs.values():
            async_job.discard_results()
            async_job.put_result(_CANCELLED_SENTINEL)
        self.jobs.clear()

        try:
            await self._wait(asyncio.wrap_future(self._executor.submit(self._close_generator)))
        finally:
            self._executor.shutdown(wait=True)

    def _close_generator(self):
        self.generator.close()
        self.generator._async_owner = None
        self.generator._async_submit = None

    async def cancel(self, job: AsyncJob):
        self.jobs.pop(job.job, None)
        try:
            if self._closing:
                await self._wait(self._close_task)
            else:
                await self.run_on_owner(self._cancel, job.job)
        finally:
            job.discard_results()
            job.put_result(_CANCELLED_SENTINEL)

    def _cancel(self, job):
        if self._owner_error is None:
            self.generator.cancel(job)


class AsyncJob:
    """
    Async wrapper for dynamic generator job. See definition of Job.
    """
    def __init__(self, generator: AsyncGenerator, *args: object, **kwargs: object):
        self._attach(generator, Job(*args, **kwargs))

    @classmethod
    def from_job(cls, generator: AsyncGenerator, job: Job, *, enqueue=True):
        """Wrap a prepared job, optionally leaving admission to enqueue_async."""
        value = cls.__new__(cls)
        value._attach(generator, job, enqueue=enqueue)
        return value

    def _attach(self, generator, job, *, enqueue=True):
        self.generator = generator
        self.job = job
        # The queue is unbounded and written with put_nowait: the shared iteration task must never block on a
        # consumer that stopped draining (see _run_iteration). Growth is paced by the sync generator's own token
        # rate and ends when the job completes or is cancelled, which is the frontend's responsibility on client
        # disconnect.
        self.queue = asyncio.Queue()
        self.cancelled = False
        if enqueue:
            self.generator.enqueue(self)

    def put_result(self, result):
        self.queue.put_nowait(result)

    def discard_results(self):
        while not self.queue.empty():
            _discard_result(self.queue.get_nowait())

    async def __aiter__(self):
        while True:
            # cancel() sets this flag after removing the job from the generator; the iterator exits cleanly instead
            # of waiting forever for an EOS event that will no longer be produced.
            if self.cancelled:
                break

            # Results are pushed by AsyncGenerator._run_iteration(). Exceptions are broadcast to all queues if the
            # shared generator loop fails, so consuming code sees the error at the await point. The sentinel wakes
            # this iterator when the job is cancelled from another task while parked here.
            result = await self.queue.get()
            if result is _CANCELLED_SENTINEL:
                break
            if isinstance(result, Exception):
                raise result
            yield result
            if result["eos"]:
                break

    def constrain_output_now(self, output: str | torch.Tensor):
        """
        Inject a fixed token string into the job's output stream; see Job.constrain_output_now. Safe to
        call from any coroutine on the generator's event loop. Note that tokens already sampled by the
        shared iteration task but still queued for this consumer precede the injection in the stream.
        """
        self.generator._check_open()
        self.generator._call_on_owner(self.job.constrain_output_now, output)

    def set_sampler(self, sampler):
        """
        Replace the sampler mid-generation; see Job.set_sampler. Safe to call from any coroutine on the
        generator's event loop, including the body of an `async for` over this job. Results already queued
        for this consumer were sampled before the change.
        """
        self.generator._check_open()
        self.generator._call_on_owner(self.job.set_sampler, sampler)

    def set_filters(self, filters):
        """
        Replace the filters mid-generation; see Job.set_filters and set_sampler above.
        """
        self.generator._check_open()
        self.generator._call_on_owner(self.job.set_filters, filters)

    def set_banned_strings(self, banned_strings):
        """
        Replace the banned strings mid-generation; see Job.set_banned_strings and set_sampler above.
        """
        self.generator._check_open()
        self.generator._call_on_owner(self.job.set_banned_strings, banned_strings)

    async def cancel(self):
        # Delegate cancellation to the wrapper so it can update both the sync generator queue and the async job map,
        # then mark this iterator as closed for any current or future consumers.
        await self.generator.cancel(self)
        self.cancelled = True
