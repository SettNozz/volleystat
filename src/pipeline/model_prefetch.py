"""Overlap frame decoding and stateless model inference with the serial analysis loop.

The analysis loop is strictly ordered: ByteTrack, the live ball gate and the FSM
all depend on the previous frame. The neural models feeding them do not. Raw
ball proposals and raw YOLO-pose detections are pure per-frame functions, which
is exactly why the model-output cache can store them individually.

This module computes those outputs ahead of the loop on background threads:

* one thread decodes frames into a bounded buffer;
* one thread per model job (ball, tiled ball, pose) runs inference on the
  frames the cache does not hold yet, one chunk at a time, in source order.

Workers call the same model methods with the same batch size, input size and
precision as the serial path, so cached values and cache keys are unchanged.
They never touch the SQLite cache (its connection belongs to the main thread)
and never touch tracker, gate or FSM state. The main thread receives
``(frame_index, image, {job: value})`` in source order and stores/consumes the
values exactly as if it had computed them itself.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import queue
import threading
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

_END = object()


class _Failure:
    def __init__(self, error: BaseException):
        self.error = error


def threaded_iter(iterable: Iterable, depth: int, name: str = 'prefetch') -> Iterator:
    """Run ``iterable`` on a background thread behind a bounded buffer.

    Meant for video decoding, whose native read releases the GIL. Closing the
    returned generator (or exhausting it) stops and joins the producer before
    returning, so the caller may reuse the underlying capture afterwards.
    """
    buffer: queue.Queue = queue.Queue(maxsize=max(1, int(depth)))
    stop = threading.Event()

    def offer(item) -> bool:
        while not stop.is_set():
            try:
                buffer.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def produce():
        try:
            for item in iterable:
                if not offer(item):
                    return
            offer(_END)
        except BaseException as error:  # surfaced on the consumer thread
            offer(_Failure(error))
        finally:
            close = getattr(iterable, 'close', None)
            if close is not None:
                try:
                    close()
                except Exception:
                    pass

    thread = threading.Thread(target=produce, name=name, daemon=True)
    thread.start()
    try:
        while True:
            item = buffer.get()
            if item is _END:
                return
            if isinstance(item, _Failure):
                raise item.error
            yield item
    finally:
        stop.set()
        while thread.is_alive():
            try:
                buffer.get_nowait()
            except queue.Empty:
                thread.join(0.05)


@dataclass
class PrefetchJob:
    """One stateless model output computed ahead of the analysis loop.

    ``needed(frame_index)`` says whether the frame lacks a cached value.
    ``batch(images)`` returns one value per image; when ``None``, ``single``
    is called per image. Both run only on this job's own worker threads.
    With ``workers > 1`` frames are spread over that many threads, so
    ``single`` must be safe to call concurrently (e.g. one model per thread);
    ``batch`` is ignored then.
    """
    needed: Callable[[int], bool]
    single: Callable[[Any], Any]
    batch: Optional[Callable[[List[Any]], Sequence[Any]]] = None
    workers: int = 1


def _run_job(job: PrefetchJob, chunk: List[Tuple[int, Any]]) -> Dict[int, Any]:
    wanted = [(index, image) for index, image in chunk if job.needed(index)]
    if not wanted:
        return {}
    if job.batch is not None and len(wanted) > 1:
        values = list(job.batch([image for _, image in wanted]))
    else:
        values = [job.single(image) for _, image in wanted]
    if len(values) != len(wanted):
        raise ValueError('Prefetched model output count differs from input frames')
    return {index: value for (index, _), value in zip(wanted, values)}


def prefetch_model_outputs(
    frames: Iterable[Tuple[int, Any]],
    jobs: Dict[str, PrefetchJob],
    chunk_size: int,
    chunks_ahead: int = 2,
    decode_depth: Optional[int] = None,
) -> Iterator[Tuple[int, Any, Dict[str, Any]]]:
    """Yield ``(index, image, outputs)`` in source order with inference run ahead.

    ``outputs`` maps job name to the value computed for that frame, and omits
    jobs for which the frame was not ``needed``. At most ``chunks_ahead`` chunks
    beyond the one being consumed are decoded and in flight.
    """
    chunk_size = max(1, int(chunk_size))
    decoded = threaded_iter(frames, decode_depth or chunk_size, name='frame-decode')
    executors = {name: ThreadPoolExecutor(max_workers=max(1, job.workers), thread_name_prefix=f'prefetch-{name}')
                 for name, job in jobs.items()}
    pending: deque = deque()

    def submit(chunk):
        futures = {}
        for name, job in jobs.items():
            if job.workers > 1:
                futures[name] = {index: executors[name].submit(job.single, image)
                                 for index, image in chunk if job.needed(index)}
            else:
                futures[name] = executors[name].submit(_run_job, job, chunk)
        pending.append((chunk, futures))

    def collect(future):
        if isinstance(future, dict):
            return {index: item.result() for index, item in future.items()}
        return future.result()

    def release():
        chunk, futures = pending.popleft()
        results = {name: collect(future) for name, future in futures.items()}
        for index, image in chunk:
            yield index, image, {name: values[index] for name, values in results.items() if index in values}

    try:
        chunk: List[Tuple[int, Any]] = []
        for item in decoded:
            chunk.append(item)
            if len(chunk) >= chunk_size:
                submit(chunk)
                chunk = []
                if len(pending) > chunks_ahead:
                    yield from release()
        if chunk:
            submit(chunk)
        while pending:
            yield from release()
    finally:
        for _, futures in pending:
            for future in futures.values():
                for item in (future.values() if isinstance(future, dict) else [future]):
                    item.cancel()
        for executor in executors.values():
            executor.shutdown(wait=True)
        decoded.close()
