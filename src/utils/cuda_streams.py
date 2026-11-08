"""Per-thread CUDA streams and a non-blocking Ultralytics timer.

The analysis loop runs the ball detector, the pose graphs, OSNet and the
event model from different Python threads. Work issued without a stream goes
to the legacy default stream, which every thread shares, so a small event-model
call on the main thread queued behind a whole ball batch and every ``.cpu()``
waited for it. Ultralytics also times each predictor stage with
``torch.cuda.synchronize()``, a device-wide barrier that made the ball thread
wait for the pose and OSNet kernels of the other threads too.

Neither changes a value: the same kernels run on another queue, and the
timer only reports elapsed time.
"""

from __future__ import annotations

import threading
import time
from contextlib import nullcontext

_local = threading.local()


def own_stream(priority: int = 0):
    """Context that sends this thread's CUDA work to its own stream.

    One non-blocking stream per thread and priority, reused across calls. Keep
    tensors that are created inside the context inside it: the caching
    allocator ties a block to the stream that allocated it.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency in practice
        return nullcontext()
    if not torch.cuda.is_available():
        return nullcontext()
    streams = getattr(_local, 'streams', None)
    if streams is None:
        streams = _local.streams = {}
    stream = streams.get(priority)
    if stream is None:
        low, high = torch.cuda.Stream.priority_range()  # (0, -N): lower value = higher priority
        stream = streams[priority] = torch.cuda.Stream(priority=max(min(priority, low), high))
        # Anything already queued for the inputs on the default stream comes first.
        stream.wait_stream(torch.cuda.current_stream())
    return torch.cuda.stream(stream)


def disable_ultralytics_device_sync() -> None:
    """Make ``ultralytics.utils.ops.Profile`` time without a device-wide barrier.

    Its ``torch.cuda.synchronize()`` only makes the per-stage ``speed`` numbers
    exact; results are already copied to the host with stream-ordered
    ``.cpu()`` calls. Process-wide and idempotent.
    """
    from ultralytics.utils import ops
    if getattr(ops.Profile, '_volleystat_no_device_sync', False):
        return

    def _time(self):
        return time.perf_counter()

    ops.Profile.time = _time
    ops.Profile._volleystat_no_device_sync = True
