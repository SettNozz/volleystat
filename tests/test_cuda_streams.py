import threading

import pytest
import torch

from src.utils import cuda_streams


def test_profile_timer_no_longer_synchronizes(monkeypatch):
    from ultralytics.utils import ops
    monkeypatch.setattr(ops.Profile, 'time', ops.Profile.time)
    monkeypatch.setattr(ops.Profile, '_volleystat_no_device_sync', False, raising=False)
    calls = []
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *a, **k: calls.append(a))
    with ops.Profile(device='cuda:0') as profile:
        pass
    assert calls  # the stock timer is a device-wide barrier
    calls.clear()
    cuda_streams.disable_ultralytics_device_sync()
    cuda_streams.disable_ultralytics_device_sync()  # idempotent
    with ops.Profile(device='cuda:0') as profile:
        pass
    assert not calls and profile.dt >= 0


def test_own_stream_is_a_no_op_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with cuda_streams.own_stream():
        assert torch.zeros(2).sum().item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_own_stream_is_per_thread_and_reused():
    seen = {}

    def run(name):
        with cuda_streams.own_stream():
            first = torch.cuda.current_stream()
        with cuda_streams.own_stream():
            second = torch.cuda.current_stream()
        seen[name] = (first, second)

    threads = [threading.Thread(target=run, args=(n,)) for n in 'ab']
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    (a1, a2), (b1, b2) = seen['a'], seen['b']
    assert a1 == a2 and b1 == b2 and a1 != b1
    assert a1 != torch.cuda.default_stream()
    with cuda_streams.own_stream(priority=-1):
        assert torch.cuda.current_stream().priority <= 0
