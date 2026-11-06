import threading
import time

import pytest

from src.pipeline.model_prefetch import PrefetchJob, prefetch_model_outputs, threaded_iter


def frames(count):
    return ((index, f'image-{index}') for index in range(count))


def test_outputs_stay_in_source_order_and_skip_cached_frames():
    calls = []
    job = PrefetchJob(needed=lambda index: index % 3 != 0,
                      single=lambda image: calls.append(image) or image.upper())
    result = list(prefetch_model_outputs(frames(10), {'ball_primary': job}, chunk_size=4))
    assert [index for index, _, _ in result] == list(range(10))
    for index, image, outputs in result:
        assert image == f'image-{index}'
        if index % 3:
            assert outputs == {'ball_primary': image.upper()}
        else:
            assert outputs == {}
    assert calls == [f'image-{i}' for i in range(10) if i % 3]


def test_batch_function_receives_only_needed_frames_per_chunk():
    batches = []
    job = PrefetchJob(needed=lambda index: index != 1, single=lambda image: 'single',
                      batch=lambda images: batches.append(list(images)) or [i + '!' for i in images])
    result = list(prefetch_model_outputs(frames(4), {'ball_primary': job}, chunk_size=3))
    # Chunk [0,1,2] batches its two needed frames; chunk [3] has one, so single.
    assert batches == [['image-0', 'image-2']]
    assert result[0][2] == {'ball_primary': 'image-0!'}
    assert result[1][2] == {}
    assert result[3][2] == {'ball_primary': 'single'}


def test_several_jobs_run_on_separate_threads():
    seen = {}
    def record(name):
        def run(image):
            seen.setdefault(name, set()).add(threading.current_thread().name)
            return name
        return run
    jobs = {'ball_primary': PrefetchJob(lambda i: True, record('ball')),
            'pose_raw': PrefetchJob(lambda i: True, record('pose'))}
    result = list(prefetch_model_outputs(frames(6), jobs, chunk_size=2))
    assert all(outputs == {'ball_primary': 'ball', 'pose_raw': 'pose'} for _, _, outputs in result)
    assert seen['ball'].isdisjoint(seen['pose'])
    assert threading.main_thread().name not in seen['ball'] | seen['pose']


def test_multi_worker_job_keeps_order_and_uses_several_threads():
    names = set()
    def slow_upper(image):
        names.add(threading.current_thread().name)
        time.sleep(0.01)
        return image.upper()
    job = PrefetchJob(lambda index: index != 4, slow_upper, workers=3)
    result = list(prefetch_model_outputs(frames(12), {'pose_raw': job}, chunk_size=6))
    assert [index for index, _, _ in result] == list(range(12))
    assert all(outputs == ({} if index == 4 else {'pose_raw': image.upper()}) for index, image, outputs in result)
    assert len(names) > 1


def test_worker_error_reaches_consumer_and_threads_stop():
    def broken(image):
        raise RuntimeError('model failed')
    job = PrefetchJob(lambda index: index == 5, broken)
    before = threading.active_count()
    with pytest.raises(RuntimeError, match='model failed'):
        list(prefetch_model_outputs(frames(20), {'pose_raw': job}, chunk_size=2))
    deadline = time.time() + 5
    while threading.active_count() > before and time.time() < deadline:
        time.sleep(0.01)
    assert threading.active_count() <= before


def test_closing_early_stops_decoding_before_returning():
    produced = []
    def source():
        for index in range(1000):
            produced.append(index)
            yield index, index
    stream = prefetch_model_outputs(source(), {'ball_primary': PrefetchJob(lambda i: True, lambda image: image)},
                                    chunk_size=4, chunks_ahead=1)
    first = [next(stream) for _ in range(3)]
    stream.close()
    count = len(produced)
    time.sleep(0.2)
    assert len(produced) == count < 1000      # producer joined: nothing decoded after close
    assert [index for index, _, _ in first] == [0, 1, 2]


def test_threaded_iter_propagates_errors_and_closes_source():
    closed = []
    def source():
        try:
            yield 1
            raise ValueError('decode failed')
        finally:
            closed.append(True)
    stream = threaded_iter(source(), depth=2)
    assert next(stream) == 1
    with pytest.raises(ValueError, match='decode failed'):
        next(stream)
    assert closed == [True]


def test_threaded_iter_close_joins_producer():
    def endless():
        index = 0
        while True:
            yield index
            index += 1
    stream = threaded_iter(endless(), depth=3, name='test-decode')
    assert next(stream) == 0
    stream.close()
    assert not any(t.name == 'test-decode' for t in threading.enumerate())
