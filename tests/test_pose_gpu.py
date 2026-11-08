"""GPU letterbox for YOLO-pose: same geometry and input as the Ultralytics CPU path."""
import numpy as np
import pytest
import torch

from src.detection.pose_gpu import GpuPoseModel, letterbox_geometry, pinned_frame


SHAPES = [((2160, 3840), 2560), ((720, 1280), 2560), ((200, 600), 640),
          ((1080, 1920), 1280), ((481, 853), 640)]


@pytest.mark.parametrize('shape,imgsz', SHAPES)
def test_geometry_matches_ultralytics_letterbox(shape, imgsz):
    from ultralytics.data.augment import LetterBox
    image = np.zeros((*shape, 3), dtype=np.uint8)
    reference = LetterBox((imgsz, imgsz), auto=True, stride=32)(image=image)
    (new_h, new_w), (top, bottom, left, right) = letterbox_geometry(shape[0], shape[1], (imgsz, imgsz), 32)
    assert reference.shape[:2] == (new_h + top + bottom, new_w + left + right)
    assert (new_h + top + bottom) % 32 == 0 and (new_w + left + right) % 32 == 0


def cpu_runner(imgsz):
    runner = GpuPoseModel.__new__(GpuPoseModel)
    runner.device, runner.half, runner.stride = torch.device('cpu'), False, 32
    runner.imgsz = (imgsz, imgsz)
    return runner


@pytest.mark.parametrize('shape,imgsz', [((216, 384), 256), ((200, 600), 640), ((481, 853), 640)])
def test_prepared_input_matches_ultralytics_preprocess(shape, imgsz):
    """Same RGB order, padding value and scaling; resampling within one grey level."""
    from ultralytics.data.augment import LetterBox
    rng = np.random.default_rng(0)
    # Smooth content: cv2's fixed-point bilinear and float bilinear agree to one level.
    base = rng.integers(0, 256, size=(shape[0] // 8 + 2, shape[1] // 8 + 2, 3)).astype(np.float32)
    image = np.ascontiguousarray(torch.nn.functional.interpolate(
        torch.from_numpy(base).permute(2, 0, 1)[None], size=shape, mode='bilinear',
        align_corners=False)[0].permute(1, 2, 0).round().clamp(0, 255).byte().numpy())
    reference = LetterBox((imgsz, imgsz), auto=True, stride=32)(image=image)
    expected = torch.from_numpy(np.ascontiguousarray(reference[..., ::-1].transpose(2, 0, 1))).float() / 255
    prepared = cpu_runner(imgsz)._prepare([image])[0]
    assert prepared.shape == expected.shape
    difference = (prepared - expected).abs() * 255
    # cv2 interpolates in fixed point: at most one level, on a minority of pixels
    # (4% on a real 4K frame -> 2560; up to ~12% on these small synthetic ones).
    assert float(difference.max()) <= 1.0 + 1e-4
    assert float((difference > 1e-3).float().mean()) < 0.25


def test_padding_is_ultralytics_grey():
    runner = cpu_runner(640)
    prepared = runner._prepare([np.zeros((200, 600, 3), dtype=np.uint8)])[0]
    (_, _), (top, bottom, _, _) = letterbox_geometry(200, 600, (640, 640), 32)
    assert top and bottom
    assert torch.allclose(prepared[:, :top], torch.full_like(prepared[:, :top], 114 / 255))
    assert torch.allclose(prepared[:, top + 1:-bottom - 1], torch.zeros(1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_pinned_frame_is_page_locked_and_survives_its_tensor():
    frame = pinned_frame((4, 6, 3))
    assert frame.shape == (4, 6, 3) and frame.dtype == np.uint8
    frame[:] = 7
    assert torch.from_numpy(frame).is_pinned()
    assert int(torch.from_numpy(frame).cuda().sum()) == 7 * frame.size


POSE_WEIGHTS = __import__('pathlib').Path(__file__).resolve().parents[1] / 'models/pretrained/yolo11m-pose.pt'


@pytest.mark.skipif(not torch.cuda.is_available() or not POSE_WEIGHTS.is_file(),
                    reason='needs CUDA and the local YOLO11m-pose checkpoint')
def test_cuda_graph_replay_equals_eager_network_and_payload_layout():
    runner = GpuPoseModel(POSE_WEIGHTS, 640)
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 256, size=(360, 640, 3), dtype=np.uint8)
    runner.prepare(frame.shape)
    assert not runner.capture_on_demand and len(runner._graphs) == 1
    image = runner._prepare([frame])
    with torch.inference_mode():
        eager = runner.backend(image)
        eager = (eager[0] if isinstance(eager, (list, tuple)) else eager).clone()
        graph = runner._forward(image)
        graph = graph[0] if isinstance(graph, (list, tuple)) else graph
        assert torch.equal(eager, graph)
    payload = runner([frame])[0]
    assert payload['boxes']['shape'][1:] == [6]
    assert payload['keypoints']['shape'][1:] == [17, 3]
    # A shape that was not prepared runs eagerly instead of capturing mid-run.
    runner([np.zeros((200, 600, 3), dtype=np.uint8)])
    assert len(runner._graphs) == 1
