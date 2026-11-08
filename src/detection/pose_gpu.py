"""YOLO-pose inference with the letterbox preprocessing done on the GPU.

Ultralytics resizes a 4K frame to the 2560 input with ``cv2.resize`` on the
CPU, then flips, transposes and uploads it. At 4K that costs about 30 ms per
frame, more than the network itself on an RTX 5070 Ti, and it holds the GIL
long enough that extra Python threads stop helping. Here the decoded uint8
frame is uploaded once and every step after that runs on the device.

The geometry is the Ultralytics ``LetterBox`` used by ``predict`` for a
PyTorch checkpoint (minimum stride rectangle, centred 114 padding) and the
postprocess is its ``non_max_suppression`` / ``scale_boxes`` /
``scale_coords``, so the raw output keeps the same layout as
``pack_pose_result``: ``boxes`` (N, 6) and ``keypoints`` (N, 17, 3) in source
pixels. Values are not bit-identical to the CPU path: bilinear resampling is
done in float32 and rounded to the uint8 grid instead of cv2's fixed-point
interpolation. That is why ``PREPROCESS`` is part of the pose cache key.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

PREPROCESS = 'gpu_letterbox_bilinear_round_v1'
PAD_VALUE = 114.0


def letterbox_geometry(height: int, width: int, imgsz: Tuple[int, int], stride: int):
    """Ultralytics LetterBox(auto=True, center=True) sizes for one frame.

    Returns ``(new_h, new_w), (top, bottom, left, right)``.
    """
    r = min(imgsz[0] / height, imgsz[1] / width)
    new_w, new_h = int(round(width * r)), int(round(height * r))
    dw, dh = (imgsz[1] - new_w) % stride, (imgsz[0] - new_h) % stride
    dw, dh = dw / 2, dh / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return (new_h, new_w), (top, bottom, left, right)


class GpuPoseModel:
    """A YOLO-pose checkpoint whose whole input path runs on one device."""

    def __init__(self, weights, imgsz: int, conf: float = 0.1, iou: float = 0.7,
                 max_det: int = 300, device=None, half=None):
        import torch
        from ultralytics.nn.autobackend import AutoBackend
        from ultralytics.utils.checks import check_imgsz

        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.half = (self.device.type == 'cuda') if half is None else bool(half)
        self.backend = AutoBackend(weights=str(weights), device=self.device, fp16=self.half,
                                   fuse=True, verbose=False)
        self.backend.eval()
        self.stride = int(max(int(self.backend.stride), 32)) if self.backend.pt else 32
        self.imgsz = tuple(check_imgsz(imgsz, stride=self.backend.stride, min_dim=2))
        self.conf, self.iou, self.max_det = float(conf), float(iou), int(max_det)
        self.nc = len(self.backend.names)
        self.kpt_shape = tuple(self.backend.kpt_shape)
        self.end2end = bool(getattr(self.backend, 'end2end', False))
        self._stream = torch.cuda.Stream(self.device) if self.device.type == 'cuda' else None
        self._graphs = {}
        # Capturing while another thread uses the legacy default stream (the
        # ball detector does) invalidates both, so a pipeline captures up front
        # with prepare() and then never captures again.
        self.capture_on_demand = True

    def prepare(self, frame_shape) -> None:
        """Capture the graph for frames of ``frame_shape`` before other CUDA threads start."""
        self([np.zeros(tuple(int(v) for v in frame_shape), dtype=np.uint8)])
        self.capture_on_demand = False

    def _forward(self, image):
        """Network output for one input shape, replayed from a CUDA graph.

        Eager YOLO11m dispatch costs ~22 ms of Python (GIL-holding) work per
        2560 frame, as much as the GPU math, so worker threads could not
        overlap. A captured graph launches the same kernels in ~0.1 ms with
        bit-identical outputs (measured max |diff| 0.0). Outputs live in the
        graph's static buffers and are consumed before the next replay.
        """
        import torch
        if self.device.type != 'cuda':
            return self.backend(image)
        key = tuple(image.shape), image.dtype
        entry = self._graphs.get(key)
        if entry is None and not self.capture_on_demand:
            return self.backend(image)
        if entry is None:
            static = image.clone()
            side = torch.cuda.Stream(self.device)
            side.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(side):
                for _ in range(2):
                    self.backend(static)
            torch.cuda.current_stream(self.device).wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            # Other pipeline threads keep using CUDA while this one captures.
            with torch.cuda.graph(graph, capture_error_mode='thread_local'):
                output = self.backend(static)
            entry = self._graphs[key] = (graph, static, output)
        graph, static, output = entry
        static.copy_(image)
        graph.replay()
        return output

    def _prepare(self, frames: Sequence[np.ndarray]):
        import torch
        import torch.nn.functional as F
        height, width = frames[0].shape[:2]
        (new_h, new_w), (top, bottom, left, right) = letterbox_geometry(height, width, self.imgsz, self.stride)
        # A frame decoded into pinned memory (pinned_frame) uploads by async DMA;
        # a pageable one falls back to a synchronous staged copy.
        x = torch.stack([torch.from_numpy(np.ascontiguousarray(frame)).to(self.device, non_blocking=True)
                         for frame in frames])                  # B,H,W,3 uint8 BGR
        x = x.permute(0, 3, 1, 2).flip(1).float()               # B,3,H,W RGB
        if (new_h, new_w) != (height, width):
            x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
            x = x.round_().clamp_(0, 255)
        if top or bottom or left or right:
            x = F.pad(x, (left, right, top, bottom), value=PAD_VALUE)
        x = x.half() if self.half else x
        return x.div_(255)

    def __call__(self, frames: Sequence[np.ndarray]) -> List[dict]:
        """Raw pose outputs for same-shaped BGR frames, one payload per frame."""
        import torch
        from ultralytics.utils import ops
        from src.models.inference_cache import array_payload
        if not len(frames):
            return []
        shapes = {frame.shape for frame in frames}
        if len(shapes) != 1:
            return [value for frame in frames for value in self([frame])]
        stream = torch.cuda.stream(self._stream) if self._stream is not None else _null()
        with torch.inference_mode(), stream:
            image = self._prepare(frames)
            preds = self._forward(image)
            preds = preds[0] if isinstance(preds, (list, tuple)) else preds
            dets = ops.non_max_suppression(preds.float(), self.conf, self.iou, None, False,
                                           max_det=self.max_det, nc=self.nc, end2end=self.end2end)
            out = []
            for det, frame in zip(dets, frames):
                det = det.clone()
                det[:, :4] = ops.scale_boxes(image.shape[2:], det[:, :4], frame.shape)
                kpts = det[:, 6:].view(len(det), *self.kpt_shape)
                kpts = ops.scale_coords(image.shape[2:], kpts, frame.shape)
                out.append((det[:, :6], kpts))
            out = [(boxes.cpu().numpy(), kpts.cpu().numpy()) for boxes, kpts in out]
        return [{'boxes': array_payload(boxes), 'keypoints': array_payload(kpts)} for boxes, kpts in out]


def pinned_frame(shape) -> np.ndarray:
    """An uninitialised uint8 frame in page-locked host memory.

    ``cv2.VideoCapture.read(image)`` decodes into it in place, which saves the
    25 MB host copy a pageable 4K frame needs before its GPU upload (measured
    5.7 ms -> 2.5 ms per frame on PCIe 3.0 x16). The array keeps its tensor,
    so the block returns to PyTorch's pinned cache when the frame is dropped.
    """
    import torch
    return torch.empty(tuple(int(v) for v in shape), dtype=torch.uint8, pin_memory=True).numpy()


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
