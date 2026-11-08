"""Faster OSNet encoding with values bit-identical to ``OSNetEncoder.encode``.

``OSNetEncoder`` (hashed into the OSNet cache key, so it stays untouched)
prepares every crop with torchvision on the CPU: BGR->RGB, PIL bilinear resize
to 256x128, ``ToTensor`` and ``Normalize``, one small tensor at a time. On a
cold 4K run that took ~1.2 ms of mostly GIL-holding Python per crop, about 70%
of the background OSNet thread, and the main loop stalled behind it.

Here only the PIL resize stays on the CPU, and it runs on a few threads (PIL
releases the GIL while resampling). The resized uint8 crops go to the device
in one copy and become normalized floats through a 3x256 lookup table built
with torchvision's own CPU operations, so every value is exactly what
``ToTensor`` + ``Normalize`` produce. Doing that arithmetic on CUDA instead is
not exact: a division by a Python scalar becomes a multiplication by its
reciprocal there (1 ulp off). The network part is ``OSNetEncoder.encode``'s,
call for call, so with the same batch the embeddings are the same bits.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

import numpy as np

RESIZE_THREADS = 4
_pool: ThreadPoolExecutor | None = None


def _resize_pool() -> ThreadPoolExecutor:
    global _pool
    if _pool is None:
        _pool = ThreadPoolExecutor(RESIZE_THREADS, thread_name_prefix='osnet-resize')
    return _pool


def _resize_bgr(crop: np.ndarray, size) -> np.ndarray:
    from PIL import Image
    # PIL resamples each band on its own, so resizing BGR and swapping the
    # channels afterwards equals resizing the RGB conversion.
    return np.asarray(Image.fromarray(np.ascontiguousarray(crop)).resize(size, Image.BILINEAR))


class ExactOSNetPreprocess:
    """Batched replica of ``OSNetEncoder.transform`` applied to BGR crops."""

    def __init__(self, transform, device):
        import torch
        from PIL import Image
        from torchvision import transforms
        resize, to_tensor, normalize = transform.transforms
        if not (isinstance(resize, transforms.Resize) and isinstance(to_tensor, transforms.ToTensor)
                and isinstance(normalize, transforms.Normalize)):
            raise ValueError('Unsupported OSNet transform for the exact batched path')
        if resize.interpolation != transforms.InterpolationMode.BILINEAR or resize.antialias is False:
            raise ValueError('Unsupported OSNet resize for the exact batched path')
        height, width = resize.size
        self.size = (int(width), int(height))  # PIL order
        # Each table entry goes through the same torchvision calls as a pixel.
        levels = np.repeat(np.arange(256, dtype=np.uint8)[None, :, None], 3, axis=2)  # 1x256x3
        table = normalize(to_tensor(Image.fromarray(levels)))  # 3x1x256
        self.table = table.reshape(3, 256).to(device)
        self.channels = torch.arange(3, device=device).view(1, 3, 1, 1)
        self.device = device
        if torch.device(device).type == 'cuda':
            torch.cuda.current_stream(device).synchronize()  # tables are read from other streams

    def resize(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        if len(crops) == 1:
            return np.stack([_resize_bgr(crops[0], self.size)])
        return np.stack(list(_resize_pool().map(lambda crop: _resize_bgr(crop, self.size), crops)))

    def __call__(self, crops: Sequence[np.ndarray]):
        import torch
        resized = torch.from_numpy(self.resize(crops)).to(self.device)  # N,H,W,3 BGR uint8
        index = resized.permute(0, 3, 1, 2).flip(1).long()           # N,3,H,W RGB levels
        # Contiguous NCHW like torch.stack: the gather keeps the index's
        # permuted strides, and a channels-last input selects other kernels.
        return self.table[self.channels, index].contiguous()


def encode_exact(encoder, crops: Sequence[np.ndarray], preprocess: ExactOSNetPreprocess) -> np.ndarray:
    """``OSNetEncoder.encode`` with the exact batched preprocessing."""
    if not len(crops):
        return np.empty((0, 512), dtype=np.float32)
    from src.utils.cuda_streams import own_stream
    torch = encoder.torch
    # Own stream: the final .cpu() then waits for this batch only, not for the
    # ball or event work other threads queued on the default stream.
    with own_stream(), torch.inference_mode():
        batch = preprocess(crops)
        original = torch.nn.functional.normalize(encoder.model(batch), dim=1)
        flipped = torch.nn.functional.normalize(encoder.model(torch.flip(batch, dims=[3])), dim=1)
        values = torch.nn.functional.normalize(original + flipped, dim=1).cpu().numpy()
    if not np.isfinite(values).all():
        raise ValueError('Non-finite OSNet embeddings')
    return values
