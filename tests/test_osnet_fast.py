"""The batched OSNet path must reproduce OSNetEncoder bit for bit."""

from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms

from src.models.osnet_fast import ExactOSNetPreprocess, encode_exact
from src.models.player_reid_encoder import OSNetEncoder


def _transform():
    # The exact Compose OSNetEncoder.__init__ builds.
    return transforms.Compose([transforms.Resize((256, 128)), transforms.ToTensor(),
                               transforms.Normalize([.485, .456, .406], [.229, .224, .225])])


def _crops(count=9, seed=0):
    rng = np.random.default_rng(seed)
    crops = [rng.integers(0, 256, (int(rng.integers(40, 420)), int(rng.integers(20, 200)), 3), dtype=np.uint8)
             for _ in range(count)]
    crops[0] = np.broadcast_to(np.arange(256, dtype=np.uint8)[:, None, None], (256, 64, 3)).copy()  # every level
    crops.append(np.ascontiguousarray(np.zeros((700, 300, 3), np.uint8)[::2, ::2]))  # strided source view
    return crops


def _devices():
    return ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


@pytest.mark.parametrize('device', _devices())
def test_preprocess_matches_torchvision_per_crop(device):
    transform = _transform()
    crops = _crops()
    expected = torch.stack([transform(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))) for c in crops])
    actual = ExactOSNetPreprocess(transform, device)(crops).cpu()
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert torch.equal(actual, expected)


def test_single_crop_skips_the_thread_pool():
    transform = _transform()
    crop = _crops(1)[0]
    expected = transform(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))[None]
    assert torch.equal(ExactOSNetPreprocess(transform, 'cpu')([crop]), expected)


@pytest.mark.parametrize('device', _devices())
def test_encode_exact_equals_osnet_encoder_encode(device):
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 5, stride=4), torch.nn.ReLU(),
                                torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
                                torch.nn.Linear(8, 512)).to(device).eval()
    encoder = SimpleNamespace(torch=torch, model=model, transform=_transform(), device=device)
    crops = _crops(12, seed=3)
    expected = OSNetEncoder.encode(encoder, crops)  # the unbound pinned implementation
    actual = encode_exact(encoder, crops, ExactOSNetPreprocess(encoder.transform, device))
    assert actual.dtype == expected.dtype and np.array_equal(actual, expected)
    assert encode_exact(encoder, [], None).shape == (0, 512)


def test_unsupported_transform_is_refused():
    with pytest.raises(ValueError):
        ExactOSNetPreprocess(transforms.Compose([transforms.Resize((256, 128), interpolation=transforms.InterpolationMode.NEAREST),
                                                 transforms.ToTensor(), transforms.Normalize([0, 0, 0], [1, 1, 1])]), 'cpu')
