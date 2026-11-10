"""Pinned offline TrOCR reader for jersey numbers on player torso crops.

The model is a frozen scene-text recogniser (``microsoft/trocr-base-str``,
MIT licence). It reads a torso crop and returns the decoded string with its
sequence probability. Only one- or two-digit readings count as jersey numbers;
everything else, including front views without a readable number, is "no
read". Assets are installed once with ``scripts/models/install_jersey_ocr.py``;
inference never downloads.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

import cv2
import numpy as np

DEFAULT_ASSETS = Path(__file__).resolve().parents[2]/'models/jersey_ocr/trocr_base_str_v1'
REPOSITORY = 'microsoft/trocr-base-str'
REVISION = '26c151f56a7983e2cfcf05d55dc14f2f5f70ce64'
WEIGHTS_SHA256 = 'eaff179cd469b751f3afa999ca3ed0ab178373e8b46ad060342cabd8cc277085'
FILES = ('config.json', 'generation_config.json', 'preprocessor_config.json', 'pytorch_model.bin',
         'merges.txt', 'vocab.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json')
NUMBER = re.compile(r'\d{1,2}')
# COCO keypoints used for the torso box.
SHOULDERS, HIPS = (5, 6), (11, 12)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def install_assets(output: Path = DEFAULT_ASSETS, source: Path | None = None) -> dict:
    """Install the pinned revision into a new directory; never replaces assets.

    ``source`` copies an already downloaded set of the same files (for example
    after resuming a slow download); otherwise the files come from the Hub.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Assets directory exists; keep it or choose a fresh version: {output}')
    staging = output.with_name(output.name + '.partial')
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    manifest = {'repository': REPOSITORY, 'revision': REVISION, 'licence': 'MIT', 'files': {}}
    for name in FILES:
        if source is not None:
            path = staging/name
            shutil.copy2(Path(source)/name, path)
        else:
            from huggingface_hub import hf_hub_download
            path = Path(hf_hub_download(REPOSITORY, name, revision=REVISION, local_dir=staging))
        manifest['files'][name] = {'sha256': file_sha256(path), 'bytes': path.stat().st_size}
    if manifest['files']['pytorch_model.bin']['sha256'] != WEIGHTS_SHA256:
        shutil.rmtree(staging)
        raise ValueError('Pinned TrOCR weights checksum mismatch')
    shutil.rmtree(staging/'.cache', ignore_errors=True)
    with (staging/'manifest.json').open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2)
    staging.rename(output)
    return manifest


def torso_box(pose: dict, min_height: float = 28.) -> list[float] | None:
    """Shoulder-to-hip box from confident keypoints, padded sideways.

    Numbers sit on the back or chest between the shoulders and the hips. A box
    too small to hold a legible digit returns ``None``.
    """
    kp = pose.get('kp') or []
    def point(i):
        if len(kp) > i and kp[i] and len(kp[i]) >= 3 and kp[i][2] >= .3:
            return float(kp[i][0]), float(kp[i][1])
        return None
    shoulders = [p for p in map(point, SHOULDERS) if p]
    hips = [p for p in map(point, HIPS) if p]
    if len(shoulders) == 2 and hips:
        top = min(p[1] for p in shoulders)
        bottom = max(p[1] for p in hips)
        left = min(p[0] for p in shoulders + hips)
        right = max(p[0] for p in shoulders + hips)
    elif pose.get('bb'):
        x1, y1, x2, y2 = pose['bb']
        height, width = y2 - y1, x2 - x1
        top, bottom = y1 + .18*height, y1 + .55*height
        left, right = x1 + .15*width, x2 - .15*width
    else:
        return None
    height = bottom - top
    if height < min_height or right <= left:
        return None
    width = max(right - left, .55*height)
    centre = (left + right)/2
    return [centre - .6*width, top - .05*height, centre + .6*width, bottom]


class JerseyNumberReader:
    """Batch TrOCR reading; deterministic greedy decoding, no training."""

    def __init__(self, assets: Path = DEFAULT_ASSETS, device: str = 'auto', batch_size: int = 32):
        assets = Path(assets)
        if file_sha256(assets/'pytorch_model.bin') != WEIGHTS_SHA256:
            raise ValueError('TrOCR weights differ from the pinned jersey reader')
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        self.torch = torch
        self.device = 'cuda' if device == 'auto' and torch.cuda.is_available() else 'cpu' if device == 'auto' else device
        self.processor = TrOCRProcessor.from_pretrained(assets, local_files_only=True)
        dtype = torch.float16 if self.device == 'cuda' else torch.float32
        self.model = VisionEncoderDecoderModel.from_pretrained(assets, local_files_only=True, torch_dtype=dtype)
        self.model.eval().to(self.device)
        self.dtype = dtype
        self.batch_size = max(1, int(batch_size))
        self.provenance = {'method': 'trocr_torso_jersey_number_v1', 'repository': REPOSITORY,
                           'revision': REVISION, 'weights_sha256': WEIGHTS_SHA256,
                           'device': self.device, 'dtype': str(dtype).removeprefix('torch.'),
                           'decoding': 'greedy, max 4 new tokens', 'accepted': r'^\d{1,2}$',
                           'crop': 'shoulder-to-hip torso box, 1.2x width', 'training': False}

    @staticmethod
    def crop(image: np.ndarray, box) -> np.ndarray | None:
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        h, w = image.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        return image[y1:y2, x1:x2]

    def read(self, crops: list[np.ndarray]) -> list[dict]:
        """Return ``{'text', 'number', 'prob'}`` per BGR crop."""
        results = []
        torch = self.torch
        for start in range(0, len(crops), self.batch_size):
            chunk = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops[start:start + self.batch_size]]
            pixels = self.processor(images=chunk, return_tensors='pt').pixel_values.to(self.device, self.dtype)
            with torch.inference_mode():
                out = self.model.generate(pixels, max_new_tokens=4, num_beams=1, do_sample=False,
                                          output_scores=True, return_dict_in_generate=True)
            scores = self.model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
            texts = self.processor.batch_decode(out.sequences, skip_special_tokens=True)
            special = set(self.processor.tokenizer.all_special_ids)
            for row, text in enumerate(texts):
                tokens = out.sequences[row, 1:]
                keep = [i for i, t in enumerate(tokens.tolist()) if t not in special]
                logp = float(scores[row, keep].float().sum()) if keep else float('-inf')
                text = text.strip().replace(' ', '')
                number = text if NUMBER.fullmatch(text) else None
                results.append({'text': text, 'number': number, 'prob': float(np.exp(logp))})
        return results
