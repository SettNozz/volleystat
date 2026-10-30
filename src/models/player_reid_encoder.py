"""Pinned offline OSNet encoder shared by production and identity experiments."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np

DEFAULT_ASSETS = Path(__file__).resolve().parents[2]/'models/player_reid/osnet_msmt17_v1'
SOURCE_SHA256 = 'c7c1c29187d6330f859c91da229271531920464c7011aec13842a086b2263cae'
WEIGHTS_SHA256 = '48df972f72887b95cf3b43b3a07c3a7d2398381aea0f9cae64a7ef11d512b727'
SOURCE_REVISION = 'f8cd150fdf77e8d9e1ed143b7f308c2c609ded50'
WEIGHTS_REVISION = 'a5c5cc037c24235cda3b21085b93ad77c9616224'
LICENSE_SHA256 = '3ac8ce2a83d170cb1c7c84152e0c1faca1f187794303514383960d2441716247'


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def asset_urls():
    weight = 'osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth'
    return {
        'osnet.py': (f'https://raw.githubusercontent.com/KaiyangZhou/deep-person-reid/{SOURCE_REVISION}/torchreid/models/osnet.py', SOURCE_SHA256),
        'LICENSE': (f'https://raw.githubusercontent.com/KaiyangZhou/deep-person-reid/{SOURCE_REVISION}/LICENSE', LICENSE_SHA256),
        'osnet_msmt17.pth': (f'https://huggingface.co/kaiyangzhou/osnet/resolve/{WEIGHTS_REVISION}/{weight}', WEIGHTS_SHA256),
    }


def install_assets(output: Path, source: Path | None = None) -> dict:
    """Explicit setup; refuses replacement of existing assets. No inference download."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Assets directory exists; keep it or choose a fresh version: {output}')
    # Validate/copy bytes before creating a destination; existing models stay intact.
    files = {}
    for name, (url, expected) in asset_urls().items():
        if source is not None:
            data = (Path(source)/name).read_bytes()
        else:
            import requests
            response = requests.get(url, timeout=90)
            response.raise_for_status()
            data = response.content
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f'Pinned OSNet asset checksum mismatch: {name}')
        files[name] = data
    output.mkdir(parents=True)
    manifest = {'architecture': 'osnet_x1_0', 'pretraining': 'MSMT17 combineall',
                'github_revision': SOURCE_REVISION, 'huggingface_revision': WEIGHTS_REVISION, 'files': {}}
    for name, data in files.items():
        url, expected = asset_urls()[name]
        with (output/name).open('xb') as stream:
            stream.write(data)
        manifest['files'][name] = {'url':url,'sha256':expected,'bytes':len(data)}
    with (output/'manifest.json').open('x',encoding='utf-8') as stream:
        json.dump(manifest,stream,indent=2)
    return manifest


class OSNetEncoder:
    """Strict pretrained features, full boxes and original/flip averaging.

    Does not change global CUDA memory limits, thread counts or random state.
    Production shares its process with YOLO and other loaded models.
    """
    def __init__(self, assets: Path = DEFAULT_ASSETS, weights: Path | None = None, device: str = 'auto'):
        assets = Path(assets)
        weights = Path(weights) if weights is not None else assets/'osnet_msmt17.pth'
        if file_sha256(assets/'osnet.py') != SOURCE_SHA256 or file_sha256(weights) != WEIGHTS_SHA256:
            raise ValueError('OSNet code/weights differ from the pinned supported encoder')
        import torch
        from torchvision import transforms
        self.torch = torch
        self.device = 'cuda' if device == 'auto' and torch.cuda.is_available() else 'cpu' if device == 'auto' else device
        if self.device not in ('cpu','cuda'):
            raise ValueError('ReID device must be auto, cpu or cuda')
        spec = importlib.util.spec_from_file_location('volleystat_pinned_osnet', assets/'osnet.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        state = torch.load(weights,map_location='cpu',weights_only=True)
        with torch.random.fork_rng(devices=[]):
            self.model = module.osnet_x1_0(num_classes=int(state['classifier.weight'].shape[0]),pretrained=False)
        self.model.load_state_dict(state,strict=True)
        self.model.eval().to(self.device)
        self.transform = transforms.Compose([transforms.Resize((256,128)),transforms.ToTensor(),
                                             transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
        self.provenance = {'method':'osnet_frozen_gallery_recovery_v1','architecture':'osnet_x1_0',
                           'pretraining':'MSMT17 combineall','weights_path':str(weights.resolve()),
                           'weights_sha256':WEIGHTS_SHA256,'code_path':str((assets/'osnet.py').resolve()),
                           'code_sha256':SOURCE_SHA256,'device':self.device,'input_hw':[256,128],
                           'normalization':'ImageNet mean/std','flip_average':True,'strict_weight_loading':True,
                           'training':False}

    @staticmethod
    def crop(image: np.ndarray, box) -> np.ndarray:
        x1,y1,x2,y2 = map(round,box)
        left,right = max(0,min(image.shape[1],x1)),max(0,min(image.shape[1],x2))
        top,bottom = max(0,min(image.shape[0],y1)),max(0,min(image.shape[0],y2))
        crop = image[top:bottom,left:right]
        if crop.size == 0:
            raise ValueError('Empty player crop')
        return crop

    def encode(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.empty((0,512),dtype=np.float32)
        from PIL import Image
        torch = self.torch
        batch = torch.stack([self.transform(Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)))
                             for crop in crops]).to(self.device)
        with torch.inference_mode():
            original = torch.nn.functional.normalize(self.model(batch),dim=1)
            flipped = torch.nn.functional.normalize(self.model(torch.flip(batch,dims=[3])),dim=1)
            values = torch.nn.functional.normalize(original+flipped,dim=1).cpu().numpy()
        if not np.isfinite(values).all():
            raise ValueError('Non-finite OSNet embeddings')
        return values
