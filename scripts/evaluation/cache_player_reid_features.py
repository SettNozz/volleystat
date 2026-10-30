"""Cache frozen OSNet features from saved poses; no identity labels or training."""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.models.player_reid_encoder import DEFAULT_ASSETS, OSNetEncoder, file_sha256
from src.tracking.player_identity_workflow import observations_from_annotations
from src.models.inference_cache import DEFAULT_CACHE,FrozenOSNet,ModelOutputCache,collect_embeddings,osnet_feature_store,video_frames
from scripts.evaluation.run_player_identity_pilot import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--side',choices=('a','b'),default='b')
    parser.add_argument('--assets',type=Path,default=DEFAULT_ASSETS)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--inspect-only',action='store_true')
    parser.add_argument('--model-cache-dir',type=Path,default=DEFAULT_CACHE)
    parser.add_argument('--require-model-cache',action='store_true')
    args=parser.parse_args()
    if args.output.exists() or args.batch_size<1 or args.threads<1:
        parser.error('Output must be fresh; batch size/threads positive')
    source=args.annotations.resolve()
    data=json.loads(source.read_text(encoding='utf-8'))
    video=Path(data['original_video_path']).resolve()
    frames=observations_from_annotations(data['frames'],args.side)
    expected=[p['key'] for f in frames for p in f['players']]
    plan={'schema':'player_reid_embedding_cache_v1','annotations':str(source),
          'annotations_sha256':file_sha256(source),'video':str(video),
          'video_sha256':file_sha256(video),'video_size':video.stat().st_size,
          'video_mtime_ns':video.stat().st_mtime_ns,'side':args.side,'fps':data['fps'],
          'width':data['frame_width'],'height':data['frame_height'],
          'source_frames':len(frames),'person_observations':len(expected),
          'label_provenance':'model detector boxes only; no identity checks or labels enter embeddings',
          'split_policy':'not a training dataset; no split or training',
          'training':False,'feature_directory':str(args.output.resolve()),
          'device':args.device,'batch_size':args.batch_size,'threads':args.threads,
          'observation_function':inspect.getsource(observations_from_annotations)}
    print(json.dumps({k:v for k,v in plan.items() if k!='observation_function'},ensure_ascii=False),flush=True)
    if args.inspect_only:
        return
    args.output.mkdir(parents=True)
    write_json(args.output/'plan.json',plan)
    write_json(args.output/'observations.json',{'fps':data['fps'],'side':args.side,'frames':frames})
    shutil.copy2(ROOT/'src/models/player_reid_encoder.py',args.output/'encoder_source.py')
    cache=ModelOutputCache(video,args.model_cache_dir,args.require_model_cache,
                           source_identity={'kind':'video','sha256':plan['video_sha256']})
    encoder=FrozenOSNet(args.assets,device=args.device,threads=args.threads,
                        cuda_memory_fraction=.1 if args.device=='cuda' else None)
    store=osnet_feature_store(cache,args.assets,args.assets/'osnet_msmt17.pth',data['frame_width'],data['frame_height'])
    started=time.monotonic()
    def progress(done,total):
        if done%300==0 or done==total:
            print(f'[cache] {done}/{total} missing source frames; {time.monotonic()-started:.1f}s',flush=True)
    features=collect_embeddings(frames,lambda needed:video_frames(video,needed,(data['frame_width'],data['frame_height'])),
                                encoder,store,args.batch_size,progress)
    values=np.stack([features[key] for key in expected])
    if len(values)!=len(expected) or values.shape[1]!=512 or not np.isfinite(values).all():
        raise ValueError('Embedding count/dimension/finiteness mismatch')
    if file_sha256(source)!=plan['annotations_sha256'] or video.stat().st_mtime_ns!=plan['video_mtime_ns']:
        raise ValueError('Source changed while caching')
    path=args.output/'embeddings.npz'
    with path.open('xb') as stream:
        np.savez_compressed(stream,keys=np.asarray(expected),embeddings=values)
    write_json(args.output/'manifest.json',{'schema':plan['schema'],'annotations_sha256':plan['annotations_sha256'],
               'observations_sha256':file_sha256(args.output/'observations.json'),
               'embedding_sha256':file_sha256(path),'count':len(values),'dimensions':512,
               'encoder':encoder.provenance,'model_cache':cache.summary(),'elapsed_seconds':time.monotonic()-started})
    cache.db.close()
    print(f'[cache] complete {values.shape} -> {path}',flush=True)


if __name__=='__main__':
    main()
