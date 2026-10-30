"""Inspect model caches or import a verified frozen OSNet experiment, without inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.models.inference_cache import DEFAULT_CACHE,ModelOutputCache,ObservationFeatures,digest,file_digest,model_spec
from src.models.player_reid_encoder import DEFAULT_ASSETS,SOURCE_SHA256,WEIGHTS_SHA256


def import_reid(folder, root):
    plan=json.loads((folder/'plan.json').read_text(encoding='utf-8'))
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    if plan['schema']!='player_reid_embedding_cache_v1':
        raise ValueError('Only the versioned full-video frozen OSNet cache is supported')
    video=Path(plan['video'])
    if (file_digest(video)!=plan['video_sha256']
            or file_digest(Path(plan['annotations']))!=plan['annotations_sha256']
            or manifest['annotations_sha256']!=plan['annotations_sha256']
            or file_digest(folder/'embeddings.npz')!=manifest['embedding_sha256']
            or file_digest(folder/'observations.json')!=manifest['observations_sha256']
            or file_digest(folder/'encoder_source.py')!=file_digest(ROOT/'src/models/player_reid_encoder.py')):
        raise ValueError('Legacy feature cache fingerprints differ')
    encoder=manifest['encoder']
    if (encoder['weights_sha256']!=WEIGHTS_SHA256 or encoder['code_sha256']!=SOURCE_SHA256
            or encoder['training'] is not False or encoder['strict_weight_loading'] is not True
            or encoder['input_hw']!=[256,128] or encoder['flip_average'] is not True
            or encoder['normalization']!='ImageNet mean/std'):
        raise ValueError('Unsupported encoder/preprocessing provenance')
    observations=json.loads((folder/'observations.json').read_text(encoding='utf-8'))
    rows=[(frame['f'],obs) for frame in observations['frames'] for obs in frame['players']]
    with np.load(folder/'embeddings.npz',allow_pickle=False) as archive:
        keys=archive['keys'].tolist()
        vectors=archive['embeddings']
    if (keys!=[obs['key'] for _,obs in rows] or vectors.shape!=(len(rows),512)
            or len(rows)!=manifest['count'] or not np.isfinite(vectors).all()):
        raise ValueError('Feature keys/count/dimensions differ')
    cache=ModelOutputCache(video,root,source_identity={'kind':'video','sha256':plan['video_sha256']})
    store=ObservationFeatures(cache,model_spec(DEFAULT_ASSETS/'osnet_msmt17.pth',
        backend=file_digest(ROOT/'src/models/player_reid_encoder.py'),code_sha256=SOURCE_SHA256,
        crop='clamped_rounded_full_bbox',flip_average=True),plan['width'],plan['height'])
    for index,((frame,obs),vector) in enumerate(zip(rows,vectors)):
        store.put(frame,obs['bb'],vector)
        if index%10000==0:
            print(f'Imported {index}/{len(rows)} frozen crops',flush=True)
    cache.put('imports',manifest['embedding_sha256'],{
        'source':str(folder.resolve()),'manifest_sha256':file_digest(folder/'manifest.json'),
        'original_encoder_provenance':encoder,'count':len(rows),
        'note':'Explicit frozen-artifact import; legacy runtime is retained as recorded, not inferred'})
    print(json.dumps(cache.summary()),flush=True)
    cache.db.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    imported=commands.add_parser('import-reid',help='Verify and reuse an existing frozen feature artifact; zero model calls')
    imported.add_argument('--cache',type=Path,required=True)
    imported.add_argument('--model-cache-dir',type=Path,default=DEFAULT_CACHE)
    inspected=commands.add_parser('inspect',help='Read stage coverage; no inference or cache creation')
    inspected.add_argument('--video',type=Path,required=True)
    inspected.add_argument('--model-cache-dir',type=Path,default=DEFAULT_CACHE)
    args=parser.parse_args()
    if args.command=='import-reid':
        import_reid(args.cache,args.model_cache_dir)
    else:
        identity={'kind':'video','sha256':file_digest(args.video)}
        directory=args.model_cache_dir/digest(identity)
        if not (directory/'outputs.sqlite3').exists():
            print(json.dumps({'source':identity,'cache_available':False}))
            return
        cache=ModelOutputCache(args.video,args.model_cache_dir,True,source_identity=identity)
        stages=[{'stage':stage,'items':count} for stage,count in cache.db.execute(
            'SELECT stage,count(*) FROM outputs GROUP BY stage')]
        print(json.dumps({'source':identity,'directory':str(directory),'stages':stages},indent=2))
        cache.db.close()


if __name__=='__main__':
    main()
