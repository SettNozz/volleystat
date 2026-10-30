"""Version candidate features from frozen skeletons without detector inference."""
from __future__ import annotations
import argparse,copy,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from action_recognition.event_clips.team_pose_features import CANDIDATE_SCHEMA,candidate_team_features,contact_candidate
from src.detection.player_pose import PlayerPose
from src.tracking.pose_feature_extractor import PoseFeatureExtractor
from src.detection.sam_pose_recovery import file_hash
from scripts.evaluation.refresh_team_pose_dataset import save


def rebuild(source,output):
    source=Path(source);output=Path(output)
    meta=json.loads((source/'preflight.json').read_text())
    if meta['status']!='complete':raise ValueError('Parent dataset incomplete')
    manifest=json.loads((source/'manifest.json').read_text())
    if any(c.get('label_source')!='human' for c in manifest['clips']):raise ValueError('Expected human labels')
    if output.exists():raise ValueError('Refuse existing output')
    print(json.dumps(dict(parent=str(source),output=str(output),features=str(output/'features'),
        sources=sorted(meta['sources']),schema=CANDIDATE_SCHEMA,clips=len(manifest['clips']),
        label_source='human',split_policy=meta['split_policy']),indent=2),flush=True)
    output.mkdir();(output/'features').mkdir()
    # Snapshot historical inputs for the existing evaluation contract; never modify parent.
    import shutil
    shutil.copytree(source/'baseline_features_snapshot',output/'baseline_features_snapshot')
    for name in ('parent_manifest.json','split_policy.json','geometry.json'):
        shutil.copy2(source/name,output/name)
    needed={}
    for c in manifest['clips']:
        needed.setdefault(c['source_stem'],set()).update(range(c['start_frame'],c['end_frame']+1))
    cache_hashes={};rows={}
    for stem,frames in needed.items():
        rows[stem]={}
        for path in sorted((source/'poses').glob(stem+'_s*.jsonl')):
            cache_hashes[str(path)]=file_hash(path);fx=PoseFeatureExtractor()
            with path.open(encoding='utf-8') as stream:
                for line in stream:
                    row=json.loads(line);f=row['f'];ball=row['features_context']['ball']
                    poses=[PlayerPose(p['team'],((p['bb'][0]+p['bb'][2])/2,(p['bb'][1]+p['bb'][3])/2),
                        tuple(p['bb']),np.asarray(p['kp'],np.float32),track_id=p['id']) for p in row['poses']]
                    fx.update(f,poses,ball,meta['geometry']['net_y'])
                    if f not in frames:continue
                    values,valid=candidate_team_features(poses,ball,meta['geometry']['net_y'],fx)
                    ids=[]
                    for team in ('team_a','team_b'):
                        candidate=contact_candidate([p for p in poses if p.team==team],ball)
                        ids.append(candidate.track_id if candidate is not None and candidate.track_id is not None else -1)
                    rows[stem][f]=(values,valid,ids)
        if set(rows[stem])!=frames:raise ValueError('Incomplete skeleton coverage: '+stem)
    hashes={}
    for clip in manifest['clips']:
        path=source/'features'/(clip['clip_id']+'.npz');hashes[clip['clip_id']]=file_hash(path)
        with np.load(path,allow_pickle=False) as data:arrays={k:data[k].copy() for k in data.files}
        selected=[rows[clip['source_stem']][int(f)] for f in arrays['frames']]
        arrays['features'][:,18:]=np.array([r[0] for r in selected])
        arrays['valid'][:,18:]=np.array([r[1] for r in selected])
        arrays['features'][~arrays['valid']]=0
        arrays['schema']=np.array(CANDIDATE_SCHEMA)
        arrays['candidate_track_ids']=np.array([r[2] for r in selected],np.int64)
        np.savez_compressed(output/'features'/path.name,**arrays)
        clip['feature_schema']=CANDIDATE_SCHEMA
    manifest['feature_schema']=CANDIDATE_SCHEMA
    manifest['candidate_policy']='Per-side nearest observed anatomical point/forearm in image pixels; all contact channels from that one person'
    save(output/'manifest.json',manifest)
    meta=copy.deepcopy(meta);meta.update(status='complete',feature_schema=CANDIDATE_SCHEMA,
        manifest=str((output/'manifest.json').resolve()),features=str((output/'features').resolve()),
        derived_from=str(source.resolve()),parent_feature_hashes=hashes,skeleton_hashes=cache_hashes,
        changed_channels='18:42 only; original trajectory and nearest-player block retained as matched control',
        sam_masks_used=False)
    save(output/'preflight.json',meta)
    save(output/'features/feature_schema.json',dict(feature_schema=CANDIDATE_SCHEMA,value_names=meta['value_names'],
        feature_names=meta['input_features'],input_dim=84,feature_files=len(manifest['clips']),label_source='human'))
    print('COMPLETE',len(manifest['clips']),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',required=True,type=Path);ap.add_argument('--output',required=True,type=Path)
    args=ap.parse_args();rebuild(args.source,args.output)
