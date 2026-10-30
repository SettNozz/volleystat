"""Add production OSNet IDs to a fresh copy of saved results, or export a human seed."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

import cv2
from src.models.player_reid_encoder import file_sha256
from src.models.inference_cache import DEFAULT_CACHE,ModelOutputCache
from src.pipeline.game_analyzer import GameAnalyzer, GameStats
from src.tracking.player_identity_workflow import observations_from_annotations


def export_seed(project:Path,output:Path):
    if output.exists():
        raise FileExistsError('Seed output exists; choose a fresh filename')
    observations_path=project/'observations.json'
    observations=json.loads(observations_path.read_text(encoding='utf-8'))
    labels_path=project/'labels.json'
    labels=json.loads(labels_path.read_text(encoding='utf-8'))
    if labels['observations_sha256']!=file_sha256(observations_path):
        raise ValueError('Review labels do not match observations')
    seed=labels.get('seed') or {}
    if seed.get('label_source')!='human' or seed.get('verified') is not True:
        raise ValueError('Confirm initial human roster IDs first')
    source=observations['source']
    if source['annotations_sha256']!=file_sha256(Path(source['annotations'])):
        raise ValueError('Source analysis changed since human initialisation')
    video=Path(source['video'])
    by_key={p['key']:p for f in observations['frames'] for p in f['players']}
    payload={'schema':'player_identity_seed_v1','video_sha256':file_sha256(video),
             'video_name':video.name,'frame':seed['frame'],'side':observations['side'],
             'label_source':'human','verified':True,
             'players':[{'player_id':pid,'bb':by_key[key]['bb']} for key,pid in seed['assignments'].items()],
             'provenance':{'review_labels_sha256':file_sha256(labels_path),
                           'observations_sha256':labels['observations_sha256'],
                           'mapping':'human identity labels with observed detector boxes; checks excluded'}}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:
        json.dump(payload,stream,ensure_ascii=False,indent=2)


def add_player(args):
    """Bind a human-confirmed new person to an observed box, not to a raw ID."""
    if args.output.exists():
        raise FileExistsError('Roster output exists; choose a fresh filename')
    if not args.confirmation_note.strip() or not args.new_id.strip() or len(args.new_id)>40:
        raise ValueError('New ID and explicit human confirmation note are required')
    data=json.loads(args.annotations.read_text(encoding='utf-8'))
    metadata=data.get('player_identity') or {}
    side=metadata.get('side','b')
    frames=observations_from_annotations(data['frames'],side)
    by_frame={f['f']:f for f in data['frames']}
    frame=by_frame.get(args.frame)
    if frame is None:
        raise ValueError('Anchor frame is not in saved annotations')
    if args.pose_index is not None:
        if not 0<=args.pose_index<len(frame['poses']):
            raise ValueError('Anchor pose index is invalid')
        index=args.pose_index
    else:
        matches=[i for i,p in enumerate(frame['poses']) if p.get('player_id')==args.from_player_id]
        if len(matches)!=1:
            raise ValueError('Current player ID does not uniquely select an observed person')
        index=matches[0]
    pose=frame['poses'][index]
    if pose.get('team') not in (side,'team_'+side):
        raise ValueError('New person is not on the selected court side')
    video=Path(data['original_video_path'])
    video_sha=file_sha256(video)
    if args.existing_roster:
        payload=json.loads(args.existing_roster.read_text(encoding='utf-8'))
        if payload.get('schema')!='player_identity_roster_v2' or payload.get('video_sha256')!=video_sha or payload.get('side')!=side:
            raise ValueError('Existing roster belongs to another source/side/schema')
    else:
        initial=metadata.get('seed') or {}
        if not initial.get('assignments') or initial.get('introductions'):
            raise ValueError('Use --existing-roster when extending a previously expanded roster')
        observations={p['key']:p for f in frames for p in f['players']}
        payload={'schema':'player_identity_roster_v2','video_sha256':video_sha,'video_name':video.name,
                 'side':side,'max_on_court':len(initial['assignments']),
                 'initial':{'frame':initial['frame'],'label_source':initial['label_source'],
                            'verified':initial['verified'],
                            'players':[{'player_id':pid,'bb':observations[key]['bb']}
                                       for key,pid in initial['assignments'].items()]},
                 'introductions':[]}
    known={p['player_id'] for p in payload['initial']['players']}|{p['player_id'] for p in payload['introductions']}
    if args.new_id in known or args.frame<=payload['initial']['frame']:
        raise ValueError('New person needs a distinct ID and an anchor after the initial roster')
    same_track=[f['f'] for f in data['frames'] for p in f.get('poses',[])
                if pose.get('id') is not None and p.get('id')==pose['id'] and p.get('team') in (side,'team_'+side)]
    start=args.available_from if args.available_from is not None else min(same_track,default=args.frame)
    if start<0 or start>args.frame:
        raise ValueError('Availability lower bound must precede the observed anchor')
    payload['introductions'].append({
        'player_id':args.new_id,'anchor':{'frame':args.frame,'bb':pose['bb'],
          'provenance':{'label_source':'model','verified':False,'pose_index':index,
                        'annotations_sha256':file_sha256(args.annotations)}},
        'available_from_frame':start,
        'availability_provenance':{'label_source':'derived','verified':False,
          'method':'first selected observation of anchor raw track' if start==min(same_track,default=args.frame)
                   else 'configured lower bound; not human timing ground truth'},
        'identity_confirmation':{'label_source':'human','verified':True,'scope':'new_person',
                                 'note':args.confirmation_note},
    })
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(payload,stream,ensure_ascii=False,indent=2)
    print(json.dumps({'new_id':args.new_id,'anchor_frame':args.frame,'available_from_frame':start,
                      'known_people':len(known)+1,'max_on_court':payload['max_on_court'],
                      'initial_verified':payload['initial']['verified'],'output':str(args.output.resolve())}),flush=True)


def apply(args):
    if args.output.exists():
        raise FileExistsError('Output directory exists; choose a fresh versioned result')
    path=args.annotations.resolve()
    if not path.name.endswith('_annotations.json'):
        raise ValueError('Expected <stem>_annotations.json')
    stem=path.name[:-len('_annotations.json')]
    stats_path=path.with_name(f'{stem}_stats.json')
    annotations=json.loads(path.read_text(encoding='utf-8'))
    original_stats=json.loads(stats_path.read_text(encoding='utf-8'))
    source=Path(annotations['original_video_path'])
    cap=cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError('Cannot open original source video')
    try:
        if abs(cap.get(cv2.CAP_PROP_FPS)-annotations['fps'])>.05:
            raise ValueError('Source video FPS differs from saved annotations')
        if (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))!=(annotations['frame_width'],annotations['frame_height']):
            raise ValueError('Source dimensions differ from saved annotations')
        def images():
            last=max((f['f'] for f in annotations['frames']),default=-1)
            for index in range(last+1):
                ok,image=cap.read()
                if not ok:break
                yield index,image
        # Same post-pass called by process_video; no detection/FSM rerun.
        analyzer=GameAnalyzer.__new__(GameAnalyzer)
        analyzer.enable_player_identity=True
        analyzer.player_identity_side=args.side
        analyzer.player_identity_seed=str(args.seed) if args.seed else None
        analyzer.player_reid_assets=str(args.assets) if args.assets else None
        analyzer.player_reid_device=args.device
        analyzer.max_players_per_team=6
        analyzer.use_model_registry=not args.no_model_registry
        analyzer.model_provenance=deepcopy(original_stats.get('models') or {})
        analyzer.model_cache_dir=args.model_cache_dir
        analyzer.require_model_cache=args.require_model_cache
        analyzer._model_cache=ModelOutputCache(source,args.model_cache_dir,args.require_model_cache)
        analyzer._cache_frame_size=(annotations['frame_width'],annotations['frame_height'])
        analyzer.progress_callback=None
        stats=GameStats(video_name=annotations['video'],fps=annotations['fps'],frame_annotations=annotations['frames'])
        analyzer._apply_player_identity(stats,images(),annotations['fps'],source)
    finally:
        cap.release()
    report=stats.player_identity
    report['input_annotations_sha256']=file_sha256(path)
    report['source_video']=str(source.resolve())
    annotations['player_identity']=report
    original_stats['player_identity']=report
    from src.tracking.player_contacts import build_player_contacts
    annotations['player_contacts']=original_stats['player_contacts']=build_player_contacts(
        original_stats.get('all_events') or [],annotations['frames'],annotations['fps'],report)
    original_stats['models']=stats.models
    annotations['model_cache']=original_stats['model_cache']=analyzer._model_cache.summary()
    analyzer._model_cache.db.close()
    original_stats['processing_time_sec']=float(original_stats.get('processing_time_sec') or 0)+report.get('processing_time_sec',0)
    args.output.mkdir(parents=True)
    for name,payload in [(f'{stem}_stats.json',original_stats),(f'{stem}_annotations.json',annotations)]:
        with (args.output/name).open('x',encoding='utf-8') as stream:
            json.dump(payload,stream,ensure_ascii=False,separators=(',',':'),allow_nan=False)
    print(json.dumps({k:v for k,v in report.items() if k not in ('gallery_keys','seed')},ensure_ascii=False),flush=True)
    print(f'Fresh result: {args.output.resolve()}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    seed=sub.add_parser('export-seed',help='Export initial human IDs only, bound to exact source video')
    seed.add_argument('--review-project',type=Path,required=True)
    seed.add_argument('--output',type=Path,required=True)
    addition=sub.add_parser('add-player',help='Add a human-confirmed substitute with a new ID; existing IDs remain reusable on return')
    addition.add_argument('--annotations',type=Path,required=True)
    addition.add_argument('--frame',type=int,required=True)
    person=addition.add_mutually_exclusive_group(required=True)
    person.add_argument('--pose-index',type=int)
    person.add_argument('--from-player-id')
    addition.add_argument('--new-id',required=True)
    addition.add_argument('--available-from',type=int)
    addition.add_argument('--confirmation-note',required=True)
    addition.add_argument('--existing-roster',type=Path)
    addition.add_argument('--output',type=Path,required=True)
    replay=sub.add_parser('apply',help='Same production identity pass on unchanged saved poses; writes a fresh result')
    replay.add_argument('--annotations',type=Path,required=True)
    replay.add_argument('--output',type=Path,required=True)
    replay.add_argument('--seed',type=Path)
    replay.add_argument('--side',choices=('a','b'),default='b')
    replay.add_argument('--assets',type=Path)
    replay.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    replay.add_argument('--no-model-registry',action='store_true')
    replay.add_argument('--model-cache-dir',type=Path,default=DEFAULT_CACHE)
    replay.add_argument('--require-model-cache',action='store_true',help='Refuse inference; use saved OSNet embeddings only')
    args=parser.parse_args()
    if args.command=='export-seed':
        export_seed(args.review_project,args.output)
        print(args.output.resolve())
    elif args.command=='add-player':
        add_player(args)
    else:
        apply(args)


if __name__=='__main__':
    main()
