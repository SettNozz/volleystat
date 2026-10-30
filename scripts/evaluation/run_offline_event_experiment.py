"""Run a frozen offline event-data refresh and matched model comparison locally."""
from __future__ import annotations
import argparse,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.detection.sam_pose_recovery import file_hash
from scripts.evaluation.refresh_team_pose_dataset import save


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',default='data/event_clips_three_full_gold_causal_120f_v1/manifest.json')
    ap.add_argument('--baseline-features',default='data/event_clips_three_full_gold_causal_120f_v1/features')
    ap.add_argument('--ball-results',default='results/gold_v1_ball_reconstruction_conservative')
    ap.add_argument('--split-policy',default='data/event_clips_4k_gold_v1/split_policy.json')
    ap.add_argument('--dataset',required=True,type=Path);ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--pose-imgsz',type=int,default=2560);ap.add_argument('--setup-name',default='setup_1')
    ap.add_argument('--epochs',type=int,default=50);ap.add_argument('--patience',type=int,default=15)
    ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args();os.chdir(ROOT);args.output.mkdir(parents=True,exist_ok=False)
    status=args.output/'queue_status.json';child=None
    def record(phase,**extra):
        save(status,dict(phase=phase,pid=os.getpid(),updated_unix=time.time(),**extra))
    try:
        record('preparing')
        historical=args.output/'historical_checkpoints';historical.mkdir()
        for folder in (ROOT/'action_recognition/models/experiments_three_full_gold_multiscale_v2').iterdir():
            for model in folder.glob('*.pt'):
                destination=historical/folder.name;destination.mkdir(exist_ok=True);shutil.copy2(model,destination/model.name)
        snapshot=args.output/'code_snapshot';snapshot.mkdir();hashes={}
        for relative in ('scripts/evaluation/run_offline_event_experiment.py','scripts/evaluation/refresh_team_pose_dataset.py',
                         'scripts/evaluation/train_team_pose_experiment.py','action_recognition/event_clips/team_pose_features.py',
                         'action_recognition/models/sequence_event_model.py','src/detection/player_pose.py',
                         'src/tracking/pose_feature_extractor.py','src/pipeline/game_analyzer.py'):
            source=ROOT/relative;destination=snapshot/relative;destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,destination);hashes[relative]=file_hash(source)
        save(args.output/'queue_config.json',dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},code_hashes=hashes))
        phases=[('extracting',[sys.executable,'-u','scripts/evaluation/refresh_team_pose_dataset.py',
            '--manifest',args.manifest,'--baseline-features',args.baseline_features,'--ball-results',args.ball_results,
            '--split-policy',args.split_policy,'--output',str(args.dataset),'--resume',
            '--pose-imgsz',str(args.pose_imgsz),'--setup-name',args.setup_name]),
            ('training',[sys.executable,'-u','scripts/evaluation/train_team_pose_experiment.py',
            '--dataset',str(args.dataset),'--output',str(args.output/'training'),'--historical',str(historical),
            '--epochs',str(args.epochs),'--patience',str(args.patience),'--seed',str(args.seed)])]
        for phase,command in phases:
            with (args.output/(phase+'.log')).open('w',encoding='utf-8') as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
                record(phase,child_pid=child.pid,command=command)
                while True:
                    try:
                        code=child.wait(timeout=2);break
                    except subprocess.TimeoutExpired:
                        if (args.output/'cancel.request').exists():
                            child.terminate();child.wait(timeout=20)
                            record('cancelled',reason='cancel.request');return
            if code:raise RuntimeError(f'{phase} exited {code}; see {phase}.log')
        record('complete',report=str(args.output/'training/report.json'))
    except KeyboardInterrupt:
        record('cancelled',reason='interrupted');raise
    except Exception as exc:
        record('failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate();child.wait(timeout=20)


if __name__=='__main__':main()
