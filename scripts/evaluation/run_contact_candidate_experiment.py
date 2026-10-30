"""Isolate candidate association and contact-centre pooling on frozen gold data."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.evaluation.rebuild_contact_candidates import rebuild
from scripts.evaluation.refresh_team_pose_dataset import save
from src.detection.sam_pose_recovery import file_hash


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',required=True,type=Path)
    ap.add_argument('--dataset',required=True,type=Path)
    ap.add_argument('--previous-results',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path)
    args=ap.parse_args();os.chdir(ROOT)
    if args.output.exists() or args.dataset.exists():raise ValueError('Use new dataset and output paths')
    args.output.mkdir(parents=True)
    status=args.output/'queue_status.json'
    def record(phase,**extra):save(status,dict(phase=phase,pid=os.getpid(),updated_unix=time.time(),**extra))
    try:
        record('rebuilding')
        snapshot=args.output/'code_snapshot';snapshot.mkdir()
        import shutil
        for rel in ('scripts/evaluation/rebuild_contact_candidates.py','scripts/evaluation/run_contact_candidate_experiment.py',
                    'scripts/evaluation/train_team_pose_experiment.py','action_recognition/event_clips/team_pose_features.py',
                    'action_recognition/models/sequence_event_model.py'):
            p=snapshot/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(rel,p)
        rebuild(args.source,args.dataset)
        old_report=args.previous_results/'training/report.json'
        save(args.output/'preflight.json',dict(parent_report=str(old_report),parent_report_sha256=file_hash(old_report),
            source=str(args.source),dataset=str(args.dataset),seed=42,epochs=50,patience=15,
            label_source='human',split='leave_one_complete_source_video_out',sam_masks_used=False,
            controls='Only team channels change in candidate data; trajectory/legacy pose values stay identical',
            code_hashes={str(p.relative_to(snapshot)):file_hash(p) for p in snapshot.rglob('*.py')}))
        variants=[('candidate_attention',args.dataset,['team_pose'],False),
                  ('original_center',args.source,['trajectory','team_pose'],True),
                  ('candidate_center',args.dataset,['team_pose'],True)]
        for name,dataset,conditions,center in variants:
            command=[sys.executable,'-u','scripts/evaluation/train_team_pose_experiment.py','--dataset',str(dataset),
                '--output',str(args.output/name),'--historical',str(args.previous_results/'historical_checkpoints'),
                '--conditions',*conditions]
            if center:command.append('--contact-center-pool')
            with (args.output/(name+'.log')).open('w',encoding='utf-8') as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
                record(name,child_pid=child.pid)
                try:
                    while child.poll() is None:
                        if (args.output/'cancel.request').exists():
                            child.terminate();child.wait();record('cancelled');return
                        time.sleep(2)
                    if child.returncode:raise RuntimeError(name+' failed; see its log')
                finally:
                    if child.poll() is None:child.terminate();child.wait()
        old=json.loads(old_report.read_text());summary={'original_attention':old['aggregate']}
        for name,*_ in variants:summary[name]=json.loads((args.output/name/'report.json').read_text())['aggregate']
        save(args.output/'report.json',dict(status='complete',variants=summary,production_promoted=False,
            caveat='One seed; best-epoch model-selection scores on three videos, not independent test results'))
        record('complete',report=str(args.output/'report.json'))
        print(json.dumps(summary,indent=2),flush=True)
    except BaseException as exc:
        record('failed',error=repr(exc));raise


if __name__=='__main__':main()
