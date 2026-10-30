"""Compact Ukrainian raw-versus-rule preview from frozen classifier predictions."""
import argparse,collections,json,sys
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from action_recognition.inference.serve_guard import ServeGuard
from src.detection.sam_pose_recovery import file_hash


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--predictions',required=True,type=Path);ap.add_argument('--output',required=True,type=Path)
    args=ap.parse_args()
    if args.output.exists():raise ValueError('Refuse existing output')
    report=json.loads(args.predictions.read_text(encoding='utf-8'))
    annotation=Path(report['annotations'])
    if file_hash(annotation)!=report['annotations_sha256']:raise ValueError('Source annotations changed')
    payload=json.loads(annotation.read_text());rows={r['f']:r for r in payload['frames']}
    fps=payload['fps'];samples=report['predictions'];guard=ServeGuard()
    for f in sorted(k for k in rows if k<samples[0]['frame']):guard.update(f/fps,rows[f].get('state',''))
    labels={'serve':'ПОДАЧА','reception':'ПРИЙОМ','pass':'ПЕРЕДАЧА','attack':'АТАКА'}
    reasons={'serve_during_active_rally':'Триває розіграш',
        'serve_start_not_confirmed':'FSM не бачить початку подачі',
        'serve_restart_too_soon':'Ще не завершилася пауза'}
    args.output.mkdir(parents=True)
    title=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',30)
    detail=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',21)
    cap=cv2.VideoCapture(report['video']);cap.set(cv2.CAP_PROP_POS_FRAMES,samples[0]['frame'])
    writer=cv2.VideoWriter(str(args.output/'comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1280,816))
    if not cap.isOpened() or not writer.isOpened():raise ValueError('Cannot open video input/output')
    changes=collections.Counter();first_examples={}
    try:
        for s in samples:
            f=s['frame'];guard.update(s['time'],rows[f].get('state',''))
            old=s.get('serve_rule',{});d=guard.decision(s['new']['label'])
            s['previous_strict_rule']=old;s['serve_rule']=d
            changes[d['suppression_reason'] or ('warning' if d['warning_reason'] else 'allowed')]+=1
            ok,frame=cap.read()
            if not ok:raise ValueError('Video decode ended early')
            frame=cv2.resize(frame,(1280,720))
            header=Image.new('RGB',(1280,96),(18,22,28));draw=ImageDraw.Draw(header)
            raw=labels[s['new']['label']];filtered=labels.get(d['label'],'ВІДХИЛЕНО')
            color=(255,120,110) if d['suppression_reason'] else (255,195,80) if d['warning_reason'] else (100,230,160)
            draw.text((20,12),f'Модель: {raw}  {s["new"]["confidence"]:.0%}',font=title,fill=(240,240,245))
            draw.text((660,12),f'Після правил: {filtered}',font=title,fill=color)
            draw.text((20,57),f'{int(s["time"]//60):02d}:{s["time"]%60:04.1f}   •   прогноз для поточного кадру',font=detail,fill=(165,175,185))
            note=reasons.get(d['suppression_reason'],'Коротка пауза: лише попередження' if d['warning_reason'] else 'Прогноз залишено')
            draw.text((660,57),note,font=detail,fill=color)
            canvas=np.concatenate([cv2.cvtColor(np.asarray(header),cv2.COLOR_RGB2BGR),frame],axis=0)
            writer.write(canvas)
            key='blocked' if d['suppression_reason'] else 'warning' if d['warning_reason'] else 'allowed'
            if key not in first_examples:
                cv2.imwrite(str(args.output/(key+'.png')),canvas);first_examples[key]=f
    finally:cap.release();writer.release()
    report['serve_rules']=dict(min_restart_seconds=3,enforce_restart_pause=False,
        pause_policy='warning only; not validated against human rally boundaries',counts=dict(changes),
        state_source='saved FSM estimates, not human ground truth')
    report['parent_predictions']=str(args.predictions);report['parent_sha256']=file_hash(args.predictions)
    (args.output/'predictions.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(frames=len(samples),counts=dict(changes)),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
