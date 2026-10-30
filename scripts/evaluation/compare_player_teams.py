"""Render preserved before/after player-team results on their shared interval."""
from __future__ import annotations
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.evaluation.sam3_people_masks import plan_frames,save_json,sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before',type=Path,required=True)
    parser.add_argument('--after',type=Path,required=True)
    parser.add_argument('--video',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--start',type=float,default=15)
    parser.add_argument('--duration',type=float,default=30)
    parser.add_argument('--ffmpeg',default='ffmpeg')
    args=parser.parse_args()
    before=json.loads(args.before.read_text(encoding='utf-8-sig'))
    after=json.loads(args.after.read_text(encoding='utf-8-sig'))
    for key in ('video','frame_width','frame_height'):
        if before[key]!=after[key]:raise ValueError(f'Mismatched {key}')
    if Path(before['video']).stem!=args.video.stem or not math.isclose(before['fps'],after['fps']):
        raise ValueError('Source video/FPS mismatch')
    b={r['f']:r for r in before['frames']};a={r['f']:r for r in after['frames']}
    first,end=plan_frames(before['fps'],args.start,args.duration)
    end=min(end,max(b)+1,max(a)+1)
    if end<=first or any(i not in b or i not in a for i in range(first,end)):
        raise ValueError('Missing comparison frames')
    args.output.mkdir(parents=True,exist_ok=False)
    fps=before['fps'];w,h=1280,720
    writer=cv2.VideoWriter(str(args.output/'comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w*2,h))
    cap=cv2.VideoCapture(str(args.video))
    if not cap.isOpened() or not writer.isOpened():raise RuntimeError('Video I/O unavailable')
    counts={'before':[],'after':[]}
    def render(frame,row,meta,title):
        view=cv2.resize(frame,(w,h))
        sx,sy=w/meta['frame_width'],h/meta['frame_height']
        totals=[0,0]
        for p in row.get('poses',[]):
            team=0 if p['team'] in ('a','team_a') else 1
            totals[team]+=1
            # OpenCV uses BGR: A is yellow, B is green.
            color=(0,235,255) if team==0 else (50,245,65)
            box=[round(v*s) for v,s in zip(p['bb'],(sx,sy,sx,sy))]
            x1,y1=max(0,box[0]),max(80,box[1])
            x2,y2=min(w,box[2]),min(h,box[3])
            if x2>x1 and y2>y1:
                region=view[y1:y2,x1:x2]
                region[:]=(region*.88+np.array(color)*.12).astype(np.uint8)
            cv2.rectangle(view,tuple(box[:2]),tuple(box[2:]),color,3)
            kp=p.get('kp',[])
            bones=((5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16))
            if len(kp)==17:
                for i,j in bones:
                    if kp[i][2]>=.3 and kp[j][2]>=.3:
                        cv2.line(view,(round(kp[i][0]*sx),round(kp[i][1]*sy)),
                                 (round(kp[j][0]*sx),round(kp[j][1]*sy)),color,2,cv2.LINE_AA)
            # A bright ground marker remains visible when boxes overlap.
            foot=(round((p['bb'][0]+p['bb'][2])*sx/2),min(h-10,round(p['bb'][3]*sy)))
            cv2.circle(view,foot,10,(10,10,10),-1,cv2.LINE_AA)
            cv2.circle(view,foot,7,color,-1,cv2.LINE_AA)
            label='A' if team==0 else 'B'
            tx=max(0,min(w-24,box[0]));ty=max(82,box[1]-24)
            cv2.rectangle(view,(tx,ty),(tx+23,ty+22),color,-1)
            cv2.putText(view,label,(tx+5,ty+17),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),2,cv2.LINE_AA)
        left,right=meta.get('net_floor_left'),meta.get('net_floor_right')
        if left and right:
            cv2.line(view,(round(left[0]*sx),round(left[1]*sy)),(round(right[0]*sx),round(right[1]*sy)),(220,80,230),2)
        cv2.rectangle(view,(0,0),(w,80),(15,15,15),-1)
        cv2.putText(view,title,(12,24),cv2.FONT_HERSHEY_SIMPLEX,.64,(255,255,255),1,cv2.LINE_AA)
        cv2.rectangle(view,(12,39),(31,58),(0,235,255),-1)
        cv2.putText(view,f'YELLOW - A / FAR: {totals[0]}/6',(42,56),cv2.FONT_HERSHEY_SIMPLEX,.62,(0,235,255),2,cv2.LINE_AA)
        cv2.rectangle(view,(445,39),(464,58),(50,245,65),-1)
        cv2.putText(view,f'GREEN - B / NEAR: {totals[1]}/6',(475,56),cv2.FONT_HERSHEY_SIMPLEX,.62,(50,245,65),2,cv2.LINE_AA)
        cv2.putText(view,f'Frame {row["f"]} | {row["f"]/fps:.2f}s',(12,74),cv2.FONT_HERSHEY_SIMPLEX,.4,(190,190,190),1,cv2.LINE_AA)
        return view,totals
    try:
        for f in range(end):
            ok,frame=cap.read()
            if not ok:raise RuntimeError(f'Video ended at {f}')
            if f<first:continue
            old,bc=render(frame,b[f],before,'BEFORE: saved workflow')
            new,ac=render(frame,a[f],after,'AFTER: improved player recall and team divider')
            counts['before'].append(bc);counts['after'].append(ac)
            composite=np.concatenate([old,new],axis=1);writer.write(composite)
            if f in (first,(first+end)//2,end-1):
                cv2.imwrite(str(args.output/f'frame_{f}.jpg'),composite)
    finally:
        cap.release();writer.release()
    summary={}
    for key,values in counts.items():
        arr=np.asarray(values)
        summary[key]=dict(frames=len(arr),mean_a=float(arr[:,0].mean()),mean_b=float(arr[:,1].mean()),
                          a_six_frames=int((arr[:,0]==6).sum()),b_six_frames=int((arr[:,1]==6).sum()),
                          both_six_frames=int((arr==6).all(axis=1).sum()),
                          a_histogram={str(v):int((arr[:,0]==v).sum()) for v in range(7)},
                          b_histogram={str(v):int((arr[:,1]==v).sum()) for v in range(7)})
    report=dict(start_frame=first,end_frame_exclusive=end,fps=fps,summary=summary,
                before_sha256=sha256(args.before),after_sha256=sha256(args.after),
                before_divider=before.get('net_floor_method'),after_divider=after.get('net_floor_method'),
                player_detection=after.get('player_detection',{}),
                accuracy_evaluated=False,warning='Detection counts are not human-verified recall or team accuracy.')
    save_json(args.output/'report.json',report)
    subprocess.run([args.ffmpeg,'-hide_banner','-loglevel','warning','-nostdin','-n','-i',str(args.output/'comparison.mp4'),
                    '-c:v','libx264','-preset','fast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart','-an','-threads','4',str(args.output/'comparison_web.mp4')],check=True)
    rows=''.join(f'<tr><td>{label}</td><td>{r["mean_a"]:.2f}</td><td>{r["mean_b"]:.2f}</td><td>{r["a_six_frames"]}/{r["frames"]}</td><td>{r["b_six_frames"]}/{r["frames"]}</td></tr>' for label,r in [('До',summary['before']),('Після',summary['after'])])
    page=f'''<!doctype html><html lang="uk"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Гравці та команди — порівняння</title><style>body{{max-width:1280px;margin:32px auto;padding:0 20px;font:17px/1.55 system-ui;color:#e6edf3;background:#10151d}}video,img{{width:100%}}td,th{{padding:12px;text-align:left;border-bottom:1px solid #394351}}a{{color:#80baff}}</style><h1>Визначення гравців та команд</h1><p>Ліворуч — попередній результат. Праворуч — повторний повний воркфлоу з входом YOLO-pose 2560, перевіркою лінії під сіткою та відновленням поз за підказками SAM (якщо ввімкнене в цьому запуску).  Жовта команда A — дальня сторона, зелена команда B — ближня; фіолетова лінія — використаний розділювач команд.</p><video src="comparison_web.mp4" controls preload="metadata"></video><table><tr><th>Результат</th><th>Середня кількість A</th><th>Середня кількість B</th><th>Кадри з 6 у A</th><th>Кадри з 6 у B</th></tr>{rows}</table><p>Порівняно {end-first} спільних кадрів ({first}–{end-1}). Шість — обмеження на реальні детекції, не штучно заповнена кількість. Закритий іншими гравець може залишитися невизначеним. Кількість детекцій не є оцінкою точності за людською розміткою.</p><p>Новий розділювач: {after.get('net_floor_method')}. Вихідна калібровка не переписана. Передбачення FSM та ознаки перераховані повним воркфлоу; історичний і новий запуски мають різну тривалість, тож порівняння тут стосується лише гравців/команд.</p><p><a href="report.json">Звіт з кількостями</a></p>'''
    for f in (first,(first+end)//2,end-1):page+=f'<h2>Кадр {f}</h2><img src="frame_{f}.jpg" alt="До та після">'
    (args.output/'review.html').write_text(page+'</html>',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
