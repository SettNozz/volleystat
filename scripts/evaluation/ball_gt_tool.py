#!/usr/bin/env python3
"""Ground-truth labeling tool for ball detection evaluation.

Click the ball, or press N when it is not visible.  One click per frame is
enough because the metric is centre-distance, not box IoU: what we need to
measure is "did the detector put a point on the ball", plus how many points it
put anywhere else.

Detector proposals (YOLO, SAM3) are drawn as hints so the eye can find a 20 px
ball quickly, but the label is always the human click — otherwise the GT would
inherit exactly the blind spots we are trying to measure.

Usage:
    python scripts/evaluation/ball_gt_tool.py --video ... --start 6000 --count 300 --stride 3
    # open http://localhost:5055
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from flask import Flask, jsonify, request, send_file

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "results" / "ball_eval"

app = Flask(__name__)
STATE: dict = {}


def extract_frames(video: Path, indices: list[int], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    missing = [i for i in indices if not (out_dir / f"{i:06d}.jpg").exists()]
    if not missing:
        return
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video}")
    wanted = set(missing)
    idx = 0
    last = max(wanted)
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(wanted))
    idx = min(wanted)
    while idx <= last:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in wanted:
            cv2.imwrite(str(out_dir / f"{idx:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        idx += 1
    cap.release()
    print(f"Extracted {len(missing)} frames to {out_dir}")


def load_proposals(indices: list[int]) -> dict[int, list[dict]]:
    """Merge every candidate dump we have for these frames, as visual hints."""
    hints: dict[int, list[dict]] = {i: [] for i in indices}
    for path in sorted(EVAL_DIR.glob("*_candidates.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        frame_indices = data.get("frame_indices", [])
        for imgsz, run in data.get("runs", {}).items():
            for fi, dets in zip(frame_indices, run.get("detections", [])):
                if fi not in hints:
                    continue
                for d in dets:
                    x1, y1, x2, y2 = d["box"]
                    hints[fi].append(
                        {
                            "x": (x1 + x2) / 2,
                            "y": (y1 + y2) / 2,
                            "r": max(x2 - x1, y2 - y1) / 2,
                            "score": d["conf"],
                            "src": f"yolo{imgsz}",
                        }
                    )
    for path in sorted(EVAL_DIR.glob("*_sam3_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for entry in data.get("frames", []):
            fi = entry["frame"]
            if fi not in hints:
                continue
            for d in entry.get("candidates", []):
                x1, y1, x2, y2 = d["box"]
                hints[fi].append(
                    {
                        "x": (x1 + x2) / 2,
                        "y": (y1 + y2) / 2,
                        "r": max(x2 - x1, y2 - y1) / 2,
                        "score": d["score"],
                        "src": "sam3",
                    }
                )
    return {i: condense(v) for i, v in hints.items()}


def condense(hints: list[dict], radius: float = 14.0, min_score: float = 0.15, keep: int = 6) -> list[dict]:
    """Collapse the same proposal seen by several runs into one marker.

    Every configuration is dumped at a low confidence floor so thresholds can be
    swept offline, but a screen with twenty circles on it slows the eye down
    instead of helping it. Hints are a search aid, not evidence.
    """
    out: list[dict] = []
    for h in sorted(hints, key=lambda h: -h["score"]):
        if h["score"] < min_score:
            continue
        near = next(
            (k for k in out if abs(k["x"] - h["x"]) < radius and abs(k["y"] - h["y"]) < radius),
            None,
        )
        if near is not None:
            if h["src"] not in near["src"]:
                near["src"] = f"{near['src']}+{h['src']}"
            continue
        out.append(dict(h))
        if len(out) >= keep:
            break
    return out


@app.get("/")
def index():
    return PAGE


@app.get("/api/task")
def api_task():
    return jsonify(
        {
            "video": STATE["video"],
            "width": STATE["width"],
            "height": STATE["height"],
            "frames": STATE["indices"],
            "hints": {str(k): v for k, v in STATE["hints"].items()},
            "labels": STATE["labels"],
            "output": str(STATE["gt_path"]),
        }
    )


@app.get("/frame/<int:idx>.jpg")
def frame(idx: int):
    return send_file(STATE["frames_dir"] / f"{idx:06d}.jpg", mimetype="image/jpeg")


@app.post("/api/label")
def api_label():
    payload = request.get_json(force=True)
    frame_idx = str(int(payload["frame"]))
    if payload.get("clear"):
        STATE["labels"].pop(frame_idx, None)
    else:
        STATE["labels"][frame_idx] = payload["label"]
    save_labels()
    return jsonify({"ok": True, "labeled": len(STATE["labels"])})


def save_labels() -> None:
    STATE["gt_path"].write_text(
        json.dumps(
            {
                "video": STATE["video"],
                "resolution": [STATE["width"], STATE["height"]],
                "frames": STATE["indices"],
                "labels": STATE["labels"],
            },
            indent=1,
        ),
        encoding="utf-8",
    )


PAGE = """
<!doctype html><meta charset="utf-8"><title>Ball GT</title>
<style>
 body{margin:0;background:#111;color:#eee;font:13px system-ui}
 #bar{padding:6px 10px;background:#1b1b1b;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
 #wrap{position:relative;display:inline-block}
 canvas{display:block;cursor:crosshair}
 #zoom{position:fixed;right:10px;top:46px;border:1px solid #444;background:#000;image-rendering:pixelated}
 .k{color:#888}
 b{color:#7fd}
 .no{color:#f77}
</style>
<div id="bar">
  <span>frame <b id="fno">-</b> <span class="k">(<span id="pos">0</span>/<span id="tot">0</span>)</span></span>
  <span>labeled <b id="done">0</b> · no-ball <span class="no" id="noball">0</span></span>
  <span class="k">click = ball · <b>N</b> not visible · <b>S</b> unsure · <b>&larr;/&rarr;</b> move · <b>D</b> delete label</span>
  <span class="k">hints: <span style="color:#0f0">yolo</span> <span style="color:#0af">sam3</span> · <b>H</b> toggle</span>
</div>
<div id="wrap"><canvas id="c"></canvas></div>
<canvas id="zoom" width="240" height="240"></canvas>
<script>
let T=null,i=0,img=new Image(),showHints=true,mouse=null;
const c=document.getElementById('c'),ctx=c.getContext('2d');
const z=document.getElementById('zoom'),zx=z.getContext('2d');
fetch('/api/task').then(r=>r.json()).then(t=>{T=t;c.width=t.width;c.height=t.height;
  document.getElementById('tot').textContent=t.frames.length;
  const first=t.frames.findIndex(f=>!(String(f) in t.labels));i=first<0?0:first;load();});
function load(){const f=T.frames[i];img.onload=draw;img.src='/frame/'+f+'.jpg?';}
function draw(){
  ctx.drawImage(img,0,0);
  const f=T.frames[i];
  if(showHints)for(const h of (T.hints[f]||[])){
    ctx.strokeStyle=h.src.includes('sam3')?'#0af':'#0f0';ctx.lineWidth=1;
    ctx.beginPath();ctx.arc(h.x,h.y,Math.max(h.r,6)+3,0,7);ctx.stroke();
    ctx.fillStyle=ctx.strokeStyle;ctx.font='10px monospace';
    ctx.fillText(h.score.toFixed(2),h.x+9,h.y-7);
  }
  // previous labeled position — the ball moves smoothly, so it points the eye
  for(let k=i-1;k>=0&&k>=i-3;k--){const p=T.labels[String(T.frames[k])];
    if(p&&p.x!=null){ctx.strokeStyle='rgba(255,255,0,.45)';ctx.beginPath();ctx.arc(p.x,p.y,12,0,7);ctx.stroke();break;}}
  const cur=T.labels[String(f)];
  if(cur&&cur.x!=null){ctx.strokeStyle='#ff0';ctx.lineWidth=2;
    ctx.beginPath();ctx.arc(cur.x,cur.y,14,0,7);ctx.stroke();
    ctx.beginPath();ctx.moveTo(cur.x-20,cur.y);ctx.lineTo(cur.x+20,cur.y);
    ctx.moveTo(cur.x,cur.y-20);ctx.lineTo(cur.x,cur.y+20);ctx.stroke();}
  if(cur&&cur.x==null){ctx.fillStyle=cur.state==='unsure'?'#fa0':'#f55';ctx.font='20px system-ui';
    ctx.fillText(cur.state==='unsure'?'UNSURE':'NO BALL',12,28);}
  document.getElementById('fno').textContent=f;
  document.getElementById('pos').textContent=i+1;
  let d=0,nb=0;for(const k in T.labels){d++;if(T.labels[k].x==null&&T.labels[k].state==='absent')nb++;}
  document.getElementById('done').textContent=d;
  document.getElementById('noball').textContent=nb;
}
c.addEventListener('mousemove',e=>{const r=c.getBoundingClientRect();
  mouse=[(e.clientX-r.left)*c.width/r.width,(e.clientY-r.top)*c.height/r.height];
  zx.imageSmoothingEnabled=false;zx.clearRect(0,0,240,240);
  zx.drawImage(img,mouse[0]-20,mouse[1]-20,40,40,0,0,240,240);
  zx.strokeStyle='#f0f';zx.strokeRect(114,114,12,12);});
c.addEventListener('click',e=>{const r=c.getBoundingClientRect();
  const x=(e.clientX-r.left)*c.width/r.width,y=(e.clientY-r.top)*c.height/r.height;
  send({x:Math.round(x*10)/10,y:Math.round(y*10)/10,state:'visible'},true);});
function send(label,advance){const f=T.frames[i];T.labels[String(f)]=label;
  fetch('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({frame:f,label:label})});
  if(advance&&i<T.frames.length-1){i++;load();}else draw();}
document.addEventListener('keydown',e=>{
  const k=e.key.toLowerCase();
  if(k==='n'){send({x:null,y:null,state:'absent'},true);}
  else if(k==='s'){send({x:null,y:null,state:'unsure'},true);}
  else if(k==='arrowright'){if(i<T.frames.length-1){i++;load();}}
  else if(k==='arrowleft'){if(i>0){i--;load();}}
  else if(k==='h'){showHints=!showHints;draw();}
  else if(k==='d'){const f=T.frames[i];delete T.labels[String(f)];
    fetch('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({frame:f,clear:true})});draw();}
  else return;
  e.preventDefault();});
</script>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--start", type=int, default=6000)
    ap.add_argument("--count", type=int, default=300)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--port", type=int, default=5055)
    ap.add_argument("--gt", default=None, help="Output JSON (default: results/ball_eval/<stem>_gt.json)")
    args = ap.parse_args()

    video = Path(args.video)
    indices = [args.start + k * args.stride for k in range(args.count)]
    frames_dir = EVAL_DIR / f"frames_{video.stem}"
    extract_frames(video, indices, frames_dir)

    cap = cv2.VideoCapture(str(video))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    gt_path = Path(args.gt) if args.gt else EVAL_DIR / f"{video.stem}_{args.start}_{indices[-1]}_gt.json"
    labels = {}
    if gt_path.exists():
        labels = json.loads(gt_path.read_text(encoding="utf-8")).get("labels", {})
        print(f"Resuming: {len(labels)} labels already in {gt_path}")

    STATE.update(
        video=str(video),
        indices=indices,
        frames_dir=frames_dir,
        width=width,
        height=height,
        hints=load_proposals(indices),
        labels=labels,
        gt_path=gt_path,
    )
    n_hints = sum(len(v) for v in STATE["hints"].values())
    print(f"{len(indices)} frames, {n_hints} proposal hints -> {gt_path}")
    print(f"Open http://localhost:{args.port}")
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
