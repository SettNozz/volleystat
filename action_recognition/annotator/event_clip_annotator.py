#!/usr/bin/env python3
"""Label event clips: one keypress per clip.

Rebuilt as a Flask page rather than the lost Streamlit app: Streamlit re-runs the
whole script on every interaction, which at ~800 clips per hour of footage is the
difference between a tolerable session and an unusable one. Here the clip list is
fetched once and each label is a single POST, so the only thing that ever reloads
is the video element.

The trajectory plot is drawn from the clip's feature Parquet, not the video. It
loads instantly and it is often the faster read of the two: a serve starts at the
back line and crosses the net once, a pass stays on one side, an attack goes
down steeply. 4K clips take a moment to decode, and waiting on them for every
clip is what makes labeling feel slow.

Usage:
    python -m action_recognition.annotator.event_clip_annotator \
        --manifest data/event_clips_pilot/manifest_predicted.json
    # open http://localhost:5056
"""

from __future__ import annotations

import argparse
import json
import mimetypes
from pathlib import Path

from flask import Flask, jsonify, request, send_file

_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from action_recognition.event_clips.config import EVENT_TYPES  # noqa: E402
from action_recognition.event_clips.manifest_schema import (  # noqa: E402
    upgrade_manifest,
    validate_context_value,
)

app = Flask(__name__)
STATE: dict = {}

CONTEXT_HELP = {
    "match_phase": {
        "label": "Match phase",
        "description": "The stage of the match visible at this moment.",
        "values": {
            "unknown": "There is not enough visual context to determine the phase.",
            "non_game": "Warm-up, timeout, between sets, referee discussion, or other activity unrelated to live play. Selecting it resets the remaining fields to unknown.",
            "dead_ball": "No rally is active; the previous point is over and the teams are resetting.",
            "serve_setup": "The server is preparing or waiting for the whistle before contacting the ball.",
            "serve_flight": "From the serve contact until the receiver touches the ball or the serve ends the point. Selecting it automatically sets serve / serve / not_applicable below.",
            "rally_live": "The ball remains in play after the service reception.",
            "rally_end": "The immediate point-ending moment, before the teams fully reset.",
        },
    },
    "physical_action": {
        "label": "Physical action",
        "description": "What the player physically does when contacting the ball.",
        "values": {
            "unknown": "The type of physical contact cannot be identified reliably.",
            "serve": "A player puts the ball into play from the service zone.",
            "forearm_pass": "A contact with joined forearms; it may be a serve reception or a defensive dig.",
            "overhead_set": "A two-hand overhead contact, usually used to prepare an attacker.",
            "attack_hit": "An intentional attacking hit that sends the ball toward the opponent.",
            "block_touch": "A touch at the net intended to stop or deflect an opponent's attack.",
            "other_touch": "A tip, poke, foot save, one-hand save, or another contact that does not fit the actions above.",
        },
    },
    "context_role": {
        "label": "Tactical role",
        "description": "Why this contact happens within the rally; this is separate from the physical technique.",
        "values": {
            "unknown": "The tactical purpose of the contact cannot be identified reliably.",
            "serve": "The contact starts the rally.",
            "serve_reception": "The team's first contact after an opponent's serve; include it in reception statistics.",
            "defensive_dig": "A defensive contact against an attack or free ball, not a serve; exclude it from reception statistics.",
            "set": "A contact intended to create the next attacking opportunity.",
            "attack": "A contact intended to score or put the opponent under pressure.",
            "block": "A net touch intended to stop an attack; it does not consume one of the team's three contacts.",
            "free_ball": "A controlled, non-aggressive ball sent to the opponent.",
            "other": "A volleyball contact whose tactical role does not fit the categories above.",
        },
    },
    "reception_quality": {
        "label": "Serve-reception quality",
        "description": "Evaluate only when Tactical role is serve_reception; otherwise select not_applicable.",
        "values": {
            "not_applicable": "This event is not a serve reception.",
            "unknown": "It is a serve reception, but its outcome is not visible or cannot be judged confidently.",
            "no_touch_ace": "The serve scores without the receiving player touching the ball.",
            "error": "The receiver touches the serve but directly loses the point or cannot keep the ball playable.",
            "overpass": "The reception crosses the net directly and gives the opponent an immediate play or attack.",
            "negative": "The ball stays playable, but the setter cannot run the planned attack or has very limited options.",
            "positive": "The ball reaches the setter area and allows an organized attack, but not every attacking option.",
            "perfect": "The pass reaches the target accurately and gives the setter all normal attacking options.",
            "controlled": "The reception is clearly playable, but the video is insufficient to grade it as negative, positive, or perfect.",
        },
    },
}


def load_trajectory(clip_id: str) -> dict:
    """Ball path and speed for one clip, from its feature Parquet."""
    path = STATE["features_dir"] / f"{clip_id}.parquet"
    if not path.exists():
        return {}
    try:
        import pandas as pd

        df = pd.read_parquet(path)
    except Exception:
        return {}
    cols = {c: df[c].tolist() for c in ("ball_x", "ball_y", "ball_speed") if c in df}
    if "ball_x" not in cols:
        return {}
    return {
        "x": cols["ball_x"],
        "y": cols.get("ball_y", []),
        "speed": cols.get("ball_speed", []),
    }


@app.get("/")
def index():
    return PAGE


@app.get("/api/task")
def api_task():
    return jsonify(
        {
            "classes": EVENT_TYPES,
            "taxonomies": STATE["manifest"].get("taxonomies", {}),
            "context_help": CONTEXT_HELP,
            "clips": STATE["clips"],
            "output": str(STATE["output"]),
            "mode": STATE.get("mode", "both"),
        }
    )


@app.get("/api/traj/<clip_id>")
def api_traj(clip_id: str):
    return jsonify(load_trajectory(clip_id))


@app.get("/clip/<clip_id>.mp4")
def clip(clip_id: str):
    path = STATE["paths"].get(clip_id)
    if not path or not Path(path).exists():
        return "", 404
    return send_file(path, mimetype="video/mp4")


@app.get("/source/<clip_id>")
def source(clip_id: str):
    """Range-stream a source video when pre-cut clips were intentionally skipped."""
    path = STATE["sources"].get(clip_id)
    if not path or not Path(path).exists():
        return "", 404
    mimetype = mimetypes.guess_type(path)[0] or "application/octet-stream"
    return send_file(path, mimetype=mimetype, conditional=True)


@app.post("/api/label")
def api_label():
    payload = request.get_json(force=True)
    clip_id = payload["clip_id"]
    entry = STATE["by_id"].get(clip_id)
    if entry is None:
        return jsonify({"ok": False}), 404
    context = payload.get("context") or {}
    try:
        validated_context = {
            str(field): validate_context_value(str(field), str(value))
            for field, value in context.items()
        }
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    entry.update(validated_context)
    if entry.get("match_phase") == "non_game":
        entry.update(
            physical_action="unknown",
            context_role="unknown",
            reception_quality="unknown",
            is_service_reception=False,
        )
    elif entry.get("match_phase") == "serve_flight":
        entry.update(
            physical_action="serve",
            context_role="serve",
            reception_quality="not_applicable",
            is_service_reception=False,
        )
    if payload.get("clear"):
        entry["event_types"] = []
        entry["label_source"] = "cleared"
    elif payload.get("label") is not None:
        label = payload["label"]
        if label not in EVENT_TYPES:
            return jsonify({"ok": False, "error": f"unknown label {label}"}), 400
        entry["event_types"] = [label]
        entry["primary_event_type"] = label
        entry["label_source"] = "human"
    if context and not payload.get("draft_context"):
        entry["context_label_source"] = "human"
        entry["context_label_confidence"] = 1.0
    if payload.get("accept_context"):
        entry["context_label_source"] = "human"
        entry["context_label_confidence"] = 1.0
    if context or payload.get("accept_context"):
        entry["is_service_reception"] = (
            entry.get("context_role") == "serve_reception"
        )
    save()
    return jsonify({"ok": True, "labeled": count_human()})


def clip_is_complete(clip: dict, mode: str) -> bool:
    action_done = clip.get("label_source") in ("human", "cleared")
    context_done = clip.get("context_label_source") == "human"
    if mode == "action":
        return action_done
    if mode == "context":
        return context_done
    return action_done and context_done


def count_human() -> int:
    mode = STATE.get("mode", "both")
    return sum(clip_is_complete(c, mode) for c in STATE["manifest"]["clips"])


def save() -> None:
    STATE["output"].write_text(json.dumps(STATE["manifest"], indent=1), encoding="utf-8")


PAGE = """
<!doctype html><meta charset="utf-8"><title>Event clips</title>
<style>
 body{margin:0;background:#111;color:#eee;font:13px system-ui}
 #bar{padding:6px 10px;background:#1b1b1b;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
 #main{display:flex;gap:12px;padding:10px;align-items:flex-start;flex-wrap:wrap}
 video{max-width:70vw;max-height:70vh;background:#000}
 #side{min-width:420px;max-width:520px}
 .cls{display:block;margin:3px 0;padding:6px 10px;border:1px solid #444;border-radius:4px;cursor:pointer}
 .cls.pre{border-color:#fa0;color:#fa0}
 .cls.cur{background:#274;border-color:#7fd;color:#fff}
 .k{color:#888}
 .vlm{margin:8px 0;padding:7px;border-left:3px solid #fa0;background:#211b10;color:#ddd}
 .guide{margin:0 0 10px;padding:9px;border-left:3px solid #7fd;background:#14201d;line-height:1.4}
 .field{margin:10px 0}.field label{display:block;color:#eee;margin-bottom:2px;font-weight:650}
 .field-help{color:#aaa;margin:0 0 5px;line-height:1.3}
 .value-help{color:#7fd;margin-top:4px;line-height:1.35;min-height:18px}
 select,button{background:#181818;color:#eee;border:1px solid #555;padding:5px;width:100%}
 b{color:#7fd}
 canvas{background:#000;border:1px solid #333;display:block;margin-top:8px}
</style>
<div id="bar">
  <span>clip <b id="pos">0</b>/<span id="tot">0</span></span>
  <span><span id="modeName">both</span>-complete <b id="done">0</b></span>
  <span id="cid" class="k"></span>
  <span class="k"><b>Enter</b> accept &amp; next &middot; <b>1-6</b> action &middot; <b>&larr;/&rarr;</b> move &middot; <b>R</b> replay &middot; <b>C</b> clip/rally context &middot; <b>D</b> clear</span>
</div>
<div id="main">
  <video id="v" autoplay loop muted playsinline></video>
  <div id="side">
    <div class="guide"><b>How to label:</b> the values are pre-filled. Correct only the fields that are wrong, then press <b>Enter</b> to confirm all fields and open the next clip.</div>
    <div id="classes"></div>
    <button id="contextToggle" onclick="toggleContext()">show full rally context</button>
    <div id="contextFields"></div>
    <button id="acceptContext" onclick="acceptContext()">accept context &amp; next (Enter)</button>
    <div class="k" id="meta"></div>
    <div class="vlm" id="vlm" style="display:none"></div>
    <canvas id="traj" width="280" height="200"></canvas>
    <div class="k">trajectory: ball path, red = fast</div>
  </div>
</div>
<script>
let T=null,i=0,clipStart=0,clipEnd=null,showRally=false;
const v=document.getElementById('v'),tc=document.getElementById('traj').getContext('2d');
v.addEventListener('loadedmetadata',()=>{
  if(clipStart>0&&Math.abs(v.currentTime-clipStart)>0.05)v.currentTime=clipStart;
  else v.play().catch(()=>{});
});
v.addEventListener('seeked',()=>v.play().catch(()=>{}));
v.addEventListener('timeupdate',()=>{
  if(clipEnd!==null&&v.currentTime>=clipEnd){
    v.currentTime=clipStart;v.play().catch(()=>{});
  }
});
fetch('/api/task').then(r=>r.json()).then(t=>{T=t;
  document.getElementById('tot').textContent=t.clips.length;
  document.getElementById('modeName').textContent=t.mode;
  const first=t.clips.findIndex(c=>!complete(c));
  i=first<0?0:first;render();});
function cur(){return T.clips[i];}
function complete(c){
  const actionDone=c.label_source==='human'||c.label_source==='cleared';
  const contextDone=c.context_label_source==='human';
  return T.mode==='action'?actionDone:T.mode==='context'?contextDone:(actionDone&&contextDone);
}
function render(){
  const c=cur();
  document.getElementById('pos').textContent=i+1;
  document.getElementById('cid').textContent=c.clip_id;
  document.getElementById('done').textContent=T.clips.filter(complete).length;
  const chosen=(c.event_types&&c.event_types[0])||null;
  const pre=c.primary_event_type||c.fsm_event_type||null;
  document.getElementById('classes').innerHTML=T.classes.map((name,n)=>{
    let cls='cls';
    if(chosen===name&&c.label_source==='human')cls+=' cur';
    else if(pre===name)cls+=' pre';
    const handler=T.mode==='context'?'':`onclick="pick('${name}')"`;
    return `<div class="${cls}" ${handler}>${n+1}. ${name}</div>`;}).join('');
  const fields=[
    ['match_phase','match_phases'],
    ['physical_action','physical_actions'],
    ['context_role','context_roles'],
    ['reception_quality','reception_qualities']
  ];
  document.getElementById('contextFields').innerHTML=fields.map(([field,tax])=>{
    const help=T.context_help[field]||{};
    const options=(T.taxonomies[tax]||[]).map(value=>
      `<option value="${value}" ${c[field]===value?'selected':''}>${value} — ${(help.values||{})[value]||''}</option>`).join('');
    const selectedHelp=(help.values||{})[c[field]]||'';
    return `<div class="field"><label>${help.label||field}</label>`+
      `<div class="field-help">${help.description||''}</div>`+
      `<select onchange="setContext('${field}',this.value)">${options}</select>`+
      `<div class="value-help">${selectedHelp}</div></div>`;
  }).join('');
  document.getElementById('meta').textContent=
    `frames ${c.start_frame}-${c.end_frame} (contact ${c.contact_frame}) · `+
    `source ${c.source_stem} · label_source ${c.label_source||'none'}`;
  const vlm=c.vlm_prediction;
  const ve=document.getElementById('vlm');
  if(vlm){
    const conf=Math.round(100*(vlm.confidence||0));
    ve.textContent=`VLM ${vlm.action} (${conf}%) · ${vlm.reason||'no reason'} · ${vlm.seconds||'?'}s`;
    ve.style.display='block';
  }else{
    ve.style.display='none';
  }
  clipStart=0;clipEnd=null;
  document.getElementById('contextToggle').textContent=showRally?'show contact clip':'show full rally context';
  if(showRally&&c.source_video){
    const fps=Number(c.fps)||30;
    clipStart=Math.max(0,Number(c.rally_context_start_frame??c.start_frame??0)/fps);
    clipEnd=(Number(c.rally_context_end_frame??c.end_frame??0)+1)/fps;
    v.src='/source/'+encodeURIComponent(c.clip_id);
  }else if(c.path){
    v.src='/clip/'+c.clip_id+'.mp4';
  }else if(c.source_video){
    const fps=Number(c.fps)||30;
    clipStart=Math.max(0,Number(c.start_frame||0)/fps);
    clipEnd=(Number(c.end_frame||c.start_frame||0)+1)/fps;
    v.src='/source/'+encodeURIComponent(c.clip_id);
  }else{
    v.removeAttribute('src');
  }
  v.load();
  fetch('/api/traj/'+c.clip_id).then(r=>r.json()).then(drawTraj);
}
function drawTraj(d){
  tc.clearRect(0,0,280,200);
  if(!d.x||!d.x.length)return;
  const xs=d.x,ys=d.y,sp=d.speed||[];
  const minx=Math.min(...xs),maxx=Math.max(...xs),miny=Math.min(...ys),maxy=Math.max(...ys);
  const sx=v=>10+(v-minx)/Math.max(maxx-minx,1)*260, sy=v=>10+(v-miny)/Math.max(maxy-miny,1)*180;
  const maxsp=Math.max(1,...sp);
  for(let k=1;k<xs.length;k++){
    const f=sp.length?sp[k]/maxsp:0;
    tc.strokeStyle=`rgb(${Math.round(80+175*f)},${Math.round(220-160*f)},${Math.round(120-100*f)})`;
    tc.lineWidth=2;tc.beginPath();
    tc.moveTo(sx(xs[k-1]),sy(ys[k-1]));tc.lineTo(sx(xs[k]),sy(ys[k]));tc.stroke();
  }
  tc.fillStyle='#7fd';tc.beginPath();tc.arc(sx(xs[0]),sy(ys[0]),4,0,7);tc.fill();
  tc.fillStyle='#f55';tc.beginPath();tc.arc(sx(xs.at(-1)),sy(ys.at(-1)),4,0,7);tc.fill();
}
function pick(label){const c=cur();
  c.event_types=[label];c.primary_event_type=label;c.label_source='human';
  fetch('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({clip_id:c.clip_id,label:label})});
  if(i<T.clips.length-1){i++;}
  render();}
function setContext(field,value){const c=cur();c[field]=value;c.context_label_source='draft';
  let context={[field]:value};
  if(field==='match_phase'&&value==='non_game'){
    c.physical_action='unknown';c.context_role='unknown';c.reception_quality='unknown';
    context={match_phase:value,physical_action:'unknown',context_role:'unknown',reception_quality:'unknown'};
  }else if(field==='match_phase'&&value==='serve_flight'){
    c.physical_action='serve';c.context_role='serve';c.reception_quality='not_applicable';
    context={match_phase:value,physical_action:'serve',context_role:'serve',reception_quality:'not_applicable'};
  }
  fetch('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({clip_id:c.clip_id,context:context,draft_context:true})});render();}
function acceptContext(){const c=cur();
  const context={match_phase:c.match_phase,physical_action:c.physical_action,
    context_role:c.context_role,reception_quality:c.reception_quality};
  c.context_label_source='human';
  fetch('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({clip_id:c.clip_id,context:context,accept_context:true})});
  if(i<T.clips.length-1){i++;}render();}
function toggleContext(){showRally=!showRally;render();}
document.addEventListener('keydown',e=>{
  const k=e.key.toLowerCase();
  const n=parseInt(k,10);
  if(n>=1&&n<=T.classes.length&&T.mode!=='context'){pick(T.classes[n-1]);}
  else if(k==='enter'){
    if(T.mode==='context'){acceptContext();}
    else if(T.mode==='both'&&cur().label_source==='human'){acceptContext();}
    else {const p=cur().primary_event_type||cur().fsm_event_type;
      if(p&&T.classes.includes(p))pick(p);}
  }
  else if(k==='arrowright'){if(i<T.clips.length-1){i++;render();}}
  else if(k==='arrowleft'){if(i>0){i--;render();}}
  else if(k==='r'){v.currentTime=clipStart;v.play();}
  else if(k==='c'){toggleContext();}
  else if(k==='d'){const c=cur();c.event_types=[];c.label_source='cleared';
    fetch('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({clip_id:c.clip_id,clear:true})});render();}
  else return;
  e.preventDefault();});
</script>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, help="manifest.json or manifest_predicted.json")
    ap.add_argument("--features-dir", default=None, help="default: <manifest dir>/features")
    ap.add_argument("--output", default=None, help="default: <manifest dir>/manifest_labeled.json")
    ap.add_argument(
        "--mode", choices=("action", "context", "both"), default="both",
        help="annotation queue and Enter-key behavior (default: both)",
    )
    ap.add_argument("--port", type=int, default=5056)
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    manifest = upgrade_manifest(json.loads(manifest_path.read_text(encoding="utf-8")))
    out = Path(args.output) if args.output else manifest_path.parent / "manifest_labeled.json"
    # Resume from a previous session rather than re-labeling from scratch.
    if out.exists():
        prior = {c["clip_id"]: c for c in json.loads(out.read_text(encoding="utf-8"))["clips"]}
        for clip in manifest["clips"]:
            saved = prior.get(clip["clip_id"])
            if saved and (
                saved.get("label_source") in ("human", "cleared")
                or saved.get("context_label_source") == "human"
            ):
                clip.update(saved)

    STATE.update(
        manifest=manifest,
        clips=manifest["clips"],
        by_id={c["clip_id"]: c for c in manifest["clips"]},
        paths={c["clip_id"]: c.get("path") for c in manifest["clips"]},
        sources={c["clip_id"]: c.get("source_video") for c in manifest["clips"]},
        features_dir=Path(args.features_dir) if args.features_dir
        else manifest_path.parent / "features",
        output=out,
        mode=args.mode,
    )
    human = count_human()
    with_clip = sum(1 for c in manifest["clips"] if c.get("path"))
    print(f"{len(manifest['clips'])} clips ({with_clip} with video, {human} already human-labeled)")
    print(f"Classes: {', '.join(f'{n+1}={c}' for n, c in enumerate(EVENT_TYPES))}")
    print(f"Saving to {out}")
    print(f"Open http://localhost:{args.port}")
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
