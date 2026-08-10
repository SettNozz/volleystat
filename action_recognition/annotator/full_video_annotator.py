#!/usr/bin/env python3
"""Annotate volleyball events directly on complete source videos.

The UI is intentionally contact-first: select the team once, then add events
with one button or keyboard shortcut while the original video keeps its full
rally context. Serve reception asks for one extra quality choice. Non-game
periods are marked as a start/end range.

Usage:
    python -m action_recognition.annotator.full_video_annotator \
        --manifest data/event_clips_4k_2026-07-31_v2/human_393/manifest.json
    # open http://127.0.0.1:5058
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from flask import Flask, jsonify, request, send_file


SCHEMA_VERSION = 1
TEAM_SIDES = ("team_b", "team_a")
SEGMENT_PHASES = ("between_rallies", "non_game")
LEGACY_SEGMENT_PHASES = {"serve_setup", "dead_ball"}
RECEPTION_QUALITIES = (
    "unknown",
    "error",
    "overpass",
    "negative",
    "positive",
    "perfect",
    "controlled",
)
EVENT_PRESETS: dict[str, dict[str, str]] = {
    "serve": {
        "label": "Serve",
        "match_phase": "serve_flight",
        "physical_action": "serve",
        "context_role": "serve",
        "reception_quality": "not_applicable",
    },
    "reception": {
        "label": "Serve reception",
        "match_phase": "rally_live",
        "physical_action": "forearm_pass",
        "context_role": "serve_reception",
        "reception_quality": "unknown",
    },
    "dig": {
        "label": "Defensive dig",
        "match_phase": "rally_live",
        "physical_action": "forearm_pass",
        "context_role": "defensive_dig",
        "reception_quality": "not_applicable",
    },
    "set": {
        "label": "Set / pass",
        "match_phase": "rally_live",
        "physical_action": "overhead_set",
        "context_role": "set",
        "reception_quality": "not_applicable",
    },
    "attack": {
        "label": "Attack",
        "match_phase": "rally_live",
        "physical_action": "attack_hit",
        "context_role": "attack",
        "reception_quality": "not_applicable",
    },
    "block": {
        "label": "Block touch",
        "match_phase": "rally_live",
        "physical_action": "block_touch",
        "context_role": "block",
        "reception_quality": "not_applicable",
    },
    "free_ball": {
        "label": "Free ball",
        "match_phase": "rally_live",
        "physical_action": "other_touch",
        "context_role": "free_ball",
        "reception_quality": "not_applicable",
    },
    "other": {
        "label": "Other touch",
        "match_phase": "rally_live",
        "physical_action": "other_touch",
        "context_role": "other",
        "reception_quality": "not_applicable",
    },
}

app = Flask(__name__)
STATE: dict[str, Any] = {}
SAVE_LOCK = threading.Lock()


@app.after_request
def disable_annotation_cache(response):
    if request.path == "/" or request.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _source_videos(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    videos: dict[str, dict[str, Any]] = {}
    for clip in manifest.get("clips", []):
        source_video = clip.get("source_video")
        if not source_video:
            continue
        video_id = str(clip.get("source_stem") or Path(source_video).stem)
        entry = videos.setdefault(
            video_id,
            {
                "video_id": video_id,
                "name": video_id,
                "path": str(source_video),
                "fps": float(clip.get("fps") or 30.0),
                "legacy_contact_count": 0,
            },
        )
        entry["legacy_contact_count"] += 1
    return list(videos.values())


def build_project(manifest: Mapping[str, Any]) -> dict[str, Any]:
    videos = _source_videos(manifest)
    if not videos:
        raise ValueError("manifest has no clips with source_video")
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "videos": videos,
        "annotations": [],
        "segments": [],
    }


def upgrade_project(project: dict[str, Any]) -> dict[str, Any]:
    """Merge legacy between-rally phase labels without losing provenance."""
    for segment in project.get("segments", []):
        phase = segment.get("match_phase")
        if phase in LEGACY_SEGMENT_PHASES:
            segment.setdefault("legacy_match_phase", phase)
            segment["match_phase"] = "between_rallies"
    return project


def _save() -> None:
    STATE["project"]["updated_at"] = _utc_now()
    output: Path = STATE["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with SAVE_LOCK:
        temporary.write_text(
            json.dumps(STATE["project"], ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        temporary.replace(output)


def _video(video_id: str) -> dict[str, Any]:
    video = STATE["videos"].get(video_id)
    if video is None:
        raise ValueError(f"unknown video_id {video_id!r}")
    return video


def normalize_annotation(payload: Mapping[str, Any]) -> dict[str, Any]:
    video_id = str(payload.get("video_id") or "")
    video = _video(video_id)
    event_type = str(payload.get("event_type") or "")
    if event_type not in EVENT_PRESETS:
        raise ValueError(f"unknown event_type {event_type!r}")
    team_side = str(payload.get("team_side") or "")
    if team_side not in TEAM_SIDES:
        raise ValueError(f"unknown team_side {team_side!r}")
    frame = int(payload.get("frame", -1))
    if frame < 0:
        raise ValueError("frame must be non-negative")

    preset = EVENT_PRESETS[event_type]
    quality = str(payload.get("reception_quality") or preset["reception_quality"])
    if event_type == "reception":
        if quality not in RECEPTION_QUALITIES:
            raise ValueError(f"unknown reception_quality {quality!r}")
    else:
        quality = "not_applicable"

    return {
        "annotation_id": str(payload.get("annotation_id") or f"ann_{uuid4().hex[:12]}"),
        "video_id": video_id,
        "frame": frame,
        "time_seconds": frame / float(video["fps"]),
        "team_side": team_side,
        "event_type": event_type,
        "match_phase": preset["match_phase"],
        "physical_action": preset["physical_action"],
        "context_role": preset["context_role"],
        "reception_quality": quality,
        "is_service_reception": event_type == "reception",
        "label_source": "human",
        "created_at": str(payload.get("created_at") or _utc_now()),
    }


@app.get("/")
def index():
    return PAGE


@app.get("/api/project")
def api_project():
    return jsonify(
        {
            **STATE["project"],
            "event_presets": EVENT_PRESETS,
            "team_labels": {
                "team_b": "Near / camera side",
                "team_a": "Far / opponent side",
            },
            "output": str(STATE["output"]),
        }
    )


@app.get("/video/<video_id>")
def stream_video(video_id: str):
    try:
        path = Path(_video(video_id)["path"])
    except ValueError:
        return "", 404
    if not path.is_file():
        return "", 404
    mimetype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return send_file(path, mimetype=mimetype, conditional=True)


@app.post("/api/annotations")
def add_annotation():
    try:
        annotation = normalize_annotation(request.get_json(force=True))
    except (TypeError, ValueError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    STATE["project"]["annotations"].append(annotation)
    STATE["project"]["annotations"].sort(
        key=lambda item: (item["video_id"], item["frame"], item["created_at"])
    )
    _save()
    return jsonify({"ok": True, "annotation": annotation})


@app.delete("/api/annotations/<annotation_id>")
def delete_annotation(annotation_id: str):
    annotations = STATE["project"]["annotations"]
    original_count = len(annotations)
    annotations[:] = [
        item for item in annotations if item.get("annotation_id") != annotation_id
    ]
    if len(annotations) == original_count:
        return jsonify({"ok": False, "error": "annotation not found"}), 404
    _save()
    return jsonify({"ok": True})


@app.post("/api/segments")
def add_segment():
    payload = request.get_json(force=True)
    try:
        video_id = str(payload.get("video_id") or "")
        video = _video(video_id)
        start_frame = int(payload.get("start_frame", -1))
        end_frame = int(payload.get("end_frame", -1))
        match_phase = str(payload.get("match_phase") or "non_game")
        if start_frame < 0 or end_frame < start_frame:
            raise ValueError("invalid phase frame range")
        if match_phase not in SEGMENT_PHASES:
            raise ValueError(f"unknown segment match_phase {match_phase!r}")
    except (TypeError, ValueError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    segment = {
        "segment_id": f"seg_{uuid4().hex[:12]}",
        "video_id": video_id,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start_seconds": start_frame / float(video["fps"]),
        "end_seconds": end_frame / float(video["fps"]),
        "match_phase": match_phase,
        "team_side": None,
        "physical_action": "unknown",
        "context_role": "unknown",
        "reception_quality": "unknown",
        "label_source": "human",
        "created_at": _utc_now(),
    }
    STATE["project"]["segments"].append(segment)
    STATE["project"]["segments"].sort(
        key=lambda item: (item["video_id"], item["start_frame"])
    )
    _save()
    return jsonify({"ok": True, "segment": segment})


@app.delete("/api/segments/<segment_id>")
def delete_segment(segment_id: str):
    segments = STATE["project"]["segments"]
    original_count = len(segments)
    segments[:] = [item for item in segments if item.get("segment_id") != segment_id]
    if len(segments) == original_count:
        return jsonify({"ok": False, "error": "segment not found"}), 404
    _save()
    return jsonify({"ok": True})


PAGE = r"""
<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Full video volleyball annotator</title>
<style>
:root{--bg:#0d1117;--panel:#151b23;--line:#2b3440;--text:#e6edf3;--muted:#8b98a7;--cyan:#4dd7c8;--near:#ffd75e;--far:#70a5ff;--danger:#ff6b6b;--serve:#b88cff;--receive:#4dd7c8;--build:#70a5ff;--finish:#ff8b5c}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px system-ui,sans-serif}
button,select{font:inherit;color:inherit;background:#111821;border:1px solid var(--line);border-radius:7px}
button{cursor:pointer}button:hover{filter:brightness(1.17)}button:disabled{opacity:.5;cursor:wait}
#top{position:sticky;top:0;z-index:5;display:flex;gap:14px;align-items:center;flex-wrap:wrap;padding:9px 14px;background:#111720;border-bottom:1px solid var(--line)}
#top select{padding:7px;min-width:250px}.status{color:var(--muted)}.status b{color:var(--cyan)}
#layout{display:grid;grid-template-columns:minmax(520px,1fr) 420px;gap:12px;padding:12px;align-items:start}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px}
video{width:100%;max-height:68vh;background:#000;border-radius:8px;display:block}
#videoStage{position:relative;width:100%;background:#000;border-radius:8px;overflow:hidden}#videoStage video{border-radius:0}
.context-rail{position:absolute;top:10px;bottom:66px;z-index:2;display:none;flex-direction:column;justify-content:center;align-items:center;padding:10px 8px;text-align:center;pointer-events:none;overflow:hidden;transition:background .15s}.context-rail.left{left:0}.context-rail.right{right:0}.rail-kicker{font-size:10px;letter-spacing:.12em;text-transform:uppercase;opacity:.75}.rail-main{margin-top:7px;font-size:17px;font-weight:800;line-height:1.12;text-transform:uppercase;overflow-wrap:anywhere}.rail-detail{margin-top:8px;font-size:11px;line-height:1.3;opacity:.82;overflow-wrap:anywhere}
.transport{display:grid;grid-template-columns:repeat(9,1fr);gap:6px;margin-top:8px}.transport button{padding:7px 3px}
#clock{font:600 16px ui-monospace,monospace;color:var(--cyan);margin:8px 2px 2px}
.section{margin-bottom:13px}.section h3{font-size:12px;text-transform:uppercase;letter-spacing:.09em;color:var(--muted);margin:0 0 7px}
.team{display:grid;grid-template-columns:1fr 1fr;gap:7px}.team button{padding:11px}.team .none-button{grid-column:1/-1}.team .active.none{border-color:#8b98a7;background:#28313c}.team .active.near{border-color:var(--near);background:#382f14}.team .active.far{border-color:var(--far);background:#15294b}
.events{display:grid;grid-template-columns:1fr 1fr;gap:7px}.events button{padding:11px;text-align:left;font-weight:650}.events small{float:right;color:#111;background:#ffffffb8;border-radius:4px;padding:1px 5px}
.serve{border-left:4px solid var(--serve)}.receive{border-left:4px solid var(--receive)}.build{border-left:4px solid var(--build)}.finish{border-left:4px solid var(--finish)}
#quality{display:none;padding:9px;background:#10221f;border:1px solid #28665e;border-radius:8px;margin:8px 0}.qualities{display:grid;grid-template-columns:repeat(2,1fr);gap:5px}.qualities button{padding:8px}
.phase-buttons{display:grid;grid-template-columns:1fr;gap:6px}.phase-buttons button{width:100%;padding:10px;text-align:left}.phase-buttons .between{border-left:4px solid var(--cyan)}.phase-buttons .nongame{border-left:4px solid #d08a43}.recording{background:#4a281c!important;border-color:#ff9c59!important}
.hint{font-size:12px;line-height:1.4;color:var(--muted);margin-top:6px}.save{color:var(--cyan)}
#timelinePanel{margin:0 12px 12px}.timeline{height:32px;background:#0b0f14;border:1px solid var(--line);border-radius:6px;position:relative;cursor:pointer;overflow:hidden}
.marker{position:absolute;top:4px;width:5px;height:22px;border-radius:2px;transform:translateX(-2px);z-index:2}.segment{position:absolute;top:0;height:100%;background:#b1793b55;border-left:1px solid #e8a557;border-right:1px solid #e8a557}
.playhead{position:absolute;top:0;width:2px;height:100%;background:white;z-index:3;pointer-events:none}
#history{max-height:245px;overflow:auto}.row{display:grid;grid-template-columns:64px 72px 1fr 30px;gap:7px;align-items:center;padding:6px;border-bottom:1px solid #242d38}.row:hover{background:#1b2430}.row button{border:0;background:transparent;color:var(--danger);font-size:17px}
.tag{font-size:11px;padding:2px 5px;border-radius:4px;text-align:center}.tag.near{background:#4a3e17;color:var(--near)}.tag.far{background:#17345e;color:var(--far)}
@media(max-width:980px){#layout{grid-template-columns:1fr}video{max-height:55vh}}
</style></head><body>
<div id="top">
  <select id="videoSelect" onchange="changeVideo(this.value)"></select>
  <span class="status">Events <b id="eventCount">0</b></span>
  <span class="status">Score from next serves: Near <b id="nearScore">0</b> · Far <b id="farScore">0</b></span>
  <span class="status save" id="saveState">autosaved</span>
</div>
<div id="layout">
  <div class="panel">
    <div id="videoStage">
      <video id="video" controls playsinline></video>
      <div id="leftRail" class="context-rail left"><div class="rail-kicker">Current phase</div><div id="leftRailMain" class="rail-main">Unlabeled</div><div id="leftRailDetail" class="rail-detail"></div></div>
      <div id="rightRail" class="context-rail right"><div class="rail-kicker">Contact class</div><div id="rightRailMain" class="rail-main">No contact</div><div id="rightRailDetail" class="rail-detail"></div></div>
    </div>
    <div id="clock">00:00.000 · frame 0</div>
    <div class="transport">
      <button onclick="seek(-10)">−10s</button><button onclick="seek(-2)">−2s</button>
      <button onclick="step(-1)">◀ frame</button><button onclick="togglePlay()">Play / pause</button>
      <button onclick="step(1)">frame ▶</button><button onclick="seek(2)">+2s</button><button onclick="seek(10)">+10s</button>
      <button onclick="undoLast()">Undo (Z)</button>
      <select id="speed" onchange="setSpeed(this.value)"><option>.25</option><option>.5</option><option selected>1</option><option>1.5</option><option>2</option></select>
    </div>
  </div>
  <div class="panel">
    <div class="section"><h3>1 · Team touching the ball — contacts only</h3><div class="team">
      <button id="team_none" class="none-button active none" onclick="setTeam(null)">No team / no ball</button>
      <button id="team_b" onclick="setTeam('team_b')">Near / camera <b>N</b></button>
      <button id="team_a" onclick="setTeam('team_a')">Far / opponent <b>F</b></button>
    </div></div>
    <div class="section"><h3>2 · Contact at the current frame</h3><div class="events">
      <button class="serve" onclick="addEvent('serve')">Serve <small>S</small></button>
      <button class="receive" onclick="startReception()">Serve reception <small>R</small></button>
      <button class="receive" onclick="addEvent('dig')">Defensive dig <small>D</small></button>
      <button class="build" onclick="addEvent('set')">Set / pass <small>E</small></button>
      <button class="finish" onclick="addEvent('attack')">Attack <small>A</small></button>
      <button class="finish" onclick="addEvent('block')">Block touch <small>B</small></button>
      <button class="build" onclick="addEvent('free_ball')">Free ball <small>G</small></button>
      <button onclick="addEvent('other')">Other touch <small>O</small></button>
    </div></div>
    <div id="quality"><h3>Reception quality</h3><div class="qualities">
      <button onclick="finishReception('error')">Error</button><button onclick="finishReception('overpass')">Overpass</button>
      <button onclick="finishReception('negative')">Negative</button><button onclick="finishReception('positive')">Positive</button>
      <button onclick="finishReception('perfect')">Perfect</button><button onclick="finishReception('controlled')">Controlled / unclear</button>
    </div><div class="hint">Esc cancels. The contact frame was captured when you pressed Reception.</div></div>
    <div class="section"><h3>3 · Period without a ball contact</h3><div class="phase-buttons">
      <button id="phase_between_rallies" class="between" onclick="toggleSegment('between_rallies')">Start between-rallies / serve-setup range</button>
      <button id="phase_non_game" class="nongame" onclick="toggleSegment('non_game')">Start non-game range</button>
    </div>
      <div class="hint">Between rallies includes the dead-ball reset and waiting for the next serve. Non-game is warm-up, timeout, between sets, referee discussion, or unrelated activity.</div>
    </div>
    <div class="hint"><b>Keyboard:</b> Space play/pause · ←/→ one frame · Shift+←/→ ±2s · J/L ±2s · Shift+J/L ±10s · Z multi-level undo.</div>
  </div>
</div>
<div id="timelinePanel" class="panel">
  <div class="section"><h3>Timeline — click to seek</h3><div id="timeline" class="timeline" onclick="timelineSeek(event)"><div id="playhead" class="playhead"></div></div></div>
  <div class="section"><h3>Annotations in this video</h3><div id="history"></div></div>
</div>
<script>
let P=null,currentVideo=null,team=null,pendingReception=null,pendingSegment=null,busy=false,undoPending=0,undoRunning=false;
const v=document.getElementById('video');
const colors={serve:'#b88cff',reception:'#4dd7c8',dig:'#39aa9e',set:'#70a5ff',attack:'#ff8b5c',block:'#ff6262',free_ball:'#76b7ff',other:'#aaa'};
const phaseLabels={between_rallies:'Between rallies / serve setup',non_game:'Non-game',serve_flight:'Serve flight',rally_live:'Rally live',unlabeled:'Unlabeled'};
const phaseColors={between_rallies:'#26756ecc',non_game:'#895522cc',serve_flight:'#6949aacc',rally_live:'#176a58cc',unlabeled:'#242b35cc'};
const manualSegmentPhases=['between_rallies','non_game'];
fetch('/api/project',{cache:'no-store'}).then(r=>r.json()).then(p=>{P=p;
  videoSelect.innerHTML=P.videos.map(x=>`<option value="${x.video_id}">${x.name}</option>`).join('');
  changeVideo(P.videos[0].video_id);
});
function videoInfo(){return P.videos.find(x=>x.video_id===currentVideo)}
function fps(){return Number(videoInfo().fps)||30}
function frame(){return Math.max(0,Math.round(v.currentTime*fps()))}
function currentAnnotations(){return P.annotations.filter(x=>x.video_id===currentVideo).sort((a,b)=>a.frame-b.frame)}
function currentSegments(){return P.segments.filter(x=>x.video_id===currentVideo).sort((a,b)=>a.start_frame-b.start_frame)}
function changeVideo(id){currentVideo=id;pendingReception=null;pendingSegment=null;
  document.getElementById('quality').style.display='none';setTeam(null);updateSegmentButtons();
  v.src='/video/'+encodeURIComponent(id);v.load();render();}
function setTeam(side){team=side;document.getElementById('team_none').className=side===null?'none-button active none':'none-button';document.getElementById('team_b').className=side==='team_b'?'active near':'';document.getElementById('team_a').className=side==='team_a'?'active far':'';document.querySelectorAll('.events button').forEach(button=>button.disabled=side===null);updateContextRails()}
function setBusy(value){busy=value;document.getElementById('saveState').textContent=value?'saving…':'autosaved';if(!value&&undoPending&&!undoRunning)queueMicrotask(drainUndo)}
function togglePlay(){v.paused?v.play():v.pause()}
function seek(seconds){v.currentTime=Math.max(0,Math.min(v.duration||Infinity,v.currentTime+seconds))}
function step(direction){v.pause();v.currentTime=Math.max(0,v.currentTime+direction/fps())}
function setSpeed(value){v.playbackRate=Number(value)}
function stamp(){return{video_id:currentVideo,frame:frame(),team_side:team}}
function requireTeam(){if(team!==null)return true;alert('Select Near or Far before adding a ball contact.');return false}
async function addEvent(type,quality=null,captured=null){if(busy||(!captured&&!requireTeam()))return;setBusy(true);
  const body={...(captured||stamp()),event_type:type};if(quality)body.reception_quality=quality;
  const response=await fetch('/api/annotations',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const data=await response.json();if(data.ok)P.annotations.push(data.annotation);else alert(data.error);setBusy(false);render();}
function startReception(){if(!requireTeam())return;pendingReception=stamp();v.pause();document.getElementById('quality').style.display='block'}
function finishReception(quality){const captured=pendingReception;pendingReception=null;document.getElementById('quality').style.display='none';addEvent('reception',quality,captured)}
async function removeAnnotation(id){if(busy)return;setBusy(true);const r=await fetch('/api/annotations/'+id,{method:'DELETE'});if(r.ok)P.annotations=P.annotations.filter(x=>x.annotation_id!==id);setBusy(false);render()}
async function removeSegment(id){if(busy)return;setBusy(true);const r=await fetch('/api/segments/'+id,{method:'DELETE'});if(r.ok)P.segments=P.segments.filter(x=>x.segment_id!==id);setBusy(false);render()}
function latestUndoable(){const items=[
  ...currentAnnotations().map(x=>({kind:'annotation',id:x.annotation_id,created_at:x.created_at,position:x.frame})),
  ...currentSegments().map(x=>({kind:'segment',id:x.segment_id,created_at:x.created_at,position:x.start_frame}))
];items.sort((a,b)=>String(a.created_at||'').localeCompare(String(b.created_at||''))||a.position-b.position);return items.at(-1)}
function undoLast(){undoPending++;drainUndo()}
async function drainUndo(){if(undoRunning||busy)return;undoRunning=true;
  while(undoPending>0){const item=latestUndoable();if(!item){undoPending=0;break}undoPending--;setBusy(true);
    const path=item.kind==='annotation'?'/api/annotations/':'/api/segments/';const r=await fetch(path+item.id,{method:'DELETE'});
    if(r.ok){if(item.kind==='annotation')P.annotations=P.annotations.filter(x=>x.annotation_id!==item.id);else P.segments=P.segments.filter(x=>x.segment_id!==item.id)}
    setBusy(false);render();
  }undoRunning=false;setBusy(false)}
function toggleSegment(matchPhase){if(pendingSegment===null){pendingSegment={match_phase:matchPhase,start_frame:frame()};setTeam(null);v.pause();updateSegmentButtons();return}if(pendingSegment.match_phase!==matchPhase){alert('Finish the active phase range before starting another one.');return}saveSegment()}
async function saveSegment(){const start=pendingSegment.start_frame,end=frame(),matchPhase=pendingSegment.match_phase;if(end<start){alert('Move forward before finishing the range.');return}setBusy(true);
  const r=await fetch('/api/segments',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({video_id:currentVideo,start_frame:start,end_frame:end,match_phase:matchPhase})});
  const data=await r.json();if(data.ok){P.segments.push(data.segment);pendingSegment=null}else alert(data.error);setBusy(false);updateSegmentButtons();render()}
function updateSegmentButtons(){for(const phase of manualSegmentPhases){const button=document.getElementById('phase_'+phase),active=pendingSegment?.match_phase===phase;button.textContent=active?`Finish ${phaseLabels[phase]} (started at frame ${pendingSegment.start_frame})`:`Start ${phaseLabels[phase].toLowerCase()}`;button.classList.toggle('recording',active)}}
function fmt(seconds){if(!Number.isFinite(seconds))return'00:00.000';const m=Math.floor(seconds/60),s=seconds-m*60;return`${String(m).padStart(2,'0')}:${s.toFixed(3).padStart(6,'0')}`}
function timelineSeek(e){if(!v.duration)return;const r=e.currentTarget.getBoundingClientRect();v.currentTime=Math.max(0,Math.min(v.duration,(e.clientX-r.left)/r.width*v.duration))}
function seekFrame(f){v.pause();v.currentTime=f/fps()}
function score(){const serves=currentAnnotations().filter(x=>x.event_type==='serve');const result={team_b:0,team_a:0};for(let i=1;i<serves.length;i++)result[serves[i].team_side]++;return result}
function frameContext(currentFrame){const anns=currentAnnotations(),segment=currentSegments().find(s=>s.start_frame<=currentFrame&&currentFrame<=s.end_frame);let phase=segment?.match_phase||null;
  let previous=null,closest=null,closestDistance=Infinity;for(const annotation of anns){if(annotation.frame<=currentFrame)previous=annotation;const distance=Math.abs(annotation.frame-currentFrame);if(distance<closestDistance){closest=annotation;closestDistance=distance}}
  if(!phase){if(previous)phase=previous.event_type==='serve'?'serve_flight':'rally_live';else phase='unlabeled'}
  return{phase,contact:closestDistance<=5?closest:null,contactDistance:closestDistance,previous};}
function updateContextRails(){if(!P||!currentVideo||!v.videoWidth||!v.clientHeight)return;const width=v.clientWidth,height=v.clientHeight,contentWidth=Math.min(width,height*(v.videoWidth/v.videoHeight)),railWidth=Math.max(0,(width-contentWidth)/2);
  const left=document.getElementById('leftRail'),right=document.getElementById('rightRail'),visible=railWidth>=72;for(const rail of[left,right]){rail.style.display=visible?'flex':'none';rail.style.width=railWidth+'px'}if(!visible)return;
  const currentFrame=frame(),ctx=frameContext(currentFrame),phase=ctx.phase in phaseLabels?ctx.phase:'unlabeled';left.style.background=`linear-gradient(135deg,${phaseColors[phase]},#05070acc)`;leftRailMain.textContent=phaseLabels[phase];leftRailDetail.textContent=phase==='serve_flight'?'From serve contact to the first receiving contact':phase==='rally_live'?'Ball is live after the serve reception':'';
  const contact=ctx.contact;if(contact){const delta=currentFrame-contact.frame,position=delta===0?'Exact contact frame':delta<0?`${Math.abs(delta)} frame${Math.abs(delta)===1?'':'s'} before contact`:`${delta} frame${delta===1?'':'s'} after contact`;right.style.background=`linear-gradient(225deg,${colors[contact.event_type]||'#666'}cc,#05070acc)`;rightRailMain.textContent=P.event_presets[contact.event_type]?.label||contact.event_type;rightRailDetail.textContent=`${contact.team_side==='team_b'?'Near / camera':'Far / opponent'} · ${position}${contact.event_type==='reception'?' · '+contact.reception_quality:''}`;
  }else{right.style.background='linear-gradient(225deg,#242b35cc,#05070acc)';rightRailMain.textContent='No contact';rightRailDetail.textContent=team===null?'No team / no ball':`Selected: ${team==='team_b'?'Near / camera':'Far / opponent'}`;}}
function render(){if(!P||!currentVideo)return;const anns=currentAnnotations(),segs=currentSegments(),duration=v.duration||0;
  eventCount.textContent=anns.length;const sc=score();nearScore.textContent=sc.team_b;farScore.textContent=sc.team_a;
  const timeline=document.getElementById('timeline');timeline.querySelectorAll('.marker,.segment').forEach(x=>x.remove());
  if(duration){segs.forEach(s=>{const el=document.createElement('div');el.className='segment';el.style.left=(s.start_seconds/duration*100)+'%';el.style.width=((s.end_seconds-s.start_seconds)/duration*100)+'%';el.style.background=phaseColors[s.match_phase]||phaseColors.non_game;el.title=`${phaseLabels[s.match_phase]||s.match_phase} ${fmt(s.start_seconds)}–${fmt(s.end_seconds)}`;timeline.appendChild(el)});
    anns.forEach(a=>{const el=document.createElement('div');el.className='marker';el.style.left=(a.time_seconds/duration*100)+'%';el.style.background=colors[a.event_type]||'#aaa';el.title=`${a.event_type} · ${a.team_side} · frame ${a.frame}`;el.onclick=e=>{e.stopPropagation();seekFrame(a.frame)};timeline.appendChild(el)})}
  const rows=[...anns.map(a=>({...a,kind:'event',sort:a.frame})),...segs.map(s=>({...s,kind:'segment',sort:s.start_frame}))].sort((a,b)=>b.sort-a.sort);
  history.innerHTML=rows.length?rows.map(x=>x.kind==='event'?`<div class="row" onclick="seekFrame(${x.frame})"><span>${fmt(x.time_seconds)}</span><span class="tag ${x.team_side==='team_b'?'near':'far'}">${x.team_side==='team_b'?'NEAR':'FAR'}</span><span>${P.event_presets[x.event_type]?.label||x.event_type}${x.event_type==='reception'?' · '+x.reception_quality:''}</span><button onclick="event.stopPropagation();removeAnnotation('${x.annotation_id}')">×</button></div>`:`<div class="row" onclick="seekFrame(${x.start_frame})"><span>${fmt(x.start_seconds)}</span><span class="tag">PHASE</span><span>${phaseLabels[x.match_phase]||x.match_phase} → ${fmt(x.end_seconds)}</span><button onclick="event.stopPropagation();removeSegment('${x.segment_id}')">×</button></div>`).join(''):'<div class="hint">No annotations yet. Mark ball contacts or no-contact phase ranges while watching the full video.</div>';updateContextRails();
}
v.addEventListener('loadedmetadata',render);v.addEventListener('durationchange',render);v.addEventListener('timeupdate',()=>{clock.textContent=`${fmt(v.currentTime)} · frame ${frame()}`;if(v.duration)playhead.style.left=(v.currentTime/v.duration*100)+'%';updateContextRails()});window.addEventListener('resize',updateContextRails);
document.addEventListener('keydown',e=>{if(e.target.matches('select,input'))return;const k=e.key.toLowerCase();
  if(k===' '){togglePlay()}else if(k==='arrowleft'){e.shiftKey?seek(-2):step(-1)}else if(k==='arrowright'){e.shiftKey?seek(2):step(1)}else if(k==='j'){seek(e.shiftKey?-10:-2)}else if(k==='l'){seek(e.shiftKey?10:2)}else if(k===','){step(-1)}else if(k==='.'){step(1)}else if(k==='n'){setTeam('team_b')}else if(k==='f'){setTeam('team_a')}else if(k==='s'){addEvent('serve')}else if(k==='r'){startReception()}else if(k==='d'){addEvent('dig')}else if(k==='e'){addEvent('set')}else if(k==='a'){addEvent('attack')}else if(k==='b'){addEvent('block')}else if(k==='g'){addEvent('free_ball')}else if(k==='o'){addEvent('other')}else if(k==='z'){undoLast()}else if(k==='escape'){pendingReception=null;quality.style.display='none'}else return;e.preventDefault()});
</script></body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="clip manifest used to discover source videos")
    parser.add_argument("--output", default=None, help="default: <manifest dir>/full_video_annotations.json")
    parser.add_argument("--port", type=int, default=5058)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output = Path(args.output) if args.output else manifest_path.parent / "full_video_annotations.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    project = upgrade_project(
        json.loads(output.read_text(encoding="utf-8"))
        if output.exists()
        else build_project(manifest)
    )
    source_by_id = {video["video_id"]: video for video in _source_videos(manifest)}
    for video in project.get("videos", []):
        if video["video_id"] in source_by_id:
            video.update(source_by_id[video["video_id"]])
    videos = {video["video_id"]: video for video in project["videos"]}
    STATE.clear()
    STATE.update(project=project, videos=videos, output=output)
    _save()

    print(f"{len(videos)} full videos, {len(project['annotations'])} saved events")
    print(f"Saving to {output}")
    print(f"Open http://127.0.0.1:{args.port}")
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
