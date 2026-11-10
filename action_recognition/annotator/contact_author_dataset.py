"""Source-bound human contact-author projects, rosters and versioned decisions.

Roster reference boxes identify people; they are never detector training boxes.
Player IDs are scoped to one source video and are never recycled on substitution.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

import cv2
import numpy as np

from src.tracking.player_identity import overlap
from src.tracking.player_contacts import side

from .contact_author_movement import movement_export, segments_for

SCHEMA = "contact_author_annotation_v1"
DECISIONS = {"player", "other_team", "unsure", "not_contact"}
EVENT_TYPES = {"serve", "reception", "pass", "set", "dig", "attack", "block", "free_ball", "other"}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1<<20), b""): value.update(block)
    return value.hexdigest()


def now(): return datetime.now(timezone.utc).isoformat()


def reviewed_contact(contact,state):
    """Apply an explicit human type override, preserving source event and timing."""
    row=deepcopy(contact);review=state.get("event_reviews",{}).get(contact["id"])
    if review:
        row["original_event"]={k:deepcopy(contact[k]) for k in
            ("type","raw_types","core_type","event_label_source","event_verified") if k in contact}
        row.update(type=review["type"],raw_types=[review["type"]],event_label_source="human",event_verified=True,event_review=deepcopy(review))
        row.pop("core_type",None)
    timing = state.get("timing_reviews", {}).get(contact["id"])
    if timing:
        row.update(frame=timing["frame"], timing_verified=True, timing_source="human", timing_review=deepcopy(timing))
    return row


def fingerprint_file(path):
    path = Path(path).resolve(); stat = path.stat()
    return {"path":str(path),"sha256":sha(path),"size":stat.st_size,"mtime_ns":stat.st_mtime_ns}


def write_new(path, value):
    raw = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x",encoding="utf-8") as stream: stream.write(raw)


def contact_queue(manifest, review):
    by_annotation = defaultdict(list)
    for event in review.get("events",[]):
        for key in event.get("annotation_ids",[]):
            by_annotation[(event["video_id"],key)].append((event,review["reviews"].get(event["id"],{})))
    groups = {}; excluded = []
    for clip in manifest["clips"]:
        stem = clip["source_stem"]
        linked = {event["id"]:(event,label) for key in clip.get("annotation_ids",[])
                  for event,label in by_annotation.get((stem,key),[])}
        if any(label.get("verified") and label.get("status")=="wrong_event" for _,label in linked.values()):
            excluded.append({"clip_id":clip["clip_id"],"source_video":stem,"reason":"human_review_wrong_event"})
            continue
        frames = {int(label["frame"]) for _,label in linked.values() if label.get("verified") and label.get("temporal_valid")}
        if len(frames)>1: raise ValueError("One training clip links different audited contact times")
        verified = bool(frames)
        frame = next(iter(frames)) if verified else int(clip["contact_frame"])
        # Merge exact audited duplicate touches, keeping all original links.
        key = (stem,frame) if verified else (stem,frame,clip["clip_id"])
        if key not in groups:
            groups[key] = {"id":f"{stem}:f{frame}:"+hashlib.sha256(str(key).encode()).hexdigest()[:10],
                "video_id":stem,"frame":frame,"type":clip["primary_event_type"],"raw_types":[],
                "clip_ids":[],"annotation_ids":[],"original_frames":[],"review_ids":[],
                "timing_verified":verified,"timing_source":"human_audited_contact" if verified else "human_original_contact_proposal",
                "event_label_source":"human","author_label_source":None}
        row = groups[key]
        row["clip_ids"].append(clip["clip_id"])
        row["annotation_ids"] += clip.get("annotation_ids",[])
        row["original_frames"].append(int(clip["contact_frame"]))
        row["review_ids"] += list(linked)
        row["raw_types"] += clip.get("full_event_types") or [clip["primary_event_type"]]
    result = sorted(groups.values(),key=lambda e:(e["video_id"],e["frame"],e["id"]))
    for row in result:
        for field in ("clip_ids","annotation_ids","original_frames","review_ids","raw_types"):
            row[field] = sorted(set(row[field]))
        row["type"] = row["raw_types"][0] if len(row["raw_types"])==1 else " / ".join(row["raw_types"])
    return result,excluded


def initial_roster(annotations, first_contact, target_side="b"):
    options = []
    for row in annotations["frames"]:
        if abs(row["f"]-first_contact)>900: continue
        players = [p for p in row.get("poses") or [] if side(p.get("team"))==target_side]
        if len(players)!=6: continue
        if any(len(p.get("bb") or [])!=4 or p["bb"][3]<=p["bb"][1] for p in players): continue
        penalty = sum(max((overlap(p["bb"],q["bb"]) for q in players if q is not p),default=0.) for p in players)
        quality = sum(np.mean([p["kp"][j][2] for j in (5,6,11,12)]) for p in players)
        # Prefer clear views around the first real contact, not early warm-up.
        score = quality-3*penalty-.003*abs(row["f"]-first_contact)
        options.append((score,row["f"],players))
    if not options: raise ValueError("No complete six-person cached frame; provide a different pose-enabled analysis")
    _,frame,players = max(options,key=lambda x:(x[0],-x[1]))
    players = sorted(players,key=lambda p:(p["bb"][0]+p["bb"][2])/2)
    return {"confirmed":False,"players":[{"player_id":f"P{i+1}","identity_confirmation":None,
             "references":[{"frame":frame,"bbox":p["bb"],"label_source":"model",
                            "scope":"identity_reference_only","raw_track_id":p.get("id")}]} for i,p in enumerate(players)]}


def prepare_project(manifest_path, analyses_dir, review_path, output):
    output = Path(output)
    if output.exists(): raise ValueError("Choose a fresh project directory")
    manifest_path,review_path = Path(manifest_path),Path(review_path)
    manifest = json.loads(manifest_path.read_bytes()); review = json.loads(review_path.read_bytes())
    if manifest.get("label_source")!="human" or not manifest.get("clips"):
        raise ValueError("Expected the human-labelled event training manifest")
    if review.get("schema")!="contact_spatial_review_v1":
        raise ValueError("Expected audited contact_spatial_review_v1 review")
    if not all(c.get("source_stem") and c.get("annotation_ids") for c in manifest["clips"]):
        raise ValueError("Training clips must retain source-video and human annotation IDs")
    contacts,excluded = contact_queue(manifest,review)
    stems = sorted({c["source_stem"] for c in manifest["clips"]})
    contract = {"manifest":str(manifest_path.resolve()),"clips":len(manifest["clips"]),
                "source_videos":stems,"feature_schema":manifest.get("feature_schema"),
                "new_feature_or_model_training":False,"label_provenance":"human event labels; contact-author labels start empty",
                "split_policy":"complete source_video; event-model training sources are development data for author evaluation",
                "contacts_after_audit_and_duplicate_grouping":len(contacts),"excluded_known_wrong_events":len(excluded)}
    print(json.dumps(contract,ensure_ascii=False,indent=2),flush=True)
    output.mkdir(parents=True)
    sources = [fingerprint_file(manifest_path),fingerprint_file(review_path)]
    videos = {}; rosters = {}
    for stem in stems:
        path = Path(analyses_dir)/f"{stem}_annotations.json"
        source = fingerprint_file(path); sources.append(source)
        ann = json.loads(path.read_bytes())
        video = Path(ann["original_video_path"])
        if video.stem!=stem: raise ValueError("Saved analysis belongs to another source video")
        cap = cv2.VideoCapture(str(video))
        try:
            if not cap.isOpened(): raise ValueError("Cannot open original video: "+str(video))
            width,height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if (width,height)!=(ann["frame_width"],ann["frame_height"]) or abs(cap.get(cv2.CAP_PROP_FPS)-ann["fps"])>.05:
                raise ValueError("Cached poses and original video geometry differ")
        finally: cap.release()
        length = max(row["f"] for row in ann["frames"])+1
        rows = [c for c in contacts if c["video_id"]==stem]
        if not rows or any(not 0<=c["frame"]<length for c in rows): raise ValueError("Contact outside saved source")
        rosters[stem] = initial_roster(ann,rows[0]["frame"])
        videos[stem] = {"video_id":stem,"name":stem,"fps":ann["fps"],"width":width,"height":height,
                        "num_frames":length,"video_source":fingerprint_file(video),"annotations_source":source,
                        "target_side":"b","initial_roster":rosters[stem],"contact_count":len(rows)}
        print(f"Prepared {stem}: {len(rows)} contacts, six proposed reference cards",flush=True)
        del ann
    project = {"schema":SCHEMA,"created_at":now(),"source_manifest_sha256":sha(manifest_path),
               "source_scope":contract,"sources":sources,"videos":videos,"contacts":contacts,
               "excluded":excluded,"initial_author_labels":0,"training":False,
               "evaluation_scope":"development_on_event_model_training_sources",
               "roster_policy":"per-video confirmed people; never recycle IDs; explicit new-person confirmation for additions"}
    write_new(output/"project.json",project)
    store = AnnotationStore(output)
    store.initialize(project)
    return project


class Conflict(ValueError): pass


class MovementOverlap(ValueError):
    def __init__(self, player_id, segment, start, end):
        super().__init__(f"{player_id} вже позначено в кадрах f{start}–{end}.")
        self.details = {"frame": start, "bbox": segment["boxes"][start - segment["start"]]}


class AnnotationStore:
    def __init__(self,directory):
        self.directory = Path(directory).resolve(); self.project_path = self.directory/"project.json"
        self.project = json.loads(self.project_path.read_bytes()); self.fingerprint = sha(self.project_path)
        if self.project.get("schema")!=SCHEMA: raise ValueError("Unknown author annotation project")
        self.contacts = {c["id"]:c for c in self.project["contacts"]}
        self.path = self.directory/"annotations.sqlite3"

    def connect(self):
        connection = sqlite3.connect(self.path,timeout=10)
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    def initialize(self,project=None):
        with self.connect() as con:
            con.executescript("CREATE TABLE IF NOT EXISTS state(id INTEGER PRIMARY KEY CHECK(id=1),body TEXT NOT NULL);"
                              "CREATE TABLE IF NOT EXISTS revisions(revision INTEGER PRIMARY KEY,created_at TEXT,action TEXT NOT NULL);")
            state = {"schema":SCHEMA,"project_sha256":self.fingerprint,"revision":0,"labels":{},
                     "rosters":{key:deepcopy(v["initial_roster"]) for key,v in self.project["videos"].items()}}
            con.execute("INSERT OR IGNORE INTO state VALUES(1,?)",(json.dumps(state,ensure_ascii=False),))
        return self.load()

    def check_sources(self,full=False):
        if sha(self.project_path)!=self.fingerprint: raise Conflict("Project queue changed; create a new project")
        sources = self.project["sources"]+[v["video_source"] for v in self.project["videos"].values()]
        for source in sources:
            p = Path(source["path"])
            try: stat = p.stat()
            except OSError: raise Conflict("Source file is unavailable: "+p.name)
            if stat.st_size!=source["size"] or stat.st_mtime_ns!=source["mtime_ns"] or (full and sha(p)!=source["sha256"]):
                raise Conflict("Source changed; original author labels were preserved: "+p.name)

    def load(self):
        with self.connect() as con: row = con.execute("SELECT body FROM state WHERE id=1").fetchone()
        if row is None: raise ValueError("Annotation database is not initialized")
        value = json.loads(row[0])
        if value.get("project_sha256")!=self.fingerprint: raise Conflict("Labels belong to another contact queue")
        return value

    def mutate(self,expected_revision,action,operation):
        self.check_sources()
        con = self.connect()
        try:
            con.execute("BEGIN IMMEDIATE")
            state = json.loads(con.execute("SELECT body FROM state WHERE id=1").fetchone()[0])
            if state["project_sha256"]!=self.fingerprint: raise Conflict("Changed project")
            if type(expected_revision) is not int or state["revision"]!=expected_revision:
                raise Conflict("Project changed in another tab; reload before saving")
            operation(state)
            state["revision"] += 1
            con.execute("UPDATE state SET body=? WHERE id=1",(json.dumps(state,ensure_ascii=False,allow_nan=False),))
            con.execute("INSERT INTO revisions VALUES(?,?,?)",(state["revision"],now(),json.dumps(action,ensure_ascii=False)))
            con.commit(); return state
        except Exception:
            con.rollback(); raise
        finally: con.close()

    def label(self,payload):
        key = payload.get("contact_id"); decision = payload.get("decision")
        if not isinstance(key,str): raise ValueError("Unknown contact")
        if decision not in DECISIONS|{"reset"}: raise ValueError("Unknown annotation decision")
        def apply(state):
            video = self.contact(key, state)["video_id"]; roster = state["rosters"][video]
            if decision=="reset": state["labels"].pop(key,None); return
            if not roster["confirmed"]: raise ValueError("Confirm this video's player cards first")
            pid = payload.get("player_id")
            if decision=="player" and pid not in {p["player_id"] for p in roster["players"]}: raise ValueError("Unknown player")
            if decision!="player" and pid is not None: raise ValueError("Only a player decision may contain a player ID")
            state["labels"][key] = {"decision":decision,"player_id":pid,"label_source":"human",
                 "verified":decision!="unsure","verification_scope":"contact_presence" if decision=="not_contact" else "contact_author_only","updated_at":now(),
                 "revision":state["revision"]+1}
        return self.mutate(payload.get("revision"),payload,apply)

    def event_action(self,payload):
        key=payload.get("contact_id");action=payload.get("action","set")
        if not isinstance(key,str): raise ValueError("Unknown contact")
        if action not in ("set","reset"): raise ValueError("Unknown event review action")
        if action=="set" and (not isinstance(payload.get("type"),str) or payload["type"] not in EVENT_TYPES): raise ValueError("Unknown contact type")
        def apply(state):
            self.contact(key, state)
            reviews=state.setdefault("event_reviews",{})
            if action=="reset": reviews.pop(key,None);return
            reviews[key]={"type":payload["type"],"label_source":"human","verified":True,
                          "verification_scope":"contact_type_only","updated_at":now(),"revision":state["revision"]+1}
        return self.mutate(payload.get("revision"),payload,apply)

    def contact(self, key, state):
        row = self.contacts.get(key) or state.get("added_contacts", {}).get(key)
        if row is None:
            raise ValueError("Unknown contact")
        return row

    def queue(self, state):
        return [*self.contacts.values(), *state.get("added_contacts", {}).values()]

    def contact_frame_action(self, payload):
        action = payload.get("action")
        frame = payload.get("frame")
        video = payload.get("video_id")
        key = payload.get("contact_id")
        metadata = self.project["videos"].get(video)
        if metadata is None or type(frame) is not int or not 0 <= frame < metadata["num_frames"]:
            raise ValueError("Кадр поза межами відео")
        if action not in ("add", "timing"):
            raise ValueError("Unknown frame action")
        kind = payload.get("type")
        if action == "add" and (not isinstance(kind, str) or kind not in EVENT_TYPES):
            raise ValueError("Оберіть тип нового контакту")

        def apply(state):
            if action == "timing" and (not isinstance(key, str) or self.contact(key, state)["video_id"] != video):
                raise ValueError("Unknown contact")
            timing = state.setdefault("timing_reviews", {})
            for row in self.queue(state):
                effective = timing.get(row["id"], row)["frame"]
                if row["video_id"] == video and row["id"] != key and effective == frame:
                    raise ValueError("На цьому кадрі вже є контакт. Відкрийте його у списку.")
            if action == "add":
                new_key = f"{video}:human:r{state['revision'] + 1}"
                state.setdefault("added_contacts", {})[new_key] = {
                    "id": new_key, "video_id": video, "frame": frame, "type": kind, "raw_types": [kind],
                    "original_frames": [frame], "event_label_source": "human", "event_verified": True,
                    "timing_source": "human", "timing_verified": True, "author_label_source": None,
                }
            else:
                timing[key] = {"frame": frame, "label_source": "human", "verified": True,
                               "updated_at": now(), "revision": state["revision"] + 1}

        return self.mutate(payload.get("revision"), payload, apply)

    def movement_action(self, payload):
        video = payload.get("video_id")
        key = payload.get("segment_id")
        decision = payload.get("decision")
        pid = payload.get("player_id")
        if video not in self.project["videos"] or not isinstance(key, str):
            raise ValueError("Unknown movement segment")
        self.check_sources()
        segments = segments_for(self.project["videos"][video])
        segment = segments.get(key)
        if segment is None or decision not in ("player", "other", "unsure", "false_positive", "undo"):
            raise ValueError("Unknown movement segment or decision")

        def apply(state):
            roster = state["rosters"][video]
            if not roster["confirmed"]:
                raise ValueError("Спочатку підтвердьте картки гравців")
            reviewed = state.setdefault("movement_reviews", {}).setdefault(video, {})
            spans = reviewed.setdefault(key, [])
            if decision == "undo":
                if spans:
                    spans.pop()
                if not spans:
                    reviewed.pop(key)
                return
            if decision == "player":
                if pid not in {p["player_id"] for p in roster["players"]}:
                    raise ValueError("Unknown player")
            elif pid is not None:
                raise ValueError("Only a player decision may contain a player ID")
            start = spans[-1]["end"] + 1 if spans else segment["start"]
            end = payload.get("end", segment["end"])
            if type(end) is not int or not start <= end <= segment["end"]:
                raise ValueError("Оберіть кадр у нерозміченій частині відрізка")
            confirm_overlap = payload.get("confirm_overlap") is True
            if pid is not None and not confirm_overlap:
                for other_key, other_spans in reviewed.items():
                    if other_key == key:
                        continue
                    for span in other_spans:
                        if span["player_id"] == pid and span["start"] <= end and span["end"] >= start:
                            raise MovementOverlap(pid, segments[other_key], max(start, span["start"]), min(end, span["end"]))
            spans.append({"start": start, "end": end, "decision": decision, "player_id": pid,
                          "label_source": "human", "verified": decision != "unsure",
                          "updated_at": now(), "revision": state["revision"] + 1})

        return self.mutate(payload.get("revision"), payload, apply)

    def roster_action(self,payload):
        video = payload.get("video_id"); action = payload.get("action")
        if video not in self.project["videos"]: raise ValueError("Unknown video")
        def apply(state):
            roster = state["rosters"][video]
            if action=="remove_player":
                player = next((p for p in roster["players"] if p["player_id"]==payload.get("player_id")),None)
                if player is None: raise ValueError("Unknown player")
                if payload.get("removal_confirmed") is not True: raise ValueError("Confirm player removal")
                for spans in state.get("movement_reviews", {}).get(video, {}).values():
                    if any(span["player_id"] == player["player_id"] for span in spans):
                        raise ValueError("Картка вже має мітки руху. Спочатку скасуйте ці мітки.")
                affected = {key:deepcopy(label) for key,label in state["labels"].items()
                            if self.contact(key, state)["video_id"]==video and label.get("player_id")==player["player_id"]}
                if affected and payload.get("unassign_contacts_confirmed") is not True:
                    raise Conflict(f"Player has {len(affected)} contact labels; explicitly confirm returning them to unlabelled")
                roster.setdefault("removed_players",[]).append({"player":deepcopy(player),"removed_at":now(),
                    "revision":state["revision"]+1,"unassigned_labels":affected})
                roster["players"].remove(player)
                for key in affected: del state["labels"][key]
                if not roster["players"]: roster["confirmed"] = False
                return
            if action=="restore_player":
                removed = roster.get("removed_players",[])
                entry = next((p for p in removed if p["player"]["player_id"]==payload.get("player_id")),None)
                if entry is None: raise ValueError("Unknown removed player")
                if any(p["player_id"]==payload.get("player_id") for p in roster["players"]): raise Conflict("Player ID already exists")
                roster["players"].append(deepcopy(entry["player"]))
                roster["players"].sort(key=lambda p:int(p["player_id"][1:]))
                for key,label in entry["unassigned_labels"].items():
                    if key not in state["labels"]: state["labels"][key] = deepcopy(label)
                roster.setdefault("restored_players",[]).append({**deepcopy(entry),"restored_at":now(),"restored_revision":state["revision"]+1})
                removed.remove(entry)
                return
            if action=="confirm":
                if len(roster["players"])<1: raise ValueError("Empty roster")
                roster["confirmed"] = True
                for p in roster["players"]:
                    p["identity_confirmation"] = {"label_source":"human","verified":True,"updated_at":now()}
                return
            if action not in ("replace_reference","add_view","add_player"): raise ValueError("Unknown roster action")
            metadata = self.project["videos"][video]
            frame = payload.get("frame"); box = payload.get("bbox")
            if type(frame) is not int or not 0<=frame<metadata["num_frames"]: raise ValueError("Invalid reference frame")
            if not isinstance(box,list) or len(box)!=4 or not all(isinstance(v,(int,float)) and np.isfinite(v) for v in box):
                raise ValueError("Invalid reference box")
            if not (0<=box[0]<box[2]<=metadata["width"] and 0<=box[1]<box[3]<=metadata["height"]) or min(box[2]-box[0],box[3]-box[1])<12:
                raise ValueError("Reference box is outside the source image or too small")
            reference = {"frame":frame,"bbox":box,"label_source":"human","scope":"identity_reference_only"}
            if action=="add_player":
                if payload.get("new_person_confirmed") is not True: raise ValueError("Confirm this is a new person")
                reserved = roster["players"]+[p["player"] for p in roster.get("removed_players",[])]
                number = max((int(p["player_id"][1:]) for p in reserved),default=0)+1
                roster["players"].append({"player_id":f"P{number}","references":[reference],
                      "identity_confirmation":{"label_source":"human","verified":True,"scope":"new_person","updated_at":now()}})
            else:
                player = next((p for p in roster["players"] if p["player_id"]==payload.get("player_id")),None)
                if player is None: raise ValueError("Unknown player")
                if action=="replace_reference":
                    if roster["confirmed"]: raise Conflict("Confirmed IDs keep their original reference; add a view of the same person")
                    player["references"] = [reference]
                else:
                    if payload.get("same_person_confirmed") is not True: raise ValueError("Confirm the same existing person")
                    player["references"].append(reference)
        return self.mutate(payload.get("revision"),payload,apply)

    def export(self):
        self.check_sources(); state = self.load()
        queue = [reviewed_contact(c, state) for c in self.queue(state)]
        queue.sort(key=lambda row: (row["video_id"], row["frame"]))
        rows = []
        for c in queue:
            label = state["labels"].get(c["id"])
            if not label or label["decision"] not in ("player","other_team") or not label["verified"]: continue
            rows.append({**c,"author":deepcopy(label),"author_label_source":"human",
                         "author_key":f'{c["video_id"]}:{label["player_id"]}' if label["player_id"] else None,
                         "team":self.project["videos"][c["video_id"]]["target_side"] if label["decision"]=="player" else "a"})
        return {"schema":"human_contact_author_dataset_v1","created_at":now(),"project_sha256":self.fingerprint,
                "revision":state["revision"],"rosters":state["rosters"],"contacts":rows,
                "decisions":deepcopy(state["labels"]),"queue":queue,
                "event_reviews":deepcopy(state.get("event_reviews",{})),
                "movement":movement_export(self.project, state),
                "excluded":deepcopy(self.project["excluded"]),"sources":deepcopy(self.project["sources"]),
                "videos":deepcopy(self.project["videos"]),"source_scope":self.project["source_scope"],
                "evaluation_scope":self.project["evaluation_scope"],"split_unit":"source_video",
                "counts":{"eligible_author_labels":len(rows),"decisions":dict(Counter(x["decision"] for x in state["labels"].values())),
                          "unlabelled":len(queue)-len(state["labels"])},
                "training_promotion":False}
