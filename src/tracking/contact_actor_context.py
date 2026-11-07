"""Observed action evidence and bounded player context for contact review.

No keypoints/ball coordinates are generated. Ground depth describes observed
court position, not rotational eligibility. A positive human identity anchor
can guide a separate overlay; negative case labels never enter the scorer.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np

from src.tracking.contact_author_features import ball, points
from src.tracking.court_setups import homography_from_corners
from src.tracking.player_contacts import _distance, side, summarize

SCHEMA = "observed_contact_actor_context_v1"


@dataclass(frozen=True)
class ActorContextConfig:
    context_seconds: float = 1.2
    ground_seconds: float = 1.
    max_track_gap_seconds: float = .30
    appearance_max: float = .30
    appearance_margin: float = .05
    identity_vote_min: int = 2
    action_radius_seconds: float = .20
    max_hand_distance: float = .45
    max_block_distance: float = .25
    actor_margin: float = .06
    prior_weight: float = .025
    arm_motion_weight: float = .10


def feet(pose):
    kp = points(pose)
    values = [kp[j] for j in (15,16) if j in kp and kp[j][2] >= .5]
    return (float(np.mean([p[0] for p in values])), max(p[1] for p in values)) if values else None


def action_evidence(pose, b, event_type, net_y=None):
    """Same-arm evidence: a nearby head/body cannot stand in for an attack hand."""
    kp = points(pose); box = pose.get("bb") or []
    if len(box) != 4 or box[3] <= box[1] or b is None:
        return {"supported":False,"reason":"missing_pose_or_ball"}
    h = box[3]-box[1]
    arms = []
    for shoulder,elbow,wrist in ((5,7,9),(6,8,10)):
        if shoulder not in kp or wrist not in kp:
            continue
        raised = (kp[shoulder][1]-kp[wrist][1])/h
        distance = math.dist(b[:2],kp[wrist][:2])/h
        part = "wrist"
        if elbow in kp:
            forearm = _distance(b,kp[elbow],kp[wrist])/h+.025
            if forearm < distance: distance,part = forearm,"forearm"
        block_height = net_y is None or kp[wrist][1] <= net_y+.15*h
        supported = True
        if event_type == "block":
            supported = raised >= .08 and block_height
        elif event_type == "attack":
            # Preserve low/underhand attacks when the HAND is almost touching.
            supported = raised >= -.04 or math.dist(b[:2],kp[wrist][:2])/h <= .09
        arms.append({"supported":supported,"distance_body_heights":distance,"arm_raise":raised,
                     "wrist":list(kp[wrist][:2]),"contact_part":part,"wrist_joint":wrist})
    supported = [a for a in arms if a["supported"]]
    if supported:
        result = min(supported,key=lambda a:a["distance_body_heights"])
        bilateral = len(arms)==2 and all(a["arm_raise"]>=.08 for a in arms)
        return {**result,"bilateral_raised":bilateral,"reason":"observed_active_arm"}
    if event_type not in ("attack","block") and kp:
        distance = min(math.dist(b[:2],p[:2])/h for p in kp.values())
        return {"supported":True,"distance_body_heights":distance+.30,"arm_raise":None,
                "contact_part":"body_fallback","reason":"body_contact_requires_extra_evidence"}
    return {"supported":False,"reason":"no_overhead_action_evidence",
            "arm_raise":max((a["arm_raise"] for a in arms),default=None)}


class ActorContext:
    def __init__(self, annotations, appearance=None, config=ActorContextConfig()):
        self.data = annotations; self.fps = float(annotations["fps"]); self.config = config
        self.frames = {int(row["f"]):row for row in annotations["frames"]}
        self.appearance = appearance or {}
        self.tracks = defaultdict(list); self.by_key = {}; self.by_track_frame = {}
        for row in annotations["frames"]:
            f = int(row["f"])
            for i,pose in enumerate(row.get("poses") or []):
                if pose.get("id") is None: continue
                value = {"f":f,"i":i,"pose":pose,"foot":feet(pose)}
                self.tracks[pose["id"]].append(value); self.by_key[(f,i)] = value
                self.by_track_frame[(pose["id"],f)] = value
        self.track_times = {key:np.array([v["f"] for v in rows]) for key,rows in self.tracks.items()}
        self.p4_source_tracks = {v["pose"]["id"] for v in self.by_key.values() if v["pose"].get("player_id")=="P4"}
        self.p4_reference_tracks = {v["pose"]["id"] for (f,i),v in self.by_key.items()
             if (d:=self.appearance.get(f"{f}:{i}")) and d["distance"]<=config.appearance_max
             and d["other_distance"]-d["distance"]>=config.appearance_margin}
        try:
            self.inverse = np.linalg.inv(homography_from_corners(annotations["court_corners"]))
        except (KeyError,ValueError,np.linalg.LinAlgError): self.inverse = None
        self._cache = {}

    def window(self, value, seconds):
        rid = value["pose"]["id"]; times = self.track_times[rid]
        radius = round(seconds*self.fps)
        lo = np.searchsorted(times,value["f"]-radius,side="left")
        hi = np.searchsorted(times,value["f"]+radius,side="right")
        rows = self.tracks[rid][lo:hi]
        # Recycled/discontinuous raw IDs must not connect two people.
        current = next(i for i,v in enumerate(rows) if v["f"]==value["f"])
        first = last = current
        for direction in (-1,1):
            at = current
            while 0 <= at+direction < len(rows):
                a,b = rows[at],rows[at+direction]
                gap = abs(a["f"]-b["f"])/self.fps
                ba,bb = a["pose"]["bb"],b["pose"]["bb"]
                h = max(1.,(ba[3]-ba[1]+bb[3]-bb[1])*.5)
                shift = math.dist([(ba[0]+ba[2])*.5,(ba[1]+ba[3])*.5],[(bb[0]+bb[2])*.5,(bb[1]+bb[3])*.5])/h
                if gap > self.config.max_track_gap_seconds or shift > .75: break
                at += direction
            if direction < 0: first = at
            else: last = at
        return rows[first:last+1]

    def ground_role(self,value):
        samples = [v for v in self.window(value,self.config.ground_seconds) if v["foot"] is not None]
        if len(samples)<5 or self.inverse is None:
            return {"position":"unknown","depth_m":None,"support_frames":len(samples)}
        level = float(np.quantile([v["foot"][1] for v in samples],.85))
        grounded = [v for v in samples if v["foot"][1] >= level]
        depth = []
        for v in grounded:
            x,y = v["foot"]; floor = self.data.get("net_floor_left"),self.data.get("net_floor_right")
            if not all(floor): continue
            (x1,y1),(x2,y2) = floor
            net_y = y1+(y2-y1)*(x-x1)/(x2-x1) if abs(x2-x1)>1e-6 else (y1+y2)/2
            world = self.inverse@np.array([x,y,1.]); net = self.inverse@np.array([x,net_y,1.])
            depth.append(abs(float(world[1]/world[2]-net[1]/net[2])))
        d = float(np.median(depth)) if depth else None
        return {"position":"near_net" if d is not None and d <= 3. else "backcourt" if d is not None else "unknown",
                "depth_m":d,"support_frames":len(samples),"ground_image_y":level,
                "method":"bounded_observed_ground_anchors","rotation_eligibility_known":False}

    def resolve(self,f,index,use_context=True,use_appearance=True):
        key = (f,index,use_context,use_appearance)
        if key in self._cache: return self._cache[key]
        value = self.by_key.get((f,index)); pose = self.frames[f]["poses"][index]
        result = {"player_id":pose.get("player_id"),"side":side(pose.get("team")),
                  "identity_method":"source_identity","position":"unknown","depth_m":None}
        if value is None or not use_context:
            return result
        rows = self.window(value,self.config.context_seconds)
        result.update(self.ground_role(value))
        clean = [v for v in rows if v["pose"].get("player_id") and
                 v["pose"].get("identity",{}).get("reason")=="associated" and
                 v["pose"].get("identity",{}).get("appearance_distance",1.) <= .35]
        before = [v for v in clean if v["f"] < f]; after = [v for v in clean if v["f"] > f]
        identities = {v["pose"]["player_id"] for v in clean}
        if len(identities)==1 and len(before)>=self.config.identity_vote_min and len(after)>=self.config.identity_vote_min:
            pid = next(iter(identities))
            result.update(player_id=pid,side="b",identity_method="two_sided_raw_track_identity",
                          identity_anchors=[before[-1]["f"],after[0]["f"]])
        if result["player_id"] == "P4" and pose.get("identity",{}).get("appearance_distance",0.) > .35:
            result.update(player_id=None,identity_method="weak_P4_source_identity_rejected")
        if use_appearance and self.appearance:
            available = [(v,self.appearance.get(f'{v["f"]}:{v["i"]}')) for v in rows]
            trusted = [(v,d) for v,d in available if d and d["distance"] <= self.config.appearance_max
                       and d["other_distance"]-d["distance"] >= self.config.appearance_margin]
            left = [(v,d) for v,d in trusted if v["f"] < f]
            right = [(v,d) for v,d in trusted if v["f"] > f]
            direct = self.appearance.get(f"{f}:{index}")
            direct_ok = direct and direct["distance"] <= self.config.appearance_max and direct["other_distance"]-direct["distance"] >= self.config.appearance_margin
            if direct_ok or (len(left)>=2 and len(right)>=2):
                evidence = direct if direct_ok else min([d for _,d in left+right],key=lambda d:d["distance"])
                result.update(player_id="P4",side="b",identity_method="human_anchor_cached_appearance_proposal",
                              appearance_anchor_distance=evidence["distance"],appearance_other_distance=evidence["other_distance"],
                              identity_anchors=[left[-1][0]["f"],right[0][0]["f"]] if left and right else [f])
            elif pose.get("player_id")=="P4" or result["player_id"]=="P4":
                result.update(player_id=None,identity_method="P4_reference_not_supported")
        self._cache[key] = result
        return result

    def frame_context(self,f,use_appearance=True,p4_only=False):
        eligible = self.p4_source_tracks | (self.p4_reference_tracks if use_appearance else set())
        values = [deepcopy(self.resolve(f,i,not p4_only or pose.get("id") in eligible,use_appearance))
                  for i,pose in enumerate(self.frames[f].get("poses") or [])]
        # Identity overlays stay one-to-one, even when two crops resemble P4.
        p4 = [i for i,v in enumerate(values) if v["player_id"]=="P4"]
        if len(p4)>1:
            ranked = sorted(p4,key=lambda i:values[i].get("appearance_anchor_distance",1.))
            best,second = ranked[:2]
            margin = values[second].get("appearance_anchor_distance",1.)-values[best].get("appearance_anchor_distance",1.)
            rejected = ranked[1:] if margin >= self.config.appearance_margin else ranked
            for i in rejected: values[i].update(player_id=None,identity_method="ambiguous_P4_frame_match")
        return values

    def arm_speed(self,f,pose,wrist):
        """Shoulder-relative observed arm motion, not whole-body/camera motion."""
        if wrist not in (9,10) or pose.get("id") is None: return None
        shoulder = 5 if wrist==9 else 6
        samples = [self.by_track_frame.get((pose["id"],f+d)) for d in (-3,3)]
        if any(v is None for v in samples): return None
        values = [points(v["pose"]) for v in samples]
        if any(shoulder not in kp or wrist not in kp for kp in values): return None
        vectors = [[kp[wrist][j]-kp[shoulder][j] for j in (0,1)] for kp in values]
        h = pose["bb"][3]-pose["bb"][1]
        return float(math.dist(*vectors)*self.fps/(6*h))


def build_action_report(events, context, priors, identity, method_id,
                        use_context=True,use_appearance=True):
    """Joint actor/side choice; side probability cannot preselect an inactive body."""
    contacts = []; fps = context.fps; config = context.config
    times = sorted(set(int(e["frame"]) for e in events))
    block_frames = [int(e["frame"]) for e in events if e["type"]=="block"]
    for event_index,event in enumerate(events):
        frame = int(event["frame"]); radius = round(config.action_radius_seconds*fps)
        neighbours = [abs(t-frame) for t in times if t != frame]
        if neighbours: radius = min(radius,max(0,(min(neighbours)-1)//2))
        buckets = {}; rejected = []
        for f in range(frame-radius,frame+radius+1):
            row = context.frames.get(f); b = ball(row)
            if b is None: continue
            if any(t not in context.frames or context.frames[t].get("ball_out_of_frame") or
                   (t>min(f,frame) and context.frames[t].get("ball_new")) for t in range(min(f,frame),max(f,frame)+1)):
                continue
            identities = context.frame_context(f,use_appearance) if use_context else None
            for i,pose in enumerate(row.get("poses") or []):
                state = identities[i] if identities else context.resolve(f,i,False,False)
                evidence = action_evidence(pose,b,event["type"],context.data.get("net_y"))
                if event["type"]=="attack" and evidence.get("bilateral_raised"):
                    # A blocker's two raised hands also look like an attack in
                    # projection. Do not count the same observed blocking action
                    # as the preceding opponent attack without distinct evidence.
                    for block_frame in block_frames:
                        if not frame < block_frame <= frame+round(.30*fps): continue
                        later = context.by_track_frame.get((pose.get("id"),block_frame))
                        later_ball = ball(context.frames.get(block_frame))
                        if later is None or later_ball is None: continue
                        block = action_evidence(later["pose"],later_ball,"block",context.data.get("net_y"))
                        if block["supported"] and block["distance_body_heights"]<=config.max_block_distance:
                            evidence.update(supported=False,reason="nearby_block_pose_not_attack_evidence")
                            break
                if not evidence["supported"]:
                    if pose.get("player_id")=="P4" or state["player_id"]=="P4":
                        rejected.append({"frame":f,"raw_track_id":pose.get("id"),"source_player_id":pose.get("player_id"),
                                         "player_id":state["player_id"],"reason":evidence["reason"],"arm_raise":evidence.get("arm_raise")})
                    continue
                team = state["side"]
                if team not in ("a","b"): continue
                distance = evidence["distance_body_heights"]
                speed = context.arm_speed(f,pose,evidence.get("wrist_joint"))
                evidence["relative_arm_speed"] = speed
                prior = float(np.clip(priors[event_index],.02,.98))
                side_cost = -config.prior_weight*np.log(prior if team=="b" else 1-prior)
                depth = state.get("depth_m")
                role_cost = min(.25,max(0.,(depth-3.)*.08)) if event["type"]=="block" and depth is not None else 0.
                motion_bonus = config.arm_motion_weight*min(1.,speed/2.) if speed is not None else 0.
                score = distance+.04*((f-frame)/max(1,radius))**2+side_cost+role_cost-motion_bonus
                candidate = {"frame":f,"pose_index":i,"raw_track_id":pose.get("id"),
                             "player_id":state["player_id"],"side":team,"source_team":side(pose.get("team")),
                             "source_player_id":pose.get("player_id"),"score":float(score),
                             "distance_px":float(distance*(pose["bb"][3]-pose["bb"][1])),
                             "distance_body_heights":float(distance),"bbox":pose["bb"],"ball":list(b),
                             "ball_source":row.get("ball_source") or ("interpolated_short" if row.get("ball_interp") else "detected"),
                             "contact_part":evidence["contact_part"],"action_evidence":evidence,
                             "identity_resolution":deepcopy(state),"identity":deepcopy(pose.get("identity") or {})}
                key = (team,"person",state["player_id"]) if state["player_id"] else ("track",pose.get("id"))
                if key not in buckets or score < buckets[key]["score"]: buckets[key] = candidate
        candidates = sorted(buckets.values(),key=lambda c:c["score"])
        best = candidates[0] if candidates else None
        margin = candidates[1]["score"]-best["score"] if len(candidates)>1 else None
        contact = {"event_index":event_index,"source_event_index":event_index,"frame":frame,"original_frame":frame,
                   "time":frame/fps,"type":event["type"],"player_id":None,"proposed_player_id":None,
                   "status":"unresolved","reason":"no_supported_active_actor","label_source":"derived","verified":False,
                   "near_side_score":float(priors[event_index]),"score_calibrated":False,
                   "side_proposal":best["side"] if best else None,"side_accepted":False,
                   "candidates":deepcopy(candidates[:4]),"margin":float(margin) if margin is not None else None,
                   "rejected_inactive_P4":rejected,"uses_estimated_joints":False,
                   "detection_evidence":{"ballistic":0.,"interaction":0.,"joint":0.}}
        if best:
            contact.update(contact_frame=best["frame"],contact_time=best["frame"]/fps,
                           identity_resolution=best["identity_resolution"])
            limit = config.max_block_distance if event["type"]=="block" else config.max_hand_distance
            if best["distance_body_heights"] > limit:
                contact["reason"] = "active_actor_too_far"
            elif margin is not None and margin < config.actor_margin:
                contact["reason"] = "ambiguous_active_actors"
            elif best["side"] == "a":
                contact.update(status="other_side",reason="observed_opponent_action",side_accepted=True)
            elif best["player_id"] is None:
                contact.update(reason="active_actor_identity_unknown",side_accepted=True)
            else:
                contact.update(player_id=best["player_id"],proposed_player_id=best["player_id"],status="assigned",
                               reason="observed_action_and_temporal_identity",side_accepted=True)
        contacts.append(contact)
    roster = sorted((identity.get("gallery_keys") or {}).keys(),key=lambda p:int(p[1:]) if p[1:].isdigit() else p)
    return summarize({"schema":"player_contacts_v1","method":SCHEMA,"method_id":method_id,
                      "side":"b","roster_ids":roster,"contacts":contacts,
                      "limitations":"Action/identity proposals; five P4 cases are development checks, not held-out accuracy"})
