"""Offline observed/derived pose and trajectory features for contact actors.

Interpolation is a feature hypothesis only, marked explicitly. It never
changes source poses, source balls, detector caches, or human contact targets.
"""
from copy import deepcopy
import math

import numpy as np

from src.tracking.contact_author_features import geometry, points, ball
from src.tracking.contact_actor_context import action_evidence
from src.tracking.player_contacts import _distance


def interpolated_pose(track,frame):
    exact = next((o["pose"] for o in track if o["frame"]==frame),None)
    if exact is not None: result = deepcopy(exact)
    else:
        before = [o for o in track if frame-6<=o["frame"]<frame]
        after = [o for o in track if frame<o["frame"]<=frame+6]
        if not before or not after: return None,0.
        a,b = max(before,key=lambda o:o["frame"]),min(after,key=lambda o:o["frame"])
        t = (frame-a["frame"])/(b["frame"]-a["frame"])
        result = {**a["pose"],"bb":[(1-t)*x+t*y for x,y in zip(a["pose"]["bb"],b["pose"]["bb"])],"kp":[[0.,0.,0.]]*17}
    derived = 0
    for j in range(17):
        if result["kp"][j][2]>=.25: continue
        before = [o for o in track if frame-6<=o["frame"]<frame and o["pose"]["kp"][j][2]>=.5]
        after = [o for o in track if frame<o["frame"]<=frame+6 and o["pose"]["kp"][j][2]>=.5]
        if not before or not after: continue
        a,b = max(before,key=lambda o:o["frame"]),min(after,key=lambda o:o["frame"])
        t = (frame-a["frame"])/(b["frame"]-a["frame"])
        left,right = a["pose"]["kp"][j],b["pose"]["kp"][j]
        result["kp"][j] = [(1-t)*left[0]+t*right[0],(1-t)*left[1]+t*right[1],min(left[2],right[2])*.5]
        derived += 1
    return result,derived/17


def polynomial_contact(by_frame,frame,direction):
    """Observed-only bounded extrapolation; off-camera/restarts/gaps stop support."""
    if by_frame.get(frame,{}).get("ball_out_of_frame"): return None
    samples = []
    for n in range(1,13):
        f = frame+direction*n; row = by_frame.get(f)
        if row is None or ball(row) is None or row.get("ball_interp") or row.get("ball_source") not in (None,"detected","observed"):
            break
        newer = by_frame.get(f if direction>0 else f+1,{})
        if newer.get("ball_new"): break
        if n>=2: samples.append((f-frame,row["ball"][:2]))
    if len(samples)<4: return None
    t = np.array([r[0] for r in samples],float); xy = np.array([r[1] for r in samples],float)
    design = np.column_stack((t*t,t,np.ones(len(t))))
    coef = np.linalg.lstsq(design,xy,rcond=None)[0]
    residual = np.linalg.norm(design@coef-xy,axis=1)
    if len(t)>=7:
        selected = residual<=np.quantile(residual,.8)
        coef = np.linalg.lstsq(design[selected],xy[selected],rcond=None)[0]
        residual = np.linalg.norm(design[selected]@coef-xy[selected],axis=1)
    return {"xy":coef[2],"velocity_per_frame":coef[1],"rmse":float(np.sqrt(np.mean(residual**2))),"support":len(samples)}


def temporal_actor(actor,track,by_frame,frame,event_type,metadata,net_y=None):
    fields = {}; source = by_frame[actor["frame"]]["poses"][actor["pose_index"]]
    fps = metadata["fps"]; h = max(1.,source["bb"][3]-source["bb"][1])
    current,derived = interpolated_pose(track,frame)
    observed = {o["frame"]:o["pose"] for o in track}; fields["estimated_joint_fraction"] = derived
    fields["pose_at_contact_observed"] = float(frame in observed)
    fields["pose_offset"] = (actor["frame"]-frame)/fps
    def geometric(p,f):
        b = ball(by_frame.get(f))
        return geometry(p,b,h) if p is not None and b is not None else None
    g = geometric(current,frame)
    for j in range(22): fields[f"contact_geometry_{j}"] = g["values"][j] if g else None
    physical = geometric(source,actor["frame"])
    fields["observed_body_distance"] = physical["cost"] if physical else None
    fields["source_side_near"] = float(source.get("team") in ("b","team_b"))
    fields["ball_confidence"] = by_frame.get(frame,{}).get("ball_confidence")
    fields["ball_observed"] = float(ball(by_frame.get(frame)) is not None and not by_frame.get(frame,{}).get("ball_interp"))
    if current is not None:
        kp = points(current); bb = current["bb"]
        hip = np.mean([kp[j][:2] for j in (11,12)],axis=0) if all(j in kp for j in (11,12)) else None
        for j in range(17):
            for axis in (0,1): fields[f"joint_{j}_relative_{axis}"] = (kp[j][axis]-hip[axis])/h if j in kp and hip is not None else None
        wrists = [kp[j][:2] for j in (9,10) if j in kp]
        fields["wrist_pair_distance"] = math.dist(*wrists)/h if len(wrists)==2 else None
        fields["body_x"] = (bb[0]+bb[2])/2/metadata["width"]
        fields["body_y"] = (bb[1]+bb[3])/2/metadata["height"]
        fields["body_image_height"] = (bb[3]-bb[1])/metadata["height"]
        feet = [kp[j][:2] for j in (15,16) if j in kp]
        fields["foot_image_y"] = max(p[1] for p in feet)/metadata["height"] if feet else None
        evidence = action_evidence(current,ball(by_frame.get(frame)),event_type,net_y)
        fields["action_supported"] = float(evidence["supported"])
        fields["active_hand_distance"] = evidence.get("distance_body_heights")
        fields["active_arm_raise"] = evidence.get("arm_raise")
    else:
        kp = {}; wrists = []
        for j in range(17):
            for axis in (0,1): fields[f"joint_{j}_relative_{axis}"] = None
        for key in ("wrist_pair_distance","body_x","body_y","body_image_height","foot_image_y","action_supported","active_hand_distance","active_arm_raise"):
            fields[key] = None
    for offset in (-12,-8,-4,0,4,8,12):
        f = frame+offset; value = geometric(observed.get(f),f)
        for j,name in ((0,"wrist"),(1,"forearm"),(3,"body"),(13,"arm_raise")):
            fields[f"curve_{offset}_{name}"] = value["values"][j] if value else None
    for wrist,shoulder in ((9,5),(10,6)):
        vectors = []
        for offset in (-4,0,4):
            q = points(observed.get(frame+offset,{}) or {})
            vectors.append(np.array(q[wrist][:2])-q[shoulder][:2] if wrist in q and shoulder in q else None)
        fields[f"arm_{wrist}_speed"] = float(np.linalg.norm(vectors[2]-vectors[0])*fps/(8*h)) if vectors[0] is not None and vectors[2] is not None else None
        fields[f"arm_{wrist}_acceleration"] = float(np.linalg.norm(vectors[2]-2*vectors[1]+vectors[0])/h) if all(v is not None for v in vectors) else None
    feet_y = []
    for o in track:
        if abs(o["frame"]-frame)>30: continue
        q = points(o["pose"]); feet_y += [max(q[j][1] for j in (15,16) if j in q)] if any(j in q for j in (15,16)) else []
    ground = float(np.quantile(feet_y,.85)) if len(feet_y)>=5 else None
    fields["ground_image_y"] = ground/metadata["height"] if ground is not None else None
    fields["jump_from_ground"] = (ground-max(kp[j][1] for j in (15,16) if j in kp))/h if ground is not None and any(j in kp for j in (15,16)) else None
    for direction,name in ((-1,"incoming"),(1,"outgoing")):
        fit = polynomial_contact(by_frame,frame,direction)
        fields[name+"_fit_rmse"] = fit["rmse"]/h if fit else None
        fields[name+"_fit_support"] = fit["support"]/11 if fit else None
        fields[name+"_hand_distance"] = min(math.dist(fit["xy"],p) for p in wrists)/h if fit and wrists else None
        forearms = [(kp[a][:2],kp[b][:2]) for a,b in ((7,9),(8,10)) if a in kp and b in kp]
        fields[name+"_forearm_distance"] = min(_distance(fit["xy"],a,b) for a,b in forearms)/h if fit and forearms else None
        for axis in (0,1): fields[name+f"_velocity_{axis}"] = float(fit["velocity_per_frame"][axis]*fps/h) if fit else None
    raw_types = set(event_type.split(" / "))
    for kind in ("serve","reception","pass","attack","block"):
        aliases = {"pass","set","dig","free_ball"} if kind=="pass" else {kind}
        fields["type_"+kind] = float(bool(raw_types & aliases))
    return fields
