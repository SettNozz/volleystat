"""Canonical, audited STES targets derived from immutable human review."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from action_recognition.annotator.contact_spatial_annotator import validate_review


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonicalize(review_path):
    path=Path(review_path)
    raw=json.loads(path.read_text(encoding="utf-8"))
    for source,expected in [(raw["manifest_path"],raw["manifest_sha256"])]+[(s["path"],s["sha256"]) for s in raw["sources"].values()]:
        if file_sha(source)!=expected:raise ValueError("Human source changed: "+source)
    lookup={e["id"]:e for e in raw["events"]}
    if set(lookup)!=set(raw["reviews"]):raise ValueError("Every event must be reviewed exactly once")
    groups=defaultdict(list);excluded=[];ignored=defaultdict(set)
    for identity,event in lookup.items():
        review=raw["reviews"][identity];video=raw["videos"][event["video_id"]]
        clean=validate_review(review,event,video)
        for key in ("spatial_valid","temporal_valid","xy","xy_normalized"):
            if clean[key]!=review[key]:raise ValueError("Inconsistent review field: "+identity+":"+key)
        if not review.get("verified"):raise ValueError("Unverified review: "+identity)
        if not review["temporal_valid"]:
            excluded.append(dict(id=identity,status=review["status"],frame=review["frame"]))
            if review["status"]!="wrong_event":
                # An uncertain touch is unknown, never a negative background target.
                lo=min(event["original_frame"],review["frame"])-6
                hi=max(event["original_frame"],review["frame"])+6
                ignored[event["video_id"]].update(range(max(0,lo),min(video["num_frames"],hi+1)))
            continue
        groups[(event["video_id"],review["frame"])].append((event,review))
    records=defaultdict(list);merges=[]
    for (stem,frame),members in sorted(groups.items()):
        labels={r["label"] for e,r in members};decision=None
        if len(labels)>1:
            ids={e["id"] for e,r in members}
            allowed={"Khe4P8QcJkg_5_4k:3759:reception","Khe4P8QcJkg_5_4k:3765:pass"}
            if (stem,frame)!=("Khe4P8QcJkg_5_4k",3761) or ids!=allowed or labels!={"reception","pass"}:
                raise ValueError(f"Unresolved class conflict: {stem}:{frame}")
            label="reception"
            decision="Authorized resolution: first touch after reviewed serve at 3684; next separate pass at 3794; identical reviewed ball centre. Keep reception."
        else:label=next(iter(labels))
        spatial=[r for e,r in members if r["spatial_valid"]]
        xy=None;spread=0.
        if spatial:
            points=np.asarray([r["xy"] for r in spatial],float)
            spread=float(np.linalg.norm(points[:,None]-points[None,:],axis=-1).max())
            if spread>32:raise ValueError(f"Duplicate XY disagreement >32 native pixels: {stem}:{frame}")
            xy=np.median(points,axis=0).tolist()
        video=raw["videos"][stem]
        record=dict(frame=frame,type=label,xy=xy,xy_normalized=[xy[0]/video["width"],xy[1]/video["height"]] if xy else None,
            spatial_valid=xy is not None,temporal_valid=True,
            review_ids=[e["id"] for e,r in members],annotation_ids=sorted({a for e,r in members for a in e["annotation_ids"]}))
        records[stem].append(record)
        if len(members)>1:
            merges.append(dict(video_id=stem,frame=frame,type=label,members=record["review_ids"],xy_rule="coordinate-wise median of verified points",max_pair_distance_native_px=spread,resolution=decision))
    return dict(schema="stes_verified_spatial_targets_v1",review_path=str(path.resolve()),review_sha256=file_sha(path),
        manifest_sha256=raw["manifest_sha256"],sources=raw["sources"],videos=raw["videos"],
        events=dict(records),ignored_exact_frames={k:sorted(v) for k,v in ignored.items()},excluded=excluded,merges=merges,
        audit=dict(reviewed=len(lookup),temporal_events=sum(map(len,records.values())),spatial_events=sum(e["spatial_valid"] for rows in records.values() for e in rows),duplicate_rows_removed=sum(len(m["members"])-1 for m in merges)),
        coordinate_system="normalized original-image xy, x/width y/height; native zero-based frames; no crop")


def spatial_targets(length,events):
    xy=np.zeros((length,2),np.float32);valid=np.zeros(length,bool)
    for event in events:
        f=event["frame"]
        if not 0<=f<length:raise ValueError("Contact outside decoded video")
        if event["spatial_valid"]:
            xy[f]=event["xy_normalized"];valid[f]=True
    return xy,valid
