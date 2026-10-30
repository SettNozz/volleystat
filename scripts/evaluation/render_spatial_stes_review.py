"""Render verified contact frames: green human XY, orange model XY (oracle time)."""
import argparse,json
from pathlib import Path
import cv2,numpy as np


def render(folder):
    folder=Path(folder);canonical=json.loads((folder/"canonical_labels.json").read_text(encoding="utf-8"))
    result=json.loads((folder/"report.json").read_text());stem=result["holdout"]
    predicted=np.load(folder/"best_predictions.npz")["xy"]
    events=[e for e in canonical["events"][stem] if e["spatial_valid"]]
    chosen=[]
    for label in ("serve","reception","pass","attack"):
        candidates=[e for e in events if e["type"]==label]
        chosen.extend(candidates[i] for i in np.linspace(0,len(candidates)-1,3,dtype=int))
    chosen.sort(key=lambda e:e["frame"])
    cap=cv2.VideoCapture(canonical["videos"][stem]["path"])
    tiles=[]
    try:
        for e in chosen:
            f=e["frame"];cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,image=cap.read()
            if not ok or int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))-1!=f:raise ValueError("Frame mismatch")
            image=cv2.resize(image,(640,360));truth=np.array(e["xy_normalized"]);pred=predicted[f]
            a=tuple(np.rint(truth*[640,360]).astype(int));b=tuple(np.rint(pred*[640,360]).astype(int))
            cv2.line(image,a,b,(200,200,200),1,cv2.LINE_AA)
            cv2.circle(image,a,9,(60,240,90),2,cv2.LINE_AA);cv2.drawMarker(image,a,(60,240,90),cv2.MARKER_CROSS,15,2)
            cv2.circle(image,b,9,(0,155,255),2,cv2.LINE_AA);cv2.drawMarker(image,b,(0,155,255),cv2.MARKER_TILTED_CROSS,15,2)
            distance=float(np.linalg.norm((pred-truth)*224))
            strip=np.full((32,640,3),25,np.uint8)
            cv2.putText(strip,f"{e['type']}  f={f}  error={distance:.1f}px @224",(10,22),cv2.FONT_HERSHEY_SIMPLEX,.55,(240,240,240),1,cv2.LINE_AA)
            tiles.append(np.vstack([strip,image]))
    finally:cap.release()
    grid=np.vstack([np.hstack(tiles[i:i+3]) for i in range(0,12,3)])
    header=np.full((75,1920,3),20,np.uint8)
    cv2.putText(header,"GREEN: human ball centre | ORANGE: STES prediction | Known human contact frames",(18,29),cv2.FONT_HERSHEY_SIMPLEX,.75,(240,240,240),1,cv2.LINE_AA)
    cv2.putText(header,"Held-out Khe video - localization only; these frames are NOT event-detection successes.",(18,58),cv2.FONT_HERSHEY_SIMPLEX,.65,(190,190,190),1,cv2.LINE_AA)
    output=folder/"spatial_contact_review.jpg"
    if not cv2.imwrite(str(output),np.vstack([header,grid]),[cv2.IMWRITE_JPEG_QUALITY,95]):raise OSError("Cannot save review")
    (folder/"spatial_contact_review_frames.json").write_text(json.dumps(dict(holdout=stem,frames=[e["frame"] for e in chosen],selection="three evenly spaced human contacts per class; not best-error examples"),indent=2))
    print(output)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("folder",type=Path);render(p.parse_args().folder)
