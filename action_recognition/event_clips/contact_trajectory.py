"""Offline image-space trajectory summaries; no gap filling or court-height claims."""
import numpy as np

SCHEMA='contact_trajectory_before_after_v1'
NAMES=('pre_vx','pre_vy','pre_speed','post_vx','post_vy','post_speed',
       'delta_vx','delta_vy','delta_speed','turn_sin','turn_cos',
       'pre_fit_error','post_fit_error','pre_coverage','post_coverage',
       'pre_reconstructed_fraction','post_reconstructed_fraction')
OFFSETS=np.arange(-12,13,dtype=np.int64)


def summarize(points,available,new_track,reconstructed,fps):
    """25 exact frames [-12,+12], normalized x/y; slopes use [-12,-2]/[2,12].

    Each slope requires its entire contiguous window. Changes additionally
    require uninterrupted context between both sides. No velocity spans a
    missing frame, off-camera point or track restart. Reconstructed coordinates
    remain derived inputs and have explicit fraction/provenance information.
    """
    p=np.asarray(points,float);a=np.asarray(available,bool)
    n=np.asarray(new_track,bool);r=np.asarray(reconstructed,bool)
    if p.shape!=(25,2) or any(x.shape!=(25,) for x in (a,n,r)):
        raise ValueError('Expected 25 absolute frames with two coordinates')
    if not np.isfinite(p).all() or not np.isfinite(fps) or fps<=0:
        raise ValueError('Nonfinite coordinates or invalid fps')
    values=np.zeros(len(NAMES),np.float32);valid=np.zeros(len(NAMES),bool)
    for side,idx in enumerate((np.arange(11),np.arange(14,25))):
        values[13+side]=a[idx].mean();valid[13+side]=True
        if a[idx].any():
            values[15+side]=r[idx][a[idx]].mean();valid[15+side]=True
        if not a[idx].all() or n[idx[1:]].any():continue
        t=OFFSETS[idx]/fps;t=t-t.mean();q=p[idx]-p[idx].mean(0)
        velocity=(t[:,None]*q).sum(0)/(t*t).sum()
        error=np.sqrt(np.mean(np.sum((q-t[:,None]*velocity)**2,axis=1)))
        base=side*3
        values[base:base+3]=[*velocity,np.linalg.norm(velocity)];valid[base:base+3]=True
        values[11+side]=error;valid[11+side]=True
    if valid[0] and valid[3] and a.all() and not n[1:].any():
        values[6:9]=values[3:6]-values[:3];valid[6:9]=True
        if values[2]>1e-6 and values[5]>1e-6:
            u=values[:2]/values[2];v=values[3:5]/values[5]
            values[9:11]=[np.clip(u[0]*v[1]-u[1]*v[0],-1,1),np.clip(u@v,-1,1)]
            valid[9:11]=True
    return values,valid
