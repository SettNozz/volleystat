import numpy as np
import torch
from action_recognition.event_clips.contact_trajectory import summarize,OFFSETS,NAMES


def inputs():
    t=OFFSETS/30
    p=np.column_stack([.5+.1*t,.5-.2*t])
    return p,np.ones(25,bool),np.zeros(25,bool),np.zeros(25,bool)


def test_constant_motion_has_correct_velocity_without_false_contact():
    p,a,n,r=inputs();v,m=summarize(p,a,n,r,30)
    np.testing.assert_allclose(v[:2],[.1,-.2],atol=1e-6)
    np.testing.assert_allclose(v[6:10],0,atol=1e-6)
    assert np.isclose(v[10],1) and m.all()
    assert max(v[11:13])<1e-7


def test_turn_uses_signed_sine_cosine_and_speed_change():
    p,a,n,r=inputs();t=OFFSETS/30
    p[:,0]=.5+np.minimum(t,0)*.1;p[:,1]=.5+np.maximum(t,0)*.2
    v,m=summarize(p,a,n,r,30)
    np.testing.assert_allclose(v[6:11],[-.1,.2,.1,1,0],atol=1e-6)
    assert m[6:11].all()


def test_missing_and_off_camera_frames_never_form_velocity_across_gap():
    p,a,n,r=inputs();a[5]=False;p[5]=0
    v,m=summarize(p,a,n,r,30)
    assert not m[:3].any() and not m[6:11].any()
    assert m[3:6].all() and v[13]<1
    assert not v[~m].any()


def test_restart_or_missing_at_contact_disables_changes_but_keeps_each_side():
    for gap in (True,False):
        p,a,n,r=inputs()
        if gap:a[12]=False
        else:n[12]=True
        v,m=summarize(p,a,n,r,30)
        assert m[:6].all() and not m[6:11].any()


def test_stationary_ball_has_no_fabricated_direction_and_reconstruction_stays_explicit():
    p,a,n,r=inputs();p[:]=.5;r[:11]=True
    v,m=summarize(p,a,n,r,30)
    assert not m[9:11].any() and v[15]==1 and v[16]==0
    a[:]=False;v,m=summarize(p,a,n,r,30)
    assert set(np.flatnonzero(m))=={13,14} and not v.any()


def test_late_fusion_starts_as_exact_legacy_model_and_keeps_rng():
    from scripts.evaluation.run_trajectory_contact_experiment import TrajectoryFusion
    from action_recognition.models.sequence_event_model import build_model
    torch.manual_seed(42);legacy=build_model('multiscale',18,sequence_len=120,short_sequence_len=60)
    legacy_rng=torch.get_rng_state()
    torch.manual_seed(42);fusion=TrajectoryFusion()
    assert torch.equal(torch.get_rng_state(),legacy_rng)
    for key,value in legacy.state_dict().items():assert torch.equal(value,fusion.base.state_dict()[key])
    legacy.eval();fusion.eval();x=torch.randn(2,120,18);z=torch.randn(2,len(NAMES)*2)
    with torch.no_grad():torch.testing.assert_close(legacy(x),fusion(x,z),rtol=0,atol=0)
