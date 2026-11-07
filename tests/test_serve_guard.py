import pytest
from action_recognition.inference.serve_guard import ServeGuard


def test_active_rally_cannot_restart_on_state_glitch_or_missing_ball():
    g=ServeGuard();g.update(0,'rally_active')
    for t,state in [(1,''),(2,'serve_ready'),(10,'serving')]:
        g.update(t,state);assert g.reason('serve')=='serve_during_active_rally'
    assert g.reason('attack') is None


def test_cooldown_starts_once_at_rally_end_not_every_end_frame():
    g=ServeGuard();g.update(0,'rally_active');g.update(5,'rally_end');g.update(6,'rally_end')
    assert g.remaining==2
    g.update(8,'serve_ready');g.update(8.1,'serving');assert g.reason('serve') is None


def test_early_episode_stays_blocked_after_timer_expires():
    g=ServeGuard(enforce_restart_pause=True);g.update(0,'rally_end');g.update(2,'serving')
    assert g.reason('serve')=='serve_restart_too_soon'
    g.update(4,'serving');assert g.reason('serve')=='serve_restart_too_soon'
    g.update(5,'serve_ready');g.update(6,'serving');assert g.reason('serve') is None


def test_timeout_alone_does_not_confirm_a_new_start_and_no_relabel():
    g=ServeGuard();g.update(0,'rally_end');g.update(10,'serve_ready')
    d=g.decision('serve');assert d['label'] is None and d['raw_label']=='serve'
    assert d['suppression_reason']=='serve_start_not_confirmed'


def test_initial_serving_and_reset_allow_independent_video():
    g=ServeGuard();g.update(0,'serving');assert g.reason('serve') is None
    g.update(1,'rally_active');g.reset();g.update(0,'serving');assert g.reason('serve') is None
    with pytest.raises(ValueError):g.update(-1,'serving')


def predictor_for_logits(logits):
    import torch
    from collections import deque
    from action_recognition.inference.streaming_event_predictor import StreamingEventPredictor
    from action_recognition.event_clips.feature_schema import RESOLUTION_NORMALIZED_SCHEMA
    class Model:
        def __call__(self,x):return torch.tensor([logits],dtype=torch.float32)
    p=StreamingEventPredictor.__new__(StreamingEventPredictor)
    p.feature_schema=RESOLUTION_NORMALIZED_SCHEMA;p.sequence_len=2;p.INFER_EVERY=1
    p._buf=deque(maxlen=2);p._recent=deque(maxlen=5);p.reset()
    p.model=Model();p.device='cpu';p.event_types=['serve','reception','pass','attack']
    p._mean=p._std=p._active_mask=None
    return p


def feed(p,t,state):
    return p.add_frame((100,100),{'team_a':[],'team_b':[]},200,30,fsm_state=state,
                       frame_size=(1280,720),video_time=t)


def test_predictor_preserves_raw_serve_but_does_not_relabel_or_reuse_cached_serve():
    p=predictor_for_logits([10,0,0,0])
    feed(p,0,'serving');assert feed(p,.1,'serving')=='serve'
    assert feed(p,.2,'rally_active') is None
    assert p.last_decision['raw_label']=='serve'
    assert p.last_decision['suppression_reason']=='serve_during_active_rally'
    p.INFER_EVERY=100
    assert feed(p,.3,'rally_active') is None


def test_phase_gating_cannot_reintroduce_serve_during_restart_pause():
    p=predictor_for_logits([9,0,0,10])
    p.serve_guard=ServeGuard(enforce_restart_pause=True)
    feed(p,0,'rally_end');assert feed(p,1,'serving') is None
    assert p.last_decision['gated_label']=='serve'
    assert p.last_decision['suppression_reason']=='serve_restart_too_soon'


def test_default_pause_warns_without_discarding_a_recognized_serving_episode():
    g=ServeGuard();g.update(0,'rally_end');g.update(2,'serving')
    d=g.decision('serve')
    assert d['label']=='serve' and d['suppression_reason'] is None
    assert d['warning_reason']=='serve_restart_pause_uncertain'
    g.update(4,'serving');assert g.reason('serve') is None
    g.update(5,'rally_active');assert g.reason('serve')=='serve_during_active_rally'
