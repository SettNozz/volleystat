import numpy as np
import torch
from torch import nn
from action_recognition.models.sequence_event_model import MultiScaleEventTransformer,save_checkpoint,load_checkpoint
from scripts.evaluation.train_team_pose_experiment import offline_sequence,condition_sequence,TeamDataset
from action_recognition.event_clips.team_pose_features import VALUE_NAMES


def test_offline_window_preserves_future_and_fixed_contact_position():
    clip=dict(start_frame=40,end_frame=159,contact_frame=100)
    v=np.zeros((120,42),np.float32);v[:,0]=np.arange(40,160);m=np.ones_like(v,bool)
    x,valid=offline_sequence(clip,v,m,np.arange(40,160))
    assert x[60,0]==100 and x[119,0]==159
    assert valid[119,0] and x[0,0]==40


def test_missing_end_context_is_right_padded_not_shifted():
    clip=dict(start_frame=40,end_frame=130,contact_frame=100)
    v=np.ones((91,42),np.float32);m=np.ones_like(v,bool)
    x,valid=offline_sequence(clip,v,m,np.arange(40,131))
    assert valid[60,0] and valid[90,0]
    assert not valid[91:,0].any() and not x[91:,0].any()


def test_control_conditions_keep_same_dimensions_without_team_information():
    seq=(np.ones((120,42),np.float32),np.ones((120,42),bool))
    a,am=condition_sequence(seq,'trajectory');b,bm=condition_sequence(seq,'pose')
    assert a.shape==b.shape==(120,42)
    assert not am[:,10:].any() and not am[:,5].any()
    assert bm[:,:18].all() and not bm[:,18:].any()


class Capture(nn.Module):
    def forward(self,x):
        self.seen=x.detach().clone()
        return torch.zeros((len(x),96))


def test_offline_short_branch_is_centered_while_old_default_stays_recent():
    x=torch.arange(120,dtype=torch.float32).reshape(1,120,1).repeat(1,1,18)
    for centered,expected in [(False,60),(True,30)]:
        model=MultiScaleEventTransformer(short_window_centered=centered)
        model.long_branch=Capture();model.short_branch=Capture();model.eval()
        with torch.no_grad():model(x)
        assert model.short_branch.seen[0,0,0].item()==expected
        assert model.short_branch.seen.shape[1]==60
        assert model.long_branch.seen.shape[1]==120


def test_offline_checkpoint_roundtrip_retains_window_contract(tmp_path):
    model=MultiScaleEventTransformer(input_dim=84,short_window_centered=True)
    path=tmp_path/'offline.pt'
    save_checkpoint(path,model,'multiscale',84,extra=dict(sequence_len=120,short_sequence_len=60,short_window_centered=True))
    restored,metadata=load_checkpoint(path)
    assert restored.short_window_centered and metadata['short_window_centered']


def test_validation_dataset_reuses_train_normalization():
    clip=dict(source_stem='train',clip_id='x')
    train=dict(clip=clip,target=0,values=np.ones((120,42),np.float32),valid=np.ones((120,42),bool))
    validation=dict(clip=dict(source_stem='validation',clip_id='y'),target=1,values=np.full((120,42),999,np.float32),valid=np.ones((120,42),bool))
    tr=TeamDataset([train],'team_pose');va=TeamDataset([validation],'team_pose',(tr.mean,tr.std))
    np.testing.assert_array_equal(va.mean,tr.mean)
    assert tr.mean[0]==1


def test_contact_pool_reads_central_encoded_frame_and_keeps_future_context():
    from action_recognition.models.sequence_event_model import _TemporalTransformerBranch
    branch=_TemporalTransformerBranch(4,d_model=4,num_layers=1,contact_center_pool=True)
    branch.norm=nn.Identity();branch.proj=nn.Identity();branch.encoder=nn.Identity()
    x=torch.arange(120,dtype=torch.float32).reshape(1,120,1).repeat(1,1,4)
    with torch.no_grad():
        branch.pos.zero_()
        assert torch.equal(branch(x),x[:,60])
    # Full encoder receives the complete input, including future frames.
    class FutureEncoder(nn.Module):
        def forward(self,h):return h+h[:,-1:]
    branch.encoder=FutureEncoder()
    assert torch.equal(branch(x),x[:,60]+x[:,-1])


def test_contact_pool_checkpoint_roundtrip_and_legacy_default(tmp_path):
    from action_recognition.models.sequence_event_model import build_model
    model=build_model('multiscale',84,short_window_centered=True,contact_center_pool=True)
    path=tmp_path/'center.pt'
    save_checkpoint(path,model,'multiscale',84,extra=dict(short_window_centered=True,contact_center_pool=True))
    restored,_=load_checkpoint(path)
    assert restored.contact_center_pool and restored.long_branch.contact_center_pool
    legacy=tmp_path/'legacy.pt'
    save_checkpoint(legacy,build_model('multiscale',18),'multiscale',18)
    old,_=load_checkpoint(legacy)
    assert not old.contact_center_pool and not old.short_window_centered


def test_rebuild_preserves_labels_frames_and_original_channels(tmp_path):
    import json
    from scripts.evaluation.rebuild_contact_candidates import rebuild
    from action_recognition.event_clips.team_pose_features import CANDIDATE_SCHEMA
    source=tmp_path/'source';source.mkdir();(source/'features').mkdir();(source/'poses').mkdir()
    (source/'baseline_features_snapshot').mkdir()
    clip=dict(clip_id='clip',source_stem='video',start_frame=0,end_frame=0,contact_frame=0,
              label_source='human',event_types=['attack'])
    for name,payload in [('manifest.json',dict(clips=[clip])),('parent_manifest.json',dict(clips=[clip])),
                         ('geometry.json',dict(net_y=90)),('split_policy.json',dict(unit='source_video')),
                         ('preflight.json',dict(status='complete',sources={'video':{}},geometry=dict(net_y=90),
                            split_policy=dict(unit='source_video'),value_names=list(VALUE_NAMES),input_features=[]))]:
        (source/name).write_text(json.dumps(payload))
    values=np.arange(42,dtype=np.float32)[None,:];mask=np.ones((1,42),bool)
    np.savez(source/'features/clip.npz',features=values,valid=mask,frames=np.array([0]),schema=np.array('old'))
    row=dict(f=0,features_context=dict(ball=None),poses=[])
    (source/'poses/video_s000.jsonl').write_text(json.dumps(row)+'\n')
    output=tmp_path/'new';rebuild(source,output)
    with np.load(output/'features/clip.npz') as data:
        np.testing.assert_array_equal(data['features'][:,:18],values[:,:18])
        np.testing.assert_array_equal(data['valid'][:,:18],mask[:,:18])
        assert data['schema'].item()==CANDIDATE_SCHEMA
        assert data['candidate_track_ids'].tolist()==[[-1,-1]]
    assert json.loads((output/'manifest.json').read_text())['clips'][0]['event_types']==['attack']
    with np.load(source/'features/clip.npz') as data:np.testing.assert_array_equal(data['features'],values)
    import pytest
    with pytest.raises(ValueError,match='existing output'):rebuild(source,output)
