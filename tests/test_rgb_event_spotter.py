import numpy as np
import torch
from action_recognition.rally_spotting.rgb_spotter import RGBEventSpotter,frame_targets,decode,evaluate,predict


def test_empty_timeline_is_background_not_a_forced_action():
    p=np.zeros((50,5),np.float32);p[:,0]=.9;p[:,1:]=.025
    assert decode(p,.1)==[]
    np.testing.assert_array_equal(frame_targets(50,[]),np.zeros(50))


def test_peaks_match_one_contact_not_one_event_per_frame():
    p=np.zeros((80,5),np.float32);p[:,0]=1
    p[38:43,0]=.1;p[38:43,1]=[.5,.7,.9,.7,.5]
    metrics,events=evaluate(p,[dict(frame=41,type='serve')],.4)
    assert len(events)==1 and events[0]['frame']==40
    assert metrics['at_1']['micro']['tp']==1 and metrics['at_1']['micro']['fp']==0


def test_unsupported_contacts_do_not_become_background_or_erase_nearby_gold():
    y=frame_targets(30,[dict(frame=13,type='pass')],ignored=[10])
    assert y[10]==-100 and y[13]==3 and y[25]==0
    y=frame_targets(30,[dict(frame=13,type='pass'),dict(frame=13,type='attack')])
    assert y[13]==-100


def test_model_shape_and_future_context():
    torch.manual_seed(2);model=RGBEventSpotter(input_dim=8,hidden=8,layers=1).eval()
    x=torch.zeros(1,20,8);changed=x.clone();changed[:,11:]=1
    with torch.no_grad():
        a=model(x);b=model(changed)
    assert a.shape==(1,20,5) and not torch.allclose(a[:,10],b[:,10])


def test_overlap_prediction_covers_end_and_short_video():
    class Model(torch.nn.Module):
        def forward(self,x):return torch.zeros((*x.shape[:2],5))
    for length in (3,127,251):
        p=predict(Model(),np.zeros((length,8),np.float32),torch.device('cpu'))
        assert p.shape==(length,5)
        np.testing.assert_allclose(p,.2,atol=1e-6)
