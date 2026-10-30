import copy,json
import pytest
import torch
from action_recognition.rally_spotting.stes_continuation import lr_factor,extend_scheduler,validate_source


def test_extend_schedule_retains_weights_and_adam_moments():
    p=torch.nn.Parameter(torch.tensor([1.]))
    opt=torch.optim.AdamW([p],lr=.001)
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda e:lr_factor(e,3))
    for _ in range(2):opt.zero_grad();p.square().sum().backward();opt.step();sched.step()
    weight=p.detach().clone();moment=opt.state[p]["exp_avg"].clone();steps=opt.state[p]["step"].clone()
    sched.lr_lambdas=[lambda e:lr_factor(e,20)]
    extend_scheduler(opt,sched,2,20)
    assert sched.last_epoch==2 and opt.param_groups[0]["lr"]==pytest.approx(.001*lr_factor(2,20))
    torch.testing.assert_close(p,weight);torch.testing.assert_close(opt.state[p]["exp_avg"],moment);torch.testing.assert_close(opt.state[p]["step"],steps)
    opt.zero_grad();p.square().sum().backward();opt.step();sched.step()
    assert sched.last_epoch==3 and opt.param_groups[0]["lr"]==pytest.approx(.001*lr_factor(3,20))


@pytest.fixture
def source(tmp_path):
    parent={key:"same" for key in ("schema","review_sha256","canonical_audit","source_revision","author_code_hashes","backbone_sha256","cache_preflight_sha256","holdout","train_sources")}
    parent.update(recipe=dict(epochs=3,batch=1,steps=128,initialization="ImageNet"),implementation_hashes={"action_recognition/models/example.py":"unchanged"})
    (tmp_path/"preflight.json").write_text(json.dumps(parent))
    for name in ("best.pt","best_metrics.json","best_predictions.npz"):(tmp_path/name).write_text("source")
    ck={k:{} for k in ("model","scaler","python_rng","numpy_rng","torch_rng","cuda_rng")}
    ck.update(next_epoch=2,best=10.,history=[dict(epoch=1),dict(epoch=2)],optimizer=dict(param_groups=[dict(lr=.00025)]),scheduler=dict(last_epoch=2,base_lrs=[.001]))
    path=tmp_path/"last.pt";torch.save(ck,path)
    target=copy.deepcopy(parent);target["recipe"]["epochs"]=20
    return path,target


def test_full_checkpoint_continues_from_third_epoch(source):
    path,target=source;result=validate_source(path,target,20)
    assert result["completed_epochs"]==2 and result["start_epoch"]==3 and result["new_epochs"]==18
    assert result["next_lr"]==pytest.approx([.001*lr_factor(2,20)])


@pytest.mark.parametrize("change",["review_sha256","batch"])
def test_changed_inputs_rejected(source,change):
    path,target=source
    if change=="batch":target["recipe"][change]=2
    else:target[change]="changed"
    with pytest.raises(ValueError,match="mismatch"):validate_source(path,target,20)
