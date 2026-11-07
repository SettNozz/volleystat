from copy import deepcopy

import numpy as np
import pytest

from scripts.evaluation.compare_contact_author_methods import align_predictions,paired,strict_transitions


def cases():
    return [{"contact":{"id":str(i),"video_id":"v","frame":i*30,"type":"pass"},
             "options":["other_team","P1","P2"],"target":t,"strict":True} for i,t in enumerate((1,2,0))]


def test_alignment_uses_ids_and_refuses_changed_truth_or_missing_test_rows():
    source=cases();records=[{**c["contact"],"truth":c["options"][c["target"]],"strict":True,"prediction":"P1"} for c in source]
    result=align_predictions(source,list(reversed(records)),lambda row:row["prediction"])
    assert [p["label"] for p in result]==["P1"]*3
    with pytest.raises(ValueError,match="missing"):align_predictions(source,records[:-1],lambda row:row["prediction"])
    changed=deepcopy(records);changed[1]["truth"]="P1"
    with pytest.raises(ValueError,match="differs"):align_predictions(source,changed,lambda row:row["prediction"])
    changed=deepcopy(records);changed[0]["strict"]=False
    with pytest.raises(ValueError,match="differs"):align_predictions(source,changed,lambda row:row["prediction"])


def test_strict_transition_counts_ignore_excluded_and_outer_test_labels_without_bridging():
    contacts=[{"video_id":v,"frame":i*30,"type":t} for v in ("train","outer")
              for i,t in enumerate(("serve","reception","pass","attack"))]
    near=np.array([1,0,0,1,1,0,0,1]);strict=np.array([True,True,False,True]*2)
    first=strict_transitions(contacts,near,strict,{"train"})
    assert first=={("serve","reception"):pytest.approx(1/3)}
    altered=near.copy();altered[2]=1;altered[4:]=1-altered[4:]
    assert strict_transitions(contacts,altered,strict,{"train"})==first
    assert strict_transitions(contacts,altered,strict,{"outer"})!={}


def test_paired_comparison_counts_each_exact_person_once_and_unknown_as_wrong():
    source=cases();left=[{"label":p} for p in ("P1","P1","other_team")]
    right=[{"label":p} for p in ("unknown","P2","other_team")]
    assert paired(source,left,right,np.ones(3,bool))=={"codex_only":1,"claude_only":1,"both_correct":1}
    assert paired(source,left,right,np.array([True,True,False]))=={"codex_only":1,"claude_only":1}
