"""Viewer edits may coexist with a frozen export, never change its frame inputs."""
from copy import deepcopy
import pytest
from scripts.render_saved_analysis import viewer_metadata_changes


def test_viewer_court_save_preserved_but_player_or_ball_edit_invalidates_export():
    source={'frames':[{'f':0,'ball':[1,2],'poses':[{'player_id':'P7'}]}],
            'fps':30,'court_corners':[[1,2]],'attack_heatmap':{'hits':[]}}
    viewer=deepcopy(source)
    viewer.update(court_corners=[[1.01,2]],attack_heatmap={'hits':[1]},
                  manual_court_correction={'source':'viewer'})
    assert set(viewer_metadata_changes(source,viewer))=={'court_corners','attack_heatmap','manual_court_correction'}
    viewer['frames'][0]['poses'][0]['player_id']='P1'
    with pytest.raises(ValueError,match='Render inputs'):
        viewer_metadata_changes(source,viewer)
    untrusted=deepcopy(source)
    untrusted['fps']=25
    with pytest.raises(ValueError):
        viewer_metadata_changes(source,untrusted)
