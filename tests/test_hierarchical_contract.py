from action_recognition.hierarchical.contract import encode_clip_context


def test_contract_masks_reception_quality_only_for_service_reception():
    reception = encode_clip_context({
        "clip_id": "r",
        "contact_frame": 10,
        "fsm_event_type": "reception",
        "team": "team_a",
        "context_label_source": "human",
    })
    attack = encode_clip_context({
        "clip_id": "a",
        "contact_frame": 20,
        "fsm_event_type": "attack",
        "team": "team_b",
    })

    assert reception["reception_quality_mask"] is True
    assert reception["label_weight"] == 1.0
    assert attack["reception_quality_mask"] is False
    assert attack["label_weight"] == 0.5
