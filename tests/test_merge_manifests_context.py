from action_recognition.event_clips.merge_manifests import merge_clip_records


def test_human_context_merges_without_replacing_stronger_action_label():
    current = {
        "clip_id": "c1",
        "event_types": ["attack"],
        "label_source": "human",
        "context_role": "attack",
        "context_label_source": "weak_fsm",
    }
    candidate = {
        "clip_id": "c1",
        "event_types": ["other"],
        "label_source": "model",
        "context_role": "free_ball",
        "reception_quality": "not_applicable",
        "context_label_source": "human",
    }

    merged, changed = merge_clip_records(current, candidate)

    assert changed is True
    assert merged["event_types"] == ["attack"]
    assert merged["label_source"] == "human"
    assert merged["context_role"] == "free_ball"
    assert merged["context_label_source"] == "human"
