from action_recognition.event_clips.manifest_schema import (
    MANIFEST_SCHEMA_VERSION,
    upgrade_manifest,
    validate_context_value,
)


def test_legacy_manifest_upgrades_without_losing_fields():
    legacy = {
        "custom": "keep-me",
        "clips": [{
            "clip_id": "game_e0001_f10",
            "event_types": ["reception"],
            "fsm_event_type": "reception",
            "team": "team_b",
            "label_source": "human",
        }],
    }

    upgraded = upgrade_manifest(legacy)
    clip = upgraded["clips"][0]

    assert upgraded["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert upgraded["custom"] == "keep-me"
    assert clip["event_types"] == ["reception"]
    assert clip["context_role"] == "serve_reception"
    assert clip["stable_team_id"] == "observed_team"
    assert clip["reception_quality"] == "unknown"
    assert clip["context_label_source"] == "weak_fsm"
    assert clip["context_label_source"] != clip["label_source"]


def test_context_validator_rejects_unknown_value():
    assert validate_context_value("match_phase", "non_game") == "non_game"
    try:
        validate_context_value("match_phase", "coffee_break")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid phase was accepted")
