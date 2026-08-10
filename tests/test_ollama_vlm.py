from __future__ import annotations

import json

from action_recognition.vlm.ollama_client import OllamaVLMClient, normalize_base_url
from action_recognition.vlm.prelabel_events import apply_prediction, should_skip_clip


class _Response:
    status_code = 200

    def __init__(self, value):
        self._value = value

    def json(self):
        return self._value

    def raise_for_status(self):
        return None


class _Session:
    def __init__(self):
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        if url.endswith("/api/tags"):
            return _Response({"models": [{"name": "qwen3-vl:8b-instruct"}]})
        if url.endswith("/api/version"):
            return _Response({"version": "1.2.3"})
        return _Response(
            {
                "message": {
                    "content": json.dumps(
                        {
                            "action": "attack",
                            "confidence": 0.82,
                            "contact_visible": True,
                            "reason": "Jump hit near net",
                        }
                    )
                },
                "total_duration": 10,
                "eval_count": 7,
            }
        )


def test_normalize_base_url_adds_scheme_and_removes_slash():
    assert normalize_base_url("macbook.local:11434/") == "http://macbook.local:11434"


def test_client_health_and_structured_chat():
    session = _Session()
    client = OllamaVLMClient(
        "macbook.local:11434",
        "qwen3-vl:8b-instruct",
        retries=0,
        session=session,
    )

    status = client.check()
    response = client.chat_json(
        "classify",
        [b"jpeg"],
        {"type": "object", "properties": {"action": {"type": "string"}}},
    )

    assert status["version"] == "1.2.3"
    assert status["model_available"] is True
    assert response.data["action"] == "attack"
    payload = session.calls[-1][2]["json"]
    assert payload["format"]["type"] == "object"
    assert payload["messages"][0]["images"]
    assert payload["think"] is False


def test_apply_prediction_keeps_auditable_vlm_metadata():
    clip = {"clip_id": "c1", "event_types": []}
    apply_prediction(
        clip,
        {
            "action": "pass",
            "confidence": 1.4,
            "contact_visible": True,
            "reason": "Controlled overhead set",
        },
        model="qwen3-vl:8b-instruct",
        seconds=2.345,
    )

    assert clip["event_types"] == ["pass"]
    assert clip["label_source"] == "vlm_prelabel"
    assert clip["vlm_prediction"]["confidence"] == 1.0
    assert clip["vlm_prediction"]["seconds"] == 2.35


def test_force_never_overwrites_human_decisions():
    assert should_skip_clip({"label_source": "human"}, force=True)
    assert should_skip_clip({"label_source": "cleared"}, force=True)
    assert should_skip_clip({"label_source": "fsm_rule"}, force=False)
    assert not should_skip_clip({"label_source": "fsm_rule"}, force=True)
