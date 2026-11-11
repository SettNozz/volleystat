from io import BytesIO
import json
import zipfile

import pytest

from src.tracking.player_contacts import build_player_contacts
from src.viewer_app.player_highlights import player_highlights


def pose(x, player_id="P1", team="b", raw_id=1):
    return {"id": raw_id, "player_id": player_id, "team": team,
            "bb": [x-20, 0, x+20, 100], "kp": [[x, 50, .99] for _ in range(17)]}


def frame(f, poses=None):
    return {"f": f, "ball": [100, 50], "poses": poses if poses is not None else [pose(100)]}


def event(f, kind):
    return {"frame": f, "type": kind, "confidence": .9, "source": "dense_contact_spotter"}


def test_highlights_keep_only_the_players_assigned_contacts_in_time_order():
    events = [event(60, "attack"), event(30, "reception"), event(90, "serve")]
    frames = [frame(30), frame(60), frame(90, [pose(100, "P2", raw_id=2)])]
    report = build_player_contacts(events, frames, 30)
    moments = player_highlights(report, events, 30, "P1")
    assert [(m["frame"], m["category"]) for m in moments] == [(30, "reception"), (60, "attack")]
    assert moments[0]["time"] == 1.0 and not moments[0]["uncertain"]
    assert [m["category"] for m in player_highlights(report, events, 30, "P2")] == ["serve"]
    with pytest.raises(ValueError):
        player_highlights(report, events, 30, "P9")


def test_unresolved_contacts_are_suggestions_only_on_request():
    events = [event(30, "attack")]
    report = build_player_contacts(events, [frame(30)], 30)
    contact = report["contacts"][0]
    contact.update(status="unresolved", player_id=None, proposed_player_id="P1")
    assert player_highlights(report, events, 30, "P1") == []
    moments = player_highlights(report, events, 30, "P1", include_suggestions=True)
    assert len(moments) == 1 and moments[0]["uncertain"]


def test_serve_fault_contacts_count_as_serves():
    events = [{"frame": 30, "type": "serve_fault"}]
    faults = build_player_contacts([dict(events[0], type="serve")], [frame(30)], 30)["contacts"]
    report = {"roster_ids": ["P1"], "contacts": []}
    moments = player_highlights(report, events, 30, "P1", extra_contacts=faults)
    assert [(m["type"], m["category"]) for m in moments] == [("serve_fault", "serve")]


@pytest.fixture
def client(tmp_path):
    import viewer
    ann_path, stats_path = tmp_path / "clip_annotations.json", tmp_path / "clip_stats.json"
    ann_path.write_text(json.dumps({"video": "clip", "fps": 30, "frames": [frame(30)]}), encoding="utf-8")
    stats_path.write_text(json.dumps({"video": "clip", "fps": 30, "all_events": [event(30, "attack")]}),
                          encoding="utf-8")
    viewer.jobs["highlight-test"] = {"job_id": "highlight-test", "video_stem": "clip", "status": "done", "annotations_path": str(ann_path),
                                     "stats_path": str(stats_path), "video_path": str(tmp_path / "clip.mp4")}
    with viewer.app.test_client() as test_client:
        yield test_client
    viewer.jobs.pop("highlight-test", None)
    viewer._player_contact_inputs.cache_clear()


def test_statistics_page_adds_personal_view_and_keeps_the_team_table(client):
    general = client.get("/player-statistics/highlight-test").data
    assert b'id="personal"' not in general and b'id="player-filter"' in general
    page = client.get("/player-statistics/highlight-test?player=P1&view=serve").data
    assert b'id="personal"' in page and b'<th scope="row">' in page
    assert b"/player-highlights/highlight-test?player=P1" in page
    assert b"Opposite hitter" in page
    assert b'id="personal"' not in client.get("/player-statistics/highlight-test?player=P9").data


def test_highlights_page_lists_moments_and_rejects_unknown_players(client):
    page = client.get("/player-highlights/highlight-test?player=P1&type=attack")
    assert page.status_code == 200
    moments = json.loads(page.data.split(b"const MOMENTS = ", 1)[1].split(b";\n", 1)[0])
    assert [(m["frame"], m["category"]) for m in moments] == [(30, "attack")]
    assert b"/api/video/highlight-test" in page.data
    assert client.get("/player-highlights/highlight-test?player=P9").status_code == 404


def test_excel_export_writes_position(client):
    profiles = {"P1": {"name": "Ann", "number": "7", "position": "setter"}}
    response = client.post("/player-statistics/highlight-test/excel", data={"profiles": json.dumps(profiles)})
    assert response.status_code == 200, response.data
    with zipfile.ZipFile(BytesIO(response.data)) as workbook:
        strings = workbook.read("xl/sharedStrings.xml").decode()
    for text in ("Position", "Ann", "Setter", "P1"):
        assert f">{text}<" in strings
    bad = {"P1": {"position": "libero"}}
    assert client.post("/player-statistics/highlight-test/excel",
                       data={"profiles": json.dumps(bad)}).status_code == 400
