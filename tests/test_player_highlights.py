from io import BytesIO
import json
from pathlib import Path
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
    assert b"Libero" in page
    assert b'id="personal"' not in client.get("/player-statistics/highlight-test?player=P9").data


def test_highlights_open_in_main_viewer_and_api_keeps_saved_moments(client):
    page = client.get("/player-highlights/highlight-test?player=P1&type=attack")
    assert page.status_code == 200
    assert b'const INITIAL_PAGE = "highlights"' in page.data
    assert b'id="playerHighlightsPanel"' in page.data
    assert b'id="overlay"' in page.data
    payload = client.get("/api/player_highlights/highlight-test?player=P1").get_json()
    moments = payload["moments"]
    assert [(m["frame"], m["category"]) for m in moments] == [(30, "attack")]
    assert payload["player"] == "P1" and payload["roster"] == ["P1"]
    assert "stats_sha256" in payload["source"]
    assert b"/api/video/highlight-test" in page.data
    assert client.get("/player-highlights/highlight-test?player=P9").status_code == 404
    assert client.get("/api/player_highlights/highlight-test?player=P9").status_code == 404


def test_main_menu_replaces_match_stats_and_removes_rally_strip(client):
    page = client.get("/results/highlight-test")
    assert page.status_code == 200
    assert page.data.count(b">Player statistics</a>") == 1
    assert b"/player-statistics/highlight-test" in page.data
    for label in (b">Player highlights</button>", b">Set analysis</button>"):
        assert label in page.data
    for removed in (b"Match Stats", b"Set-to-attack trajectories", b'id="rallyClips"', b'id="page-table"'):
        assert removed not in page.data
    assert b'data-page="contacts"' not in page.data and b'id="playerContactsPanel"' not in page.data
    assert b'id="teamRosterPanel"' in page.data and b'id="roster-picker"' in page.data
    nav = page.data.split(b'<nav class="nav"', 1)[1].split(b'</nav>', 1)[0]
    assert b'Set analysis' not in nav and b'Player highlights' not in nav
    tools = page.data.split(b'<aside class="card video-workspace"', 1)[1].split(b'</aside>', 1)[0]
    for panel in (b'teamRosterPanel', b'setTrajectoryPanel', b'playerHighlightsPanel'):
        assert b'id="' + panel + b'"' in tools
    assert b'id="eventList"' not in page.data
    video, charts = page.data.split(b'id="page-charts"', 1)
    assert b'id="attackHeatmapPanel"' not in video and b'id="attackHeatmapPanel"' in charts
    assert b'id="togEvents" checked' in video
    assert b'id="videoEventSubstitutions"' in video


def test_roster_metadata_uses_saved_players_and_source_scoped_portraits(client):
    response = client.get("/api/player_roster/highlight-test")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["roster"] == ["P1"]
    assert "annotations_sha256" in payload["source"] and "setter" in payload["positions"]
    assert payload["positions"]["libero"] == "Libero"
    assert payload["lineup"] == {}
    assert client.get("/api/player_roster/no-such-result").status_code == 404


def test_statistics_share_navigation_and_move_explanation_below_tables(client):
    page = client.get("/player-statistics/highlight-test").data
    assert b"viewer_page.css" in page and b"viewer_statistics.css" in page
    assert b"player_profiles.js" in page
    assert b"Change photo for" not in page  # Controls are created only in Edit players.
    assert page.index(b'class="reconciliation-note"') > page.index(b"</table>")
    assert b"view=video" in page and b"view=charts" in page
    assert b">Set analysis</a>" not in page and b">Player highlights</a>" not in page
    assert b'const INITIAL_PAGE = "trajectories"' in client.get("/results/highlight-test?view=trajectories").data


def test_highlights_without_roster_return_empty_state(client):
    import viewer
    path = viewer.jobs["highlight-test"]["annotations_path"]
    annotations = json.loads(Path(path).read_text())
    annotations["frames"] = []
    Path(path).write_text(json.dumps(annotations))
    viewer._player_contact_inputs.cache_clear()
    payload = client.get("/api/player_highlights/highlight-test").get_json()
    assert payload["player"] is None and payload["moments"] == [] and payload["roster"] == []
    assert client.get("/player-highlights/highlight-test").status_code == 200


@pytest.mark.parametrize(("position", "label"), [("setter", "Setter"), ("libero", "Libero")])
def test_excel_export_writes_position(client, position, label):
    profiles = {"P1": {"name": "Ann", "number": "7", "position": position}}
    response = client.post("/player-statistics/highlight-test/excel", data={"profiles": json.dumps(profiles)})
    assert response.status_code == 200, response.data
    with zipfile.ZipFile(BytesIO(response.data)) as workbook:
        strings = workbook.read("xl/sharedStrings.xml").decode()
    for text in ("Position", "Ann", label, "P1"):
        assert f">{text}<" in strings
    bad = {"P1": {"position": "unknown_position"}}
    assert client.post("/player-statistics/highlight-test/excel",
                       data={"profiles": json.dumps(bad)}).status_code == 400
