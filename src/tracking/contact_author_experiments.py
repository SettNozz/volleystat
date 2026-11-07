"""Read-only, source-bound experimental contact reports for the local viewer."""
import json
from pathlib import Path

from src.tracking.player_contacts import CONTACT_TYPES


class ExperimentSourceChanged(ValueError):
    pass


def experiment_path(stats_path):
    path = Path(stats_path)
    return path.with_name(path.name.removesuffix("_stats.json")+"_player_contact_experiment.json")


def load_experiment(path, source, stats):
    bundle = json.loads(Path(path).read_bytes())
    if bundle.get("schema") != "player_contact_experiment_v1" or bundle.get("source") != source:
        raise ExperimentSourceChanged("Contact experiment belongs to changed analysis files")
    reports = bundle.get("reports") or {}
    if not reports or bundle.get("selected_method") not in reports:
        raise ValueError("Experiment has no selected report")
    events = stats.get("all_events") or []
    expected = {i for i, event in enumerate(events) if event.get("type") in CONTACT_TYPES}
    for method, report in reports.items():
        contacts = report.get("contacts") or []
        if (report.get("schema") != "player_contacts_v1" or method not in (bundle.get("methods") or {})
                or len(contacts) != len(expected) or {c.get("event_index") for c in contacts} != expected):
            raise ValueError("Experiment contacts do not match source events")
        for contact in contacts:
            event = events[contact["event_index"]]
            if (contact.get("frame") != event.get("frame") or contact.get("type") != event.get("type")
                    or contact.get("label_source") != "derived" or contact.get("verified") is not False):
                raise ValueError("Experiment changed an event or claimed human verification")
            if contact.get("player_id") is not None and contact["player_id"] not in report.get("roster_ids", []):
                raise ValueError("Experiment author is outside its roster")
    return bundle
