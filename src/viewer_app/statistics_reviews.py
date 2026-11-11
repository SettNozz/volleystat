"""Append-only point reviews bound to both saved analysis files."""

import json
from datetime import datetime, timezone


class StatisticsSourceChanged(ValueError):
    pass


def review_path(stats_path):
    return stats_path.with_name(stats_path.name.removesuffix("_stats.json") + "_point_reviews.jsonl")


def load_point_reviews(stats_path, source):
    path = review_path(stats_path)
    reviews = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("schema") != "point_review_v1" or record.get("source") != source:
                raise StatisticsSourceChanged("Analysis changed; saved point reviews belong to another source")
            reviews.append(record["review"])
    return reviews


def validate_point_review(body):
    if not isinstance(body.get("attempt_ref"), str):
        raise ValueError("attempt_ref must be a string")
    if type(body.get("reset", False)) is not bool:
        raise ValueError("reset must be a boolean")
    reset = body.get("reset", False)
    if not reset and ("winner_side" not in body or body["winner_side"] not in ("a", "b", None)):
        raise ValueError("Point winner must be a, b or null")
    return {"attempt_ref": body["attempt_ref"], "reset": reset,
            "winner_side": body.get("winner_side")}


def append_point_review(stats_path, source, review):
    record = {"schema": "point_review_v1", "source": source,
              "saved_at": datetime.now(timezone.utc).isoformat(), "review": review}
    with review_path(stats_path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
