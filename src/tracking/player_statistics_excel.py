"""Export the viewer's reconciled player statistics and supporting evidence."""

from io import BytesIO


PLAYER_POSITIONS = {
    "outside_hitter": "Outside hitter",
    "middle_blocker": "Middle blocker",
    "setter": "Setter",
    "opposite_hitter": "Opposite hitter",
    "libero": "Libero",
}

STATISTICS_VIEWS = {
    "attack": ("Attacks", [
        ("total", "Attempts"), ("point", "Point-winning"), ("continued", "Play continued"),
        ("out", "Out"), ("net", "Into net"), ("lost", "Point lost, cause unknown"),
        ("unknown", "Unresolved"), ("errors", "Errors"),
    ]),
    "serve": ("Serves", [
        ("total", "Attempts"), ("in_play", "In play"), ("ace", "Aces"),
        ("error", "Errors"), ("unknown", "Unresolved"), ("rate", "Error %"),
    ]),
    "reception": ("Serve reception", [
        ("total", "Attempts"), ("controlled", "Controlled"), ("overpass", "Overpass"),
        ("error", "Errors"), ("unknown", "Unresolved"), ("rate", "Controlled %"),
    ]),
}


def player_statistic_value(kind, counts, key):
    if key == "errors":
        return counts["out"] + counts["net"]
    if key == "rate":
        classified = counts["total"] - counts["unknown"]
        return counts["error" if kind == "serve" else "controlled"] / classified if classified else None
    return counts[key]


def export_player_statistics(summary, profiles):
    import xlsxwriter

    output = BytesIO()
    with xlsxwriter.Workbook(output, {"in_memory": True, "strings_to_formulas": False,
                                      "strings_to_urls": False}) as workbook:
        header = workbook.add_format({"bold": True, "bg_color": "#E7EEFF", "font_color": "#183367"})
        percent = workbook.add_format({"num_format": "0.0%"})
        for kind, (title, columns) in STATISTICS_VIEWS.items():
            sheet = workbook.add_worksheet(title)
            coverage_columns = ["Human-confirmed authors", "Automatic authors", "Derived outcomes", "Unresolved outcomes"]
            headers = ["Player", "Jersey", "Position", "Roster ID"] + [label for _, label in columns] + coverage_columns
            sheet.write_row(0, 0, headers, header)
            sheet.set_column(0, 0, 24)
            for index, label in enumerate(headers[1:], start=1):
                sheet.set_column(index, index, max(11, len(label) + 2))
            for row, (player, stats) in enumerate(summary["players"].items(), start=1):
                profile = profiles.get(player, {})
                number = profile.get("number", "")
                label = profile.get("name") or (f"#{number}" if number else player)
                position = PLAYER_POSITIONS.get(profile.get("position", ""), "")
                sheet.write_row(row, 0, [label, number, position, player])
                for index, (key, _) in enumerate(columns, start=4):
                    value = player_statistic_value(kind, stats[kind], key)
                    if value is not None:
                        sheet.write_number(row, index, value, percent if key == "rate" else None)
                coverage = summary.get("coverage", {}).get(player, {}).get(kind, {})
                sheet.write_row(row, 4 + len(columns), [coverage.get(key, 0) for key in ("human", "automatic", "classified", "unknown")])
            sheet.freeze_panes(1, 1)
        evidence = workbook.add_worksheet("Contact evidence")
        keys = ["event_index", "frame", "player_id", "author_status", "author_source", "kind", "outcome",
                "outcome_source", "reason", "attempt_ref", "point_source", "point_reviewed", "event_verified"]
        evidence.write_row(0, 0, keys, header)
        evidence.set_column(0, len(keys) - 1, 22)
        for row, contact in enumerate(summary.get("reconciliation", {}).get("contacts", []), start=1):
            evidence.write_row(row, 0, [contact.get(key) for key in keys])
        evidence.freeze_panes(1, 0)
        points = workbook.add_worksheet("Point evidence")
        keys = ["attempt_ref", "rally_id", "set_id", "start_frame", "end_frame", "winner_side", "status", "source", "evidence", "verified"]
        points.write_row(0, 0, keys, header)
        points.set_column(0, len(keys) - 1, 22)
        for row, point in enumerate(summary.get("reconciliation", {}).get("points", []), start=1):
            points.write_row(row, 0, [point.get(key) for key in keys])
        points.freeze_panes(1, 0)
    output.seek(0)
    return output
