"""Write the viewer's player statistics as a three-sheet Excel workbook."""

from io import BytesIO


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
            headers = ["Player", "Jersey", "Roster ID"] + [label for _, label in columns]
            sheet.write_row(0, 0, headers, header)
            sheet.set_column(0, 0, 24)
            for index, label in enumerate(headers[1:], start=1):
                sheet.set_column(index, index, max(11, len(label) + 2))
            for row, (player, stats) in enumerate(summary["players"].items(), start=1):
                profile = profiles.get(player, {})
                number = profile.get("number", "")
                label = profile.get("name") or (f"#{number}" if number else player)
                sheet.write_row(row, 0, [label, number, player])
                for index, (key, _) in enumerate(columns, start=3):
                    value = player_statistic_value(kind, stats[kind], key)
                    if value is not None:
                        sheet.write_number(row, index, value, percent if key == "rate" else None)
            sheet.freeze_panes(1, 1)
    output.seek(0)
    return output
