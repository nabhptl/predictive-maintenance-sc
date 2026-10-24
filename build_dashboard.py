#!/usr/bin/env python
"""Build dashboard.html: one standalone page (double-click to open) that replays the trained models' output.

    python build_dashboard.py

Run it after train.py. It reads models/<domain>/replay.json for every machine type that has been trained.
A machine type without trained output shows the simulated preview instead.
"""
import json

import config
import fuzzy


def build_html():
    data = {
        "fuzzy": {
            "life": {k: list(v) for k, v in fuzzy.LIFE_SETS.items()},
            "wear": {k: list(v) for k, v in fuzzy.WEAR_SETS.items()},
            "warn": config.WARNING_AT,
            "crit": config.CRITICAL_AT,
        }
    }
    found = []
    for domain in config.DOMAINS:
        path = config.MODELS_DIR / domain / "replay.json"
        if path.exists():
            data[domain] = json.loads(path.read_text())
            found.append(domain)
    template = (config.ROOT / "templates" / "dashboard.html").read_text()
    assert "/*__PM_DATA__*/null" in template
    body = template.replace("/*__PM_DATA__*/null", json.dumps(data, separators=(",", ":")))
    html = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n</head>\n<body>\n'
            + body + "\n</body>\n</html>\n")
    return html, found


if __name__ == "__main__":
    html, found = build_html()
    out = config.ROOT / "dashboard.html"
    out.write_text(html)
    print(f"Wrote {out} with real data for: {', '.join(found) if found else 'nothing (simulated preview only)'}")
