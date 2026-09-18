"""Generate a self-contained HTML review report from a completed JSON run."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def render(source: Path, destination: Path) -> None:
    report = json.loads(source.read_text(encoding="utf-8"))
    summary = report["summary"]
    def escape(value):
        return html.escape(str(value))
    metrics = "".join(f"<dt>{escape(k)}</dt><dd>{escape(summary.get(k))}</dd>" for k in (
        "model", "attempted", "samples", "failed", "p50_latency_seconds", "p95_latency_seconds",
        "mean_real_time_factor", "automated_acceptance_coverage"))
    rows = "".join("<tr>" + "".join(f"<td>{escape(row.get(k, ''))}</td>" for k in (
        "sample_id", "session_id", "hypothesis", "decision", "decision_reason", "error")) + "</tr>"
        for row in report["samples"])
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1"><title>SwitchSafe evaluation</title>
    <style>body{{font:15px system-ui;margin:30px;color:#16324f}}h1{{color:#125366}}
    dt{{font-weight:bold}}dd{{margin:0 0 10px}}table{{border-collapse:collapse;width:100%}}
    td,th{{text-align:left;border-bottom:1px solid #ccd6dd;padding:10px;vertical-align:top}}
    .warning{{padding:16px;background:#fff0c2}}input{{padding:10px;width:320px;max-width:90%}}</style>
    <h1>SwitchSafe: IMDA corpus evaluation</h1>
    <p class="warning">{escape(summary['warning'])}<br>WER/CER and unsafe-acceptance rate require independent labels.</p>
    <dl>{metrics}</dl><h2>Session breakdown</h2><pre>{escape(json.dumps(summary['sessions'], indent=2))}</pre>
    <h2>Transcript review</h2><input id="filter" placeholder="Filter transcript, session or decision" aria-label="Filter recordings">
    <table><thead><tr><th>Sample</th><th>Session</th><th>Transcript</th><th>Decision</th><th>Reason</th><th>Error</th></tr></thead>
    <tbody>{rows}</tbody></table>
    <script>document.getElementById('filter').addEventListener('input', function() {{
    const term=this.value.toLowerCase();document.querySelectorAll('tbody tr').forEach(row=>{{
    row.hidden=!row.textContent.toLowerCase().includes(term);}});}});</script></html>"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/derived/report.html"))
    args = parser.parse_args()
    render(args.source, args.output)
    print(args.output.resolve())
