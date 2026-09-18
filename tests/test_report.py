import json
from switchsafe.report import render


def test_report_escapes_untrusted_transcripts(tmp_path):
    source = tmp_path / "input.json"
    source.write_text(json.dumps({
        "summary": {"warning": "uncalibrated", "sessions": {}},
        "samples": [{"hypothesis": "<script>alert(1)</script>", "decision": "accept"}],
    }))
    destination = tmp_path / "output.html"
    render(source, destination)
    document = destination.read_text()
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in document
    assert "<script>alert(1)</script>" not in document
