from pathlib import Path
import json

from switchsafe.corpus import run_directory
from switchsafe.transcribe import Transcript
from switchsafe.policy import RiskSignals, decide, Decision


class FakeTranscriber:
    def transcribe(self, path):
        if path.stem == "bad":
            raise ValueError("corrupt file")
        return Transcript("test request", 1, -.2, .01)


def test_session_report_and_corrupt_file(tmp_path: Path):
    for session, name in [("SESSION0", "good"), ("SESSION1", "bad")]:
        folder = tmp_path / "SPEAKER0001" / session
        folder.mkdir(parents=True)
        (folder / f"{name}.WAV").touch()
    output = tmp_path / "report.json"
    summary = run_directory(tmp_path / "SPEAKER0001", output, "fake", transcriber=FakeTranscriber())
    assert summary["attempted"] == 2
    assert summary["failed"] == 1
    assert summary["sessions"]["SESSION0"]["samples"] == 1
    assert output.with_suffix(".review.csv").exists()
    assert len(json.loads(output.read_text())["samples"]) == 2


def test_missing_entity_confidence_requires_confirmation():
    assert decide(RiskSignals(-.2, .01, critical_action=True))[0] == Decision.CONFIRM


def test_invalid_signals_fail_closed():
    assert decide(RiskSignals(float("nan"), .01))[0] == Decision.ESCALATE
