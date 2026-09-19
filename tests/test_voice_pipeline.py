from pathlib import Path

from switchsafe.agentbench import Evidence
from switchsafe.transcribe import Transcript
from switchsafe.voice_pipeline import VoiceAgentPipeline


class FakeTranscriber:
    def __init__(self, text="Device HP-42 shows E7", confidence=-0.2):
        self.result = Transcript(text, 2.0, confidence, 0.01)

    def transcribe(self, audio_path: Path) -> Transcript:
        return self.result


class FakePlanner:
    def __init__(self, steps=None):
        self.steps = steps or ["retrieve_guidance", "check_device_status"]

    def plan(self, query: str) -> dict:
        return {"steps": self.steps, "schema_valid": True, "latency_ms": 1.0}


CORPUS = [Evidence("manual-e7", "Error E7 requires diagnostics.")]


def test_integrated_pipeline_passes_transcript_to_agents():
    result = VoiceAgentPipeline(FakeTranscriber(), FakePlanner(), CORPUS).run(Path("sample.wav"))
    assert result["status"] == "completed"
    assert result["transcript"] == "Device HP-42 shows E7"
    assert result["tool_calls"][0]["tool"] == "get_device_status"
    assert result["trace"][-1] == {"agent": "verifier", "grounded": True}


def test_asr_gate_blocks_low_confidence_audio_before_planning():
    result = VoiceAgentPipeline(FakeTranscriber(confidence=-1.2), FakePlanner(), CORPUS).run(
        Path("sample.wav")
    )
    assert result["status"] == "retry_stronger_model"
    assert len(result["trace"]) == 1


def test_mutating_plan_requires_confirmation():
    planner = FakePlanner(["retrieve_guidance", "check_device_status", "create_service_request"])
    result = VoiceAgentPipeline(FakeTranscriber(), planner, CORPUS).run(Path("sample.wav"))
    assert result["status"] == "awaiting_confirmation"
