from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import time
from typing import Protocol

from switchsafe.policy import Decision, RiskSignals, decide
from switchsafe.task_store import TaskTools
from switchsafe.transcribe import Transcript


class Transcriber(Protocol):
    def transcribe(self, audio_path: Path) -> Transcript: ...


class ActionPlanner(Protocol):
    def task_actions(self, query: str) -> dict: ...


@dataclass
class TaskAgentState:
    input_source: str
    transcript: str = ""
    asr_decision: str = "not_applicable"
    actions: list[dict] = field(default_factory=list)
    results: list[dict] = field(default_factory=list)
    status: str = "running"
    answer: str = ""
    warnings: list[str] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)


class VoiceTaskPipeline:
    def __init__(self, planner: ActionPlanner, tools: TaskTools,
                 transcriber: Transcriber | None = None):
        self.planner, self.tools, self.transcriber = planner, tools, transcriber

    def run_text(self, text: str, approved: bool = False) -> dict:
        state = TaskAgentState("text", transcript=text, asr_decision="bypassed_for_text")
        return self._continue(state, approved)

    def run_audio(self, audio_path: Path, approved: bool = False) -> dict:
        if self.transcriber is None:
            raise ValueError("an audio transcriber is required")
        started = time.perf_counter()
        state = TaskAgentState(str(audio_path))
        transcript = self.transcriber.transcribe(audio_path)
        decision, reason = decide(RiskSignals(transcript.mean_log_probability,
                                              transcript.no_speech_probability))
        if not transcript.text.strip():
            decision, reason = Decision.ESCALATE, "empty transcript"
        state.transcript, state.asr_decision = transcript.text, decision
        state.trace.append({"agent": "speech_gate", "decision": decision, "reason": reason})
        if decision != Decision.ACCEPT:
            state.status = decision
            state.answer = "Audio was blocked before planning."
            result = asdict(state)
            result["pipeline_latency_ms"] = (time.perf_counter() - started) * 1000
            return result
        return self._continue(state, approved, started)

    def _continue(self, state: TaskAgentState, approved: bool,
                  started: float | None = None) -> dict:
        started = started or time.perf_counter()
        plan = self.planner.task_actions(state.transcript)
        state.actions = plan.get("actions", [])
        state.trace.append({"agent": "planner", "actions": state.actions,
                            "schema_valid": plan.get("schema_valid"),
                            "latency_ms": plan.get("latency_ms")})
        if not plan.get("schema_valid"):
            state.status, state.answer = "human_escalation", "The planner returned invalid output."
        elif not state.actions:
            state.status, state.answer = "unsupported_action", (
                "I could not map that request to a supported task or note action."
            )
        else:
            for action in state.actions:
                try:
                    output = self.tools.execute(action["tool"], action.get("arguments", {}), approved)
                    state.results.append({"tool": action["tool"], "output": output})
                except PermissionError as error:
                    state.status = "awaiting_confirmation"
                    state.warnings.append(str(error))
                    break
                except (KeyError, TypeError, ValueError, OSError) as error:
                    state.status = "human_escalation"
                    state.warnings.append(f"tool failure: {error}")
                    break
            if state.status == "running":
                state.status = "completed"
            state.answer = self._verify(state)
        state.trace.append({"agent": "executor_verifier", "status": state.status,
                            "executed": [row["tool"] for row in state.results]})
        result = asdict(state)
        result["pipeline_latency_ms"] = (time.perf_counter() - started) * 1000
        return result

    @staticmethod
    def _verify(state: TaskAgentState) -> str:
        if state.status == "awaiting_confirmation":
            return "Confirmation is required before changing local data."
        if state.status != "completed":
            return "The requested action was not completed."
        return f"Completed {len(state.results)} verified action(s): " + ", ".join(
            row["tool"] for row in state.results
        )
