from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
import time
from typing import Protocol

from switchsafe.agentbench import Evidence, ToolRegistry, default_tools, hybrid_retrieve
from switchsafe.policy import Decision, RiskSignals, decide
from switchsafe.transcribe import Transcript


class Transcriber(Protocol):
    def transcribe(self, audio_path: Path) -> Transcript: ...


class Planner(Protocol):
    def plan(self, query: str) -> dict: ...


@dataclass
class VoiceAgentState:
    audio_path: str
    transcript: str = ""
    asr_decision: str = ""
    asr_reason: str = ""
    plan: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    answer: str = ""
    status: str = "running"
    warnings: list[str] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)


class SpeechGateAgent:
    def __init__(self, transcriber: Transcriber):
        self.transcriber = transcriber

    def run(self, state: VoiceAgentState, audio_path: Path) -> Transcript:
        result = self.transcriber.transcribe(audio_path)
        decision, reason = decide(RiskSignals(result.mean_log_probability,
                                              result.no_speech_probability))
        if not result.text.strip():
            decision, reason = Decision.ESCALATE, "empty transcript"
        state.transcript, state.asr_decision, state.asr_reason = result.text, decision, reason
        state.trace.append({"agent": "speech_gate", "decision": decision, "reason": reason})
        return result


class PlannerAgent:
    def __init__(self, planner: Planner):
        self.planner = planner

    def run(self, state: VoiceAgentState) -> None:
        result = self.planner.plan(state.transcript)
        state.plan = result.get("steps", [])
        state.trace.append({"agent": "planner", "steps": state.plan,
                            "latency_ms": result.get("latency_ms")})
        if not result.get("schema_valid", True):
            state.status = "human_escalation"
            state.warnings.append("planner returned an invalid schema")


class RetrievalAgent:
    def __init__(self, corpus: list[Evidence]):
        self.corpus = corpus

    def run(self, state: VoiceAgentState) -> None:
        if "retrieve_guidance" not in state.plan:
            state.trace.append({"agent": "retrieval", "skipped": True})
            return
        hits = hybrid_retrieve(state.transcript, self.corpus)
        state.evidence = [hit.evidence_id for hit in hits]
        state.trace.append({"agent": "retrieval", "evidence": state.evidence})


class ExecutionAgent:
    def __init__(self, tools: ToolRegistry):
        self.tools = tools

    def run(self, state: VoiceAgentState, approved: bool) -> None:
        match = re.search(r"\b([A-Z]{1,3}-\d{2,})\b", state.transcript, re.IGNORECASE)
        device_id = match.group(1).upper() if match else None
        if "ask_clarification" in state.plan or not device_id:
            state.status = "needs_clarification"
            state.answer = "Please confirm the device identifier before I continue."
            state.trace.append({"agent": "executor", "status": state.status})
            return
        for step in state.plan:
            if step == "check_device_status":
                call = {"tool": "get_device_status", "arguments": {"device_id": device_id}}
                state.tool_calls.append(call)
                try:
                    state.observations.append(self.tools.call(call["tool"], call["arguments"]))
                except (OSError, TimeoutError, ValueError) as error:
                    state.status = "human_escalation"
                    state.warnings.append(f"tool failure: {error}")
                    break
            elif step == "create_service_request":
                call = {"tool": step, "arguments": {"device_id": device_id}}
                state.tool_calls.append(call)
                try:
                    state.observations.append(self.tools.call(step, call["arguments"], approved))
                except PermissionError as error:
                    state.status = "awaiting_confirmation"
                    state.warnings.append(str(error))
                    break
        if state.status == "running":
            state.status = "completed"
        state.trace.append({"agent": "executor", "status": state.status,
                            "tool_calls": state.tool_calls})


class VerificationAgent:
    def run(self, state: VoiceAgentState) -> None:
        if state.status != "completed":
            state.trace.append({"agent": "verifier", "skipped": True})
            return
        if not state.evidence:
            state.status = "human_escalation"
            state.warnings.append("verification rejected an unsupported response")
            state.trace.append({"agent": "verifier", "grounded": False})
            return
        observations = "; ".join(state.observations) or "No tool observation"
        state.answer = f"{observations}. Evidence: {', '.join(state.evidence)}."
        state.trace.append({"agent": "verifier", "grounded": True})


class VoiceAgentPipeline:
    """Integrated speech-to-plan-to-tool workflow with explicit agent boundaries."""

    def __init__(self, transcriber: Transcriber, planner: Planner, corpus: list[Evidence],
                 tools: ToolRegistry | None = None):
        self.speech = SpeechGateAgent(transcriber)
        self.planner = PlannerAgent(planner)
        self.retrieval = RetrievalAgent(corpus)
        self.executor = ExecutionAgent(tools or default_tools())
        self.verifier = VerificationAgent()

    def run(self, audio_path: Path, approved: bool = False) -> dict:
        started = time.perf_counter()
        state = VoiceAgentState(str(audio_path))
        transcript = self.speech.run(state, audio_path)
        if state.asr_decision != Decision.ACCEPT:
            state.status = state.asr_decision
            state.answer = "The audio was not passed to the agent workflow."
        else:
            self.planner.run(state)
            if state.status == "running":
                self.retrieval.run(state)
                self.executor.run(state, approved)
                self.verifier.run(state)
        result = asdict(state)
        result["audio_duration_seconds"] = transcript.duration_seconds
        result["pipeline_latency_ms"] = (time.perf_counter() - started) * 1000
        return result
