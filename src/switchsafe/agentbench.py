from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import hashlib
import math
import re
import time
from typing import Callable


class Strategy(StrEnum):
    DIRECT = "direct"
    PLAN_EXECUTE = "plan_execute"
    VERIFIED = "plan_execute_verified"


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    text: str


@dataclass
class AgentState:
    query: str
    plan: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    answer: str = ""
    warnings: list[str] = field(default_factory=list)
    status: str = "running"


TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def _hash_vector(text: str, dimensions: int = 128) -> list[float]:
    vector = [0.0] * dimensions
    for token in _tokens(text):
        digest = hashlib.sha256(token.encode()).digest()
        vector[int.from_bytes(digest[:2], "big") % dimensions] += 1.0
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def hybrid_retrieve(query: str, corpus: list[Evidence], top_k: int = 3) -> list[Evidence]:
    """Fuse exact-token and deterministic vector rankings using reciprocal rank fusion."""
    query_tokens = set(_tokens(query))
    lexical = sorted(corpus, key=lambda item: len(query_tokens & set(_tokens(item.text))), reverse=True)
    query_vector = _hash_vector(query)
    semantic = sorted(
        corpus,
        key=lambda item: sum(a * b for a, b in zip(query_vector, _hash_vector(item.text))),
        reverse=True,
    )
    scores: dict[str, float] = {}
    by_id = {item.evidence_id: item for item in corpus}
    for ranking in (lexical, semantic):
        for rank, item in enumerate(ranking, start=1):
            scores[item.evidence_id] = scores.get(item.evidence_id, 0.0) + 1 / (60 + rank)
    ordered = sorted(scores, key=scores.get, reverse=True)
    return [by_id[item_id] for item_id in ordered[:top_k]]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, tuple[Callable[[dict], str], bool]] = {}

    def register(self, name: str, function: Callable[[dict], str], mutating: bool = False) -> None:
        self._tools[name] = (function, mutating)

    def call(self, name: str, arguments: dict, approved: bool = False) -> str:
        if name not in self._tools:
            raise ValueError(f"tool not allowlisted: {name}")
        function, mutating = self._tools[name]
        if mutating and not approved:
            raise PermissionError(f"confirmation required for {name}")
        return function(arguments)


def default_tools() -> ToolRegistry:
    tools = ToolRegistry()
    tools.register("get_device_status", lambda args: f"{args['device_id']}: diagnostic required")
    tools.register("create_service_request", lambda args: f"created:{args['device_id']}", mutating=True)
    return tools


def run_agent(
    query: str,
    corpus: list[Evidence],
    strategy: Strategy = Strategy.VERIFIED,
    approved: bool = False,
    tools: ToolRegistry | None = None,
) -> tuple[AgentState, float]:
    """Execute an inspectable plan-and-execute trajectory with bounded tools."""
    started = time.perf_counter()
    state = AgentState(query=query)
    registry = tools or default_tools()
    state.evidence = hybrid_retrieve(query, corpus)
    wants_service = "service" in query.lower() or "appointment" in query.lower()
    device_match = re.search(r"\b(?:device\s*)?([A-Z]{1,3}-?\d{2,})\b", query, re.IGNORECASE)
    device_id = device_match.group(1).upper() if device_match else None
    state.plan = ["retrieve supporting guidance", "check device status"]
    if wants_service:
        state.plan.append("request confirmation and create service request")
    if strategy == Strategy.DIRECT:
        state.plan = ["respond directly"]
        if not device_id:
            state.status, state.answer = "needs_clarification", "Please provide the device identifier."
        else:
            state.status, state.answer = "completed", f"Device {device_id} requires diagnosis."
        return state, (time.perf_counter() - started) * 1000
    if not state.evidence:
        state.status, state.answer = "abstained", "I could not find supporting guidance."
    elif not device_id:
        state.status, state.answer = "needs_clarification", "Please provide the device identifier."
    else:
        call = {"tool": "get_device_status", "arguments": {"device_id": device_id}}
        state.tool_calls.append(call)
        state.observations.append(registry.call(call["tool"], call["arguments"]))
        if wants_service:
            call = {"tool": "create_service_request", "arguments": {"device_id": device_id}}
            state.tool_calls.append(call)
            try:
                state.observations.append(registry.call(call["tool"], call["arguments"], approved))
                state.status = "completed"
            except PermissionError as error:
                state.warnings.append(str(error))
                state.status = "awaiting_confirmation"
        else:
            state.status = "completed"
        citations = ", ".join(item.evidence_id for item in state.evidence)
        state.answer = f"Device {device_id} requires diagnosis. Evidence: {citations}."
    if strategy == Strategy.VERIFIED and "Evidence:" not in state.answer and state.status == "completed":
        state.status = "abstained"
        state.warnings.append("verification rejected an unsupported answer")
    return state, (time.perf_counter() - started) * 1000


def evaluate(scenarios: list[dict], corpus: list[Evidence], strategy: Strategy) -> dict:
    rows = []
    for scenario in scenarios:
        state, latency = run_agent(
            scenario["query"], corpus, strategy, bool(scenario.get("approved", False))
        )
        tools_used = [call["tool"] for call in state.tool_calls]
        expected_tools = scenario.get("expected_tools", [])
        task_success = state.status == scenario["expected_status"] and tools_used == expected_tools
        unauthorized = "create_service_request" in tools_used and not scenario.get("approved", False)
        rows.append({
            "id": scenario["id"], "task_success": task_success,
            "status": state.status, "tools": tools_used,
            "unauthorized_action": unauthorized and state.status == "completed",
            "grounded": "Evidence:" in state.answer or state.status != "completed",
            "latency_ms": latency,
        })
    latencies = sorted(row["latency_ms"] for row in rows)
    percentile = lambda p: latencies[min(len(latencies) - 1, math.ceil(p * len(latencies)) - 1)]
    return {
        "strategy": strategy,
        "scenarios": len(rows),
        "task_success_rate": sum(row["task_success"] for row in rows) / len(rows),
        "grounded_rate": sum(row["grounded"] for row in rows) / len(rows),
        "unauthorized_action_rate": sum(row["unauthorized_action"] for row in rows) / len(rows),
        "p50_latency_ms": percentile(0.50), "p95_latency_ms": percentile(0.95),
        "rows": rows,
    }
