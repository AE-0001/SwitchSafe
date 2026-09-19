from __future__ import annotations

import json
import time
from urllib.request import Request, urlopen


class OllamaPlanner:
    """Local model adapter using Ollama's documented HTTP generation API."""

    def __init__(self, model: str = "qwen3:4b", base_url: str = "http://127.0.0.1:11434"):
        self.model, self.base_url = model, base_url.rstrip("/")

    def plan(self, query: str) -> dict:
        prompt = (
            "Return JSON only with a 'steps' array. Allowed steps are "
            "retrieve_guidance, check_device_status, create_service_request, ask_clarification. "
            "Never create a service request without confirmation. Request: " + query
        )
        body = json.dumps({"model": self.model, "prompt": prompt, "stream": False,
                           "think": False, "format": "json",
                           "options": {"temperature": 0}}).encode()
        started = time.perf_counter()
        request = Request(f"{self.base_url}/api/generate", data=body,
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=120) as response:
            payload = json.load(response)
        parsed = json.loads(payload["response"])
        raw_steps = parsed.get("steps", [])
        steps = [item.get("step") if isinstance(item, dict) else item for item in raw_steps]
        allowed = {"retrieve_guidance", "check_device_status", "create_service_request",
                   "ask_clarification"}
        schema_valid = isinstance(raw_steps, list) and all(
            isinstance(step, str) and step in allowed for step in steps
        )
        return {
            "steps": steps,
            "schema_valid": schema_valid,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "prompt_tokens": payload.get("prompt_eval_count"),
            "completion_tokens": payload.get("eval_count"),
        }

    def task_actions(self, query: str) -> dict:
        """Translate natural language into allowlisted local task actions."""
        prompt = """Return JSON only as {"actions":[{"tool":"...","arguments":{...}}]}.
Allowed tools and arguments:
- create_task: title string
- list_tasks: no arguments
- complete_task: task_id integer
- save_note: text string
- search_notes: query string
Use only these tools. If unsupported or required information is missing, return an empty actions list.
Use the exact tool names above. Examples:
"Add submit application to my tasks" -> {"actions":[{"tool":"create_task","arguments":{"title":"submit application"}}]}
"Show my tasks" -> {"actions":[{"tool":"list_tasks","arguments":{}}]}
"Save a note that my interview is Tuesday" -> {"actions":[{"tool":"save_note","arguments":{"text":"my interview is Tuesday"}}]}
User request: """ + query
        body = json.dumps({"model": self.model, "prompt": prompt, "stream": False,
                           "think": False, "format": "json",
                           "options": {"temperature": 0}}).encode()
        started = time.perf_counter()
        request = Request(f"{self.base_url}/api/generate", data=body,
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=120) as response:
            payload = json.load(response)
        parsed = json.loads(payload["response"])
        actions = parsed.get("actions", [])
        allowed = {"create_task", "list_tasks", "complete_task", "save_note", "search_notes"}
        valid = isinstance(actions, list) and all(
            isinstance(item, dict) and item.get("tool") in allowed
            and isinstance(item.get("arguments", {}), dict) for item in actions
        )
        return {"actions": actions if valid else [], "schema_valid": valid,
                "latency_ms": (time.perf_counter() - started) * 1000,
                "prompt_tokens": payload.get("prompt_eval_count"),
                "completion_tokens": payload.get("eval_count")}


def evaluate_planner(planner: OllamaPlanner, scenarios: list[dict]) -> dict:
    rows = []
    for scenario in scenarios:
        required = ["retrieve_guidance"]
        if scenario["expected_tools"]:
            required.append("check_device_status")
        if "create_service_request" in scenario["expected_tools"]:
            required.append("create_service_request")
        if scenario["expected_status"] == "needs_clarification":
            required.append("ask_clarification")
        try:
            result = planner.plan(scenario["query"])
            steps = result["steps"] if isinstance(result["steps"], list) else []
            recall = len(set(required) & set(steps)) / len(set(required))
            valid = bool(result.get("schema_valid", True))
            rows.append({"id": scenario["id"], "valid": valid, "required_step_recall": recall,
                         **result})
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            rows.append({"id": scenario["id"], "valid": False,
                         "required_step_recall": 0.0, "error": str(error)})
    valid = [row for row in rows if row["valid"]]
    latencies = sorted(row["latency_ms"] for row in valid)
    p95 = latencies[max(0, min(len(latencies) - 1, int(0.95 * len(latencies))))] if valid else None
    return {
        "model": planner.model, "scenarios": len(rows),
        "schema_valid_rate": len(valid) / len(rows),
        "mean_required_step_recall": sum(row["required_step_recall"] for row in rows) / len(rows),
        "p95_latency_ms": p95,
        "prompt_tokens": sum(row.get("prompt_tokens") or 0 for row in valid),
        "completion_tokens": sum(row.get("completion_tokens") or 0 for row in valid),
        "rows": rows,
    }
