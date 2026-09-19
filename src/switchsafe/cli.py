from __future__ import annotations

import argparse
import json
from pathlib import Path

from switchsafe.agentbench import Evidence, Strategy, evaluate
from switchsafe.benchmark import run
from switchsafe.corpus import run_directory
from switchsafe.policy import RiskSignals, decide, policy_metrics


def demo() -> dict:
    """Deterministic policy demonstration, not a corpus benchmark."""
    scenarios = [
        ("clear request", RiskSignals(-0.20, 0.01), True),
        ("uncertain critical account number", RiskSignals(-0.25, 0.02, 0.42, True), False),
        ("low-confidence accented/noisy speech", RiskSignals(-1.10, 0.04), False),
        ("silence or non-speech", RiskSignals(-2.00, 0.91), False),
    ]
    rows = []
    for name, signals, is_safe in scenarios:
        decision, reason = decide(signals)
        rows.append({"scenario": name, "decision": decision, "reason": reason, "is_safe": is_safe})
    return {"mode": "deterministic_policy_demo", "scenarios": rows, "metrics": policy_metrics(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark failure-aware ASR")
    parser.add_argument("manifest", type=Path, nargs="?")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--agent-eval", type=Path, help="evaluate agent trajectories from JSON")
    parser.add_argument("--audio-dir", type=Path, help="transcribe unlabelled WAV files")
    parser.add_argument("--output", type=Path, default=Path("data/derived/benchmark.json"))
    parser.add_argument("--model", default="tiny.en")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.demo:
        print(json.dumps(demo(), indent=2))
        return
    if args.agent_eval:
        scenarios = json.loads(args.agent_eval.read_text(encoding="utf-8"))
        corpus = [
            Evidence("manual-e7", "Error E7 requires diagnostics before arranging service."),
            Evidence("safety", "Service creation is a mutating action that requires confirmation."),
            Evidence("fallback", "Escalate when guidance or required identifiers are missing."),
        ]
        results = [evaluate(scenarios, corpus, strategy) for strategy in Strategy]
        print(json.dumps({"mode": "agent_strategy_evaluation", "results": results}, indent=2))
        return
    if args.audio_dir:
        print(json.dumps(run_directory(args.audio_dir, args.output, args.model, args.limit), indent=2))
        return
    if args.manifest is None:
        parser.error("manifest is required unless --demo is used")
    print(json.dumps(run(args.manifest, args.output, args.model, args.limit), indent=2))


if __name__ == "__main__":
    main()
