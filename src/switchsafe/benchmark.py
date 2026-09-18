from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from switchsafe.dataset import read_manifest
from switchsafe.metrics import character_error_rate, word_error_rate
from switchsafe.policy import Decision, RiskSignals, decide, policy_metrics
from switchsafe.transcribe import FasterWhisperTranscriber


def percentile(values: list[float], percentile_value: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * percentile_value))
    return ordered[index]


def run(manifest: Path, output: Path, model: str, limit: int | None = None) -> dict:
    samples = read_manifest(manifest)
    if limit is not None:
        samples = samples[:limit]
    transcriber = FasterWhisperTranscriber(model)
    rows: list[dict] = []

    for sample in samples:
        started = time.perf_counter()
        result = transcriber.transcribe(sample.audio_path)
        latency = time.perf_counter() - started
        wer = word_error_rate(sample.reference, result.text)
        decision, reason = decide(
            RiskSignals(result.mean_log_probability, result.no_speech_probability,
                        sample.critical_entity_confidence, sample.critical_action)
        )
        if not result.text.strip():
            decision, reason = Decision.ESCALATE, "empty transcript"
        rows.append(
            {
                "sample_id": sample.sample_id,
                "speaker_id": sample.speaker_id,
                "condition": sample.condition,
                "audio_path": str(sample.audio_path),
                "reference": sample.reference,
                "hypothesis": result.text,
                **wer,
                "cer": character_error_rate(sample.reference, result.text),
                "audio_seconds": result.duration_seconds,
                "latency_seconds": latency,
                "real_time_factor": latency / result.duration_seconds if result.duration_seconds else 0.0,
                "mean_log_probability": result.mean_log_probability,
                "no_speech_probability": result.no_speech_probability,
                "decision": decision,
                "decision_reason": reason,
            }
        )
        if sample.is_safe is not None:
            rows[-1]["is_safe"] = sample.is_safe

    if not rows:
        raise ValueError("Manifest did not contain any samples")
    latencies = [row["latency_seconds"] for row in rows]
    summary = {
        "model": model,
        "samples": len(rows),
        "speakers": len({row["speaker_id"] for row in rows}),
        "mean_wer": statistics.fmean(row["wer"] for row in rows),
        "corpus_wer": sum(row["substitutions"] + row["deletions"] + row["insertions"] for row in rows) / sum(row["reference_words"] for row in rows),
        "mean_cer": statistics.fmean(row["cer"] for row in rows),
        "p50_latency_seconds": percentile(latencies, 0.50),
        "p95_latency_seconds": percentile(latencies, 0.95),
        "mean_real_time_factor": statistics.fmean(row["real_time_factor"] for row in rows),
        "decision_counts": {
            decision: sum(row["decision"] == decision for row in rows)
            for decision in sorted({row["decision"] for row in rows})
        },
        "policy_metrics": policy_metrics(rows),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": summary, "samples": rows}, indent=2), encoding="utf-8")
    return summary
