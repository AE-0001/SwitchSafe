"""Checkpointed corpus inference; no accuracy claims without reference labels."""
from __future__ import annotations

import csv
import json
import platform
import statistics
import time
from pathlib import Path

from switchsafe.benchmark import percentile
from switchsafe.policy import Decision, RiskSignals, decide
from switchsafe.transcribe import FasterWhisperTranscriber


def summarize(rows: list[dict]) -> dict:
    valid = [row for row in rows if row["status"] == "ok"]
    latencies = [row["latency_seconds"] for row in valid]
    return {
        "attempted": len(rows), "samples": len(valid), "failed": len(rows) - len(valid),
        "p50_latency_seconds": percentile(latencies, .5) if valid else None,
        "p95_latency_seconds": percentile(latencies, .95) if valid else None,
        "mean_real_time_factor": statistics.fmean(row["real_time_factor"] for row in valid) if valid else None,
        "decision_counts": {str(d): sum(row["decision"] == d for row in rows) for d in Decision},
        "automated_acceptance_coverage": sum(row["decision"] == Decision.ACCEPT for row in rows) / len(rows) if rows else 0,
    }


def save(output: Path, rows: list[dict], model: str, total: int, load_seconds: float) -> dict:
    summary = {
        "mode": "unlabelled_real_audio", "model": model, "device": "cpu", "compute_type": "int8",
        "python": platform.python_version(), "platform": platform.platform(),
        "discovered_recordings": total, "model_load_seconds": load_seconds,
        "latency_scope": "per-file inference, excludes model loading, includes first-file warmup",
        **summarize(rows),
        "sessions": {s: summarize([r for r in rows if r["session_id"] == s]) for s in sorted({r["session_id"] for r in rows})},
        "accuracy_metrics": "unavailable without independent reference transcripts",
        "safety_metrics": "unsafe acceptance unavailable without human safe/unsafe labels",
        "warning": "Thresholds are uncalibrated. Accept is not proof of correctness. No downstream actions are executed.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps({"summary": summary, "samples": rows}, indent=2), encoding="utf-8")
    temporary.replace(output)
    with output.with_suffix(".review.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "audio_path", "speaker_id", "session_id", "hypothesis", "decision", "reference", "is_safe", "notes"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in writer.fieldnames})
    return summary


def run_directory(audio_dir: Path, output: Path, model: str, limit: int | None = None,
                  transcriber=None) -> dict:
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    files = sorted(p for p in audio_dir.rglob("*") if p.suffix.lower() == ".wav")
    total = len(files)
    files = files[:limit] if limit is not None else files
    if not files:
        raise ValueError(f"No WAV files under {audio_dir}")
    started = time.perf_counter()
    transcriber = transcriber or FasterWhisperTranscriber(model)
    load_seconds = time.perf_counter() - started
    rows = []
    for index, path in enumerate(files, 1):
        row = {"sample_id": str(path.relative_to(audio_dir)), "audio_path": str(path.resolve()),
               "speaker_id": path.parent.parent.name, "session_id": path.parent.name}
        try:
            started = time.perf_counter()
            result = transcriber.transcribe(path)
            latency = time.perf_counter() - started
            if result.duration_seconds <= 0:
                raise ValueError("audio duration must be positive")
            decision, reason = decide(RiskSignals(result.mean_log_probability, result.no_speech_probability))
            if not result.text.strip():
                decision, reason = Decision.ESCALATE, "empty transcript"
            row.update({"status": "ok", "hypothesis": result.text, "duration_seconds": result.duration_seconds,
                        "latency_seconds": latency, "real_time_factor": latency / result.duration_seconds,
                        "mean_log_probability": result.mean_log_probability,
                        "no_speech_probability": result.no_speech_probability,
                        "decision": decision, "decision_reason": reason})
        except Exception as error:
            row.update({"status": "error", "error": str(error), "decision": Decision.ESCALATE,
                        "decision_reason": "transcription failed"})
        rows.append(row)
        if index % 25 == 0 or index == len(files):
            save(output, rows, model, total, load_seconds)
            print(f"Processed {index}/{len(files)} WAVs", flush=True)
    return save(output, rows, model, total, load_seconds)
