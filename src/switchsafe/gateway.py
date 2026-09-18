from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from switchsafe.benchmark import percentile
from switchsafe.policy import RiskSignals, decide
from switchsafe.transcribe import FasterWhisperTranscriber


def run_directory(audio_dir: Path, output: Path, model: str, limit: int | None = None) -> dict:
    """Transcribe real audio without claiming accuracy metrics when references are absent."""
    files = sorted(path for path in audio_dir.rglob("*") if path.suffix.lower() == ".wav")
    if limit is not None:
        files = files[:limit]
    if not files:
        raise ValueError(f"No WAV files found under {audio_dir}")
    transcriber = FasterWhisperTranscriber(model)
    rows = []
    for audio_path in files:
        started = time.perf_counter()
        result = transcriber.transcribe(audio_path)
        latency = time.perf_counter() - started
        decision, reason = decide(RiskSignals(result.mean_log_probability, result.no_speech_probability))
        rows.append({
            "sample_id": audio_path.stem,
            "audio_path": str(audio_path),
            "hypothesis": result.text,
            "duration_seconds": result.duration_seconds,
            "latency_seconds": latency,
            "real_time_factor": latency / result.duration_seconds if result.duration_seconds else 0.0,
            "mean_log_probability": result.mean_log_probability,
            "no_speech_probability": result.no_speech_probability,
            "decision": decision,
            "decision_reason": reason,
        })
    latencies = [row["latency_seconds"] for row in rows]
    summary = {
        "mode": "unlabelled_real_audio",
        "model": model,
        "samples": len(rows),
        "p50_latency_seconds": percentile(latencies, 0.50),
        "p95_latency_seconds": percentile(latencies, 0.95),
        "mean_real_time_factor": statistics.fmean(row["real_time_factor"] for row in rows),
        "decision_counts": {
            decision: sum(row["decision"] == decision for row in rows)
            for decision in sorted({row["decision"] for row in rows})
        },
        "accuracy_metrics": "unavailable: archive did not include reference transcripts",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": summary, "samples": rows}, indent=2), encoding="utf-8")
    return summary
