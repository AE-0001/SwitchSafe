from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math


class Decision(StrEnum):
    ACCEPT = "accept"
    CONFIRM = "request_confirmation"
    RETRY = "retry_stronger_model"
    ESCALATE = "human_escalation"


@dataclass(frozen=True)
class RiskSignals:
    mean_log_probability: float
    no_speech_probability: float
    critical_entity_confidence: float | None = None
    critical_action: bool = False


def decide(signals: RiskSignals) -> tuple[Decision, str]:
    """Conservative, inspectable routing policy for a downstream voice agent."""
    if not math.isfinite(signals.mean_log_probability) or not math.isfinite(signals.no_speech_probability):
        return Decision.ESCALATE, "invalid confidence signal"
    if not 0 <= signals.no_speech_probability <= 1:
        return Decision.ESCALATE, "invalid no-speech probability"
    if signals.no_speech_probability >= 0.70:
        return Decision.ESCALATE, "audio is probably silence or non-speech"
    if signals.critical_action:
        confidence = signals.critical_entity_confidence
        if confidence is None or not math.isfinite(confidence) or not 0.65 <= confidence <= 1:
            return Decision.CONFIRM, "critical entity is uncertain"
    if signals.mean_log_probability < -0.85:
        return Decision.RETRY, "transcription confidence is low"
    return Decision.ACCEPT, "risk checks passed"


def policy_metrics(rows: list[dict]) -> dict[str, float | int | None]:
    """Evaluate routing quality when benchmark rows include an `is_safe` label."""
    labelled = [row for row in rows if isinstance(row.get("is_safe"), bool)]
    if not labelled:
        return {"labelled_samples": 0}
    accepted = [row for row in labelled if row["decision"] == Decision.ACCEPT]
    unsafe = [row for row in labelled if not row["is_safe"]]
    unsafe_accepted = [row for row in accepted if not row["is_safe"]]
    return {
        "labelled_samples": len(labelled),
        "automated_processing_coverage": len(accepted) / len(labelled),
        "unsafe_transcript_acceptance_rate": len(unsafe_accepted) / len(unsafe) if unsafe else None,
        "unsafe_accepted": len(unsafe_accepted),
        "unsafe_samples": len(unsafe),
    }
