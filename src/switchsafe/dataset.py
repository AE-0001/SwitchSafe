from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Sample:
    sample_id: str
    audio_path: Path
    reference: str
    speaker_id: str
    condition: str = "clean"
    critical_action: bool = False
    critical_entity_confidence: float | None = None
    is_safe: bool | None = None


def parse_bool(value: str | None) -> bool | None:
    if value is None or not value.strip():
        return None
    normalized = value.strip().lower()
    if normalized in ("true", "1", "yes"):
        return True
    if normalized in ("false", "0", "no"):
        return False
    raise ValueError(f"Expected boolean label, got {value!r}")


def read_manifest(path: Path) -> list[Sample]:
    samples: list[Sample] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if not row.get("reference", "").strip():
                raise ValueError("Labelled manifests require independent non-empty references")
            audio_path = Path(row["audio_path"])
            if not audio_path.is_absolute():
                audio_path = (path.parent / audio_path).resolve()
            samples.append(
                Sample(
                    sample_id=row["sample_id"],
                    audio_path=audio_path,
                    reference=row["reference"],
                    speaker_id=row.get("speaker_id", "unknown"),
                    condition=row.get("condition", "clean"),
                    critical_action=parse_bool(row.get("critical_action")) or False,
                    critical_entity_confidence=(float(row["critical_entity_confidence"]) if row.get("critical_entity_confidence") else None),
                    is_safe=parse_bool(row.get("is_safe")),
                )
            )
    return samples
