from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Transcript:
    text: str
    duration_seconds: float
    mean_log_probability: float
    no_speech_probability: float


class FasterWhisperTranscriber:
    def __init__(self, model_name: str = "tiny.en", device: str = "cpu") -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("Install the project dependencies before benchmarking") from exc
        compute_type = "int8" if device == "cpu" else "float16"
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)

    def transcribe(self, audio_path: Path) -> Transcript:
        segments, info = self.model.transcribe(
            str(audio_path),
            language="en",
            beam_size=5,
            vad_filter=True,
        )
        materialized = list(segments)
        text = " ".join(segment.text.strip() for segment in materialized).strip()
        mean_log_probability = (
            sum(segment.avg_logprob for segment in materialized) / len(materialized)
            if materialized
            else -10.0
        )
        no_speech_probability = (
            max(segment.no_speech_prob for segment in materialized) if materialized else 1.0
        )
        return Transcript(text, info.duration, mean_log_probability, no_speech_probability)

