# CPU benchmark: 682 recordings

Measured on a local Windows laptop using Python 3.11.9, Faster-Whisper `tiny.en`,
CPU inference and INT8 computation. One IMDA Singapore-English speaker, two sessions:
353 recordings in SESSION0 and 329 in SESSION1.

| Measurement | Result |
| --- | ---: |
| Attempted / completed / failed | 682 / 682 / 0 |
| p50 inference latency | 0.499 seconds |
| p95 inference latency | 0.729 seconds |
| Mean real-time factor | 0.114 |
| Separate model-load time | 3.394 seconds |
| Accept / retry / confirm / escalate | 677 / 3 / 0 / 2 |

Latency excludes model loading and includes first-file warmup. Results depend on
hardware, clip duration and model settings; this is not a throughput guarantee.
Mean real-time factor is inference duration divided by audio duration, averaged
over clips. The acceptance count is a policy outcome, not transcription accuracy.

No independent reference transcripts or human safety labels were available for
this run, so WER, CER and unsafe-acceptance reduction are not reported. The code
supports reference-based evaluation separately. One speaker does not demonstrate
accent or speaker generalisation. Raw audio, hypotheses and private local paths
are intentionally not published.
