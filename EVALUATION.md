# Evaluation contract

## Available data

One IMDA speaker, two sessions: SESSION0 has 353 WAV recordings and SESSION1 has 329.
Two sessions of the same speaker are not independent speaker-level evaluation groups.
The audio files are excluded from source control.

## Performance run

The full-corpus command evaluates CPU INT8 Faster-Whisper tiny.en. It reports per-file
latency, nearest-rank-style p50/p95 latency, arithmetic mean real-time factor, routing
counts, failures and session breakdowns. Model construction is timed separately. File
latency includes decoding, VAD and materialization of the lazy transcription generator.
The first-file warmup is included. These numbers depend on hardware and background load.
There is no model training in this project: pretrained Whisper weights are used for inference.

## Accuracy run

Use a manifest with independent reference text to calculate WER/CER. Text normalization
lowercases ASCII English words and numbers and removes punctuation; this is an explicit
English-only baseline and can lose distinctions in code-switched text. Mean per-utterance
WER and corpus WER answer different questions; the report includes both. Never use the
model's own hypothesis as the reference. Review and correct references by listening to audio.

## Policy validation

Unit tests use constructed risk signals and fake transcribers to exercise all four branches;
they are not evidence of real-world safety. Mean log probability is not a calibrated
probability of transcript correctness. Critical-entity confidence must be supplied by an
upstream verifier. Critical requests without that confidence require confirmation.
Confirmation, retry and human escalation are instructions for a caller, not automatically
executed integrations. This application executes no consequential downstream action.

For safety evaluation, label whether the candidate transcript is safe to use for its intended
action; this is model-output- and action-dependent, not an intrinsic property of a WAV.
Human-reviewed labels are required. Report unsafe accepted / unsafe labelled, and accepted /
all labelled, together with counts and the annotation rubric. With no unsafe labels the
unsafe-acceptance rate is undefined, not zero. Calibrate thresholds on a separate partition
and evaluate on held-out speakers when more speakers are available. The current one-speaker
corpus cannot validate population-level accent robustness.
