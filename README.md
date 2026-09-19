# SwitchSafe

[![CI](https://github.com/AE-0001/SwitchSafe/actions/workflows/ci.yml/badge.svg)](https://github.com/AE-0001/SwitchSafe/actions/workflows/ci.yml)

**A confidence-aware speech gateway for voice agents.** Instead of passing every
transcript straight to an agent, SwitchSafe produces an inspectable accept,
retry, confirm or escalate decision.

Built with Python, Faster-Whisper, CTranslate2 and Pytest. CPU INT8 evaluation on
**682 real IMDA Singapore-English recordings** measured **0.499 s p50**, **0.729 s
p95** and **0.114 mean real-time factor**. See [benchmark evidence](evaluation/CPU_RESULTS.md)
for measurement scope and limitations.

## Try it in minutes

```sh
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
python -m switchsafe.cli --demo
```

The deterministic policy demo needs no audio, API key or downloaded ASR model.
It is a functional demonstration, not a measured safety benchmark.

## How it works

```text
Local audio → Faster-Whisper CPU INT8 → transcript + confidence signals
                                      ↓
                          risk-aware routing policy
                                      ↓
                   accept / retry / confirm / escalate
                                      ↓
                      checkpointed JSON + review report
```

- Fail-closed handling of invalid signals, empty transcripts and corrupt audio.
- Critical-action confirmation when caller-supplied entity confidence is missing.
- Corpus checkpoints every 25 clips, with speaker/session provenance.
- Per-session latency, real-time factor and routing counts.
- Reference-based WER/CER and edit-error evaluation in labelled-manifest mode.
- Searchable HTML reports and CSV exports for independent human review.

## Run real audio

Use an authorised local dataset. Raw audio and transcript outputs are not committed.
The first inference run downloads the selected Whisper model.

```sh
python -m switchsafe.cli --audio-dir data/raw/imda --model tiny.en --output data/derived/run.json
python -m switchsafe.report data/derived/run.json --output data/derived/run.html
```

For a quick pilot, add `--limit 10`. Open the generated HTML report in your browser.
On Windows, `./run_imda.ps1 -AudioDir 'path/to/audio'` runs the same corpus workflow.

For accuracy evaluation, create a CSV manifest with independently transcribed references:

```csv
sample_id,audio_path,reference,speaker_id,condition
clip001,raw/imda/001.wav,independent reference transcript,speaker001,clean
```

```sh
python -m switchsafe.cli data/manifest.csv --model tiny.en --output data/derived/labelled.json
```

Audio paths are relative to the manifest. Optional `critical_action`,
`critical_entity_confidence` and `is_safe` columns support policy evaluation.
Do not use model hypotheses as ground-truth references.

## Evidence and boundaries

The real-audio run covers **one speaker across two sessions**, not 100 speakers.
It measures inference performance and routing, not validated transcript accuracy.
Confidence thresholds are uncalibrated; acceptance does not prove correctness.
Retry, confirmation and escalation are routing outputs, not external integrations.
No downstream agent actions are executed.

- [CPU results](evaluation/CPU_RESULTS.md)
- [Evaluation methodology](EVALUATION.md)
- [AI-assisted build and verification workflow](BUILD_NOTES.md)
- [Automated tests](tests)

Next steps: independent transcript review, held-out-speaker evaluation and
threshold calibration against action-specific failure labels.

## AgentBench extension

SwitchSafe also contains an evaluation-first Plan-and-Execute harness. It combines
hybrid lexical/vector retrieval, an explicit shared state, allowlisted tools,
confirmation gates for mutating actions and a post-execution grounding check.

```sh
python -m switchsafe.cli --agent-eval evaluation/agent_scenarios.json
```

The command compares direct, Plan-and-Execute and verified Plan-and-Execute
strategies over labelled trajectories. It reports exact task success, grounding,
unauthorized-action rate and p50/p95 orchestration latency. Included tools are
deterministic sandbox simulations, making evaluation reproducible without an API
key. A model provider can later replace planning and synthesis without changing
the tool-security or evaluation contracts.

The current benchmark contains 120 deterministic labelled trajectories spanning
diagnostics, paraphrases, missing identifiers, confirmation gates and injected
tool timeouts. A real local Qwen3 4B planner is evaluated separately through
Ollama so model-schema adherence and planning recall are not conflated with the
deterministic orchestration tests. See [AgentBench results](evaluation/AGENT_RESULTS.md).
