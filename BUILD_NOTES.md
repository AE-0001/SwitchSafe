# Building with AI, verifying with evidence

AI-assisted development supported implementation, debugging and test iteration. The
verification loop is concrete: run deterministic routing tests, exercise failure cases,
then run the actual CPU audio benchmark and inspect its structured outputs.

Recent hardening added checkpointed corpus processing, session-level metrics,
corrupt-audio handling and searchable review reports. The project was not built by
an autonomous agent swarm, and its benchmark is not a production deployment.

## Reviewer quick path

1. Install with `python -m pip install -e '.[dev]'`.
2. Run `python -m pytest -q`.
3. Run `python -m switchsafe.cli --demo` without downloading models.
4. Read `evaluation/CPU_RESULTS.md` for the real-audio benchmark scope.
5. Read `EVALUATION.md` for accuracy and policy-evaluation methodology.

Raw IMDA audio and transcript outputs are excluded from this repository. Supply
your own authorised local copy to reproduce inference. Policy decisions are
inspectable outputs; downstream agent actions are deliberately not executed.

The AgentBench extension begins with deterministic planner and tool adapters. This
isolates orchestration and evaluation correctness before adding a non-deterministic
model provider. It is not presented as frontier-model evaluation until such a
provider has actually been run.
