# AgentBench results

## Deterministic orchestration matrix

Run with:

```sh
python -m switchsafe.cli --agent-matrix
```

The suite generates 120 labelled trajectories from a fixed matrix of diagnostic,
service, paraphrase, missing-identifier, approval and injected-timeout cases.
These are controlled contract tests, not sampled production conversations.

| Strategy | Task success | Grounded | Tool arguments | Unauthorized actions | Tool-failure recovery |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct baseline | 16.7% | 16.7% | 16.7% | 0% | 0% |
| ReAct | 100% | 100% | 100% | 0% | 100% |
| Plan-and-Execute | 100% | 100% | 100% | 0% | 100% |
| Plan-and-Execute + verification | 100% | 100% | 100% | 0% | 100% |

Hybrid retrieval achieved Hit@3 of 73.3% and MRR of 0.633. This deliberately
visible retrieval gap is the next improvement target; orchestration success does
not imply that retrieval is optimal.

## Local-model planner

Run with an installed Ollama model:

```sh
python -m switchsafe.cli --llm-eval 12 --llm-model qwen3:4b
```

On a 12-case subset, local Qwen3 4B produced 100% schema-valid plans and 91.7%
mean required-step recall. Planning p95 was 5.741 seconds, using 707 prompt and
1,117 completion tokens across the run. Temperature was zero and the thinking
channel was disabled for structured output.

The subset is too small to claim general model quality. The detailed JSON output
is retained alongside this document for auditability. Future evaluation should
use manually authored held-out cases, prompt-injection documents, additional
models and repeated stochastic trials.
