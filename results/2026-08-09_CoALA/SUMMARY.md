# Raman agent benchmark

- Agent: **CoALA**
- Run: `2026-08-09_CoALA`
- Generated: 2026-08-09T21:59:47
- Instrument: 532.021 nm, centre 1200.0 cm-1, 1024 px

## Result

**Solved 3 / 4 gradable tasks (75.0%)**

A task counts as solved only when every one of its checks passes. There is no partial credit.

| | tasks |
|---|---|
| Solved | 3 |
| Failed | 1 |
| Not gradable - instrument limits (`blocked`) | 0 |
| Not gradable - run failed (`error`) | 3 |
| **Total defined** | **7** |

`blocked` and `error` are excluded from the solve rate: they record cases where the harness or the instrument, not the agent, prevented an answer from being graded.

## By capability axis

| Axis | Solved | Gradable | Rate | Excluded |
|---|---:|---:|---:|---:|
| data processing | 2 | 2 | 100.0% | 0 |
| identification | 0 | 0 | 0.0% | 2 |
| procedure | 1 | 2 | 50.0% | 1 |

## Per task

| Task | Axis | Result | Checks | First failure |
|---|---|---|---:|---|
| T047 | data processing | pass | 2/2 |  |
| T059 | procedure | **fail** | 3/4 | peaks after correction: 3/4 matched (submitted 5) [-26.536, 507.974, 966.517, 1366.834, 2128.038] vs [115.879, 507.974, 966.517, 1366.834] |
| T067 | procedure | _error_ | 3/4 | The model returned an empty reply (no text, no tool call). This is usually a context-window overflow - check the Ollama host and context size. |
| T073 | data processing | pass | 1/1 |  |
| T110 | procedure | pass | 2/2 |  |
| T125 | identification | _error_ | 0/1 | The model returned an empty reply (no text, no tool call). This is usually a context-window overflow - check the Ollama host and context size. |
| T128 | identification | _error_ | 0/1 | The model returned an empty reply (no text, no tool call). This is usually a context-window overflow - check the Ollama host and context size. |

## Excluded from the solve rate

- **Not gradable (run failed)** (3): T067, T125, T128

---

Each task has a companion `<TASK>.json` in this folder with the full prompt, the agent's answer, every check with its expected and observed value, and the tool calls it made.
