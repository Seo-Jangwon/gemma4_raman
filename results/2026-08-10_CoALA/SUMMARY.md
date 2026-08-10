# Raman agent benchmark

- Agent: **CoALA**
- Run: `2026-08-10_CoALA`
- Generated: 2026-08-10T12:56:07
- Instrument: 532.021 nm, centre 1200.0 cm-1, 1024 px

## Result

**Solved 2 / 2 gradable tasks (100.0%)**

A task counts as solved only when every one of its checks passes. There is no partial credit.

| | tasks |
|---|---|
| Solved | 2 |
| Failed | 0 |
| Not gradable - instrument limits (`blocked`) | 0 |
| Not gradable - run failed (`error`) | 1 |
| **Total defined** | **3** |

`blocked` and `error` are excluded from the solve rate: they record cases where the harness or the instrument, not the agent, prevented an answer from being graded.

## By capability axis

| Axis | Solved | Gradable | Rate | Excluded |
|---|---:|---:|---:|---:|
| identification | 1 | 1 | 100.0% | 1 |
| procedure | 1 | 1 | 100.0% | 0 |

## Per task

| Task | Axis | Result | Checks | First failure |
|---|---|---|---:|---|
| T110 | procedure | pass | 2/2 |  |
| T125 | identification | pass | 1/1 |  |
| T128 | identification | _error_ | 0/1 | The model returned an empty reply (no text, no tool call). This is usually a context-window overflow - check the Ollama host and context size. |

## Excluded from the solve rate

- **Not gradable (run failed)** (1): T128

---

Each task has a companion `<TASK>.json` in this folder with the full prompt, the agent's answer, every check with its expected and observed value, and the tool calls it made.
