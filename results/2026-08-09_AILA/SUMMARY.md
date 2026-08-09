# Raman agent benchmark

- Agent: **AILA**
- Run: `2026-08-09_AILA`
- Generated: 2026-08-09T21:22:55
- Instrument: 532.021 nm, centre 1200.0 cm-1, 1024 px

## Result

**Solved 2 / 3 gradable tasks (66.7%)**

A task counts as solved only when every one of its checks passes. There is no partial credit.

| | tasks |
|---|---|
| Solved | 2 |
| Failed | 1 |
| Not gradable - instrument limits (`blocked`) | 0 |
| Not gradable - run failed (`error`) | 3 |
| **Total defined** | **6** |

`blocked` and `error` are excluded from the solve rate: they record cases where the harness or the instrument, not the agent, prevented an answer from being graded.

## By capability axis

| Axis | Solved | Gradable | Rate | Excluded |
|---|---:|---:|---:|---:|
| data processing | 1 | 1 | 100.0% | 0 |
| identification | 0 | 1 | 0.0% | 3 |
| procedure | 1 | 1 | 100.0% | 0 |

## Per task

| Task | Axis | Result | Checks | First failure |
|---|---|---|---:|---|
| T035 | procedure | pass | 5/5 |  |
| T073 | data processing | pass | 1/1 |  |
| T120 | identification | _error_ | 0/1 | The model returned an empty reply (no text, no tool call). This is usually a context-window overflow - check _NUM_CTX and the Ollama host. |
| T124 | identification | _error_ | 0/1 | The model returned an empty reply (no text, no tool call). This is usually a context-window overflow - check _NUM_CTX and the Ollama host. |
| T125 | identification | _error_ | 0/1 | The model returned an empty reply (no text, no tool call). This is usually a context-window overflow - check _NUM_CTX and the Ollama host. |
| T129 | identification | **fail** | 1/2 | shift: 3 (expected 2.7, \|Δ\|=0.3 ≤ 0.2)  [answer] |

## Excluded from the solve rate

- **Not gradable (run failed)** (3): T120, T124, T125

---

Each task has a companion `<TASK>.json` in this folder with the full prompt, the agent's answer, every check with its expected and observed value, and the tool calls it made.
