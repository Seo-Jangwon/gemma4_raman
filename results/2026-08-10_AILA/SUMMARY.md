# Raman agent benchmark

- Agent: **AILA**
- Run: `2026-08-10_AILA`
- Generated: 2026-08-10T14:00:21
- Instrument: 532.021 nm, centre 1200.0 cm-1, 1024 px

## Result

**Solved 3 / 3 gradable tasks (100.0%)**

A task counts as solved only when every one of its checks passes. There is no partial credit.

| | tasks |
|---|---|
| Solved | 3 |
| Failed | 0 |
| Not gradable - instrument limits (`blocked`) | 0 |
| Not gradable - run failed (`error`) | 0 |
| **Total defined** | **3** |

`blocked` and `error` are excluded from the solve rate: they record cases where the harness or the instrument, not the agent, prevented an answer from being graded.

## By capability axis

| Axis | Solved | Gradable | Rate | Excluded |
|---|---:|---:|---:|---:|
| identification | 3 | 3 | 100.0% | 0 |

## Per task

| Task | Axis | Result | Checks | First failure |
|---|---|---|---:|---|
| T120 | identification | pass | 1/1 |  |
| T124 | identification | pass | 1/1 |  |
| T125 | identification | pass | 1/1 |  |

---

Each task has a companion `<TASK>.json` in this folder with the full prompt, the agent's answer, every check with its expected and observed value, and the tool calls it made.
