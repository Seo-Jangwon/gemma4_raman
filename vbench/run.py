# -*- coding: utf-8 -*-
"""러너 —  python -m vbench.run --agent careful --level medium --episodes 5

에피소드를 여러 번 돌리고 결과를 모은다. **에피소드 순서가 곧 학습 곡선의 x축**이다
(설계문서 §6.4) — 같은 난이도에서 시드만 바꿔 돌리고, 점수가 오르는지를 본다.

예:
  python -m vbench.run --agent careful --episodes 5
  python -m vbench.run --agent AILA --level hard --episodes 5 --seed0 100
  python -m vbench.run --agent oracle,naive,greedy,careful --episodes 3   # 기준선 눈금
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from vbench.episode import run_episode, BASELINES

RESULTS = Path(__file__).resolve().parent / "results"


def summarise(rows: list) -> dict:
    """에피소드들에서 한 줄 요약을 뽑는다."""
    ok = [r for r in rows if r.get("error") is None]
    sc = [r["score"] for r in ok]
    if not sc:
        return {"episodes": len(rows), "graded": 0}
    # 학습 곡선의 기울기 — 에피소드를 거치며 나아지는가(설계문서 §6.4).
    slope = float(np.polyfit(np.arange(len(sc)), sc, 1)[0]) if len(sc) >= 3 else None
    beh = [r["behaviour"] for r in ok]
    backed = [b["backoff"]["backed_off"] for b in beh if b.get("backoff")]
    return {
        "episodes": len(rows), "graded": len(ok),
        "errors": len(rows) - len(ok),
        "score_mean": round(float(np.mean(sc)), 4),
        # 거리도 함께 싣는다 — 점수 1/(1+d) 는 d 가 클수록 눌리므로, 못한 답들끼리의
        # 차이는 점수보다 거리에서 더 잘 보인다.
        "distance_mean": round(float(np.mean([r["distance"] for r in ok
                                              if r.get("distance") is not None])), 3),
        "score_sd": round(float(np.std(sc)), 4),
        "score_first": round(sc[0], 4), "score_last": round(sc[-1], 4),
        "learning_slope": None if slope is None else round(slope, 5),
        "unsafe_answers": sum(1 for r in ok if r.get("safe") is False),
        "no_answer": sum(1 for r in ok if r.get("missing_keys")),
        "measurements_mean": round(float(np.mean([b["measurements"] for b in beh])), 1),
        "burned_mean": round(float(np.mean([b["burned"] for b in beh])), 2),
        "saturated_mean": round(float(np.mean([b["saturated"] for b in beh])), 2),
        "best_snr_mean": round(float(np.mean([b["best_snr"] for b in beh])), 1),
        "snr_ratio_mean": round(float(np.mean([r.get("snr_ratio", 0.0) for r in ok])), 3),
        # 경고를 본 뒤 실제로 물러섰는가 — 가장 날카로운 적응 지표(§6.3).
        "backed_off": f"{sum(backed)}/{len(backed)}" if backed else "n/a",
    }


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="virtual Raman benchmark runner")
    ap.add_argument("--agent", default="careful",
                    help="AILA / CoALA / " + " / ".join(BASELINES) + " (쉼표로 여러 개)")
    ap.add_argument("--level", default="medium", help="easy / medium / hard (쉼표로 여러 개)")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=0, help="첫 에피소드의 시드")
    ap.add_argument("--out", default="", help="결과 폴더(기본: vbench/results/<시각>)")
    a = ap.parse_args(argv)

    out = Path(a.out) if a.out else RESULTS / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    agents = [s.strip() for s in a.agent.split(",") if s.strip()]
    levels = [s.strip() for s in a.level.split(",") if s.strip()]
    summary = {}

    for level in levels:
        for agent in agents:
            rows = []
            print(f"\n=== {agent} @ {level} ===", flush=True)
            print("%-4s %7s %8s %7s %6s  %s" % ("ep", "점수", "거리", "안전", "측정", "비고"),
                  flush=True)
            for i in range(a.episodes):
                seed = a.seed0 + i
                r = run_episode(agent, seed=seed, level=level, out_dir=out)
                rows.append(r)
                note = (r.get("error") or r.get("unsafe_why")
                        or (f"키 없음 {r['missing_keys']}" if r.get("missing_keys") else ""))
                print("%-4d %7.3f %8s %7s %6d  %s" % (
                    i, r["score"], r.get("distance"), r.get("safe"),
                    r["behaviour"]["measurements"], note), flush=True)
            s = summarise(rows)
            summary[f"{agent}@{level}"] = s
            print("  → 평균 %.3f (첫 %.3f → 끝 %.3f), 기울기 %s, 소각 %.1f, 후퇴 %s"
                  % (s["score_mean"], s["score_first"], s["score_last"],
                     s["learning_slope"], s["burned_mean"], s["backed_off"]), flush=True)

    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n결과: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
