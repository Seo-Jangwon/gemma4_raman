# -*- coding: utf-8 -*-
"""oracle.py 눈으로 확인하기 —  python -m vbench.check_oracle

  좌  (파워, 노출) 평면의 SNR 지형 + 안전 영역 + 정답과 허용 대역
  우  전략별 점수 — 암기가 통하지 않는지, 정답이 1.00 을 받는지
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vbench.world import VirtualWorld
from vbench import oracle as O

OUT = Path(__file__).resolve().parent / "_check"
N_SEEDS = 8

STRATEGIES = {
    "memorised 14.5% / 9.0s": lambda o: (14.5, 9.0),
    "default 20% / 1s":       lambda o: (20.0, 1.0),
    "cautious 8% / 3s":       lambda o: (8.0, 3.0),
    "oracle answer":          lambda o: (o.power_pct, o.exposure_s),
}


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    OUT.mkdir(parents=True, exist_ok=True)

    w = VirtualWorld(seed=0, level="medium")
    orc = O.build(w)
    bx, by = orc.best_spots_mm[0]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6))

    # ── 좌: (P, t) 평면의 SNR 지형 ───────────────────────────────────────────
    Ps = np.linspace(4, 45, 70)
    Ts = np.linspace(0.2, 20, 70)
    snr = np.full((len(Ts), len(Ps)), np.nan)
    unsafe_why = np.zeros((len(Ts), len(Ps)))          # 1=포화, 2=손상
    for i, t in enumerate(Ts):
        for j, P in enumerate(Ps):
            r = w.predict(bx, by, P, t)
            if r["safe"]:
                snr[i, j] = r["snr"]
            else:
                unsafe_why[i, j] = 2 if r["u"] > 1.0 else 1

    ext = [Ps[0], Ps[-1], Ts[0], Ts[-1]]
    axL.imshow(np.where(unsafe_why == 1, 1.0, np.nan), extent=ext, origin="lower",
               aspect="auto", cmap="Blues", vmin=0, vmax=2)
    axL.imshow(np.where(unsafe_why == 2, 1.0, np.nan), extent=ext, origin="lower",
               aspect="auto", cmap="Reds", vmin=0, vmax=2)
    im = axL.imshow(snr, extent=ext, origin="lower", aspect="auto", cmap="viridis")
    fig.colorbar(im, ax=axL, label="SNR (safe region only)")

    axL.plot(orc.power_pct, orc.exposure_s, "w*", ms=20, mec="k", mew=1.2,
             label=f"answer  {orc.power_pct:.2f}% / {orc.exposure_s:.2f}s  (SNR {orc.snr:.0f})")
    axL.plot(orc.power_band, [orc.exposure_s] * 2, "w-", lw=2.5,
             label=f"power band {orc.power_band}")
    axL.plot([orc.power_pct] * 2,
             [orc.exposure_s - orc.exposure_tol_s, orc.exposure_s], "w--", lw=2.5,
             label=f"exposure tol  ±{orc.exposure_tol_s:.2f}s")
    axL.set_xlabel("laser power (%)")
    axL.set_ylabel("CCD exposure (s)")
    axL.set_title("SNR over (power, exposure) at the best spot\n"
                  "blue = saturated,  red = burned,  colour = usable", fontsize=10)
    axL.legend(fontsize=8, loc="upper right")

    # ── 우: 전략별 점수 ──────────────────────────────────────────────────────
    res = {k: [] for k in STRATEGIES}
    rows = []
    for s in range(N_SEEDS):
        ws = VirtualWorld(seed=s, level="medium")
        os_ = O.build(ws)
        sx, sy = os_.best_spots_mm[0]
        rows.append((s, ws.e0, ws.fluor_scale, os_.power_pct, os_.exposure_s, os_.snr))
        for k, f in STRATEGIES.items():
            P, T = f(os_)
            res[k].append(os_.score(sx, sy, P, T, ws.predict(sx, sy, P, T)["safe"]))

    xs = np.arange(N_SEEDS)
    for k, v in res.items():
        axR.plot(xs, v, "o-", label=f"{k}   mean {np.mean(v):.2f}")
    axR.set_ylim(-0.05, 1.08)
    axR.set_xlabel("episode (seed)")
    axR.set_ylabel("score  =  1/(1+d)")
    axR.set_title("a memorised recipe must NOT work\n"
                  "(position is given as correct for all strategies)", fontsize=10)
    axR.grid(alpha=0.3)
    axR.legend(fontsize=8, loc="center right")

    fig.tight_layout()
    path = OUT / "oracle_check.png"
    fig.savefig(path, dpi=110)
    print(f"saved: {path}")

    # ── 콘솔 ─────────────────────────────────────────────────────────────────
    print("\n에피소드마다 정답이 달라지는가 (medium):")
    print("  seed     e0  형광배율      P*      t*   SNR*")
    for s, e0, fs, P, T, S in rows:
        print(f"  {s:4d} {e0:6.1f} {fs:8.2f} {P:7.2f} {T:7.2f} {S:6.1f}")
    arr = np.array([[r[3], r[4], r[5]] for r in rows])
    for i, nm in enumerate(("P*", "t*", "SNR*")):
        print(f"    {nm:5s} {arr[:,i].min():6.2f} ~ {arr[:,i].max():6.2f}"
              f"   (변동계수 {100*arr[:,i].std()/arr[:,i].mean():.0f}%)")

    print("\n전략별 점수:")
    for k, v in res.items():
        zeros = sum(1 for x in v if x == 0)
        print(f"  {k:26s} {' '.join(f'{x:.2f}' for x in v)}   평균 {np.mean(v):.2f}"
              f"   0점 {zeros}/{N_SEEDS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
