# -*- coding: utf-8 -*-
"""world.py 눈으로 확인하기 —  python -m vbench.check

에이전트 없이 월드만 돌려 보고, 사람이 보고 판단할 그림 한 장을 만든다.
  1행  난이도별 월드 조감도 (하/중/상)  ← 에이전트에게는 절대 주지 않는 그림
  2행  난이도별 실제 화면 (타겟이 많은 자리)  ← 에이전트가 보는 그림
  3행  상황별 스펙트럼 4종 (정상 / 경고 / 탄화 / 포화)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vbench.world import VirtualWorld, FIELD_DS, LEVELS, TARGET_CM1

OUT = Path(__file__).resolve().parent / "_check"


def rich_spot(w: VirtualWorld, q: float = 0.5):
    """타겟이 많은 자리 하나의 스테이지 좌표."""
    ft = w.field[..., 2]
    idx = np.argwhere(ft > 0.9)
    r, c = idx[int(len(idx) * q)]
    return w.map_to_stage(c * FIELD_DS, r * FIELD_DS)


def best_recipe(w, x, y):
    """이 자리의 안전한 최적 (P, t) 을 수치로 찾는다 (oracle 의 축소판)."""
    best = (0.0, None)
    for P in np.arange(4.0, 80.0, 0.5):
        lo, hi = 0.02, 60.0
        for _ in range(40):                     # 안전한 최대 노출 이분탐색
            mid = (lo + hi) / 2
            if w.predict(x, y, P, mid)["safe"]:
                lo = mid
            else:
                hi = mid
        s = w.predict(x, y, P, lo)["snr"]
        if s > best[0]:
            best = (s, (float(P), float(lo)))
    return best


def main() -> int:
    # 이 프로젝트는 cp949 콘솔에서 돈다 — 한글도 '—' 도 그대로 print 하면 죽는다.
    # backend/llm_client.py 가 쓰는 것과 같은 처리.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(15, 12))
    levels = list(LEVELS)

    # ── 1행: 조감도 ──────────────────────────────────────────────────────────
    for i, lv in enumerate(levels):
        w = VirtualWorld(seed=0, level=lv)
        ax = fig.add_subplot(3, 3, i + 1)
        ax.imshow(w.overview(600)[..., ::-1])
        ax.set_title(f"[{lv}] world overview  (target {w.target_area:.0%} of map)", fontsize=10)
        ax.axis("off")
        if i == 0:
            ax.text(0.02, 0.02, "agent never sees this", transform=ax.transAxes,
                    color="w", fontsize=8, va="bottom")

    # ── 2행: 에이전트가 보는 화면 ────────────────────────────────────────────
    for i, lv in enumerate(levels):
        w = VirtualWorld(seed=0, level=lv)
        x, y = rich_spot(w)
        ax = fig.add_subplot(3, 3, i + 4)
        ax.imshow(w.render(x, y)[..., ::-1])
        ax.set_title(f"[{lv}] camera view @ f_tgt={w.composition_at(x, y)[2]:.2f}", fontsize=10)
        ax.axis("off")

    # ── 3행: 스펙트럼 4종 ────────────────────────────────────────────────────
    w = VirtualWorld(seed=0, level="medium")
    x, y = rich_spot(w)
    snr_star, (p_star, t_star) = best_recipe(w, x, y)

    # 좌: 상황 4종 (매번 새 시료)
    cases = [
        ("safe",      p_star, t_star * 0.45),
        ("optimum",   p_star, t_star),
        ("burned",    50.0,   3.0),
        ("saturated", 10.0,   25.0),
    ]
    ax = fig.add_subplot(3, 2, 5)
    for name, P, t in cases:
        w.dose[:] = 0                                   # 매번 새 시료로
        res = w.measure(x, y, P, t)
        rec = res["record"]
        lbl = (f"{name}: P={P:.0f}% t={t:.1f}s  SNR={rec['snr']:.0f} u={rec['u']:.2f}"
               + ("  SAT" if rec["saturated"] else "") + ("  WARN" if rec["warned"] else "")
               + (f"  surv={rec['survival']:.2f}" if rec["survival"] < 0.99 else ""))
        ax.plot(res["axis"], res["data"], lw=1.0, label=lbl)
    ax.axvline(TARGET_CM1, color="k", ls=":", lw=0.8)
    ax.set_xlabel("Raman shift (cm$^{-1}$)"); ax.set_ylabel("counts")
    ax.set_title(f"four situations, fresh sample each time\n"
                 f"optimum P={p_star:.1f}% t={t_star:.1f}s SNR*={snr_star:.0f}", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left")

    # 우: **같은 자리에 반복 측정** — 경고가 뜨고, 그 다음 무너지는가.
    #     이 벤치마크가 '적응형'을 판별하는 메커니즘 전부가 이 그림 하나에 있다.
    ax = fig.add_subplot(3, 2, 6)
    w.dose[:] = 0
    P_rep, t_rep = p_star, t_star * 0.35
    rows = []
    for k in range(6):
        res = w.measure(x, y, P_rep, t_rep)
        rec = res["record"]
        flag = ("WARN" if rec["warned"] else "ok  ") + (" BURNED" if rec["survival"] < 0.99 else "")
        ax.plot(res["axis"], res["data"], lw=1.0,
                label=f"#{k+1}  u={rec['u']:.2f}  SNR={rec['snr']:.0f}  {flag}")
        rows.append((k + 1, rec["u"], rec["snr"], rec["survival"], rec["warned"]))
    ax.axvline(TARGET_CM1, color="k", ls=":", lw=0.8)
    ax.set_xlabel("Raman shift (cm$^{-1}$)"); ax.set_ylabel("counts")
    ax.set_title(f"SAME spot measured 6x  (P={P_rep:.1f}%, t={t_rep:.1f}s each)\n"
                 f"warning must appear BEFORE the peak collapses", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left")

    print("\n같은 자리 반복 측정 (P=%.1f%%, t=%.1fs):" % (P_rep, t_rep))
    print("  #   u     SNR   생존율  경고")
    for k, u, s, sv, wn in rows:
        print("  %d  %5.2f %6.1f  %5.2f   %s" % (k, u, s, sv, "WARN" if wn else ""))

    fig.tight_layout()
    path = OUT / "world_check.png"
    fig.savefig(path, dpi=110)
    print(f"saved: {path}")

    # ── 콘솔 요약 ────────────────────────────────────────────────────────────
    print(f"\noptimum at this spot:  P={p_star:.1f}%   t={t_star:.2f}s   SNR*={snr_star:.1f}")
    print("\n색 대비(타겟 vs 버퍼) — 난이도가 실제로 색으로만 갈리는지 확인:")
    for lv in levels:
        ww = VirtualWorld(seed=0, level=lv)
        d = float(np.linalg.norm(ww.c_buf - np.array([225.0, 170.0, 55.0])))
        print(f"  [{lv}]  buffer RGB = ({ww.c_buf[0]:5.1f},{ww.c_buf[1]:5.1f},{ww.c_buf[2]:5.1f})"
              f"   |target - buffer| = {d:6.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
