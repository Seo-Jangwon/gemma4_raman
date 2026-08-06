# -*- coding: utf-8 -*-
"""drivers.py 검증 —  python -m vbench.check_drivers

정적 검사만으로는 부족하다. 도구가 드라이버의 어떤 메서드를 어떤 인자로 부르는지는
런타임에야 드러난다(기본값 생략, getattr 폴백, 예외 경로). 그래서 **에이전트가 쓰는
그 TOOL_DISPATCH 를 그대로 통과시켜** 45개 하드웨어 도구를 전부 눌러 본다.

두 가지를 본다:
  1. 도구가 예외 없이 돌고 ok:True 를 주는가 (= 드라이버 표면이 맞는가)
  2. 물리가 도구를 통해서도 살아 있는가 (측정하면 손상이 쌓이는가)
"""
from __future__ import annotations

import sys

from vbench.world import VirtualWorld
from vbench import drivers as D


def _fmt(v, n=90):
    s = str(v)
    return s if len(s) <= n else s[:n] + "..."


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    from backend.hw_tools.raman_tools import TOOL_DISPATCH

    w = VirtualWorld(seed=0, level="medium")
    rig = D.attach(w)
    bx0, bx1, by0, by1 = w.bounds_mm()
    cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2

    print(f"TOOL_DISPATCH 도구 수: {len(TOOL_DISPATCH)}")
    print(f"월드 범위: X {bx0:.3f}~{bx1:.3f}, Y {by0:.3f}~{by1:.3f} mm\n")

    # ── 1. 핵심 경로를 순서대로 ─────────────────────────────────────────────
    seq = [
        ("get_hardware_status",     {}),
        ("get_stage_position",      {}),
        ("get_stage_speed",         {}),
        ("set_stage_speed",         {"x_speed_mm_s": 2.0, "y_speed_mm_s": 2.0}),
        ("move_stage",              {"x": cx, "y": cy}),
        ("start_camera_stream",     {}),
        ("analyze_microscope_image", {"question": "where is the target?"}),
        ("move_to_pixel",           {"pixel_x": 700, "pixel_y": 300}),
        ("get_ccd_info",            {}),
        ("set_ccd_exposure",        {"exposure_time": 2.0}),
        ("set_laser_power",         {"percent": 15.0}),
        ("get_laser_status",        {}),
        ("acquire_spectrum",        {"exposure": 2.0, "power": 15.0}),
        ("capture_scene",           {}),
        ("set_ccd_acquisition_mode", {"mode": "accumulate", "num_accumulations": 2}),
        ("acquire_spectrum",        {"exposure": 1.0, "power": 10.0}),
        ("set_ccd_acquisition_mode", {"mode": "single"}),
        ("set_ccd_read_mode",       {"mode": "fvb"}),
        ("set_ccd_preamp_gain",     {"index": 1}),
        ("set_ccd_temperature",     {"temp": -60}),
        ("set_ccd_cooler",          {"on": True}),
        ("set_ccd_shutter",         {"mode": "auto"}),
        ("set_ccd_trigger_mode",    {"mode": "internal"}),
        ("set_camera_exposure",     {"ms": 30}),
        ("set_guide_beam_mode",     {}),
        ("laser_off",               {}),
        ("move_stage_relative",     {"dx": 0.01, "dy": 0.0}),
        ("set_guide_beam_mode",     {}),
        ("laser_on",                {}),
        ("run_autofocus",           {}),
        ("preview_grid_scan",       {"rows": 2, "cols": 2, "spacing_mm": 0.02}),
        ("run_grid_scan",           {"rows": 2, "cols": 2, "spacing_mm": 0.02,
                                     "exposure": 0.5, "power": 8.0, "autofocus": "none"}),
        ("set_ccd_shift_speeds",    {"vs_index": 0, "hs_index": 0}),
        ("set_ccd_read_mode",       {"mode": "image"}),
        ("set_ccd_image_flip",      {"hflip": False, "vflip": False}),
        ("set_ccd_read_mode",       {"mode": "fvb"}),
        ("set_camera_auto_exposure", {"enabled": False}),
        ("reconnect_hardware",      {"component": "all"}),
        ("list_results",            {}),
        ("stop_camera_stream",      {}),
    ]

    bad = []
    print("%-28s %-6s %s" % ("도구", "결과", "요약"))
    print("-" * 100)
    for name, args in seq:
        fn = TOOL_DISPATCH.get(name)
        if fn is None:
            print("%-28s %-6s %s" % (name, "없음", "TOOL_DISPATCH 에 없다"))
            bad.append((name, "not in dispatch"))
            continue
        try:
            r = fn(args)          # 디스패치 항목은 dict 하나를 받는다
        except Exception as e:
            print("%-28s %-6s %s" % (name, "예외", f"{type(e).__name__}: {e}"))
            bad.append((name, f"{type(e).__name__}: {e}"))
            continue
        ok = isinstance(r, dict) and r.get("ok", True)
        brief = {k: v for k, v in r.items()
                 if k in ("error", "position", "x", "y", "z", "max_intensity", "length",
                          "width", "height", "exposure_time", "laser_power_pct",
                          "power_percent", "beam", "temperature_C", "points", "count",
                          "sharpness_score", "saved")} if isinstance(r, dict) else r
        print("%-28s %-6s %s" % (name, "OK" if ok else "실패", _fmt(brief)))
        if not ok:
            bad.append((name, r.get("error", "?")))

    # ── 2. 디스패치에 있는데 위에서 안 눌러 본 도구 ─────────────────────────
    hw_only = {n for n in TOOL_DISPATCH
               if not n.startswith(("list_", "load_", "combine_", "aggregate_",
                                    "bundle_", "save_", "apply_", "get_bg", "run_analysis",
                                    "web_search", "search_", "recall", "remember",
                                    "inspect_", "reflect"))}
    untested = sorted(hw_only - {n for n, _ in seq})
    print(f"\n안 눌러 본 도구 {len(untested)}개: {', '.join(untested) if untested else '없음'}")

    # ── 3. 물리가 도구를 통해서도 살아 있는가 ───────────────────────────────
    print("\n같은 자리 반복 측정 — 도구 경로로도 손상이 쌓이는가")
    D.detach()
    w2 = VirtualWorld(seed=0, level="medium")
    rig2 = D.attach(w2)
    from vbench import oracle as O
    orc = O.build(w2)
    sx, sy = orc.best_spots_mm[0]
    TOOL_DISPATCH["move_stage"]({"x": sx, "y": sy})
    P = orc.power_pct
    T = orc.exposure_s * 0.4
    print("  정답 %0.2f%% / %0.2fs 인 자리에서 %0.2f%% / %0.2fs 로 6회" % (
        orc.power_pct, orc.exposure_s, P, T))
    print("   #   max_intensity      u   생존율  경고")
    for k in range(6):
        r = TOOL_DISPATCH["acquire_spectrum"]({"exposure": T, "power": P})
        rec = rig2.log[-1]
        print("   %d  %13.0f  %5.2f  %5.2f   %s" % (
            k + 1, r.get("max_intensity", 0), rec["u"], rec["survival"],
            "WARN" if rec["warned"] else ""))

    print("\n가상 경과시간 %.1fs (실제로는 자지 않는다), 이동 %d회, 측정 %d회"
          % (rig2.virtual_time_s, rig2.moves, len(rig2.log)))
    D.detach()

    print("\n" + ("실패한 도구 없음" if not bad
                  else f"실패 {len(bad)}개: " + ", ".join(f"{n}({e})" for n, e in bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
