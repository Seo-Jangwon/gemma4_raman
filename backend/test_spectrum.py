"""
test_spectrum.py — acquire_spectrum() 파라미터 대화형 검증 스크립트
================================================================

[실행]
  python backend/test_spectrum.py
  → 파라미터를 하나씩 입력받은 뒤 촬영 → 검증 → 출력/저장

[흐름]
  1. HardwareManager.startup()  (냉각 -40°C, factory calibration 자동 주입)
  2. 파라미터 대화형 입력
  3. acquire_spectrum() 호출
  4. 결과 유효성 검증 (shape / 강도 / calibration)
  5. 콘솔 출력 + CSV 저장 + 선택적 플롯
  6. HardwareManager.shutdown()  (온도 복구 블로킹)
"""
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from hardware_manager import HardwareManager
from backend.agents import raman_tools
from backend.agents.USE_andor_test import save_spectrum_csv


# ── 대화형 입력 ───────────────────────────────────────────────────────────────

def _ask(prompt: str, default, cast=str, choices=None):
    """한 줄 프롬프트. Enter 입력 시 default 반환."""
    if choices:
        opts = " / ".join(str(c) for c in choices)
        line = input(f"  {prompt} [{opts}] (기본: {default}): ").strip()
    else:
        line = input(f"  {prompt} (기본: {default}): ").strip()

    if not line:
        return default
    try:
        val = cast(line)
    except (ValueError, TypeError):
        print(f"    !! 잘못된 입력 — 기본값 {default} 사용")
        return default
    if choices is not None and val not in choices:
        print(f"    !! 허용값 아님 — 기본값 {default} 사용")
        return default
    return val


def _ask_yn(prompt: str, default: bool) -> bool:
    label = "Y/n" if default else "y/N"
    line = input(f"  {prompt} [{label}]: ").strip().lower()
    if not line:
        return default
    return line in ("y", "yes")


def prompt_params() -> dict:
    """파라미터를 대화형으로 입력받아 dict 반환."""
    print("\n" + "=" * 60)
    print("  측정 파라미터 입력  (Enter = 기본값)")
    print("=" * 60)

    p = {}

    # ── 기본 ──
    print("\n  [레이저 / 기본]")
    p["exposure"]     = _ask("노출 시간 [초]", 0.2, float)
    p["power"]        = _ask("레이저 출력 [%]", 20, int, [20, 40, 60, 80, 100])
    p["stabilize_sec"]= _ask("안정화 대기 [초]", 0.5, float)

    # ── 취득 모드 ──
    print("\n  [취득 모드]")
    p["acq_mode"] = _ask("취득 모드", "single", str,
                         ["single", "accumulate", "kinetic"])

    if p["acq_mode"] in ("accumulate", "kinetic"):
        p["num_accumulations"] = _ask("누적 횟수", 1, int)
    else:
        p["num_accumulations"] = 1

    if p["acq_mode"] == "kinetic":
        p["kinetic_count"]      = _ask("kinetic 프레임 수", 3, int)
        cyc = input("  kinetic 프레임 간격 [초] (기본: SDK 자동): ").strip()
        p["kinetic_cycle_time"] = float(cyc) if cyc else None
    else:
        p["kinetic_count"]      = 1
        p["kinetic_cycle_time"] = None

    # ── 읽기 모드 ──
    print("\n  [읽기 모드]")
    p["read_mode"] = _ask("읽기 모드", "fvb", str, ["fvb", "single_track"])
    p["hbin"]      = _ask("수평 비닝 [픽셀]", 1, int)

    if p["read_mode"] == "single_track":
        while True:
            center = input("  single_track 중심 행 번호 (필수): ").strip()
            if center:
                try:
                    p["single_track_center"] = int(center)
                    break
                except ValueError:
                    print("    !! 정수를 입력하세요")
            else:
                print("    !! 필수 항목입니다")
        p["single_track_width"] = _ask("트랙 폭 [픽셀]", 1, int)
    else:
        p["single_track_center"] = None
        p["single_track_width"]  = 1

    # ── 트리거 ──
    print("\n  [트리거]")
    p["trigger_mode"] = _ask(
        "트리거 모드", "internal", str,
        ["internal", "external", "external_start",
         "external_exposure", "external_fvb_em", "software"],
    )

    # ── 출력 / 저장 ──
    print("\n  [출력 / 저장]")
    p["full"]    = _ask_yn("전체 픽셀 데이터 출력?", False)
    p["plot"]    = _ask_yn("matplotlib 플롯?", False)
    p["no_save"] = _ask_yn("CSV 저장 생략?", False)

    # ── 요약 ──
    print("\n" + "=" * 60)
    print("  입력된 파라미터 요약")
    print("=" * 60)
    print(f"  exposure         : {p['exposure']} 초")
    print(f"  power            : {p['power']} %")
    print(f"  stabilize_sec    : {p['stabilize_sec']} 초")
    print(f"  acq_mode         : {p['acq_mode']}")
    if p["acq_mode"] in ("accumulate", "kinetic"):
        print(f"  num_accumulations: {p['num_accumulations']}")
    if p["acq_mode"] == "kinetic":
        print(f"  kinetic_count    : {p['kinetic_count']}")
        print(f"  kinetic_cycle    : {p['kinetic_cycle_time']} 초")
    print(f"  read_mode        : {p['read_mode']}")
    print(f"  hbin             : {p['hbin']}")
    if p["read_mode"] == "single_track":
        print(f"  track_center     : {p['single_track_center']}")
        print(f"  track_width      : {p['single_track_width']}")
    print(f"  trigger_mode     : {p['trigger_mode']}")

    return p


# ── 검증 ──────────────────────────────────────────────────────────────────────

def validate(result: dict, acq_mode: str, kinetic_count: int) -> list:
    if not result.get("ok"):
        return [f"ok=False — {result.get('error', '원인 불명')}"]

    errors = []
    if acq_mode == "kinetic":
        frames = result.get("frames", [])
        if len(frames) != kinetic_count:
            errors.append(f"프레임 수 불일치: 기대 {kinetic_count}, 실제 {len(frames)}")
        for f in frames:
            if not f.get("intensity"):
                errors.append(f"frame[{f['frame_index']}] intensity 비어있음")
            elif max(f["intensity"]) == 0:
                errors.append(f"frame[{f['frame_index']}] 전체 강도 0 (레이저/셔터 확인)")
    else:
        data = result.get("data", [])
        if not data:
            errors.append("data 비어있음")
        elif max(data) == 0:
            errors.append("전체 강도 0 (레이저/셔터 확인)")
        expected = result.get("length")
        if expected is not None and expected != len(data):
            errors.append(f"length 불일치: header={expected}, actual={len(data)}")

    return errors


# ── 콘솔 출력 ─────────────────────────────────────────────────────────────────

def _sep(label: str = ""):
    print(f"\n{'=' * 60}")
    if label:
        print(f"  {label}")
        print("=" * 60)


def _print_data(data: list, shifts, wavelengths, full: bool):
    if full:
        if shifts and wavelengths:
            print(f"  {'pixel':>6}  {'intensity':>10}  {'Δν (cm⁻¹)':>12}  {'λ (nm)':>10}")
            print("  " + "-" * 46)
            for i, v in enumerate(data):
                print(f"  {i:>6}  {v:>10}  {shifts[i]:>12.2f}  {wavelengths[i]:>10.4f}")
        else:
            print(f"  {'pixel':>6}  {'intensity':>10}")
            print("  " + "-" * 20)
            for i, v in enumerate(data):
                print(f"  {i:>6}  {v:>10}")
    else:
        top5 = sorted(enumerate(data), key=lambda x: x[1], reverse=True)[:5]
        print("  상위 5 픽셀:")
        for px, cnt in top5:
            extra = f"  Δν={shifts[px]:.1f} cm⁻¹" if shifts else ""
            print(f"    pixel {px:4d}  →  {cnt}{extra}")


def print_result(result: dict, full: bool):
    if not result.get("ok"):
        _sep("결과")
        print(f"  [FAIL] {result.get('error')}")
        return

    mode = result.get("mode", "?")

    if mode == "kinetic":
        _sep(f"결과 — Kinetic ({result.get('num_frames')} frames)")
        print(f"  노출시간    : {result.get('exposure_time')} 초")
        print(f"  레이저 출력 : {result.get('laser_power_pct')} %")
        for frame in result.get("frames", []):
            data = frame["intensity"]
            print(f"\n  ── Frame {frame['frame_index']:02d}  "
                  f"max={frame['max_intensity']:.1f}  "
                  f"sum={frame['sum_intensity']:.1f}  "
                  f"pixels={frame['length']}  "
                  f"cal={'예' if frame.get('calibrated') else '아니오'}")
            _print_data(data, frame.get("raman_shift_cm-1"),
                        frame.get("wavelength_nm"), full)
    else:
        _sep(f"결과 — {mode.capitalize()}")
        print(f"  노출시간    : {result.get('exposure_time')} 초")
        print(f"  레이저 출력 : {result.get('laser_power_pct')} %")
        if result.get("num_accumulations"):
            print(f"  누적 횟수   : {result['num_accumulations']}")
        print(f"  픽셀 수     : {result.get('length')}")
        print(f"  최대 강도   : {result.get('max_intensity', 0):.1f}")
        print(f"  합산 강도   : {result.get('sum_intensity', 0):.1f}")
        print(f"  캘리브레이션: {'예' if result.get('calibrated') else '아니오'}")
        _print_data(result.get("data", []),
                    result.get("raman_shift_cm-1"),
                    result.get("wavelength_nm"), full)


# ── CSV 저장 ──────────────────────────────────────────────────────────────────

def save_csv(result: dict, p: dict) -> list:
    out_dir = Path(__file__).resolve().parent / "spectra"
    out_dir.mkdir(exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"exp{p['exposure']}s_pwr{p['power']}pct_{p['acq_mode']}"
    paths = []

    if result.get("mode") == "kinetic":
        for frame in result.get("frames", []):
            rec = {
                "pixel":            list(range(len(frame["intensity"]))),
                "intensity":        frame["intensity"],
                "calibrated":       frame.get("calibrated", False),
                "raman_shift_cm-1": frame.get("raman_shift_cm-1"),
                "wavelength_nm":    frame.get("wavelength_nm"),
            }
            fp = out_dir / f"spectrum_{ts}_{tag}_f{frame['frame_index']:02d}.csv"
            save_spectrum_csv(rec, fp)
            paths.append(str(fp))
    else:
        data = result.get("data", [])
        rec = {
            "pixel":            list(range(len(data))),
            "intensity":        data,
            "calibrated":       result.get("calibrated", False),
            "raman_shift_cm-1": result.get("raman_shift_cm-1"),
            "wavelength_nm":    result.get("wavelength_nm"),
        }
        fp = out_dir / f"spectrum_{ts}_{tag}.csv"
        save_spectrum_csv(rec, fp)
        paths.append(str(fp))

    return paths


# ── 플롯 ──────────────────────────────────────────────────────────────────────

def plot_spectrum(result: dict):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib 없음 — pip install matplotlib")
        return

    fig, ax = plt.subplots(figsize=(10, 4))

    if result.get("mode") == "kinetic":
        frames = result.get("frames", [])
        calibrated = frames[0].get("calibrated") if frames else False
        for frame in frames:
            data = frame["intensity"]
            x = frame.get("raman_shift_cm-1") or list(range(len(data)))
            ax.plot(x, data, linewidth=0.8, label=f"frame {frame['frame_index']}")
        ax.legend(fontsize=8)
        ax.set_xlabel("Raman shift (cm⁻¹)" if calibrated else "Pixel")
    else:
        data = result.get("data", [])
        x = result.get("raman_shift_cm-1") or list(range(len(data)))
        ax.plot(x, data, linewidth=0.8)
        ax.set_xlabel("Raman shift (cm⁻¹)" if result.get("calibrated") else "Pixel")

    ax.set_ylabel("Intensity (counts)")
    ax.set_title(f"Raman Spectrum — {result.get('mode')}")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    sys.stdout.reconfigure(encoding="utf-8")

    p = prompt_params()

    hw = HardwareManager()
    try:
        hw.startup()

        result = raman_tools.acquire_spectrum(
            exposure=p["exposure"],
            power=p["power"],
            stabilize_sec=p["stabilize_sec"],
            acq_mode=p["acq_mode"],
            num_accumulations=p["num_accumulations"],
            kinetic_count=p["kinetic_count"],
            kinetic_cycle_time=p["kinetic_cycle_time"],
            read_mode=p["read_mode"],
            hbin=p["hbin"],
            single_track_center=p["single_track_center"],
            single_track_width=p["single_track_width"],
            trigger_mode=p["trigger_mode"],
        )

        errors = validate(result, p["acq_mode"], p["kinetic_count"])
        if errors:
            print("\n[FAIL] 검증 오류:")
            for e in errors:
                print(f"  - {e}")
        else:
            print("\n[PASS] 데이터 검증 통과")

        print_result(result, p["full"])

        if result.get("ok") and not p["no_save"]:
            for path in save_csv(result, p):
                print(f"\n  [CSV] {path}")

        if result.get("ok") and p["plot"]:
            plot_spectrum(result)

    except KeyboardInterrupt:
        print("\n[!] Ctrl+C — 종료")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        hw.shutdown()


if __name__ == "__main__":
    main()
