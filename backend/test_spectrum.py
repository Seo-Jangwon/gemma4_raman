"""
test_spectrum.py — acquire_spectrum() 단독 테스트 스크립트 (v2)
==============================================================

[역할]
  시스템 전체 초기화 후 스펙트럼 1회 측정 → 결과 출력/저장/플롯.
  캘리브레이션 적용 여부에 따라 Raman shift 또는 pixel 축으로 자동 전환.

[실행 예시]
  python backend/test_spectrum.py
  python backend/test_spectrum.py --exposure 0.5 --power 60
  python backend/test_spectrum.py --plot         # matplotlib 그래프
  python backend/test_spectrum.py --full         # 전체 pixel 데이터 출력

[흐름]
  1. HardwareManager.startup()
     → 스테이지 homing, CCD Config.txt 적용 + 냉각 -40°C,
       factory calibration 자동 주입, 카메라/레이저/Ollama 연결
  2. raman_tools.acquire_spectrum() 호출
     → 레이저 ON → CCD 촬영 → 레이저 OFF
  3. 결과 출력 + CSV 저장 (+ 선택적 플롯)
  4. HardwareManager.shutdown()
     → CCD -5°C 복구 → 전체 연결 해제
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

# 프로젝트 루트를 sys.path 에 추가
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from hardware_manager import HardwareManager
from backend.agents import raman_tools
from backend.agents.USE_andor_test import save_spectrum_csv
from backend.agents.USE_autofocus_local import AutoFocusLocal


def parse_args():
    """명령줄 인자 파싱."""
    p = argparse.ArgumentParser(description="acquire_spectrum() 테스트")
    p.add_argument("--exposure",   type=float, default=0.2,
                   help="CCD 노출 시간 (초, 기본 0.2)")
    p.add_argument("--power",      type=int,   default=20,
                   choices=[20, 40, 60, 80, 100],
                   help="레이저 출력 %% (기본 20)")
    p.add_argument("--stabilize",  type=float, default=0.5,
                   help="레이저 ON 후 안정화 대기 (초, 기본 0.5)")
    p.add_argument("--plot",       action="store_true",
                   help="matplotlib 으로 스펙트럼 그래프 출력")
    p.add_argument("--full",       action="store_true",
                   help="전체 pixel 데이터 출력 (기본: 상위 5개만)")
    return p.parse_args()


def print_result(result: dict, exposure: float, power: int, full: bool = False):
    """측정 결과 콘솔 출력."""
    print("\n" + "=" * 60)
    print("  acquire_spectrum() 결과")
    print("=" * 60)

    if not result["ok"]:
        print(f"  [FAIL] {result['error']}")
        return

    # 기본 정보
    print(f"  노출시간    : {exposure} 초")
    print(f"  레이저 출력 : {power} %")
    print(f"  픽셀 수     : {result['length']}")
    print(f"  최대 강도   : {result['max_intensity']:.1f}")
    print(f"  합산 강도   : {result['sum_intensity']:.1f}")
    print(f"  캘리브레이션: {'예 (factory)' if result.get('calibrated') else '아니오'}")

    data = result.get("data", [])
    if not data:
        return

    if full:
        # 전체 데이터 출력
        print(f"\n  전체 데이터:")
        if result.get("calibrated"):
            print(f"  {'pixel':>6}  {'intensity':>10}  {'Δν (cm⁻¹)':>12}  {'λ (nm)':>12}")
            print("  " + "-" * 50)
            for i in range(len(data)):
                print(f"  {i:>6}  {data[i]:>10}  "
                      f"{result['raman_shift_cm-1'][i]:>10.2f}  "
                      f"{result['wavelength_nm'][i]:>10.4f}")
        else:
            print(f"  {'pixel':>6}  {'intensity':>10}")
            print("  " + "-" * 20)
            for px, cnt in enumerate(data):
                print(f"  {px:>6}  {cnt:>10}")
    else:
        # 상위 5개 픽셀만
        top5 = sorted(enumerate(data), key=lambda x: x[1], reverse=True)[:5]
        print("\n  상위 5개 픽셀:")
        for px, cnt in top5:
            extra = ""
            if result.get("calibrated"):
                extra = f"  Δν={result['raman_shift_cm-1'][px]:.1f} cm⁻¹"
            print(f"    pixel {px:4d}  →  {cnt}{extra}")

    print("=" * 60)


def save_csv(result: dict, exposure: float, power: int) -> str:
    """타임스탬프 붙여서 CSV 저장. save_spectrum_csv 재사용."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).resolve().parent / "spectra"
    out_dir.mkdir(exist_ok=True)
    filepath = out_dir / f"spectrum_{timestamp}_exp{exposure}s_pwr{power}pct.csv"
    
    if "pixel" not in result and "data" in result:
        result["pixel"] = list(range(len(result["data"])))
        result["intensity"] = result["data"]

    save_spectrum_csv(result, filepath)
    return str(filepath)


def plot_spectrum(result: dict):
    """matplotlib 으로 스펙트럼 플롯. calibrated 면 x축이 Raman shift."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib 없음 — pip install matplotlib")
        return

    data = result.get("data", [])
    if not data:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    if result.get("calibrated"):
        ax.plot(result["raman_shift_cm-1"], data, linewidth=0.8)
        ax.set_xlabel("Raman shift (cm⁻¹)")
    else:
        ax.plot(data, linewidth=0.8)
        ax.set_xlabel("Pixel")
    ax.set_ylabel("Intensity (counts)")
    ax.set_title("Raman Spectrum")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()

    hw = HardwareManager()
    try:
        # 1. 전체 HW 초기화 (CCD -40°C + factory calibration 자동 적용)
        hw.startup()

        # 2. 스펙트럼 측정
        print(f"\n[TEST] acquire_spectrum("
              f"exposure={args.exposure}, "
              f"power={args.power}, "
              f"stabilize_sec={args.stabilize})")

        result = raman_tools.acquire_spectrum(
            exposure=args.exposure,
            power=args.power,
            stabilize_sec=args.stabilize,
        )

        # 3. 결과 출력
        print_result(result, args.exposure, args.power, full=args.full)

        # 4. CSV 저장
        if result["ok"]:
            csv_path = save_csv(result, args.exposure, args.power)
            print(f"\n  [CSV] 저장 완료: {csv_path}")

        # 5. 선택적 플롯
        if args.plot and result["ok"]:
            plot_spectrum(result)

    except KeyboardInterrupt:
        print("\n[!] Ctrl+C — 종료 시퀀스 진입")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 6. 안전 종료 (CCD 온도 복구 블로킹)
        hw.shutdown()


if __name__ == "__main__":
    main()
