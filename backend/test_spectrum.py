"""
acquire_spectrum() 단독 테스트 스크립트
========================================
실행:
    python backend/test_spectrum.py
    python backend/test_spectrum.py --exposure 0.5 --power 60 --stabilize 1.0
    python backend/test_spectrum.py --plot   # matplotlib으로 스펙트럼 그래프 출력

흐름:
    1. HardwareManager.startup()
       - 스테이지 homing (0,0) → (max,max) → 중점
       - CCD 냉각 -40°C 안정화 (블로킹)
       - 카메라 / 레이저 / Ollama 연결
    2. raman_tools.init_hardware() 로 HW 객체 주입
    3. acquire_spectrum() 호출 → 결과 출력 (+ 선택적 그래프)
    4. HardwareManager.shutdown()
       - CCD -5°C 복구 (블로킹)
       - 모든 연결 해제
"""

import argparse
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.hardware_manager import HardwareManager

# raman_tools는 backend/agents/ 안에 있으므로 직접 경로로 import
sys.path.insert(0, str(_PROJECT_ROOT / "backend" / "agents"))
import raman_tools


def parse_args():
    p = argparse.ArgumentParser(description="acquire_spectrum() 테스트")
    p.add_argument("--exposure",   type=float, default=0.1,
                   help="CCD 노출 시간 (초, 기본 0.1)")
    p.add_argument("--power",      type=int,   default=20,
                   choices=[20, 40, 60, 80, 100],
                   help="레이저 출력 %% (기본 100)")
    p.add_argument("--stabilize",  type=float, default=0.5,
                   help="레이저 ON 후 안정화 대기 (초, 기본 0.5)")
    p.add_argument("--plot",       action="store_true",
                   help="결과를 matplotlib으로 출력")
    return p.parse_args()


def print_result(result: dict, exposure: float, power: int):
    print("\n" + "=" * 50)
    print("  acquire_spectrum() 결과")
    print("=" * 50)

    if not result["ok"]:
        print(f"  [FAIL] {result['error']}")
        return

    print(f"  노출시간    : {exposure} 초")
    print(f"  레이저 출력 : {power} %")
    print(f"  픽셀 수     : {result['length']}")
    print(f"  최대 강도   : {result['max_intensity']:.1f}")
    print(f"  합산 강도   : {result['sum_intensity']:.1f}")

    data = result.get("data", [])
    if data:
        top5 = sorted(enumerate(data), key=lambda x: x[1], reverse=True)[:5]
        print("\n  상위 5개 픽셀:")
        for px, cnt in top5:
            print(f"    pixel {px:4d}  →  {cnt}")

    print("=" * 50)


def plot_spectrum(result: dict):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib 없음 — pip install matplotlib")
        return

    data = result.get("data", [])
    if not data:
        return

    plt.figure(figsize=(10, 4))
    plt.plot(data, linewidth=0.8)
    plt.xlabel("Pixel")
    plt.ylabel("Intensity (counts)")
    plt.title("Raman Spectrum")
    plt.tight_layout()
    plt.show()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()

    hw = HardwareManager()
    try:
        # ── 1. 전체 HW 초기화 (CCD -40°C 안정화까지 블로킹) ──────────────────
        hw.startup()

        # ── 2. raman_tools에 HW 객체 주입 ─────────────────────────────────────
        raman_tools.init_hardware(
            stage=hw.stage,
            laser=hw.laser,
            ccd=hw.ccd,
        )

        # ── 3. 스펙트럼 측정 ───────────────────────────────────────────────────
        print(f"\n[TEST] acquire_spectrum("
              f"exposure={args.exposure}, "
              f"power={args.power}, "
              f"stabilize_sec={args.stabilize})")

        result = raman_tools.acquire_spectrum(
            exposure=args.exposure,
            power=args.power,
            stabilize_sec=args.stabilize,
        )

        # ── 4. 결과 출력 ───────────────────────────────────────────────────────
        print_result(result, args.exposure, args.power)

        if args.plot and result["ok"]:
            plot_spectrum(result)

    except KeyboardInterrupt:
        print("\n[!] Ctrl+C — 종료 시퀀스 진입")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        # ── 5. 안전 종료 (CCD -5°C 복구 블로킹 후 전체 해제) ─────────────────
        hw.shutdown()


if __name__ == "__main__":
    main()
