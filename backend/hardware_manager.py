"""
하드웨어 초기화 및 종료 관리자
================================
[시작 시]
  1. 스테이지 연결  →  (0,0) → (max,max) → 중점 이동 (블로킹)
  2. Andor CCD 연결 →  냉각기 ON, -40°C 안정화까지 대기 (블로킹)
  ── 위 두 단계가 모두 완료된 뒤에야 아래로 진행 ──
  3. TUCam 카메라 연결
  4. 레이저 연결
  5. Ollama 연결 확인

[사용 중]
  모든 HW 및 API 연결을 유지.

[종료 시]
  CCD 냉각기 OFF → 온도가 -5°C 이상으로 복구될 때까지 블로킹.
  그 뒤에만 레이저 / 카메라 / 스테이지 순으로 해제.

사용 예:
    hw = HardwareManager()
    hw.startup()          # 블로킹
    # ... 작업 ...
    hw.shutdown()         # 블로킹

    # 또는 컨텍스트 매니저로:
    with HardwareManager() as hw:
        ...
"""

from __future__ import annotations

import ctypes
import signal
import sys
import threading
import time
from pathlib import Path

# ── 경로 / 설정 ───────────────────────────────────────────────────────────────
_BACKEND      = Path(__file__).resolve().parent   # .../backend
_PROJECT_ROOT = _BACKEND.parent                   # .../gemma4_raman

# 프로젝트 루트를 sys.path 최상단에 추가 → Pylance 및 런타임 모두 해결
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── 하드웨어 모듈 import (Pylance가 추적 가능한 절대 경로) ────────────────────
from backend.agents.USE_stage_test       import TangoController   # noqa: E402
from backend.agents.USE_laser_with_power import LaserController    # noqa: E402
from backend.agents                      import raman_tools         # noqa: E402
from backend.andor_codes                 import AndorCCD            # noqa: E402
# StreamingTUCam 은 _init_camera() 안에서 lazy import
# → TUCam.py 가 import 시점에 TUCam.dll 을 바로 로드하기 때문에
#   DLL 이 없는 환경에서 모듈 import 자체가 실패하는 것을 방지

try:
    from backend.andor_codes import RamanCalibrator                 # noqa: E402
except ImportError:
    RamanCalibrator = None

STAGE_DLL_PATH   = str(_BACKEND / "agents" / "Tango_DLL.dll")

LASER_PORT       = "COM4"
OLLAMA_HOST      = "http://192.168.1.16:11434"
OLLAMA_MODEL     = "gemma4:31b"

# 스테이지 한계 (mm)
STAGE_MAX_X    = 75.7518    # Config.ini [STAGE_INFO] MaxX
STAGE_MAX_Y    = 50.4961    # Config.ini [STAGE_INFO] MaxY
STAGE_CENTER_X = 37.8759    # Config.ini [STAGE_INFO] CenterX (교정된 중점)
STAGE_CENTER_Y = 25.24805   # Config.ini [STAGE_INFO] CenterY (교정된 중점)

# CCD 온도 목표 (°C)
CCD_COOL_TARGET = -40   # 냉각 목표 — 안정화될 때까지 블로킹
CCD_WARM_TARGET = -5    # 종료 시 복구 목표 — 이 온도 이상 될 때까지 블로킹

# Andor SDK 상태 코드
DRV_SUCCESS             = 20002
DRV_TEMP_OFF            = 20034
DRV_TEMP_STABILIZED     = 20036
DRV_TEMP_NOT_REACHED    = 20037
DRV_TEMP_NOT_STABILIZED = 20035
DRV_TEMP_DRIFT          = 20040

# 스테이지 초기화 속도 (mm/s)  — 일반 스캔보다 빠르게
INIT_SPEED_MM_S = 10.0

DEFAULT_OUTPUT_AMP      = 1

class HardwareManager:
    """
    모든 하드웨어 연결 수명주기 관리.
    startup() 완료 후 .stage / .ccd / .camera / .laser / .ollama 로 접근 가능.
    """

    def __init__(self):
        self.stage  = None   # TangoController
        self.ccd    = None   # AndorCCD
        self.camera = None   # StreamingTUCam
        self.laser  = None   # LaserController
        self.ollama = None   # ollama.Client

        self._shutdown_called = False
        self._lock = threading.Lock()

        # Ctrl+C / 프로세스 종료 신호 처리
        signal.signal(signal.SIGINT,  self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    # ── 신호 처리 ──────────────────────────────────────────────────────────────

    def _signal_handler(self, sig, frame):
        print("\n\n[!] 종료 신호 수신 — CCD 온도 복구 후 안전 종료합니다 (강제 중단 금지)")
        self.shutdown()
        sys.exit(0)

    # ════════════════════════════════════════════════════════════════════════════
    # 개별 초기화
    # ════════════════════════════════════════════════════════════════════════════

    def _init_stage(self):
        """
        Tango 스테이지:
          연결 → 초기화 속도 설정
          → (0, 0) 이동 → (max_x, max_y) 이동 → 중점 복귀
          모두 wait=True (블로킹).
        """
        print("\n[STAGE] 스테이지 연결 중...")
        tango = TangoController(STAGE_DLL_PATH)

        if not tango.load_dll():
            raise RuntimeError("스테이지 DLL 로드 실패")
        if not tango.create_session():
            raise RuntimeError("스테이지 세션 생성 실패")
        if not tango.connect():
            raise RuntimeError("스테이지 연결 실패 (케이블/전원/드라이버 확인)")

        # 초기화 전용 고속 이동
        v = INIT_SPEED_MM_S
        tango.set_velocity(v, v, v * 0.1, v)

        print("[STAGE] 초기화 이동: (0,0) → (max,max) → 중점")
        pos = tango.get_position()
        if pos is None:
            raise RuntimeError("스테이지 현재 위치 조회 실패 — get_position() 반환값 없음")
        z = pos[2]

        # ③ 중점
        tango.move_absolute(STAGE_CENTER_X, STAGE_CENTER_Y, z, wait=True)
        pos = tango.get_position()
        if pos is not None:
            print(f"        → 중점        도달  X={pos[0]:.4f}  Y={pos[1]:.4f}")
        time.sleep(1)

        tango.set_velocity(0.1, 0.1, 0.1, 0.1)
        print("[STAGE] 초기화 완료")
        self.stage = tango

    def _init_ccd(self):
        """
        Andor CCD 초기화.

        흐름:
          1) RamanCalibrator.from_factory_calibration() → 외부 파일 없이 캘리브레이터 생성
          2) AndorCCD(initialize_to_defaults=False) 생성 — SDK 초기화 + DLL 자동 탐색
          3) _calibrator 주입 (start_acquisition_cycle 에서 캘리브레이션 적용)
          4) 냉각 -40°C 안정화 대기 (블로킹)
        """
        print("\n[CCD]   Andor CCD 연결 중...")

        # ── 캘리브레이터 생성 (외부 파일 불필요) ──
        calibrator = None
        if RamanCalibrator is not None:
            calibrator = RamanCalibrator.from_factory_calibration(laser_nm=532.021, f_mm=580.0)
            print(f"[CCD]   Factory calibration 적용: "
                  f"{calibrator._lut.min():.0f}~{calibrator._lut.max():.0f} cm⁻¹")

        # ── CCD 초기화 ──
        # initialize_to_defaults=False: 냉각/셔터/gain 등을 여기서 직접 제어
        ccd = AndorCCD(initialize_to_defaults=False)
        ccd._calibrator = calibrator

        # ── 취득 전 하드웨어 파라미터 명시 설정 ──
        # initialize_to_defaults=False이면 SDK가 이전 세션 상태를 그대로 유지하므로
        # 라만 측정에 적합한 값으로 명시적으로 초기화한다.
        ccd.set_ad_channel(0)

        # VS Speed: SDK 권장 안전 최대 속도 (전하 전송 비효율/스미어링 방지)
        if ccd.numVSSpeeds > 0:
            vs_idx, vs_spd = ccd.get_fastest_recommended_vs_speed()
            ccd.set_vs_speed(vs_idx)
            print(f"[CCD]   VS Speed 설정: index={vs_idx}, {vs_spd:.2f} µs/pixel")

        # HS Speed: 가장 느린 속도(마지막 인덱스) = 최저 Read Noise (라만 필수)
        if ccd.numHSSpeeds_Conventional and ccd.numHSSpeeds_Conventional[0] > 0:
            slowest_idx = ccd.numHSSpeeds_Conventional[0] - 1
            ccd.set_hs_speed_conventional(slowest_idx)
            hs_spd = ccd.HSSpeeds_Conventional[0][slowest_idx]
            print(f"[CCD]   HS Speed 설정: index={slowest_idx}, {hs_spd:.3f} MHz")

        # PreAmp Gain: 인덱스 0 (최저 이득 → 동적 범위 최대)
        ccd.set_preamp_gain(0)

        # 출력 앰프: Conventional (0=EMCCD, 1=Conventional)
        ccd.set_output_amp(DEFAULT_OUTPUT_AMP)

        # 셔터: 닫힘 (촬영 직전까지 광원 차단)
        ccd.set_shutter_close()

        # 우주선(Cosmic Ray) 필터: accumulate 모드에서 연속 스캔 비교로 스파이크 제거
        ccd.set_cosmic_ray_filter(True)

        # ── 냉각 시작 ──
        ccd.set_temperature(CCD_COOL_TARGET)
        ccd.set_cooler(True)

        # 냉각 중에도 온도 조회가 가능하도록 즉시 할당
        # (server.py의 백그라운드 스레드에서 /api/ccd/status 폴링 지원)
        self.ccd = ccd

        # ── 온도 안정화 대기 (블로킹) ──
        print(f"[CCD]   냉각 시작 → {CCD_COOL_TARGET}°C 안정화까지 대기")
        _LABELS = {
            DRV_TEMP_STABILIZED:     "안정화",
            DRV_TEMP_NOT_REACHED:    "냉각 중",
            DRV_TEMP_NOT_STABILIZED: "미안정",
            DRV_TEMP_DRIFT:          "드리프트",
            DRV_TEMP_OFF:            "냉각기 OFF",
        }
        while True:
            t      = ccd.get_temperature()
            status = ccd.temperature_status_num
            label  = _LABELS.get(status, f"status={status}")
            print(f"\r        현재: {t:5d}°C  [{label}]  "
                  f"(목표: {CCD_COOL_TARGET}°C) ...", end="", flush=True)
            if status == DRV_TEMP_STABILIZED:
                print(f"\n[CCD]   온도 안정화 완료: {t}°C")
                break
            time.sleep(3)

    def _init_camera(self):
        """TUCam 카메라 연결 + 스트리밍 시작."""
        from backend.agents.USE_camera_stream import StreamingTUCam  # lazy: DLL 로드
        print("\n[CAM]   TUCam 카메라 연결 중...")
        cam = StreamingTUCam(exposure_ms=50.0)
        cam.start_stream()
        print("[CAM]   연결 완료")
        self.camera = cam

    def _init_laser(self):
        """레이저 컨트롤러 연결."""
        print(f"\n[LASER] 레이저 연결 중... (포트: {LASER_PORT})")
        laser = LaserController(port=LASER_PORT)
        if not (laser.ser and laser.ser.is_open):
            raise RuntimeError(f"레이저 포트 연결 실패 ({LASER_PORT})")
        print("[LASER] 연결 완료")
        self.laser = laser

    def _init_ollama(self):
        """Ollama API 연결 확인."""
        import ollama

        print(f"\n[LLM]   Ollama 연결 확인 중... ({OLLAMA_HOST})")
        client = ollama.Client(host=OLLAMA_HOST)
        try:
            client.list()   # 실제 통신으로 연결 검증
        except Exception as e:
            raise RuntimeError(f"Ollama 연결 실패: {e}")
        print(f"[LLM]   연결 완료 (모델: {OLLAMA_MODEL})")
        self.ollama = client

    # ════════════════════════════════════════════════════════════════════════════
    # 공개 API
    # ════════════════════════════════════════════════════════════════════════════

    def startup(self):
        """
        전체 하드웨어 초기화.

        스테이지 초기화 + CCD -40°C 냉각 → 둘 다 완료될 때까지 블로킹.
        그 이후에만 카메라 / 레이저 / Ollama 초기화 진행.
        """
        print("=" * 60)
        print("  시스템 초기화 시작")
        print("=" * 60)

        # ── Phase 1: 스테이지 + CCD (병렬 실행, 둘 다 끝날 때까지 블로킹) ──────
        stage_exc: list[Exception | None] = [None]
        ccd_exc:   list[Exception | None] = [None]
        stage_done = threading.Event()
        ccd_done   = threading.Event()

        def _run_stage():
            try:
                self._init_stage()
            except Exception as e:
                stage_exc[0] = e
            finally:
                stage_done.set()

        def _run_ccd():
            try:
                self._init_ccd()
            except Exception as e:
                ccd_exc[0] = e
            finally:
                ccd_done.set()

        t_stage = threading.Thread(target=_run_stage, daemon=True, name="init-stage")
        t_ccd   = threading.Thread(target=_run_ccd,   daemon=True, name="init-ccd")
        t_stage.start()
        t_ccd.start()

        # 스테이지와 CCD 초기화가 모두 끝날 때까지 메인 스레드 블로킹
        stage_done.wait()
        ccd_done.wait()

        if stage_exc[0] is not None:
            print(f"[WARN] [STAGE] 초기화 실패 (건너뜀): {stage_exc[0]}")
        if ccd_exc[0] is not None:
            print(f"[WARN] [CCD]   초기화 실패 (건너뜀): {ccd_exc[0]}")

        print("\n" + "-" * 60)
        print("  [Phase 1 완료] → Phase 2 진행: 카메라 / 레이저 / Ollama")
        print("-" * 60)

        # ── Phase 2: 카메라 / 레이저 / Ollama (순차, 실패해도 계속) ──────────
        for name, init_fn in [
            ("CAM",   self._init_camera),
            ("LASER", self._init_laser),
            ("LLM",   self._init_ollama),
        ]:
            try:
                init_fn()
            except Exception as e:
                print(f"[WARN] [{name}]   초기화 실패 (건너뜀): {e}")

        connected = [
            k for k, v in [
                ("STAGE", self.stage), ("CCD", self.ccd),
                ("CAM",   self.camera), ("LASER", self.laser),
                ("LLM",   self.ollama),
            ] if v is not None
        ]
        print("\n" + "=" * 60)
        print(f"  초기화 완료 — 연결된 장비: {', '.join(connected) if connected else '없음'}")
        print("=" * 60 + "\n")

        raman_tools.init_hardware(
            stage=self.stage, 
            laser=self.laser, 
            ccd=self.ccd, 
            camera=self.camera
        )

    def shutdown(self):
        """
        안전 종료.

        CCD 온도가 -5°C 이상이 될 때까지 블로킹 — 그 전까지 절대 종료 불가.
        온도 복구 후 레이저 → 카메라 → 스테이지 순으로 해제.
        """
        with self._lock:
            if self._shutdown_called:
                return
            self._shutdown_called = True

        print("\n" + "=" * 60)
        print("  시스템 종료 시작")
        print("=" * 60)

        # ── Step 1: CCD 온도 복구 (최우선, 블로킹) ────────────────────────────
        self._ccd_warmup_and_shutdown()

        # ── Step 2: 레이저 ────────────────────────────────────────────────────
        if self.laser is not None:
            try:
                self.laser.laser_off()
                if self.laser.ser and self.laser.ser.is_open:
                    self.laser.ser.close()
                print("[LASER] 연결 해제 완료")
            except Exception as e:
                print(f"[LASER] 해제 중 예외: {e}")

        # ── Step 3: 카메라 ────────────────────────────────────────────────────
        if self.camera is not None:
            try:
                self.camera.stop_stream()
                self.camera.close()
                print("[CAM]   연결 해제 완료")
            except Exception as e:
                print(f"[CAM]   해제 중 예외: {e}")

        # ── Step 4: 스테이지 ──────────────────────────────────────────────────
        if self.stage is not None:
            try:
                self.stage.disconnect()
                self.stage.free_session()
                print("[STAGE] 연결 해제 완료")
            except Exception as e:
                print(f"[STAGE] 해제 중 예외: {e}")

        print("\n" + "=" * 60)
        print("  모든 하드웨어 연결 해제 완료")
        print("=" * 60 + "\n")

    def _ccd_warmup_and_shutdown(self):
        """
        냉각기 OFF → 온도가 CCD_WARM_TARGET(−5°C) 이상이 될 때까지 절대 반환하지 않음.
        온도 복구 완료 후 Andor SDK ShutDown 호출.
        """
        if self.ccd is None:
            return

        print(f"[CCD]   냉각기 OFF → {CCD_WARM_TARGET}°C 이상 복구까지 대기")
        print("        (이 단계가 끝나기 전에는 절대 종료되지 않습니다)")

        # 취득 중이면 먼저 중단 (DRV_ACQUIRING 상태에서는 CoolerOFF 불가)
        try:
            if self.ccd.get_status() == 'ACQUIRING':
                self.ccd.sdk.AbortAcquisition()
                time.sleep(0.5)
        except Exception:
            pass

        try:
            self.ccd.set_cooler(False)
        except Exception as e:
            print(f"[CCD]   [WARN] CoolerOFF 예외: {e}")

        # 온도가 목표 이상 오를 때까지 폴링
        _LABELS = {
            DRV_TEMP_STABILIZED:     "안정",
            DRV_TEMP_NOT_REACHED:    "냉각 중",
            DRV_TEMP_NOT_STABILIZED: "미안정",
            DRV_TEMP_DRIFT:          "드리프트",
            DRV_TEMP_OFF:            "냉각기 OFF",
        }
        while True:
            try:
                t      = self.ccd.get_temperature()
                status = self.ccd.temperature_status_num
                label  = _LABELS.get(status, f"status={status}")

                print(f"\r        온도 복구 중: {t:5d}°C  [{label}]  (목표: ≥ {CCD_WARM_TARGET}°C) ...",
                      end="", flush=True)

                if t >= CCD_WARM_TARGET:
                    print(f"\n[CCD]   온도 복구 완료: {t}°C")
                    break
            except Exception as e:
                print(f"\n[CCD]   [WARN] 온도 조회 오류: {e} — 재시도...")

            time.sleep(5)

        try:
            self.ccd.close()
            print("[CCD]   Andor SDK 종료 완료")
        except Exception as e:
            print(f"[CCD]   [WARN] SDK 종료 예외: {e}")

    # ── 컨텍스트 매니저 ────────────────────────────────────────────────────────

    def __enter__(self) -> "HardwareManager":
        self.startup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False   # 예외 재전파


# ── 전역 싱글톤 (여러 모듈에서 공유) ──────────────────────────────────────────
_manager: HardwareManager | None = None


def get_manager() -> HardwareManager:
    """초기화된 HardwareManager 싱글톤 반환. startup()은 호출자가 직접 해야 함."""
    global _manager
    if _manager is None:
        _manager = HardwareManager()
    return _manager


# ── 단독 실행 시 초기화 테스트 ────────────────────────────────────────────────
if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

    hw = HardwareManager()
    try:
        hw.startup()
        print("시스템 준비 완료. 종료하려면 Ctrl+C 를 누르세요.\n")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\n[ERROR] 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
    finally:
        hw.shutdown()
