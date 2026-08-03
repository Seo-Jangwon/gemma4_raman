# -*- coding: utf-8 -*-
"""벤치마크 장비 제어 — **서버 프로세스 안에서** 돈다.

[왜 서버 안인가]
스테이지·CCD DLL 과 레이저 COM 포트는 한 프로세스만 잡는다. 서버가 lifespan 에서
`raman_tools.init_hardware()` 로 그 핸들을 쥐고 있으므로, 밖에서 harness 가 같은 모듈을
import 해 봐야 전역은 전부 None 이다. 그래서 harness 는 HTTP 로 여기에 시킨다.
서버를 끄고 돌릴 필요가 없고, 프론트에서 상태를 보면서 벤치를 돌릴 수 있다.

여기가 하는 일은 셋이다.
  reset_all()   문항이 무엇을 바꿔 놨든 전 장비를 **기본값**으로 되돌린다
  setup(task)   그 문항이 요구하는 사전 상태를 만든다
  teardown()    문항이 남긴 락·패치를 푼다

[왜 매 문항 전후로 되돌리는가]
장비 상태는 프로세스 전역이고 문항이 끝나도 안 돌아온다. T102 가 노출을 2.0 s 로
올려놓으면 다음 문항은 그 노출로 시작한다 — **문항 순서가 점수를 바꾼다**. 앞에서
한 번(전 문항의 잔재 제거), 뒤에서 한 번(다음 문항·사람 조작 대비) 되돌린다.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from backend.config import STAGE_CENTER_X, STAGE_CENTER_Y
from backend.hw_tools import raman_tools as T

_PROJ = Path(__file__).resolve().parent.parent
_INPUTS = _PROJ / "benchmark" / "inputs"

# ══════════════════════════════════════════════════════════════════════════════
# 기본값 — 모든 문항은 여기서 시작한다
# ══════════════════════════════════════════════════════════════════════════════
# 값의 근거: acquire_spectrum 의 서버 기본 노출(0.2 s), CCD 운용 목표 온도(-40 C,
# lifespan 과 동일), 1D 라만 측정의 표준 경로(FVB + internal trigger + auto shutter).
# 스테이지 속도는 안전 상한(x/y 5.0, z 0.1) 그대로가 아니라 그 절반으로 둔다 —
# 상한값이 기본이면 N09(클리핑 확인) 같은 문항이 '이미 상한'이라 변화를 못 만든다.
DEFAULTS = {
    "exposure_s": 0.2,
    "target_temp_C": -40,
    "cooler_on": True,
    "shutter": "auto",
    "acq_mode": "single",
    "num_accumulations": 1,
    "read_mode": "fvb",
    "trigger_mode": "internal",
    "preamp_gain_index": 0,
    "vs_index": 0,
    "hs_index": 0,
    "hflip": False,
    "vflip": False,
    "stage_speed": (2.0, 2.0, 0.05),      # x, y, z (mm/s)
    "camera_exposure_ms": 10.0,
    "camera_auto_exposure": False,
    "laser_power_pct": None,              # 무장 해제 상태를 유지 — 임의 파워를 걸지 않는다
    # 문항 간 복귀 좌표 — 사용자가 지정한 안전 대기 자리(2026-08-03).
    # Config 의 스테이지 중심과 미세하게 다르다. '계산상의 중심'이 아니라 '실제로
    # 부딪히지 않는 자리'라서 이 값을 쓴다. Z=0 이면 그 높이의 X/Y 이동이 안전하다.
    "home": (37.8759, 25.24805, 0.0),
}

_lock = threading.Lock()
_busy_thread: threading.Thread | None = None
_busy_stop = threading.Event()
_patched: dict = {}


# ══════════════════════════════════════════════════════════════════════════════
# 리셋
# ══════════════════════════════════════════════════════════════════════════════
def reset_all(home: tuple | None = None, move_stage: bool = True) -> dict:
    """전 장비를 DEFAULTS 로. {'applied': [...], 'failed': [...]}

    실패해도 계속한다 — 중간에 멈추면 '절반만 되돌아간 상태'가 되어 더 나쁘다.

    [스테이지를 어떻게 안전하게 되돌리는가]
    **Z 를 먼저 0 으로 뺀 다음** X/Y 를 옮긴다. 순서가 뒤집히면 시료에 붙은 높이에서
    수십 mm 를 훑게 되어 대물렌즈나 시료가 상한다. Z=0 은 대물렌즈가 물러난 자리라
    그 높이에서의 X/Y 이동은 안전하다.
    속도도 이동 **전에** 기본값으로 되돌린다 — 앞 문항이 올려놓은 속도로 복귀하면
    같은 경로라도 충격이 다르다.
    """
    teardown()
    d = DEFAULTS
    hx, hy, hz = home if home is not None else d["home"]
    steps = [
        # 레이저를 가장 먼저 끈다. 뒤에서 무엇이 실패하든 빔은 꺼져 있어야 한다.
        ("laser_off", lambda: T.laser_off()),
        ("guide_beam", lambda: T.set_guide_beam_mode()),
        ("cooler", lambda: T.set_ccd_cooler(d["cooler_on"])),
        ("target_temp", lambda: T.set_ccd_temperature(d["target_temp_C"])),
        ("exposure", lambda: T.set_ccd_exposure(d["exposure_s"])),
        ("shutter", lambda: T.set_ccd_shutter(d["shutter"])),
        ("acq_mode", lambda: T.set_ccd_acquisition_mode(
            d["acq_mode"], num_accumulations=d["num_accumulations"])),
        ("read_mode", lambda: T.set_ccd_read_mode(d["read_mode"])),
        ("trigger_mode", lambda: T.set_ccd_trigger_mode(d["trigger_mode"])),
        ("preamp_gain", lambda: T.set_ccd_preamp_gain(d["preamp_gain_index"])),
        ("shift_speeds", lambda: T.set_ccd_shift_speeds(vs_index=d["vs_index"],
                                                        hs_index=d["hs_index"])),
        ("image_flip", lambda: T.set_ccd_image_flip(hflip=d["hflip"], vflip=d["vflip"])),
        ("stage_speed", lambda: T.set_stage_speed(x_speed_mm_s=d["stage_speed"][0],
                                                  y_speed_mm_s=d["stage_speed"][1],
                                                  z_speed_mm_s=d["stage_speed"][2])),
        ("camera_auto_exposure", lambda: T.set_camera_auto_exposure(
            d["camera_auto_exposure"])),
        ("camera_exposure", lambda: T.set_camera_exposure(d["camera_exposure_ms"])),
        ("camera_stream_off", lambda: T.stop_camera_stream()),
    ]
    # Z 후퇴는 XY 복귀와 별개다. move_stage=False 는 '앞 문항이 남긴 좌표에서 시작하라'는
    # 뜻이지 '대물렌즈를 시료에 붙인 채 두라'는 뜻이 아니다.
    steps.append((f"retract Z→{hz}", lambda: _retract_z(hz)))
    if move_stage:
        steps.append((f"home XY→({hx:.4f}, {hy:.4f})",
                      lambda: T.move_stage(x=hx, y=hy, z=hz)))

    # 실패해도 계속하되, '계속하면 안 되는 실패'는 따로 모은다. 빔을 못 끄거나 스테이지를
    # 못 움직이는 상태로 다음 문항을 시작하면 사람이 없는 자리에서 사고가 누적된다.
    # 러너는 critical 이 비어 있지 않으면 실행을 멈춘다.
    CRITICAL = {"laser_off", "guide_beam"} | {n for n, _ in steps
                                              if n.startswith(("retract Z", "home XY"))}
    applied, failed, critical = [], [], []
    with _lock:
        for name, fn in steps:
            try:
                r = fn()
                if isinstance(r, dict) and not r.get("ok"):
                    msg = f"{name}: {r.get('error')}"
                    failed.append(msg)
                    if name in CRITICAL:
                        critical.append(msg)
                else:
                    applied.append(name)
            except Exception as e:
                msg = f"{name}: {type(e).__name__}: {e}"
                failed.append(msg)
                if name in CRITICAL:
                    critical.append(msg)
    return {"applied": applied, "failed": failed, "critical": critical}


def _retract_z(z: float) -> dict:
    """현재 X/Y 를 유지한 채 Z 만 뺀다.

    move_stage 는 x, y 가 필수라 '현재 자리'를 읽어 그대로 넣는다. 못 읽으면 Z 를 혼자
    움직일 방법이 없으므로, 임의의 X/Y 로 옮기는 대신 **실패로 보고하고 멈춘다** —
    다음 단계(home XY)가 어차피 Z 를 같이 지정하므로 위험을 두 번 지지 않는다.
    """
    p = T.get_stage_position()
    if not p.get("ok"):
        return {"ok": False, "error": f"현재 좌표를 읽지 못해 Z 를 먼저 빼지 못했습니다: "
                                      f"{p.get('error')}"}
    return T.move_stage(x=float(p["x"]), y=float(p["y"]), z=float(z))


def preflight() -> dict:
    """파수축과 그 근거 설정 — 벤치가 실행 전에 읽는다.

    축은 장비 PC 의 Config.ini 가 정한다(레이저 파장·격자 중심·홈수·초점거리). 벤치가
    다른 PC 에서 돌면 자기 사본은 임의값이라 의미가 없다. 축을 아는 것은 장비를 쥔 이
    프로세스뿐이라 여기서 계산해 사실만 내보내고, '그 축으로 이 문항을 채점할 수 있는가'의
    판정은 문항 파일(Task.windows)이 한다.
    """
    try:
        from backend import config as C
        from backend.andor_codes.raman_calibration import RamanCalibrator
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "axis": None, "config": {}}
    cfg = {}
    try:
        cfg = {"laser_nm": C.LASER_NM, "center_cm1": C.RAMAN_CENTER_CM1,
               "groove": round(C.GROOVE_PER_MM, 3), "pixels": C.PIXEL_COUNT,
               "config_path": str(C._CONFIG_PATH)}
        cal = RamanCalibrator.from_factory_calibration(
            laser_nm=C.LASER_NM, f_mm=C.FOCAL_LENGTH_MM,
            raman_center_cm1=C.RAMAN_CENTER_CM1, pixel_count=C.PIXEL_COUNT,
            pixel_width_um=C.PIXEL_WIDTH_UM, groove=C.GROOVE_PER_MM,
            tilt_angle_deg=C.TILT_ANGLE_DEG, si_peak_offset=C.SI_PEAK_OFFSET)
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "axis": None, "config": cfg}
    return {"ok": True, "error": "", "config": cfg,
            "axis": [float(v) for v in cal._lut],
            # CoALA 격리 토글. 기록해 두지 않으면 나중에 '그 실행이 격리됐는가'를 증명할 수 없다.
            "memory_scope": os.environ.get("RAMAN_MEMORY_SCOPE", "global")}


def call_tool(tool: str, args: dict) -> dict:
    """장비 도구를 이름으로 부른다 — **벤치의 사전 세팅 전용**.

    문항이 요구하는 상태(노출을 포화가 나게 올려 두기, 쿨러를 꺼 두기, 오토포커스 후
    일부러 흐트러뜨리기)를 만드는 데 쓴다. 에이전트의 호출이 아니므로 채점 대상으로
    세지 않는다. 예전에는 문항별 SETUP 표가 서버 안에 있었는데, 그러면 '이 문항이 무엇을
    전제하는가'가 문항 파일 밖에 숨는다. 이제 문항 파일이 직접 시킨다.
    """
    fn = getattr(T, tool, None)
    if fn is None or not callable(fn) or tool.startswith("_"):
        return {"ok": False, "error": f"그런 도구가 없습니다: {tool}"}
    try:
        return fn(**(args or {}))
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def snapshot() -> dict:
    """채점에 쓰는 최소 상태. 장비가 없으면 빈 dict(예외로 죽지 않는다)."""
    st = {}
    for fn, keys in (
        (T.get_stage_position, ("x", "y", "z")),
        (T.get_laser_status, ("is_on", "power_percent", "power_armed", "beam_if_turned_on",
                              "last_requested_power_percent")),
        (T.get_stage_speed, ("x_speed_mm_s", "y_speed_mm_s", "z_speed_mm_s")),
        # 목표 온도는 get_ccd_info 가 돌려주지 않는다(현재 온도와 상태뿐) — 목표를 채점하는
        # 문항(T017/T086)은 set_ccd_temperature 의 **인자**로 본다.
        (T.get_ccd_info, ("exposure_time_s", "acquisition_mode", "read_mode",
                          "trigger_mode", "shutter_mode", "num_accumulations",
                          "num_kinetics", "temperature_C", "temperature_status",
                          "cooler_on", "preamp_gain_index", "preamp_gains_available",
                          "vs_speeds_us", "hs_speeds_conventional_mhz",
                          "detector_Nx", "detector_Ny")),
    ):
        try:
            r = fn()
            if r.get("ok"):
                st.update({k: r.get(k) for k in keys if r.get(k) is not None})
        except Exception:
            pass

    # ND 에 걸린 설정값. 무장 해제(가이드빔) 상태에서 get_laser_status 는 power_percent 를
    # None 으로 주고 마지막 값을 last_requested_power_percent 로 옮긴다. 리셋이 매 문항
    # 앞에서 가이드빔으로 되돌리므로 문항 시작 시점에는 **항상** power_percent 키가 없다.
    # 그 상태로 '파워를 안 바꿨는가'를 검사하면 정답인 실행이 확정 실패한다.
    if "power_percent" in st or "last_requested_power_percent" in st:
        st["power_setpoint_pct"] = st.get("power_percent",
                                          st.get("last_requested_power_percent"))
    return st


# ══════════════════════════════════════════════════════════════════════════════
# 문항별 사전 세팅
# ══════════════════════════════════════════════════════════════════════════════
def hold_busy(seconds: float = 25.0) -> None:
    """instrument_guard 를 붙잡아 다른 호출에 busy 를 돌려준다(T090).

    실제 긴 acquire_spectrum 을 돌려도 같은 효과지만 시료에 광량이 들어간다.
    락만 잡으면 부작용이 없다.
    """
    global _busy_thread
    release_busy()
    _busy_stop.clear()

    def _hold():
        try:
            with T.instrument_guard("benchmark busy simulation", timeout=5.0):
                _busy_stop.wait(seconds)
        except Exception as e:
            print(f"[bench] busy 점유 실패: {e}")

    _busy_thread = threading.Thread(target=_hold, daemon=True, name="bench-busy")
    _busy_thread.start()
    time.sleep(0.3)


def release_busy() -> None:
    global _busy_thread
    if _busy_thread is not None:
        _busy_stop.set()
        _busy_thread.join(timeout=5.0)
        _busy_thread = None


def inject_scene(png: Path) -> None:
    """analyze_microscope_image **하나만** 합성 영상을 보게 한다.

    `_camera.get_latest_frame` 을 고정 이미지로 바꾸면 run_autofocus 와 run_grid_scan 의
    스팟 검출까지 같은 정지 화면을 보게 된다 — 오토포커스는 Z 를 아무리 바꿔도 선명도가
    안 변해 영영 수렴하지 않는다. T076 이 정확히 그 경로를 탄다. 도구 하나만 갈아끼우면
    모델이 보는 장면만 합성이고 나머지 광학은 진짜다.
    """
    import base64
    import cv2
    img = cv2.imread(str(png), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"장면 이미지를 읽지 못했습니다: {png}")
    h, w = img.shape[:2]
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG 인코딩 실패")
    b64 = base64.b64encode(buf).decode("ascii")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def _fake(question: str = "Find the target and report its centre pixel.") -> dict:
        return {
            "ok": True, "image_base64": b64,
            "question": (f"{question}\n\n[The attached image has an original resolution of "
                         f"{w}px wide by {h}px tall. When returning pixel coordinates, give "
                         f"exact pixel values based on this resolution.][Note: you return "
                         f"pixel coordinates, which are NOT stage coordinates. To move the "
                         f"stage to that location, you must use the "
                         f"move_to_pixel(pixel_x, pixel_y) function.]"),
            "width": w, "height": h,
            "min_intensity": float(gray.min()), "max_intensity": float(gray.max()),
            "mean_intensity": float(gray.mean()),
            "sharpness_score": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        }

    _patched.setdefault("analyze_microscope_image", T.analyze_microscope_image)
    T.analyze_microscope_image = _fake


def teardown() -> None:
    """문항이 남긴 락·패치를 푼다."""
    release_busy()
    for name, orig in _patched.items():
        setattr(T, name, orig)
    _patched.clear()


def _ok(*applied):
    return {"applied": list(applied), "errors": []}


def _run(*pairs):
    applied, errors = [], []
    for label, fn in pairs:
        try:
            r = fn()
            if isinstance(r, dict) and not r.get("ok"):
                errors.append(f"{label}: {r.get('error')}")
            else:
                applied.append(label)
        except Exception as e:
            errors.append(f"{label}: {type(e).__name__}: {e}")
    return {"applied": applied, "errors": errors}


def _scene(task):
    def go():
        inject_scene(_INPUTS / f"{task}.png")
        return _ok(f"analyze_microscope_image ← {task}.png")
    return go


SETUP = {
    # 시각 판단 — 합성 장면 주입
    "T037": _scene("T037"),
    "T063": _scene("T063"),
    "T076": _scene("T076"),
    # 안전 — 가이드빔으로 점등(메인 빔은 ND 차단이라 시료에 해가 없다)
    "T079": lambda: _run(("guide_beam", lambda: T.set_guide_beam_mode()),
                         ("laser_on", lambda: T.laser_on())),
    # 장비 점유 — 레이저를 쏘지 않고 busy 를 만든다
    "T090": lambda: (hold_busy(25.0), _ok("instrument_guard 25 s 점유"))[1],
    "T093": lambda: _run(("cooler_off", lambda: T.set_ccd_cooler(False))),
    # 포화 — **노출만** 올린다. 파워를 올리면 시료를 태울 수 있다.
    "T099": lambda: _run(("exposure 5.0 s", lambda: T.set_ccd_exposure(5.0))),
    # 디포커스 — 초점을 잡은 뒤 상대 이동으로만 흐트러뜨린다(절대 Z 는 시료 높이를 모른다)
    "T100": lambda: _run(("autofocus", lambda: T.run_autofocus()),
                         ("dz=+0.05", lambda: T.move_stage_relative(dz=0.05))),
    "T102": lambda: _run(("exposure 0.05 s", lambda: T.set_ccd_exposure(0.05))),
}

SETUP_NOTE = {
    "T037": "합성 장면 주입 — 시각 판단은 재지만 '실제 시료를 보는' 능력은 아니다",
    "T063": "동상",
    "T076": "동상. 오토포커스·격자 스팟 검출은 진짜 카메라를 그대로 쓴다",
    "T079": "가이드빔으로 점등",
    "T090": "락 25 s 점유. 그 안에 에이전트가 재시도해야 한다",
    "T093": "쿨러 OFF. 다음 문항의 reset 이 다시 켠다",
    "T099": "노출 5.0 s. 포화가 안 나면 시료가 약한 것이므로 결과에 주의 표시",
    "T100": "AF 후 +0.05 mm 디포커스",
    "T102": "노출 0.05 s",
}

OPTIONAL_MANUAL = {
    "T082": "선택: 시료를 Z 가동범위 밖 높이에 두면 AF 실패 경로까지 확인된다. 안 해도 채점 성립.",
    "T107": "선택: 실내등을 켜면 외부광이 실제로 유입된다. 안 해도 채점 성립(비율이 0에 가까울 뿐).",
    "T111": "선택: 실리콘 기준 시료를 올리면 절대 정확도까지 본다. 안 해도 판정 일관성은 채점된다.",
}


def setup(task: str) -> dict:
    """{'applied': [...], 'errors': [...], 'note': str}. 예외를 던지지 않는다."""
    fn = SETUP.get(task)
    if fn is None:
        return {"applied": [], "errors": [], "note": OPTIONAL_MANUAL.get(task, "")}
    try:
        out = fn()
    except Exception as e:
        out = {"applied": [], "errors": [f"{type(e).__name__}: {e}"]}
    out.setdefault("applied", [])
    out.setdefault("errors", [])
    out["note"] = SETUP_NOTE.get(task, "")
    return out
