# -*- coding: utf-8 -*-
"""가짜 하드웨어 드라이버 — 45개 도구를 **한 줄도 안 고치고** 시뮬레이터 위에서 돌린다.

`raman_tools.init_hardware(stage=, laser=, ccd=, camera=)` 에 이 네 객체를 넣으면 끝이다.

[왜 도구 함수를 패치하지 않는가]
도구를 하나씩 갈아끼우면 (1) 빠뜨린 도구를 알아채기 어렵고, (2) AILA/CoALA 가 보는
도구 표면이 달라져 **비교의 독립변수(오케스트레이션)가 오염된다.** 드라이버를 주입하면
두 에이전트가 완전히 같은 49/53개 도구를 그대로 쓰므로 능력 동등성이 구조적으로 보장된다.

[표면은 추측하지 않고 세어서 맞췄다]
    grep -oE '_(stage|laser|ccd|camera)\\.[a-zA-Z_]+' backend/hw_tools/raman_tools.py
로 뽑은 목록이 아래 클래스들이 구현하는 전부다. 속성도 마찬가지로 getattr 호출을
전부 뽑아 맞췄다 — 하나라도 빠지면 도구가 None 을 받아 조용히 이상하게 동작한다.

[한 가지 진실]
네 드라이버는 상태를 각자 들고 있지 않고 `VirtualRig` 하나를 공유한다. CCD 가 스펙트럼을
만들려면 **스테이지 위치와 레이저 파워**를 알아야 하는데, 실장비에서는 그 셋이 물리적으로
한 광학계에 묶여 있기 때문이다. 따로 두면 '어느 자리에서 몇 %로 쐈는가'가 갈라진다.
"""
from __future__ import annotations

import numpy as np

from vbench.world import VirtualWorld, VIEW_W, VIEW_H, N_BINS, axis as _axis
from backend.config import STAGE_MIN_Z, STAGE_MAX_Z


class VirtualRig:
    """월드 + 장비 상태. 네 드라이버가 공유하는 유일한 진실."""

    # 예산. 초과하면 측정이 **관측 가능한 에러**로 거부된다 — 예외를 던지거나 조용히
    # 멈추면 에이전트가 이유를 모른 채 같은 호출을 반복한다. 이 프로젝트의 도구 계층이
    # 전부 그렇게 하고 있으므로(raman_tools 의 busy/범위 에러들) 규약을 맞춘다.
    max_measurements = 40
    max_dose = 60.0                   # 기준초 누계
    max_virtual_time_s = 900.0

    def __init__(self, world: VirtualWorld, z0: float = 0.0):
        self.world = world
        bx0, bx1, by0, by1 = world.bounds_mm()
        # 시작 위치는 월드 중앙. 에이전트가 어디서 출발하는지가 시드마다 달라지면
        # 탐색 난이도가 갈리므로 고정한다.
        self.x = 0.5 * (bx0 + bx1)
        self.y = 0.5 * (by0 + by1)
        self.z = float(z0)
        self.vx = self.vy = 1.0
        self.vz = 0.1

        self.laser_power = 0.0        # 마지막으로 '적용된' 파워(%)
        self.laser_armed = False      # ND 가 측정 위치에 있는가
        self.laser_on = False         # SSPW 발진 중인가

        self.virtual_time_s = 0.0     # 실제로 자지 않고 누적만 한다(설계문서 §8)
        self.moves = 0
        self.log: list[dict] = []     # 측정 이력 — 채점이 읽는다

        self.total_dose = 0.0

    def spend(self, seconds: float) -> None:
        self.virtual_time_s += float(seconds)

    def budget_block(self, power_pct: float, exposure_s: float, n: int = 1) -> str | None:
        """이 측정을 허용해도 되는가. 막아야 하면 **사유 문장**, 괜찮으면 None.

        문장은 그대로 에이전트에게 간다 — 무엇이 소진됐고 그래서 무엇을 해야 하는지까지
        적는다. '거부됐다'만 알려 주면 모델은 파라미터만 바꿔 계속 재시도한다.
        """
        if len(self.log) >= self.max_measurements:
            return (f"Measurement budget exhausted: you have already taken "
                    f"{len(self.log)} of {self.max_measurements} allowed measurements. "
                    f"No further spectra can be acquired in this episode. Report your best "
                    f"answer now using the JSON block described in the task.")
        d = self.world._dose_of(power_pct, exposure_s) * max(1, int(n))
        if self.total_dose + d > self.max_dose:
            return (f"Laser dose budget exhausted: this episode allows {self.max_dose:.0f} "
                    f"reference-seconds in total and {self.total_dose:.1f} have been used; "
                    f"this measurement would add {d:.1f}. Nothing was measured and the sample "
                    f"was not exposed. Either use a much lower power/exposure, or report your "
                    f"best answer now.")
        if self.virtual_time_s > self.max_virtual_time_s:
            return (f"Time budget exhausted: {self.virtual_time_s:.0f} s of instrument time "
                    f"used out of {self.max_virtual_time_s:.0f} s. Report your best answer now.")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 스테이지
# ══════════════════════════════════════════════════════════════════════════════

class VirtualStage:
    """필요한 메서드 5개: get_position move_absolute move_relative get_velocity set_velocity."""

    MAX_SPEED_XY = 5.0
    MAX_SPEED_Z = 0.1

    def __init__(self, rig: VirtualRig):
        self.rig = rig
        self.dead = False              # raman_tools._stage_unavailable() 가 본다
        self.dead_reason = ""

    def get_position(self):
        r = self.rig
        return (r.x, r.y, r.z)

    def move_absolute(self, x=None, y=None, z=None, a=None, wait=True):
        r = self.rig
        nx = r.x if x is None else float(x)
        ny = r.y if y is None else float(y)
        nz = r.z if z is None else float(z)
        nz = max(STAGE_MIN_Z, min(STAGE_MAX_Z, nz))
        # 이동 시간은 가상 시계에만 더한다 — 실제로 자면 벤치가 몇 시간짜리가 된다.
        dist = float(np.hypot(nx - r.x, ny - r.y))
        r.spend(dist / max(r.vx, 1e-6) + abs(nz - r.z) / max(r.vz, 1e-6) + 0.2)
        r.x, r.y, r.z = nx, ny, nz
        r.moves += 1
        return True

    def move_relative(self, dx=0.0, dy=0.0, dz=0.0, da=0.0, wait=True):
        r = self.rig
        return self.move_absolute(r.x + float(dx), r.y + float(dy), r.z + float(dz), wait=wait)

    def get_velocity(self):
        r = self.rig
        return {"ok": True, "x_speed_mm_s": r.vx, "y_speed_mm_s": r.vy, "z_speed_mm_s": r.vz}

    def set_velocity(self, x=None, y=None, z=None, a=None):
        r = self.rig
        if x is not None:
            r.vx = max(-self.MAX_SPEED_XY, min(self.MAX_SPEED_XY, float(x)))
        if y is not None:
            r.vy = max(-self.MAX_SPEED_XY, min(self.MAX_SPEED_XY, float(y)))
        if z is not None:
            r.vz = max(-self.MAX_SPEED_Z, min(self.MAX_SPEED_Z, float(z)))
        return True

    # 재연결 경로가 부르는 것들 (도구가 실패하지 않도록 no-op 로 둔다)
    def disconnect(self):
        return True

    def free_session(self):
        return True

    def mark_dead(self, reason=""):
        self.dead, self.dead_reason = True, str(reason)


# ══════════════════════════════════════════════════════════════════════════════
# 레이저
# ══════════════════════════════════════════════════════════════════════════════

class VirtualLaser:
    """필요한 메서드 4개 + 속성 5개.

    `_power_set` 이 핵심이다 — raman_tools._beam_state() 가 이 값으로 '지금 켜면 측정빔이
    나가는가 가이드빔이 나가는가'를 판정한다. set_guide_beam() 이 이걸 내리는 것까지
    실장비와 같게 맞춰야, 측정 후 카메라가 다시 보이는 동작이 재현된다.
    """

    ND_MIN_PCT = 0.004
    ND_MAX_PCT = 100.0

    def __init__(self, rig: VirtualRig):
        self.rig = rig
        self.power_pct = None
        self.is_on = False
        self._power_set = False
        self.ser = None                 # 재연결 경로가 getattr 로 본다

    def set_power(self, percent):
        p = float(percent)
        if not (self.ND_MIN_PCT <= p <= self.ND_MAX_PCT):
            return False
        self.power_pct = p
        self._power_set = True
        self.rig.laser_power = p
        self.rig.laser_armed = True
        self.rig.spend(0.15)            # ND 필터 정착
        return True

    def laser_on(self):
        self.is_on = True
        self.rig.laser_on = True
        return True

    def laser_off(self):
        self.is_on = False
        self.rig.laser_on = False
        return True

    def set_guide_beam(self):
        # 측정빔 무장 해제 — 실드라이버와 같이 _power_set 을 내린다.
        self._power_set = False
        self.rig.laser_armed = False
        self.rig.spend(0.15)
        return True


# ══════════════════════════════════════════════════════════════════════════════
# CCD
# ══════════════════════════════════════════════════════════════════════════════

class VirtualCCD:
    """가장 큰 표면(37개)이지만 실제 로직이 필요한 것은 4개뿐이다:
    start_acquisition_cycle / get_acquired_data / get_exposure_time / get_status.
    나머지는 값을 기억만 하면 된다 — get_ccd_info 가 그 값을 그대로 돌려주기 때문이다.
    """

    def __init__(self, rig: VirtualRig):
        self.rig = rig
        # 실드라이버(andor_ccd_interface)의 초기값과 같게 맞춘다.
        self.Nx, self.Ny = N_BINS, 256
        self.Nx_ro, self.Ny_ro = N_BINS, 1
        self.aq_mode = "single"
        self.ro_mode = "FULL_VERTICAL_BINNING"
        self.num_acc = 1
        self.num_kin = 1
        self.exposure_time = 1.0
        self.trigger_mode = "internal"
        self.shutter_mode = "auto"
        self.shutter_explicit = False
        self.cooler_on = True
        self.temperature = -60
        self.preamp_gain_i = 0
        self.preamp_gains = [1.0, 2.0, 4.0]
        self.VSSpeeds = [1.7, 3.3]
        self.HSSpeeds_Conventional = [[0.05, 1.0]]
        self.em_mode = False
        self.em_gain = 0
        self.output_amp = 1
        self.hbin = 1
        self.ro_single_track_center = None
        self.ro_single_track_width = None
        self._calibrator = None         # 파수축은 아래 _cal_block() 이 직접 싣는다
        self._status = "IDLE"
        self._pending = None            # kinetic 경로용 버퍼

    # ── 조회 ────────────────────────────────────────────────────────────────
    def get_status(self):
        return self._status

    def get_exposure_time(self):
        return float(self.exposure_time)

    def get_temperature(self):
        return int(self.temperature)

    def get_temperature_status(self):
        return "DRV_TEMPERATURE_STABILIZED" if self.cooler_on else "DRV_TEMPERATURE_OFF"

    def get_current_hbin(self):
        return int(self.hbin)

    # ── 설정 (값만 기억한다) ────────────────────────────────────────────────
    def set_exposure_time(self, exposure):
        self.exposure_time = float(exposure)
        return True

    def set_aq_single_scan(self, exposure=None):
        self.aq_mode = "single"
        if exposure is not None:
            self.exposure_time = float(exposure)
        return True

    def set_aq_accumulate_scan(self, exposure_time=None, num_acc=None):
        self.aq_mode = "accumulate"
        if exposure_time is not None:
            self.exposure_time = float(exposure_time)
        if num_acc is not None:
            self.num_acc = int(num_acc)
        return True

    def set_aq_kinetic_scan(self, exposure_time=None, num_acc=None, num_kin=None,
                            kinetic_cycle_time=None, **kw):
        self.aq_mode = "kinetic"
        if exposure_time is not None:
            self.exposure_time = float(exposure_time)
        if num_acc is not None:
            self.num_acc = int(num_acc)
        if num_kin is not None:
            self.num_kin = int(num_kin)
        return True

    def set_aq_run_till_abort_scan(self):
        self.aq_mode = "run_till_abort"
        return True

    def set_num_accumulations(self, n):
        self.num_acc = int(n)
        return True

    def set_num_kinetics(self, n):
        self.num_kin = int(n)
        return True

    def set_ro_full_vertical_binning(self, hbin=1):
        self.ro_mode = "FULL_VERTICAL_BINNING"
        self.hbin = int(hbin or 1)
        self.Nx_ro, self.Ny_ro = int(self.Nx / self.hbin), 1
        return True

    def set_ro_single_track(self, center=None, width=None, hbin=1, **kw):
        self.ro_mode = "SINGLE_TRACK"
        self.ro_single_track_center = center
        self.ro_single_track_width = width
        self.hbin = int(hbin or 1)
        self.Nx_ro, self.Ny_ro = int(self.Nx / self.hbin), 1
        return True

    def set_ro_image_mode(self, hbin=1, **kw):
        # 실드라이버(andor_ccd_interface)의 표기는 'IMG' 다. 'IMAGE' 로 쓰면
        # raman_tools._RO_MODE_TO_ARG 매핑에 안 걸려 조용히 'fvb' 로 폴백된다
        # (실측: set_ccd_image_flip 이 image 모드인데도 거부됐다).
        self.ro_mode = "IMG"
        self.hbin = int(hbin or 1)
        self.Nx_ro, self.Ny_ro = int(self.Nx / self.hbin), self.Ny
        return True

    def set_trigger_mode(self, mode):
        self.trigger_mode = str(mode)
        return True

    def set_shutter_auto(self):
        self.shutter_mode = "auto"
        return True

    def set_shutter_open(self, permanently=True):
        self.shutter_mode = "open"
        return True

    def set_shutter_close(self):
        self.shutter_mode = "close"
        return True

    def set_cooler(self, on):
        self.cooler_on = bool(on)
        return True

    def set_temperature(self, temp):
        self.temperature = int(temp)
        return True

    def set_preamp_gain(self, index):
        self.preamp_gain_i = int(index)
        return True

    def set_vs_speed(self, index):
        return True

    def set_hs_speed_conventional(self, index):
        return True

    def set_image_flip(self, hflip=False, vflip=False):
        return True

    def free_internal_memory(self):
        return True

    def abort_acquisition(self):
        self._status = "IDLE"
        return True

    def send_software_trigger(self):
        return True

    def close(self):
        return True

    # ── 촬영 ────────────────────────────────────────────────────────────────
    def _cal_block(self) -> dict:
        ax = _axis()
        return {"calibrated": True,
                "raman_shift_cm-1": [float(v) for v in ax],
                "wavelength_nm": [float(785.0 / (1.0 - v * 785.0e-7)) for v in ax],
                "laser_nm": 785.0}

    def _shoot(self, n_frames: int = 1):
        """실제로 월드를 조사한다. **여기서만** 손상이 누적된다."""
        r = self.rig
        expo = float(self.exposure_time)
        n_acc = max(1, int(self.num_acc))

        # 셔터가 닫혀 있거나 레이저가 꺼져 있으면 빛이 안 들어온다 = 다크 프레임.
        # 도구가 laser_on() 을 부르기 전에 촬영하는 경로가 실제로 있으므로 재현해야 한다.
        dark = (self.shutter_mode == "close") or (not r.laser_on) or (not r.laser_armed)
        power = 0.0 if dark else float(r.laser_power)

        # 예산은 시료에 빔이 들어가기 **전에** 본다. 다크 프레임은 조사가 없으므로 면제한다
        # (셔터를 닫고 배경만 재는 것은 정당한 절차인데 예산을 깎을 이유가 없다).
        if not dark:
            why = r.budget_block(power, expo, n_acc * max(1, n_frames))
            if why:
                raise RuntimeError(why)

        frames = []
        for _ in range(max(1, n_frames)):
            res = r.world.measure(r.x, r.y, power, expo, accumulations=n_acc)
            rec = dict(res["record"])
            rec["dark"] = dark
            rec["virtual_time_s"] = r.virtual_time_s
            r.log.append(rec)
            if not dark:
                r.total_dose += rec["dose_shot"]
            r.spend(expo * n_acc + 0.3)         # 노출 + 판독
            frames.append(res["data"])
        return frames

    def start_acquisition_cycle(self, trigger_mode_str=None, timeout_ms=None):
        """single / accumulate 경로. 도구가 기대하는 dict 를 돌려준다."""
        try:
            data = self._shoot(1)[0]
        except Exception as e:                  # 도구는 dict 의 'error' 를 본다
            return {"error": f"{type(e).__name__}: {e}"}
        self._status = "IDLE"
        out = {"intensity": [float(v) for v in data]}
        out.update(self._cal_block())
        return out

    def prepare_acquisition(self):
        self._pending = None
        return True

    def start_acquisition(self):
        """kinetic 경로. 도구가 get_status() 로 IDLE 을 기다린 뒤 get_acquired_data() 를 부른다."""
        self._pending = self._shoot(max(1, int(self.num_kin)))
        self._status = "IDLE"
        return True

    def get_acquired_data(self):
        if self._pending is None:
            return None
        # 도구가 기대하는 모양: (num_kin, Ny_ro, Nx_ro)
        arr = np.stack([f.reshape(1, -1) for f in self._pending], axis=0)
        self._pending = None
        return arr


# ══════════════════════════════════════════════════════════════════════════════
# 카메라
# ══════════════════════════════════════════════════════════════════════════════

class VirtualCamera:
    """필요한 메서드 4개. get_latest_frame() 이 월드를 렌더한 BGR 프레임을 돌려준다.

    `analyze_microscope_image` 는 이 프레임을 vision.to_view_bgr 로 정규화한 뒤 PNG 로
    인코딩해 에이전트에게 **이미지 블록으로** 넘긴다 — 즉 여기서 만든 그림을 gemma4 가
    실제로 본다.
    """

    def __init__(self, rig: VirtualRig):
        self.rig = rig
        self.is_streaming = False
        self.exposure_ms = 20.0
        # set_camera_auto_exposure 가 _camera.TUCAMOPEN.hIdxTUCam 을 읽는다 —
        # 단순 bool 로 두면 AttributeError 가 난다(실측).
        import types as _t
        self.TUCAMOPEN = _t.SimpleNamespace(hIdxTUCam=0)

    def start_stream(self):
        self.is_streaming = True
        return True

    def stop_stream(self):
        self.is_streaming = False
        return True

    def set_exposure(self, ms):
        self.exposure_ms = float(ms)
        return True

    def get_latest_frame(self):
        r = self.rig
        r.spend(0.05)
        frame = r.world.render(r.x, r.y, VIEW_W, VIEW_H)
        # 가이드빔이 켜져 있으면 화면 한가운데 밝은 스팟이 보인다.
        # (측정빔 모드에서는 빔이 분광기로 가므로 카메라에는 아무것도 안 보인다.)
        if r.laser_on and not r.laser_armed:
            frame = self._add_guide_spot(frame)
        return frame

    def _add_guide_spot(self, frame):
        """가이드빔 스팟을 그려 넣는다. 초점에서 멀수록 크고 흐려진다.

        [왜 필요한가 — 실측으로 걸린 구멍]
        run_autofocus 는 vision.capture_laser_diff() 로 '레이저 OFF 프레임'과
        'ON 프레임'의 차분을 떠서 스팟 **면적**을 재고, 그것이 최소가 되는 Z 를 찾는다.
        스팟을 안 그리면 면적이 늘 0 이라 도구가
        "every sample came back with zero spot area" 로 실패한다(실제로 그랬다).
        에이전트는 그 실패를 이해할 방법이 없어 같은 도구를 반복하기 쉽다.

        [초점이 신호에는 영향을 주지 않는다 — 의도한 것이다]
        결합하면 파워·노출·위치 말고 **네 번째 숨은 변수**가 생겨서, 에이전트가 Z 를
        건드린 에피소드가 우리가 재려는 것과 무관한 이유로 무너진다. 설계는 세 변수다.
        시작 Z(=0)가 곧 초점면이라 오토포커스는 '해도 그만'인 도구로 남는다 —
        도구 표면이 온전해지고, 아무것도 망가뜨리지 않는다.
        """
        z_focus = 0.0
        defocus = abs(float(self.rig.z) - z_focus)
        sigma = 6.0 + 220.0 * defocus            # px — 초점에서 가장 작다
        yy, xx = np.mgrid[0:frame.shape[0], 0:frame.shape[1]]
        g = np.exp(-(((xx - frame.shape[1] / 2) ** 2 + (yy - frame.shape[0] / 2) ** 2)
                     / (2.0 * sigma ** 2)))
        out = frame.astype(np.float32) + (200.0 * g)[..., None]
        return np.clip(out, 0, 255).astype(np.uint8)

    def close(self):
        self.is_streaming = False
        return True


# ══════════════════════════════════════════════════════════════════════════════
# 주입
# ══════════════════════════════════════════════════════════════════════════════

class _VirtualManager:
    """`hardware_manager.HardwareManager` 자리에 들어가는 최소 대역품.

    [왜 필요한가 — 실측]
    get_hardware_status / reconnect_hardware 는 raman_tools 전역 핸들이 아니라
    **매니저**를 읽는다. 그런데 이 개발 PC 에서는 backend.hardware_manager 를 import 조차
    할 수 없다 — Andor SDK(pyAndorSDK2)가 실장비에만 깔리기 때문이다. 그대로 두면 두 도구가
    항상 `HardwareManager unavailable: cannot import name 'AndorCCD'` 를 돌려주는데,
    에이전트 입장에서는 장비가 고장난 것처럼 보여 엉뚱한 복구를 시도하게 된다.

    가상 환경에서는 '장비가 늘 정상'이 참이므로, 그 사실을 말해 주는 매니저를 세운다.
    """

    def __init__(self, stage, laser, ccd, camera):
        import threading
        # 원본을 따로 보관한다 — reconnect_hardware 는 재초기화 전에 _teardown_component()
        # 로 `mgr.<comp> = None` 을 먼저 실행한다. 처음엔 _init_*() 가 self.camera 를
        # 그대로 돌려주게 짰는데, 그 시점엔 이미 None 이라 네 장비가 전부 None 으로 재주입됐다
        # (실측: reconnect 직후 stop_camera_stream 이 "Camera is not initialized").
        self._devices = {"stage": stage, "laser": laser, "ccd": ccd, "camera": camera}
        self.stage, self.laser, self.ccd, self.camera = stage, laser, ccd, camera
        self._locks = {k: threading.RLock() for k in ("stage", "laser", "ccd", "camera")}

    def component_lock(self, component: str):
        return self._locks[component]

    # reconnect_hardware 가 부르는 재초기화. 가상 장비는 끊길 일이 없으므로 원본을 되꽂는다.
    def _reinit(self, name):
        setattr(self, name, self._devices[name])
        return self._devices[name]

    def _init_stage(self):
        return self._reinit("stage")

    def _init_laser(self):
        return self._reinit("laser")

    def _init_ccd(self):
        return self._reinit("ccd")

    def _init_camera(self):
        return self._reinit("camera")


def _install_manager_stub(stage, laser, ccd, camera):
    """`backend.hardware_manager` 를 에피소드 동안만 가상 매니저로 바꿔 끼운다.

    raman_tools 는 이 모듈을 **함수 안에서** import 하므로 sys.modules 치환이 통한다.
    실장비 PC 에서 실수로 돌려도 원본을 detach() 가 되돌려 놓는다.
    """
    import sys
    import types

    mod = types.ModuleType("backend.hardware_manager")
    mgr = _VirtualManager(stage, laser, ccd, camera)
    mod.get_manager = lambda: mgr                      # type: ignore[attr-defined]
    mod.HardwareManager = _VirtualManager              # type: ignore[attr-defined]
    saved = {"backend.hardware_manager": sys.modules.get("backend.hardware_manager")}
    sys.modules["backend.hardware_manager"] = mod

    # set_camera_auto_exposure 만은 드라이버 객체를 우회해 TuCam SDK 를 직접 부른다
    # (TUCAM_Capa_SetValue(_camera.TUCAMOPEN.hIdxTUCam, ...)). 그 DLL 은 실장비 PC 에만
    # 있으므로 여기서도 대역품을 세운다 — 45개 중 하나만 늘 에러를 뱉으면 에이전트가
    # 장비 이상으로 오해한다.
    tucam = types.ModuleType("backend.TuCam.TUCam")
    tucam.TUCAM_Capa_SetValue = lambda *a, **k: 0        # type: ignore[attr-defined]
    tucam.TUCAM_IDCAPA = types.SimpleNamespace(          # type: ignore[attr-defined]
        TUIDC_ATEXPOSURE=types.SimpleNamespace(value=0))
    saved["backend.TuCam.TUCam"] = sys.modules.get("backend.TuCam.TUCam")
    sys.modules["backend.TuCam.TUCam"] = tucam
    return saved


_saved_manager_module = None
_manager_stub_installed = False


def attach(world: VirtualWorld) -> VirtualRig:
    """네 가짜 드라이버를 만들어 raman_tools 에 주입한다. rig 를 돌려준다.

    이 한 줄 뒤로는 45개 하드웨어 도구가 전부 이 월드 위에서 돈다.
    """
    global _saved_manager_module, _manager_stub_installed
    from backend.hw_tools import raman_tools as T

    rig = VirtualRig(world)
    stage, laser = VirtualStage(rig), VirtualLaser(rig)
    ccd, camera = VirtualCCD(rig), VirtualCamera(rig)
    T.init_hardware(stage=stage, laser=laser, ccd=ccd, camera=camera)

    _saved_manager_module = _install_manager_stub(stage, laser, ccd, camera)
    _manager_stub_installed = True
    return rig


def detach() -> None:
    """주입을 되돌린다. 에피소드가 끝나면 반드시 부를 것 — 전역 핸들이 남으면
    다음 에피소드가 **앞 에피소드의 월드**를 조사하게 된다."""
    global _saved_manager_module, _manager_stub_installed
    import sys
    from backend.hw_tools import raman_tools as T

    T.init_hardware(stage=None, laser=None, ccd=None, camera=None)
    if _manager_stub_installed:
        for name, mod in (_saved_manager_module or {}).items():
            if mod is not None:
                sys.modules[name] = mod
            else:
                sys.modules.pop(name, None)
        _saved_manager_module = None
        _manager_stub_installed = False
