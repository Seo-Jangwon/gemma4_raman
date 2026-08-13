"""
가이드빔 + 카메라 스트리밍 통합 로컬 오토포커스 모듈
- LaserController  : USE_laser_with_power.py 의 클래스 재사용
- StreamingTUCam   : USE_camera_stream.py 의 클래스 재사용
ESC 키로 종료
"""

import sys
import os
import cv2
import time
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

# 같은 디렉터리의 모듈 임포트
sys.path.append(os.path.dirname(__file__))
from backend.tools.hw_tools.hao.USE_laser_with_power import LaserController
from backend.tools.hw_tools.hao.USE_camera_stream import StreamingTUCam
from backend.tools.hw_tools.hao.USE_stage_test import TangoController
from backend.tools.hw_tools.config import CAMERA_WIDTH as STREAM_WIDTH, CAMERA_HEIGHT as STREAM_HEIGHT  # noqa: E402
# 스팟 면적(오토포커스 목적함수)과 프레임 전처리는 공용 모듈을 쓴다(2026-07-30).
# 예전에는 이 파일과 camera_tools.run_autofocus 가 같은 알고리즘을 각자 구현했고,
# 목적함수가 갈라지면 같은 시료에서 서로 다른 Z 에 수렴한다.
from backend.service.vision import vision  # noqa: E402

COARSE_STEP  = 0.010   # mm (10µm)
COARSE_RANGE = 0.050   # mm (±50µm, 총 11포인트)


class AutoFocusLocal:
    """
    가이드빔을 쏘면서 카메라(1060x800)로 실시간 스트리밍하는 클래스.
    레이저/카메라 제어는 각각의 원본 클래스 인스턴스에 위임합니다.
    """

    def __init__(self, laser_port: str = 'COM4', exposure_ms: float = 10.0):
        print("=== AutoFocusLocal 초기화 ===")

        # 레이저 컨트롤러 (USE_laser_with_power.py)
        self.laser = LaserController(port=laser_port)

        # 카메라 스트리머 (USE_camera_stream.py)
        self.camera = StreamingTUCam(exposure_ms=exposure_ms)

        self._exposure_ms = exposure_ms

        # 스테이지 컨트롤러 (USE_stage_test.py)
        self.stage = TangoController()
        self.stage.load_dll()
        self.stage.create_session()
        self.stage.connect()

    # ------------------------------------------------------------------
    # 차분 이미지 생성 헬퍼
    # ------------------------------------------------------------------
    def _capture_diff(self):
        """
        레이저 OFF → 레퍼런스 취득 → 레이저 ON → 레이저 프레임 취득.
        구현은 vision.capture_laser_diff 단일 출처 — camera_tools.run_autofocus 가
        쓰는 것과 **같은 함수**라 두 오토포커스가 같은 Z 로 수렴한다.

        반환: (ref, laser_frame, diff_absdiff, diff_clip, spot_area)
          - diff_absdiff : cv2.absdiff 버전 (양방향 차이, 조명 변화에도 반응)
          - diff_clip    : clip(laser - ref, 0) 버전 (레이저보다 밝아진 영역만)
        """
        d = vision.capture_laser_diff(self.camera, self.laser, n_avg=3)
        return (d["ref"], d["laser_frame"], d["diff_absdiff"],
                d["diff_clip"], d["area_px"])

    # ------------------------------------------------------------------
    # 가이드빔 제어
    # ------------------------------------------------------------------
    def guide_beam_on(self):
        """가이드빔 모드 활성화 후 레이저 ON"""
        self.laser.set_guide_beam()   # 필터를 가이드빔 위치로 이동
        
        self.laser.laser_on()         # 레이저 출력 ON

    def guide_beam_off(self):
        """레이저 OFF"""
        self.laser.laser_off()

    # ------------------------------------------------------------------
    # 카메라 스트리밍
    # ------------------------------------------------------------------
    def autofocus_local(self):
        """Stage 1: 스테이지 연결 확인 및 현재 Z 위치 출력"""
        pos = self.stage.get_position()
        if pos is None:
            print("[ERROR] 스테이지 위치 조회 실패")
            return
        x, y, z, a = pos
        print(f"[Stage] 현재 위치 - X={x:.4f} mm, Y={y:.4f} mm, Z={z:.4f} mm, A={a:.4f}")

        print(f"스트리밍 해상도: {STREAM_WIDTH}x{STREAM_HEIGHT}")
        print("ESC: 종료 | E: 노출 증가 | D: 노출 감소")

        # 1. 가이드빔 필터 세팅
        print("가이드빔 필터 세팅 중...")
        self.laser.set_guide_beam()

        # 2. 카메라 스트리밍 시작
        self.camera.start_stream()

        # ==========================================
        # 🎯 적응형 힐클라이밍 + 역대 최솟값 추적기
        # ==========================================
        MAX_STEPS   = 100
        step_size   = 0.030    # [초기 보폭] (Coarse)
        min_step    = 0.001    # [최소 보폭] (Fine)
        
        sweep_state = 'init'   
        
        # 힐클라이밍 진행용 지역 변수
        best_z      = z
        best_area   = float('inf')
        direction   = 1        
        step_count  = 0
        
        # 역대 최솟값(Global Minimum)
        global_best_area = float('inf')
        global_best_z = z
        
        phase       = 'sweep'
        diff_clip_disp = None

        print(f"\n[오토포커스] 탐색 시작 (역대 최솟값 자동 저장 기능 활성화)")
        
        try:
            while True:
                # ── 1. 화면 업데이트 (항상 실행) ──
                frame = self.camera.get_latest_frame()
                if frame is not None:
                    disp = vision.to_view_bgr(frame, STREAM_WIDTH, STREAM_HEIGHT)

                    phase_label = f"[AF/{sweep_state}]" if phase == 'sweep' else "[Stream]"
                    pos_disp = self.stage.get_position()
                    cur_z = pos_disp[2] if pos_disp else z
                    
                    cv2.putText(disp, f"{phase_label} Z={cur_z:.4f}mm | Exp:{self._exposure_ms:.1f}ms",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.imshow("AutoFocus Local - Guide Beam Stream", disp)

                # ── 2. 오토포커스 (Sweep) 로직 ──
                if phase == 'sweep':
                    # 사진 찍고 레이저 면적 계산
                    _, _, _, diff_clip, spot_area = self._capture_diff()
                    
                    # 🌟 [핵심] 노이즈(면적 0)를 제외하고, 역대 최솟값이면 무조건 저장
                    if 0 < spot_area < global_best_area:
                        global_best_area = spot_area
                        global_best_z = cur_z

                    print(f"  [{sweep_state}] Z={cur_z:.4f} mm | 면적={spot_area:4d} px | (역대최소: {global_best_area:4d} px)")

                    if diff_clip is not None:
                        diff_clip_disp = cv2.resize(cv2.cvtColor(diff_clip, cv2.COLOR_GRAY2BGR),
                                                    (STREAM_WIDTH, STREAM_HEIGHT))
                        cv2.putText(diff_clip_disp, f"Area: {spot_area} px (Best: {global_best_area})",
                                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

                    # 상태 머신 시작
                    if sweep_state == 'init':
                        best_area = spot_area
                        best_z = cur_z
                        self.stage.move_absolute(x, y, best_z + (direction * step_size), a)
                        time.sleep(0.3) #  스테이지 진동/이동 안정화 대기
                        sweep_state = 'check'

                    elif sweep_state == 'check':
                        step_count += 1
                        
                        if spot_area < best_area:
                            # 좋아짐 -> 같은 방향 전진
                            best_area = spot_area
                            best_z = cur_z
                            self.stage.move_absolute(x, y, best_z + (direction * step_size), a)
                            time.sleep(0.3)
                        else:
                            # 나빠짐 -> 방향 반전 및 보폭 축소
                            print(f"    -> 면적 증가. 방향 전환 및 보폭 축소 ({step_size*1000:.0f}µm -> {step_size*500:.0f}µm)")
                            direction *= -1       
                            step_size /= 2.0
                            
                            if step_size < min_step or step_count >= MAX_STEPS:
                                sweep_state = 'done'
                            else:
                                next_z = best_z + (direction * step_size)
                                self.stage.move_absolute(x, y, next_z, a)
                                time.sleep(0.3)

                    elif sweep_state == 'done':
                        # 탐색 로직이 어디서 끝났든 상관없이, 우리가 기록해둔 "역대 최솟값" 좌표로 꽂아버림
                        print(f"\n[탐색 종료] 가장 선명했던 역대 최솟값 위치 Z={global_best_z:.4f} mm 로 최종 귀환")
                        self.stage.move_absolute(x, y, global_best_z, a)
                        time.sleep(0.5) # 이동 완료 넉넉히 대기
                        
                        # 최종 확인용 사진 1장 찰칵
                        _, _, _, diff_clip, final_area = self._capture_diff()
                        print(f"   -> [최종 확인] 도착 후 실제 측정 면적: {final_area} px (목표: {global_best_area} px)")
                        
                        phase = 'stream'
                        print("[Stream] 라이브 스트림 유지 중. (수동 측정: F 키, 레이저 격발: L 키)")

                # ── 3. 화면 업데이트 (차분 이미지) ──
                if diff_clip_disp is not None:
                    cv2.imshow("Diff - clip", diff_clip_disp)

                # ── 4. 키보드 입력 처리 ──
                key = cv2.waitKey(1) & 0xFF
                if key == 27:   # ESC
                    break
                elif key in (ord('f'), ord('F')) and phase == 'stream':
                    _, _, _, diff_clip, spot_area = self._capture_diff()
                    if diff_clip is not None:
                        diff_clip_disp = cv2.resize(cv2.cvtColor(diff_clip, cv2.COLOR_GRAY2BGR),
                                                    (STREAM_WIDTH, STREAM_HEIGHT))
                        cv2.putText(diff_clip_disp, f"Manual Check | Area: {spot_area} px",
                                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                        print(f"[Manual] spot_area = {spot_area} px")
                elif key in (ord('e'), ord('E')):
                    self._exposure_ms += 5.0
                    self.camera.set_exposure(self._exposure_ms)
                elif key in (ord('d'), ord('D')):
                    self._exposure_ms = max(1.0, self._exposure_ms - 5.0)
                    self.camera.set_exposure(self._exposure_ms)
                elif key in (ord('l'), ord('L')):
                    print("⚡ 수동 레이저 격발!")
                    self.laser.laser_on()

        except KeyboardInterrupt:
            pass
        finally:
            self.close()
   
    # ------------------------------------------------------------------
    # 정리
    # ------------------------------------------------------------------
    def close(self):
        print("종료 중...")
        self.guide_beam_off()
        self.camera.close()
        cv2.destroyAllWindows()
        self.laser.close()
        self.stage.disconnect()
        self.stage.free_session()
        print("AutoFocusLocal 종료 완료.")


# ----------------------------------------------------------------------
# 단독 실행 진입점
# ----------------------------------------------------------------------
def main():
    port_input = input("레이저 COM 포트 (기본값: COM4): ").strip()
    port = port_input.upper() if port_input else 'COM4'

    exp_input = input("초기 노출 시간 ms (기본값: 10.0): ").strip()
    try:
        exposure = float(exp_input) if exp_input else 10.0
    except ValueError:
        exposure = 10.0

    af = AutoFocusLocal(laser_port=port, exposure_ms=exposure)
    af.autofocus_local()


if __name__ == "__main__":
    main()
