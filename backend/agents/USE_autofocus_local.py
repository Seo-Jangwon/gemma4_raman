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
import numpy as np

# 같은 디렉터리의 모듈 임포트
sys.path.append(os.path.dirname(__file__))
from USE_laser_with_power import LaserController
from USE_camera_stream import StreamingTUCam
from USE_stage_test import TangoController

STREAM_WIDTH  = 1060
STREAM_HEIGHT = 800

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
    def _flush_frames(self, n: int = 3):
        """레이저 상태 변경 후 카메라 버퍼에 남은 이전 프레임 버리기"""
        for _ in range(n):
            self.camera.get_latest_frame()

    def _to_uint8(self, frame):
        """프레임을 uint8 grayscale로 정규화"""
        img = frame.copy()
        if img.dtype == np.uint16:
            img = (img / 256).astype(np.uint8)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def _capture_diff(self):
        """
        레이저 OFF → 레퍼런스 취득 → 레이저 ON → 레이저 프레임 취득
        반환: (ref, laser_frame, diff_absdiff, diff_clip)
          - diff_absdiff : cv2.absdiff 버전 (양방향 차이, 조명 변화에도 반응)
          - diff_clip    : clip(laser - ref, 0) 버전 (레이저보다 밝아진 영역만)
        """
        # 1. 레이저 OFF → ref 취득
        self.laser.laser_off()
        self._flush_frames(3)
        ref_frames = [self._to_uint8(self.camera.get_latest_frame()) for _ in range(3)
                      if self.camera.get_latest_frame() is not None]
        ref = np.mean(ref_frames, axis=0).astype(np.uint8) if ref_frames else None

        # 2. 레이저 ON → laser 프레임 취득
        self.laser.laser_on()
        self._flush_frames(3)
        laser_frames = [self._to_uint8(self.camera.get_latest_frame()) for _ in range(3)
                        if self.camera.get_latest_frame() is not None]
        laser_frame = np.mean(laser_frames, axis=0).astype(np.uint8) if laser_frames else None

        if ref is None or laser_frame is None:
            return None, None, None, None

        # 3-A. absdiff 버전: 양방향 절댓값 차이
        diff_absdiff = cv2.absdiff(laser_frame, ref)

        # 3-B. clip 버전: 레이저로 인해 밝아진 영역만 (음수 = 0)
        diff_clip = np.clip(laser_frame.astype(np.int16) - ref.astype(np.int16), 0, 255).astype(np.uint8)

        # 4. 면적 계산: Otsu threshold로 노이즈/배경 제거 후 레이저 스팟 픽셀 수
        blurred = cv2.GaussianBlur(diff_clip, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        spot_area = int(np.count_nonzero(binary))

        return ref, laser_frame, diff_absdiff, diff_clip, spot_area

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
        step_size   = 0.030    # [초기 보폭] 50µm (Coarse)
        min_step    = 0.001    # [최소 보폭] 5µm (Fine)
        
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
                    disp = frame.copy()
                    if disp.dtype == np.uint16:
                        disp = (disp / 256).astype(np.uint8)
                    if len(disp.shape) == 2:
                        disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
                    disp = cv2.resize(disp, (STREAM_WIDTH, STREAM_HEIGHT))

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
        # Hill-climbing
        # ==========================================
        MAX_STEPS   = 50
        step_size   = 0.020    # [초기 보폭] 50µm (Coarse)
        min_step    = 0.001    # [최소 보폭] 5µm (Fine - 이 이하면 탐색 종료)
        
        sweep_state = 'init'   # init -> move -> check -> done
        best_z      = z
        best_area   = float('inf')
        direction   = 1        # 1: 위로, -1: 아래로
        reversal_count = 0     # 방향을 몇 번 바꿨는지 기록
        step_count  = 0
        phase       = 'sweep'

        diff_clip_disp = None

        print(f"\n[오토포커스] 적응형 탐색 시작 (초기 스텝: {step_size*1000:.0f}µm)")
        
        try:
            while True:
                # ── 1. 화면 업데이트 (항상 실행) ──
                frame = self.camera.get_latest_frame()
                if frame is not None:
                    disp = frame.copy()
                    if disp.dtype == np.uint16:
                        disp = (disp / 256).astype(np.uint8)
                    if len(disp.shape) == 2:
                        disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
                    disp = cv2.resize(disp, (STREAM_WIDTH, STREAM_HEIGHT))

                    phase_label = f"[AF/{sweep_state}]" if phase == 'sweep' else "[Stream]"
                    pos_disp = self.stage.get_position()
                    cur_z = pos_disp[2] if pos_disp else best_z
                    
                    cv2.putText(disp, f"{phase_label} Z={cur_z:.4f}mm | Exp:{self._exposure_ms:.1f}ms",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.imshow("AutoFocus Local - Guide Beam Stream", disp)

                # ── 2. 오토포커스 (Sweep) 로직 ──
                if phase == 'sweep':
                    # 사진 찍고 레이저 면적 계산
                    _, _, _, diff_clip, spot_area = self._capture_diff()
                    print(f"  [{sweep_state}] Z={cur_z:.4f} mm | 면적={spot_area} px | 보폭={step_size*1000:.0f}µm")

                    if diff_clip is not None:
                        diff_clip_disp = cv2.resize(cv2.cvtColor(diff_clip, cv2.COLOR_GRAY2BGR),
                                                    (STREAM_WIDTH, STREAM_HEIGHT))
                        cv2.putText(diff_clip_disp, f"Area: {spot_area} px",
                                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

                    # 상태 머신 시작
                    if sweep_state == 'init':
                        best_area = spot_area
                        best_z = cur_z
                        # 일단 위(+방향)로 한 걸음 가봅니다.
                        self.stage.move_absolute(x, y, best_z + (direction * step_size), a)
                        sweep_state = 'check'

                    elif sweep_state == 'check':
                        step_count += 1
                        
                        if spot_area < best_area:
                            # 🎯 좋아졌다! (골짜기를 향해 잘 가고 있음)
                            best_area = spot_area
                            best_z = cur_z
                            reversal_count = 0 # 연속 성공이므로 반전 카운트 초기화
                            
                            # 같은 방향으로 한 걸음 더 전진
                            self.stage.move_absolute(x, y, best_z + (direction * step_size), a)
                            
                        else:
                            # ❌ 나빠졌다! (골짜기를 지났거나, 애초에 반대 방향임)
                            print(f"    -> 면적 증가! (방향 전환 및 보폭 축소)")
                            direction *= -1       # 방향 반대로
                            reversal_count += 1
                            
                            # 보폭(Step)을 절반으로 줄입니다. (미세 탐색 돌입)
                            step_size /= 2.0
                            
                            if step_size < min_step or step_count >= MAX_STEPS:
                                # 보폭이 한계치보다 작아지면 탐색을 종료합니다.
                                sweep_state = 'done'
                            else:
                                # 나빠지기 전의 가장 좋았던 위치(best_z)로 돌아간 뒤, 
                                # 줄어든 보폭으로 반대 방향으로 이동합니다.
                                next_z = best_z + (direction * step_size)
                                self.stage.move_absolute(x, y, next_z, a)

                    elif sweep_state == 'done':
                        print(f"\n[오토포커스 완료] 최적 Z={best_z:.4f} mm (면적: {best_area}px)")
                        self.stage.move_absolute(x, y, best_z, a) # 최고 위치로 최종 이동
                        phase = 'stream'
                        print("[Stream] 라이브 스트림 유지 중. (수동 측정: F 키)")

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
                # 레이저 수동 격발 (카메라 화면 보면서)
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
