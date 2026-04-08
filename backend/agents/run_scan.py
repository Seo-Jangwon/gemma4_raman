#!/usr/bin/env python
# coding: utf-8
"""
라만 스캔 메인 스크립트
=======================
오토포커스가 완료된 상태에서 실행합니다.

흐름:
  1. TUCam 카메라 연결 → 스트리밍 창 즉시 오픈
  2. Tango 스테이지 연결 (스캔 속도 9 μm/s 설정)
  3. 스냅샷 촬영
  4. SAM3 세그멘테이션 (백그라운드 스레드, 창 라이브 유지)
  5. 예상 경로 이미지 저장
  6. 창에서 SPACE 눌러 스캔 시작 (Q/ESC 취소)
  7. 경로 순서대로 move_absolute(wait=False) + 실시간 스트리밍

실행:
  python run_scan.py
  python run_scan.py --dry-run          # 이동 없이 경로만 계산
  python run_scan.py --mag 50x --step 3
"""

import argparse
import sys
import time
import threading
from pathlib import Path
import numpy as np

# ── 설정 ─────────────────────────────────────────────────────────────────────

DLL_PATH        = str(Path(__file__).resolve().parent / "backend" / "util" / "stage_move" / "Tango_DLL.dll")
SNAPSHOT_PATH   = "./outputs/snapshot.png"
PATH_PREVIEW    = "./outputs/scan_path_preview.png"
TEXT_PROMPTS    = ["cell, circle, particle, object, dust"]
MAG_LEVEL       = "20x"
SCAN_STEP_UM    = 5.0
CONF_THRESHOLD  = 0.5
OUTPUT_DIR      = "./outputs"

EXPOSURE_MS     = 50.0
WARMUP_FRAMES   = 5

SCAN_SPEED_UM_S = 9.0                          # 스캔 이동 속도 (μm/s)
SCAN_SPEED_MM_S = SCAN_SPEED_UM_S / 1000.0  # → mm/s (0.009 mm/s)

# 카메라 해상도
IMG_WIDTH  = 1060
IMG_HEIGHT = 800

# Width=1060
# Height=800

# 라이브 디스플레이 — 작게 줄여 랙 방지
STREAM_WIDTH = 640
STREAM_H     = int(IMG_HEIGHT * STREAM_WIDTH / IMG_WIDTH)
STREAM_SCALE = STREAM_WIDTH / IMG_WIDTH

# 스테이지 도달 판정 허용오차 (mm)
ARRIVE_TOL_MM = 0.002   # 2 μm

# ─────────────────────────────────────────────────────────────────────────────


def _to_bgr_u8(frame):
    """카메라 프레임 → 8-bit BGR 변환."""
    import cv2
    if frame.dtype.itemsize == 2:
        frame = (frame / 256).astype("uint8")
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


def _draw_laser_cross(disp):
    """
    화면 정중앙에 레이저 위치 마커를 그린다.
    밝은 시야에서도 보이도록 검은 테두리 + 빨간 선.
    """
    import cv2
    cx, cy = disp.shape[1] // 2, disp.shape[0] // 2
    R = 10
    # 검은 테두리 (밝은 배경에서 가시성 확보)
    cv2.circle(disp, (cx, cy), R + 1, (0, 0, 0), 3)
    cv2.line(disp, (cx - R - 4, cy), (cx + R + 4, cy), (0, 0, 0), 3)
    cv2.line(disp, (cx, cy - R - 4), (cx, cy + R + 4), (0, 0, 0), 3)
    # 빨간 마커
    cv2.circle(disp, (cx, cy), R, (30, 30, 220), 2)
    cv2.line(disp, (cx - R - 4, cy), (cx + R + 4, cy), (30, 30, 220), 1)
    cv2.line(disp, (cx, cy - R - 4), (cx, cy + R + 4), (30, 30, 220), 1)


def _draw_overlay(disp, all_pts, done_pts, cur_px, obj_id, pt_idx, total_pts, status=""):
    """스캔 진행 오버레이 (경로·완료점·현재위치) + 레이저 마커."""
    import cv2
    s = STREAM_SCALE

    # 전체 계획 경로 (회색)
    for px, py in all_pts:
        cv2.circle(disp, (int(px * s), int(py * s)), 2, (80, 80, 80), -1)

    # 완료 포인트 (초록)
    for px, py in done_pts:
        cv2.circle(disp, (int(px * s), int(py * s)), 3, (0, 200, 0), -1)

    # 현재 포인트 (파란 원)
    if cur_px is not None:
        cx, cy = int(cur_px[0] * s), int(cur_px[1] * s)
        cv2.circle(disp, (cx, cy), 7, (200, 80, 0), 2)

    # 레이저 마커 (항상 중앙)
    _draw_laser_cross(disp)

    # 상단 HUD
    hud = status or f"Obj {obj_id}  |  {pt_idx}/{total_pts}  ({pt_idx*100//max(total_pts,1)}%)"
    cv2.rectangle(disp, (0, 0), (disp.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(disp, hud, (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 220), 1, cv2.LINE_AA)

    # 하단 안내
    h = disp.shape[0]
    cv2.rectangle(disp, (0, h - 22), (300, h), (0, 0, 0), -1)
    cv2.putText(disp, "[Q/ESC] 중단", (6, h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1, cv2.LINE_AA)


def _show_frame(cam, status1="", status2=""):
    """
    카메라에서 한 프레임을 가져와 창에 표시.
    레이저 마커 + 상태 텍스트 포함.
    반환: waitKey(1) 결과
    """
    import cv2
    raw = cam.get_latest_frame() if cam is not None else None
    if raw is not None:
        disp = cv2.resize(_to_bgr_u8(raw), (STREAM_WIDTH, STREAM_H))
        _draw_laser_cross(disp)
        if status1:
            cv2.rectangle(disp, (0, 0), (disp.shape[1], 30), (0, 0, 0), -1)
            cv2.putText(disp, status1, (6, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 220), 1, cv2.LINE_AA)
        if status2:
            h = disp.shape[0]
            cv2.rectangle(disp, (0, h - 22), (500, h), (0, 0, 0), -1)
            cv2.putText(disp, status2, (6, h - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
        cv2.imshow("Raman Scan", disp)
    return cv2.waitKey(1) & 0xFF


def _move_live(stage, cam, tx, ty, tz, overlay_fn=None):
    """
    move_absolute(wait=False) + 실시간 프레임 표시.
    목표점 도달 또는 Q/ESC 입력 시 반환.
    반환값: False = 사용자 중단 요청.
    """
    import cv2
    stage.move_absolute(tx, ty, tz, wait=False)
    deadline = time.time() + 180  # 최대 3분
    while time.time() < deadline:
        pos = stage.get_position()
        if pos and abs(pos[0] - tx) < ARRIVE_TOL_MM and abs(pos[1] - ty) < ARRIVE_TOL_MM:
            break

        raw = cam.get_latest_frame() if cam is not None else None
        if raw is not None:
            disp = cv2.resize(_to_bgr_u8(raw), (STREAM_WIDTH, STREAM_H))
            if overlay_fn:
                overlay_fn(disp)
            else:
                _draw_laser_cross(disp)
            cv2.imshow("Raman Scan", disp)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            return False
    return True


# ── 경로 미리보기 저장 ─────────────────────────────────────────────────────────

def save_path_preview(snapshot_path: str, objects: list, agent, step_px: int) -> str:
    """스냅샷 위에 스캔 경로를 그려 PATH_PREVIEW 에 저장."""
    import cv2

    img = cv2.imread(snapshot_path)
    if img is None:
        return ""

    # 디버깅: 경로 생성 정보
    print(f"\n[DEBUG] 경로 생성 정보:")
    print(f"  step_px: {step_px} 픽셀")
    print(f"  μm/pixel: {agent.mapper.config['um_per_pixel']:.4f}")
    
    palette = [
        (0, 200, 255), (0, 255, 128), (255, 128, 0),
        (200, 0, 255), (0, 128, 255), (255, 0, 128),
    ]

    for obj in objects:
        color = palette[obj["id"] % len(palette)]
        cx_px, cy_px = obj["center_x"], obj["center_y"]

        _, pca_angle = agent._precompute_endpoints(obj.get("pixels", []))
        path = agent.generate_pca_snake_scan_path(obj.get("pixels", []), step_px, pca_angle)
        
        # 디버깅: 각 객체별 경로 정보
        print(f"  객체 {obj['id']}: 중심({cx_px}, {cy_px}), PCA각도={np.degrees(pca_angle):.1f}도, 경로={len(path)}개 포인트")
        
        if not path:
            path = [(cx_px, cy_px)]
            print(f"    ⚠️  경로 생성 실패 → 중심점만 사용")

        for i in range(len(path) - 1):
            cv2.line(img, (int(path[i][0]), int(path[i][1])),
                     (int(path[i+1][0]), int(path[i+1][1])), color, 1)
        for px, py in path:
            cv2.circle(img, (int(px), int(py)), 2, color, -1)

        cv2.circle(img, (int(path[0][0]),  int(path[0][1])),  6, (255, 255, 255), 2)
        cv2.circle(img, (int(path[-1][0]), int(path[-1][1])), 6, (0, 0, 200),     2)
        cv2.putText(img, f"Obj {obj['id']}", (cx_px - 20, cy_px - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    # 레이저 위치 마커 (원본 해상도)
    icx, icy = IMG_WIDTH // 2, IMG_HEIGHT // 2
    cv2.circle(img, (icx, icy), 20, (0, 0, 0), 4)
    cv2.circle(img, (icx, icy), 20, (30, 30, 220), 3)
    cv2.line(img, (icx - 28, icy), (icx + 28, icy), (0, 0, 0), 3)
    cv2.line(img, (icx, icy - 28), (icx, icy + 28), (0, 0, 0), 3)
    cv2.line(img, (icx - 28, icy), (icx + 28, icy), (30, 30, 220), 1)
    cv2.line(img, (icx, icy - 28), (icx, icy + 28), (30, 30, 220), 1)

    Path(PATH_PREVIEW).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(PATH_PREVIEW, img)
    print(f"       경로 미리보기 저장: {PATH_PREVIEW}\n")
    return PATH_PREVIEW


# ── 카메라 연결 ───────────────────────────────────────────────────────────────

def open_camera_and_window():
    """카메라 연결 → 스트리밍 창 즉시 오픈 → 스냅샷 저장. (cam, path) 반환."""
    import cv2
    from backend.autofocus.autofocus import StreamingTUCam

    print("[1/5] 카메라 연결 중...")
    Path(SNAPSHOT_PATH).parent.mkdir(parents=True, exist_ok=True)

    cam = StreamingTUCam(exposure_ms=EXPOSURE_MS)
    cam.start_stream()

    cv2.namedWindow("Raman Scan", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Raman Scan", STREAM_WIDTH, STREAM_H)

    for i in range(WARMUP_FRAMES):
        _show_frame(cam, f"cammera connected ({i+1}/{WARMUP_FRAMES})")
        time.sleep(0.05)

    raw = cam.get_latest_frame()
    if raw is None:
        cam.stop_stream(); cam.close()
        raise RuntimeError("카메라 프레임 획득 실패")

    snap = _to_bgr_u8(raw)
    import cv2 as _cv2
    _cv2.imwrite(SNAPSHOT_PATH, snap)
    h, w = snap.shape[:2]
    print(f"       카메라 연결 완료  ({w}×{h})  스냅샷: {SNAPSHOT_PATH}")
    return cam, SNAPSHOT_PATH


# ── 스테이지 연결 ─────────────────────────────────────────────────────────────

def connect_stage():
    """Tango 스테이지 연결 + 스캔 속도(9 μm/s) 설정 후 인스턴스 반환."""
    from backend.util.stage_move.stage_test import TangoController

    print("[2/5] 스테이지 연결 중...")
    tango = TangoController(DLL_PATH)

    if not tango.load_dll():
        raise RuntimeError("DLL 로드 실패 — Tango_DLL.dll 경로 확인")
    if not tango.create_session():
        raise RuntimeError("세션 생성 실패")
    if not tango.connect():
        raise RuntimeError("스테이지 연결 실패 — 케이블/전원 확인")

    v = SCAN_SPEED_MM_S
    if tango.set_velocity(v, v, v, v):
        print(f"       속도 설정: {SCAN_SPEED_UM_S} μm/s ({v} mm/s)")

    pos = tango.get_position()
    print(f"       연결 완료 | X={pos[0]:.4f}  Y={pos[1]:.4f}  Z={pos[2]:.4f} mm")
    return tango


# ── SAM3 세그멘테이션 ─────────────────────────────────────────────────────────

def run_segmentation_live(cam, image_path: str) -> list:
    """SAM3를 백그라운드 스레드에서 실행, 창은 라이브 유지."""
    from backend.scan.sam3 import segment_with_text_prompt

    print(f"\n[3/5] SAM3 세그멘테이션  프롬프트={TEXT_PROMPTS}  conf={CONF_THRESHOLD}")

    result  = {"objects": None, "error": None}
    t_start = time.time()

    def _seg():
        try:
            result["objects"] = segment_with_text_prompt(
                image_path=image_path,
                text_prompts=TEXT_PROMPTS,
                output_dir=OUTPUT_DIR,
                conf_threshold=CONF_THRESHOLD,
            )
        except Exception as e:
            result["error"] = e

    worker = threading.Thread(target=_seg, daemon=True)
    worker.start()

    while worker.is_alive():
        elapsed = time.time() - t_start
        key = _show_frame(cam, "SAM3 analysing...", f"time {elapsed:.0f}s  [Q] quit")
        if key in (ord("q"), ord("Q"), 27):
            print("\n[!] SAM3 중 취소")
            sys.exit(0)

    if result["error"]:
        raise result["error"]

    objects = result["objects"] or []
    print(f"       → {len(objects)}개 객체 검출")
    
    # 디버깅: SAM3 출력 데이터 구조 확인
    if objects:
        print(f"\n[DEBUG] SAM3 출력 데이터 구조 확인:")
        for i, obj in enumerate(objects[:3]):  # 처음 3개만
            print(f"  객체 {obj.get('id', i)}:")
            print(f"    - Keys: {list(obj.keys())}")
            print(f"    - 중심: ({obj.get('center_x')}, {obj.get('center_y')})")
            pixels = obj.get('pixels', [])
            print(f"    - 픽셀 개수: {len(pixels)}")
            if pixels:
                first_px = pixels[0]
                print(f"    - 첫 픽셀 타입: {type(first_px)}")
                print(f"    - 첫 픽셀 값: {first_px}")
        
        # JSON으로 샘플 저장
        import json
        debug_path = Path(OUTPUT_DIR) / "sam_debug.json"
        debug_data = []
        for obj in objects:
            debug_obj = {k: v for k, v in obj.items() if k != 'pixels'}
            if 'pixels' in obj:
                debug_obj['pixels_count'] = len(obj['pixels'])
                debug_obj['pixels_sample'] = obj['pixels'][:5]
            debug_data.append(debug_obj)
        with open(debug_path, 'w') as f:
            json.dump(debug_data, f, indent=2)
        print(f"  → 디버그 데이터 저장: {debug_path}\n")
    
    for obj in objects:
        print(f"         [{obj['id']:>3}]  중심({obj['center_x']:4}, {obj['center_y']:4})  "
              f"픽셀 {len(obj['pixels']):>6,}개")
    return objects


# ── 스캔 실행 ─────────────────────────────────────────────────────────────────

def execute_scan(cam, stage, objects: list, mag_level: str, scan_step_um: float,
                 origin_x_mm: float, origin_y_mm: float, origin_z_mm: float,
                 dry_run: bool) -> int:
    """
    각 포인트마다 move_absolute(wait=False) 후 도달까지 라이브 스트리밍.
    Q / ESC 로 중단.
    """
    from backend.scan.scanner_agent import ScannerAgent

    agent     = ScannerAgent(mag_level=mag_level)
    um_per_px = agent.mapper.config["um_per_pixel"]
    sign_x    = agent.mapper.config["sign_x"]
    sign_y    = agent.mapper.config["sign_y"]
    img_cx    = IMG_WIDTH  / 2.0
    img_cy    = IMG_HEIGHT / 2.0
    step_px   = max(1, int(scan_step_um / um_per_px))

    print(f"\n[5/5] 스캔 실행  dry_run={dry_run}")
    print(f"       배율={mag_level}  {um_per_px:.4f} μm/px  스텝={scan_step_um} μm ({step_px} px)")
    print(f"       속도={SCAN_SPEED_UM_S} μm/s  기준 X={origin_x_mm:.4f} Y={origin_y_mm:.4f} Z={origin_z_mm:.4f}\n")

    # 디버깅: 좌표 변환 정보
    print(f"[DEBUG] 좌표 변환 설정:")
    print(f"  이미지 중심 (레이저): ({img_cx}, {img_cy}) 픽셀")
    print(f"  sign_x={sign_x}, sign_y={sign_y}")
    print(f"  Origin Stage 좌표: ({origin_x_mm:.4f}, {origin_y_mm:.4f}) mm\n")

    total_points = 0
    stop_flag    = False

    for obj in objects:
        if stop_flag:
            break

        cx_px = obj["center_x"]
        cy_px = obj["center_y"]
        obj_x_mm = origin_x_mm + (cx_px - img_cx) * um_per_px / 1000.0 * sign_x
        obj_y_mm = origin_y_mm + (cy_px - img_cy) * um_per_px / 1000.0 * sign_y

        _, pca_angle  = agent._precompute_endpoints(obj.get("pixels", []))
        scan_path_px  = agent.generate_pca_snake_scan_path(obj.get("pixels", []), step_px, pca_angle)
        if not scan_path_px:
            scan_path_px = [(cx_px, cy_px)]

        print(f"  ▶ Obj {obj['id']:>3}  →  ({obj_x_mm:.4f}, {obj_y_mm:.4f}) mm  [{len(scan_path_px)} pts]")
        
        # 디버깅: 좌표 변환 상세
        dx_px = cx_px - img_cx
        dy_px = cy_px - img_cy
        dx_um = dx_px * um_per_px
        dy_um = dy_px * um_per_px
        dist_um = np.sqrt(dx_um**2 + dy_um**2)
        print(f"      픽셀 오프셋: ({dx_px:.1f}, {dy_px:.1f}) px → ({dx_um:.2f}, {dy_um:.2f}) μm → 거리 {dist_um:.2f} μm")

        # 객체 중심으로 이동
        if not dry_run:
            def _ov_pos(d):
                _draw_overlay(d, scan_path_px, [], (cx_px, cy_px),
                              obj["id"], 0, len(scan_path_px),
                              f"Obj {obj['id']} 이동 중...")
            ok = _move_live(stage, cam, obj_x_mm, obj_y_mm, origin_z_mm, overlay_fn=_ov_pos)
            if not ok:
                stop_flag = True; break

        done_pts = []

        for pt_idx, (px, py) in enumerate(scan_path_px, start=1):
            if stop_flag:
                break

            pt_x_mm = origin_x_mm + (px - img_cx) * um_per_px / 1000.0 * sign_x
            pt_y_mm = origin_y_mm + (py - img_cy) * um_per_px / 1000.0 * sign_y

            if not dry_run:
                done_snapshot = list(done_pts)  # closure capture
                cur_snapshot  = (px, py)
                def _ov_scan(d, _dp=done_snapshot, _cp=cur_snapshot):
                    _draw_overlay(d, scan_path_px, _dp, _cp,
                                  obj["id"], pt_idx, len(scan_path_px))
                ok = _move_live(stage, cam, pt_x_mm, pt_y_mm, origin_z_mm, overlay_fn=_ov_scan)
                if not ok:
                    stop_flag = True; break
            else:
                # dry-run: 그냥 프레임만 표시
                key = _show_frame(cam,
                                  f"[DRY] Obj {obj['id']}  {pt_idx}/{len(scan_path_px)}")
                if key in (ord("q"), ord("Q"), 27):
                    stop_flag = True; break

            # ── 라만 측정 트리거 자리 ─────────────────────────────────
            # ccd.acquire() 등
            # ─────────────────────────────────────────────────────────

            total_points += 1
            done_pts.append((px, py))
            print(f"       {pt_idx:4}/{len(scan_path_px)} pts", end="\r")

        print(f"       ✓ {len(scan_path_px)} pts 완료          ")

    print(f"\n{'─'*60}")
    print(f"스캔 완료  |  {len(objects)}개 객체  |  총 {total_points} 포인트")
    return total_points


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    import cv2

    parser = argparse.ArgumentParser()
    parser.add_argument("--mag",        default=MAG_LEVEL,    help="배율 (20x/50x/100x)")
    parser.add_argument("--step",       default=SCAN_STEP_UM, type=float, help="스캔 스텝 μm")
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--no-confirm", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("라만 스캔  (오토포커스 완료 상태)")
    print("=" * 60)

    cam   = None
    stage = None

    try:
        if args.dry_run:
            snapshot = SNAPSHOT_PATH
            if not Path(snapshot).exists():
                print(f"[!] 스냅샷 없음: {snapshot}")
                sys.exit(1)
            print(f"[1/5] DRY-RUN — 기존 스냅샷 사용: {snapshot}")
            origin_x_mm = origin_y_mm = origin_z_mm = 0.0

            cv2.namedWindow("Raman Scan", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Raman Scan", STREAM_WIDTH, STREAM_H)
        else:
            cam, snapshot = open_camera_and_window()
            _show_frame(cam, "스테이지 연결 중...")
            stage = connect_stage()
            pos = stage.get_position()
            origin_x_mm, origin_y_mm, origin_z_mm = pos[0], pos[1], pos[2]

        # SAM3
        objects = run_segmentation_live(cam, snapshot)
        if not objects:
            print("\n[!] 검출된 객체 없음.")
            return

        # 경로 미리보기 저장
        print(f"\n[4/5] 경로 미리보기 저장")
        from backend.scan.scanner_agent import ScannerAgent
        agent   = ScannerAgent(mag_level=args.mag)
        step_px = max(1, int(args.step / agent.mapper.config["um_per_pixel"]))
        save_path_preview(snapshot, objects, agent, step_px)

        # 확인 대기 — 창에서 SPACE
        if not args.no_confirm and not args.dry_run:
            print(f"\n  배율={args.mag}  스텝={args.step} μm  속도={SCAN_SPEED_UM_S} μm/s  객체={len(objects)}개")
            print("  → 스캔 창에서  [SPACE] 시작  /  [Q/ESC] 취소")
            while True:
                key = _show_frame(cam,
                                  f"준비 완료 — {len(objects)}개 객체",
                                  "[SPACE] 스캔 시작   [Q/ESC] 취소")
                if key == ord(" "):
                    break
                if key in (ord("q"), ord("Q"), 27):
                    print("취소됨"); return

        # 스캔 실행
        execute_scan(cam, stage, objects, args.mag, args.step,
                     origin_x_mm, origin_y_mm, origin_z_mm,
                     dry_run=args.dry_run)

        # 완료 후 ESC 누를 때까지 라이브 유지
        if cam is not None:
            print("\n[완료] 창에서 ESC 또는 Q 로 종료")
            while True:
                key = _show_frame(cam, "scan completed  — ESC / Q to quit")
                if key in (27, ord("q"), ord("Q")):
                    break

    except KeyboardInterrupt:
        print("\n\n[!] Ctrl+C — 중단")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)
    finally:
        cv2.destroyAllWindows()
        if cam is not None:
            cam.stop_stream(); cam.close()
        if stage is not None:
            stage.disconnect(); stage.free_session()


if __name__ == "__main__":
    main()