import sys
import os
import cv2
import numpy as np
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

sys.path.append(os.path.dirname(__file__))
from USE_camera_stream import StreamingTUCam
from USE_stage_test import TangoController
from config import (  # noqa: E402
    CAMERA_WIDTH  as STREAM_WIDTH,
    CAMERA_HEIGHT as STREAM_HEIGHT,
    STAGE_MAX_X,
    STAGE_MAX_Y,
    LENS_WIDTH_UM  as LENS_WIDTH,
    LENS_HEIGHT_UM as LENS_HEIGHT,
)

# ── 축 부호 ───────────────────────────────────
# pixel +X(right) -> stage -X  (stage +X 이동시 이미지는 -X로 이동)
# pixel +Y(down)  -> stage -Y
SIGN_X = -1
SIGN_Y = +1

# um/px 스케일  (StartPosX=250, StartPosY=135은 카메라 ROI 시작점 — 좌표 계산에 불필요)
UM_PER_PX_X = LENS_WIDTH  / STREAM_WIDTH    # 305/1060 = 0.2877 um/px
UM_PER_PX_Y = LENS_HEIGHT / STREAM_HEIGHT   # 230/800  = 0.2875 um/px

def main():
    camera = StreamingTUCam()
    camera.start_stream()
    stage  = TangoController()
    stage.load_dll()
    stage.create_session()
    stage.connect()

    pos = stage.get_position()
    if pos is not None:
        print(f"[CENTER] stage X={pos[0]:.4f}  Y={pos[1]:.4f} mm")
    else:
        print("[CENTER] stage position unavailable")

    pending_click = [None]  # [( px, py )]

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pending_click[0] = (x, y)

    try:
        window_ready = False
        while True:
            frame = camera.get_latest_frame()
            if frame is None:
                continue

            disp = frame.copy()
            if disp.dtype == np.uint16:
                disp = (disp / 256).astype(np.uint8)
            if len(disp.shape) == 2:
                disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)

            if disp.shape[:2] != (STREAM_HEIGHT, STREAM_WIDTH):
                disp = cv2.resize(disp, (STREAM_WIDTH, STREAM_HEIGHT))

            # 중심 십자선
            cx, cy = STREAM_WIDTH // 2, STREAM_HEIGHT // 2
            cv2.line(disp, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 1)
            cv2.line(disp, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 1)

            cv2.imshow("RamanGPT Camera", disp)

            if not window_ready:
                cv2.setMouseCallback("RamanGPT Camera", on_mouse)
                window_ready = True

            if pending_click[0] is not None:
                px, py = pending_click[0]
                pending_click[0] = None
                pos = stage.get_position()
                if pos is not None:
                    dx_px = px - STREAM_WIDTH  / 2.0
                    dy_px = py - STREAM_HEIGHT / 2.0
                    
                    CALIB_FACTOR_X = 1.4
                    CALIB_FACTOR_Y = 1.285
                    abs_x = pos[0] + (dx_px * UM_PER_PX_X * CALIB_FACTOR_X) / 1000.0 * SIGN_X
                    abs_y = pos[1] + (dy_px * UM_PER_PX_Y * CALIB_FACTOR_Y) / 1000.0 * SIGN_Y
                    print(f"[CLICK] pixel=({px}, {py})  stage X={abs_x:.4f}  Y={abs_y:.4f} mm")
                else:
                    print("[CLICK] stage position unavailable")

            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
                break

    except KeyboardInterrupt:
        pass
    finally:
        camera.stop_stream()
        camera.close()
        stage.disconnect()
        stage.free_session()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
