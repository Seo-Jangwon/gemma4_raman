import sys
import os
import cv2
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

sys.path.append(os.path.dirname(__file__))
from USE_camera_stream import StreamingTUCam
from USE_stage_test import TangoController
from config import (  # noqa: E402
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    STAGE_MAX_X,
    STAGE_MAX_Y,
)
# 픽셀→스테이지 변환과 프레임 전처리는 공용 모듈을 쓴다(2026-07-30).
# 예전에는 여기서 SIGN_*/UM_PER_PX_* 를 따로 정의해, raman_tools.move_to_pixel 이나
# server 의 클릭 이동과 상수가 갈라질 수 있었다.
import optics_map  # noqa: E402
import vision      # noqa: E402

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

            # 뷰 해상도로 정규화 — 이 화면의 픽셀 좌표계 = 도구 좌표계
            disp = vision.to_view_bgr(frame)

            # 중심 십자선
            cx, cy = CAMERA_WIDTH // 2, CAMERA_HEIGHT // 2
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
                    # raman_tools.move_to_pixel 과 정확히 같은 변환(optics_map 단일 출처)
                    abs_x, abs_y = optics_map.pixel_to_stage(px, py, pos[0], pos[1])
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
