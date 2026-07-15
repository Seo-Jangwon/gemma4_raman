"""
카메라 실시간 스트리밍 뷰어
ESC 키로 종료

[스레드 안전성]
이 객체는 여러 스레드가 동시에 만진다:
  - MJPEG 라이브 스트림 (server.py /api/camera/stream) — 브라우저가 열려 있는 동안 계속
  - 에이전트 (hw_manager → analyze_microscope_image / capture_camera_frame)
TUCam SDK는 스레드 안전하지 않고, 게다가 아래 메서드들은 인스턴스 하나뿐인
self.m_frame 버퍼를 공유한다 — WaitForFrame이 채운 m_frame을 memmove로 읽는
사이에 다른 스레드가 같은 m_frame으로 또 WaitForFrame을 걸면 프레임이 찢어지거나
SDK가 멈춘다. 그래서 start/get/stop 전체를 _lock으로 직렬화한다.

RLock인 이유: 같은 스레드에서 stop_stream→get_latest_frame 같은 중첩 호출이
생겨도 자기 자신에게 데드락이 걸리지 않게 하기 위함.
대기 시간은 WaitForFrame의 500ms 타임아웃으로 상한이 잡히므로, 락을 기다리는
쪽도 최대 ~500ms만 지연된다(무한 대기 없음).
"""
import numpy as np
import cv2
import sys
import os
import threading
from ctypes import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from backend.TuCam.TUCam import *


class StreamingTUCam:
    def __init__(self, exposure_ms=10.0):
        self.Path = './'
        self.TUCAMINIT = TUCAM_INIT(0, self.Path.encode('utf-8'))
        self.TUCAMOPEN = TUCAM_OPEN(0, 0)
        TUCAM_Api_Init(pointer(self.TUCAMINIT), 5000)
        if self.TUCAMINIT.uiCamCount == 0:
            raise RuntimeError("No camera found!")
        self.TUCAMOPEN = TUCAM_OPEN(0, 0)
        TUCAM_Dev_Open(pointer(self.TUCAMOPEN))
        if self.TUCAMOPEN.hIdxTUCam == 0:
            raise RuntimeError("Open failure!")
        self.set_exposure(exposure_ms)
        self.is_streaming = False
        self.m_frame = TUCAM_FRAME()
        self.m_capmode = TUCAM_CAPTURE_MODES
        self._lock = threading.RLock()   # 파일 상단 [스레드 안전성] 참고

    def set_exposure(self, ms: float):
        TUCAM_Capa_SetValue(self.TUCAMOPEN.hIdxTUCam, TUCAM_IDCAPA.TUIDC_ATEXPOSURE.value, 0)
        TUCAM_Prop_SetValue(self.TUCAMOPEN.hIdxTUCam, TUCAM_IDPROP.TUIDP_EXPOSURETM.value, c_double(ms), 0)

    def start_stream(self):
        with self._lock:
            if self.is_streaming:
                return
            self.m_frame.pBuffer = 0
            self.m_frame.ucFormatGet = TUFRM_FORMATS.TUFRM_FMT_USUAl.value
            self.m_frame.uiRsdSize = 1
            TUCAM_Buf_Alloc(self.TUCAMOPEN.hIdxTUCam, pointer(self.m_frame))
            TUCAM_Cap_Start(self.TUCAMOPEN.hIdxTUCam, self.m_capmode.TUCCM_SEQUENCE.value)
            self.is_streaming = True

    def get_latest_frame(self):
        with self._lock:
            if not self.is_streaming:
                return None
            try:
                TUCAM_Buf_WaitForFrame(self.TUCAMOPEN.hIdxTUCam, pointer(self.m_frame), 500)
            except:
                return None
            buf = create_string_buffer(self.m_frame.uiImgSize)
            pointer_data = c_void_p(self.m_frame.pBuffer + self.m_frame.usHeader)
            memmove(buf, pointer_data, self.m_frame.uiImgSize)
            dtype = np.uint8 if self.m_frame.ucElemBytes == 1 else np.uint16
            image_np = np.frombuffer(buf, dtype=dtype)
            if self.m_frame.ucChannels == 3:
                image_np = image_np.reshape((self.m_frame.usHeight, self.m_frame.usWidth, 3))
            else:
                image_np = image_np.reshape((self.m_frame.usHeight, self.m_frame.usWidth))
            # buf는 호출마다 새로 만들고 memmove로 SDK 버퍼에서 복사해 온 것이라
            # 호출자끼리 공유되지 않는다 → 추가 복사 불필요.
            return image_np

    def stop_stream(self):
        with self._lock:
            if self.is_streaming:
                TUCAM_Buf_AbortWait(self.TUCAMOPEN.hIdxTUCam)
                TUCAM_Cap_Stop(self.TUCAMOPEN.hIdxTUCam)
                TUCAM_Buf_Release(self.TUCAMOPEN.hIdxTUCam)
                self.is_streaming = False

    def close(self):
        self.stop_stream()
        if self.TUCAMOPEN.hIdxTUCam != 0:
            TUCAM_Dev_Close(self.TUCAMOPEN.hIdxTUCam)
        TUCAM_Api_Uninit()


def main():
    print("=== Camera Stream Viewer ===")
    print("ESC: 종료")
    print("E: 노출 증가 (+5ms)")
    print("D: 노출 감소 (-5ms)")
    print()

    exposure = 10.0
    camera = StreamingTUCam(exposure_ms=exposure)
    camera.start_stream()

    try:
        while True:
            frame = camera.get_latest_frame()
            if frame is None:
                continue

            # 디스플레이용 변환
            disp = frame.copy()
            if disp.dtype == np.uint16:
                disp = (disp / 256).astype(np.uint8)
            if len(disp.shape) == 2:
                disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)

            # 정보 표시
            cv2.putText(disp, f"Exposure: {exposure:.1f}ms", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 리사이즈
            h, w = disp.shape[:2]
            if w > 1280:
                disp = cv2.resize(disp, (1280, int(1280 * h / w)))

            cv2.imshow("Camera Stream", disp)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('e') or key == ord('E'):
                exposure += 5.0
                camera.set_exposure(exposure)
                print(f"Exposure: {exposure:.1f}ms")
            elif key == ord('d') or key == ord('D'):
                exposure = max(1.0, exposure - 5.0)
                camera.set_exposure(exposure)
                print(f"Exposure: {exposure:.1f}ms")

    except KeyboardInterrupt:
        pass
    finally:
        camera.close()
        cv2.destroyAllWindows()
        print("Stream closed.")


if __name__ == "__main__":
    main()
