"""
요청·응답 스키마 전부. 라우터는 여기서 가져다 쓰기만 한다.

프론트엔드가 보내는 JSON 의 '계약서'다 — 기본값을 바꾸면 프론트가 그 필드를 생략했을 때의
동작이 바뀐다.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from backend.hw_tools.config import CAMERA_WIDTH, CAMERA_HEIGHT


# ── 장비 연결 ─────────────────────────────────────────────────────────────────

class CameraConnectRequest(BaseModel):
    exposure_ms: float = 10.0


class StageConnectRequest(BaseModel):
    dll_path: str = ""


class LaserConnectRequest(BaseModel):
    port: str = "COM4"
    baud: int = 115200


class CCDConnectRequest(BaseModel):
    target_temp: int = -40


class CCDStatusResponse(BaseModel):
    connected: bool
    temperature: Optional[int]
    temp_status: Optional[str]


# ── 장비 조작 ─────────────────────────────────────────────────────────────────

class MovePixelRequest(BaseModel):
    px: float
    py: float
    # 스트림 해상도는 프론트가 임의로 정한다(µm/px 가 해상도에 반비례하므로 함께 받아야 한다).
    stream_width: int = CAMERA_WIDTH
    stream_height: int = CAMERA_HEIGHT


class StageSpeedRequest(BaseModel):
    x: float = 5.0
    y: float = 5.0
    z: float = 5.0


class AcquireSpectrumRequest(BaseModel):
    exposure: float = 0.2
    power: int = 40
    acq_mode: str = 'single'
    num_accumulations: int = 1
    kinetic_count: int = 1
    kinetic_cycle_time: Optional[float] = None
    read_mode: str = 'fvb'
    hbin: int = 1
    single_track_center: Optional[int] = None
    single_track_width: int = 1
    trigger_mode: str = 'internal'


# ── 에이전트 실행 ─────────────────────────────────────────────────────────────

class ExperimentRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    # 어떤 에이전트 아키텍처로 실행할지 선택. 기본값 "CoALA".
    # 이 필드를 보내지 않던 기존 프론트엔드/벤치마크도 그대로 동작한다.
    agent: Optional[str] = "CoALA"


