"""
LLM에게 전달할 tool 스키마 정의 (Ollama tool calling 포맷)
"""

RAMAN_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "move_stage",
            "description": "스테이지를 절대 좌표(mm)로 이동한다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "X축 위치 (mm, 0~75.3)"},
                    "y": {"type": "number", "description": "Y축 위치 (mm, 0~50.2)"},
                    "z": {"type": "number", "description": "Z축 위치 (mm, -1.0~1.0). 생략 가능"},
                },
                "required": ["x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_stage_speed",
            "description": (
                "스테이지의 이동 속도를 설정한다. "
                "x_speed_mm_s, y_speed_mm_s, z_speed_mm_s 필드로 각 축의 이동 속도가 포함된다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "x_speed_mm_s": {
                        "type": "number",
                        "description": "X축 이동 속도 (mm/s), 최대 5.0mm/s",
                    },
                    "y_speed_mm_s": {
                        "type": "number",
                        "description": "Y축 이동 속도 (mm/s), 최대 5.0mm/s",
                    },
                    "z_speed_mm_s": {
                        "type": "number",
                        "description": "Z축 이동 속도 (mm/s), 최대 0.1mm/s",
                    },
                },
                "required": ["x_speed_mm_s", "y_speed_mm_s"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stage_position",
            "description": "현재 스테이지의 X, Y, Z 위치(mm)를 읽어온다.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_stage_relative",
            "description": "현재 위치 기준으로 스테이지를 상대 이동한다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dx": {"type": "number", "description": "X 방향 이동량 (mm)"},
                    "dy": {"type": "number", "description": "Y 방향 이동량 (mm)"},
                    "dz": {"type": "number", "description": "Z 방향 이동량 (mm)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "laser_on",
            "description": "레이저를 켠다.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "laser_off",
            "description": "레이저를 끈다.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_laser_power",
            "description": "레이저 출력(ND 필터 투과율)을 설정한다. 0.004~100 % 범위의 임의 실수.",
            "parameters": {
                "type": "object",
                "properties": {
                    "percent": {
                        "type": "number",
                        "description": "출력 퍼센트 (투과율 %). 0.004~100 범위의 실수. 예: 1, 2.5, 40, 100.",
                        "minimum": 0.004,
                        "maximum": 100,
                    }
                },
                "required": ["percent"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "acquire_spectrum",
            "description": (
                "현재 위치에서 라만 스펙트럼을 수집한다. "
                "Single(1회) / Accumulate(누적 평균, 고SNR) / Kinetic(시계열 연속) 3가지 취득 모드를 지원한다. "
                "레이저 ON → 출력 안정화 → CCD 촬영 → 레이저 OFF 흐름을 자동 처리한다. "
                "캘리브레이터 연결 시 raman_shift_cm-1, wavelength_nm, laser_nm 필드가 포함된다. "
                "Kinetic 모드는 frames 배열로 각 프레임 데이터를 반환한다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "exposure": {
                        "type": "number",
                        "description": "CCD 노출 시간 (초). 기본값 0.2",
                    },
                    "power": {
                        "type": "number",
                        "description": "레이저 출력 (투과율 %). 0.004~100 범위의 실수. 기본값 40.",
                        "minimum": 0.004,
                        "maximum": 100,
                    },
                    "stabilize_sec": {
                        "type": "number",
                        "description": "레이저 ON 후 출력 안정화 대기 시간 (초). 기본값 0.5",
                    },
                    "acq_mode": {
                        "type": "string",
                        "enum": ["single", "accumulate", "kinetic"],
                        "description": (
                            "CCD 취득 모드. "
                            "'single': 1회 촬영(기본). "
                            "'accumulate': num_accumulations회 누적 합산 → 고SNR 스펙트럼. "
                            "'kinetic': kinetic_count개 프레임을 연속 취득 → 시계열 분석."
                        ),
                    },
                    "num_accumulations": {
                        "type": "integer",
                        "description": "accumulate/kinetic 모드에서 프레임당 누적 횟수. 기본값 1.",
                        "minimum": 1,
                    },
                    "kinetic_count": {
                        "type": "integer",
                        "description": "kinetic 모드에서 수집할 총 프레임 수. 기본값 1.",
                        "minimum": 1,
                    },
                    "kinetic_cycle_time": {
                        "type": "number",
                        "description": "kinetic 모드 프레임 간격 (초). 생략하면 SDK가 최소값으로 자동 계산.",
                    },
                    "read_mode": {
                        "type": "string",
                        "enum": ["fvb", "single_track"],
                        "description": (
                            "CCD 읽기 모드. "
                            "'fvb': Full Vertical Binning — 수직 전체 합산, 1D 스펙트럼(기본). "
                            "'single_track': 특정 행만 읽기 — single_track_center 필수."
                        ),
                    },
                    "hbin": {
                        "type": "integer",
                        "description": "수평 비닝 픽셀 수. 기본값 1.",
                        "minimum": 1,
                    },
                    "single_track_center": {
                        "type": "integer",
                        "description": "read_mode='single_track' 시 중심 픽셀 행 번호.",
                    },
                    "single_track_width": {
                        "type": "integer",
                        "description": "read_mode='single_track' 시 트랙 폭 (픽셀). 기본값 1.",
                        "minimum": 1,
                    },
                    "trigger_mode": {
                        "type": "string",
                        "enum": ["internal", "external", "external_start", "external_exposure", "external_fvb_em", "software"],
                        "description": "CCD 트리거 모드. 기본값 'internal'.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "start_camera_stream",
            "description": "카메라 실시간 스트리밍(미리보기)을 시작한다. 시편의 위치를 확인하거나 초점을 맞출 때 사용한다.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop_camera_stream",
            "description": "카메라 실시간 스트리밍을 중지한다.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    # ── CCD 파라미터 설정 ────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_ccd_info",
            "description": (
                "현재 CCD의 모든 설정값과 상태를 조회한다. "
                "온도, 냉각 상태, 노출 시간, 취득 모드, 읽기 모드, 이득, 시프트 속도, "
                "픽셀 수 등을 한 번에 반환한다. 파라미터 변경 전후에 사용해 현재 상태를 확인한다."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_ccd_exposure",
            "description": "CCD 노출 시간(초)을 설정한다. 값이 클수록 신호가 강해지지만 측정 시간이 늘어난다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "exposure_time": {
                        "type": "number",
                        "description": "노출 시간 [초]. 예: 0.1, 0.5, 1.0",
                    }
                },
                "required": ["exposure_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_ccd_acquisition_mode",
            "description": (
                "CCD 취득 모드를 설정한다. "
                "'single': 1회 촬영. "
                "'accumulate': num_accumulations 회 누적 후 합산. "
                "'kinetic': num_kinetics 프레임을 연속 취득. "
                "'run_till_abort': 중단 명령 전까지 연속 취득."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "description": "취득 모드",
                        "enum": ["single", "accumulate", "kinetic", "run_till_abort"],
                    },
                    "num_accumulations": {
                        "type": "integer",
                        "description": "누적 횟수 (accumulate/kinetic 모드에서 사용). 기본 1",
                    },
                    "num_kinetics": {
                        "type": "integer",
                        "description": "총 취득 프레임 수 (kinetic 모드에서 사용)",
                    },
                },
                "required": ["mode"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_ccd_trigger_mode",
            "description": (
                "CCD 트리거 모드를 설정한다. "
                "'internal': 소프트웨어로 취득 시작 (기본). "
                "'external': 외부 TTL 신호로 취득 시작. "
                "'external_start': 외부 트리거로 시작 후 내부 타이밍. "
                "'external_exposure': 외부 TTL HIGH 동안 노출. "
                "'software': SendSoftwareTrigger 사용."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "description": "트리거 모드",
                        "enum": ["internal", "external", "external_start", "external_exposure", "software"],
                    }
                },
                "required": ["mode"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_ccd_read_mode",
            "description": (
                "CCD 읽기 모드(readout mode)를 설정한다. "
                "'fvb': Full Vertical Binning — 수직 전체 합산 → 1D 스펙트럼 (기본, 라만에 사용). "
                "'single_track': 특정 수직 행만 읽음. center 파라미터 필수. "
                "'image': 2D 이미지 전체 또는 ROI."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "description": "읽기 모드",
                        "enum": ["fvb", "single_track", "image"],
                    },
                    "hbin": {
                        "type": "integer",
                        "description": "수평 빈닝 계수 (기본 1 = 빈닝 없음)",
                    },
                    "center": {
                        "type": "integer",
                        "description": "single_track 모드에서 읽을 행의 중심 번호 (1-based). single_track 모드 사용 시 반드시 지정 필요. 예: 256",
                    },
                    "width": {
                        "type": "integer",
                        "description": "single_track 모드에서 읽을 행 수 (기본 1)",
                    },
                },
                "required": ["mode"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_mcp_gain",
            "description": "iStar ICCD 카메라의 MCP(Micro-Channel Plate) 이득을 설정한다. 허용 범위는 get_mcp_gain_range()로 확인한다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "gain": {
                        "type": "integer",
                        "description": "설정할 MCP 이득값"
                    }
                },
                "required": ["gain"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_mcp_gain_range",
            "description": "iStar ICCD 카메라의 MCP 이득 허용 범위(min, max)를 반환한다.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_ccd_preamp_gain",
            "description": (
                "프리앰프(Pre-Amplifier) 이득 인덱스를 설정한다. "
                "사용 가능한 이득 목록은 get_ccd_info()의 preamp_gains_available 참조. "
                "인덱스가 클수록 이득이 높아 미약한 신호 측정에 유리하지만 노이즈도 증가한다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": "프리앰프 이득 인덱스 (0-based). 보통 0~2 범위",
                    }
                },
                "required": ["index"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_ccd_em_gain",
            "description": (
                "EM(Electron Multiplication) 이득을 설정한다. EMCCD 전용. "
                "높은 값일수록 신호를 크게 증폭하지만 과다 시 이미지가 포화될 수 있다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "gain": {
                        "type": "integer",
                        "description": "EM 이득값. get_ccd_info()의 em_gain_range 범위 내",
                    }
                },
                "required": ["gain"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_ccd_output_amp",
            "description": (
                "출력 앰프를 선택한다. "
                "0 = EMCCD 앰프 (EM 이득 활성화). "
                "1 = Conventional 앰프 (EM 비활성, 저노이즈)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "amp": {
                        "type": "integer",
                        "description": "앰프 선택: 0(EM) 또는 1(Conventional)",
                        "enum": [0, 1],
                    }
                },
                "required": ["amp"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_ccd_shift_speeds",
            "description": (
                "수직(VS) 및 수평(HS) 픽셀 시프트 속도를 설정한다. "
                "VS: 인덱스가 클수록 느리지만 전하 전송 노이즈 감소. "
                "HS: 인덱스가 클수록 느리지만 읽기 노이즈 감소. "
                "둘 중 하나만 지정해도 된다. 사용 가능한 속도 목록은 get_ccd_info() 참조."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "vs_index": {
                        "type": "integer",
                        "description": "수직 시프트 속도 인덱스",
                    },
                    "hs_index": {
                        "type": "integer",
                        "description": "수평 시프트 속도 인덱스",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_ccd_temperature",
            "description": (
                "CCD 냉각 목표 온도(°C)를 설정한다. "
                "낮은 온도일수록 암전류(dark current) 노이즈가 줄어든다. "
                "실제 도달까지 수 분이 걸릴 수 있으며 get_ccd_info()로 진행 상태를 확인한다. "
                "일반적 범위: -80 ~ 20°C."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "temp": {
                        "type": "integer",
                        "description": "목표 온도 [°C]. 예: -40, -60, -80",
                    }
                },
                "required": ["temp"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_ccd_cooler",
            "description": "CCD 펠티에 냉각기를 켜거나(true) 끈다(false). 종료 전에는 반드시 꺼야 한다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "on": {
                        "type": "boolean",
                        "description": "true = 냉각기 ON, false = 냉각기 OFF",
                    }
                },
                "required": ["on"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_ccd_shutter",
            "description": (
                "CCD 셔터 모드를 설정한다. "
                "'auto'  — 취득 시 자동으로 열고 닫음 (정상 측정). "
                "'open'  — 강제로 열어둠. "
                "'close' — 강제로 닫아둠 (다크 프레임/배경 측정 시)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "description": "셔터 모드",
                        "enum": ["auto", "open", "close"],
                    }
                },
                "required": ["mode"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_ccd_image_flip",
            "description": "취득 이미지의 수평/수직 반전을 설정한다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "hflip": {
                        "type": "boolean",
                        "description": "true = 수평 좌우 반전",
                    },
                    "vflip": {
                        "type": "boolean",
                        "description": "true = 수직 상하 반전",
                    },
                },
                "required": ["hflip", "vflip"],
            },
        },
    },
    # ── 스테이지 속도 (버그 수정: 이전에 TOOL_DISPATCH 미등록이었음) ─────────────
    {
        "type": "function",
        "function": {
            "name": "get_stage_speed",
            "description": "현재 스테이지 이동 속도(mm/s)를 조회한다. x_speed_mm_s, y_speed_mm_s, z_speed_mm_s 필드가 반환된다.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_stage_speed",
            "description": "스테이지 이동 속도를 설정한다. X/Y 최대 5.0 mm/s, Z 최대 0.1 mm/s.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x_speed_mm_s": {"type": "number", "description": "X축 이동 속도 (mm/s, 최대 5.0)"},
                    "y_speed_mm_s": {"type": "number", "description": "Y축 이동 속도 (mm/s, 최대 5.0)"},
                    "z_speed_mm_s": {"type": "number", "description": "Z축 이동 속도 (mm/s, 최대 0.1). 생략 가능"},
                },
                "required": ["x_speed_mm_s", "y_speed_mm_s"],
            },
        },
    },
    # ── 레이저 — 가이드빔 ─────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "set_guide_beam_mode",
            "description": (
                "레이저를 가이드빔 대기 상태로 전환한다. "
                "빔 스플리터를 대기 위치로, ND 필터를 메인 빔 차단 위치로 이동한다. "
                "시편 정렬·초점 확인 시 사용한다."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    # ── 카메라 확장 ───────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "set_camera_exposure",
            "description": "카메라(TUCam) 노출 시간(ms)을 설정한다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ms": {"type": "number", "description": "노출 시간 [ms]. 예: 10.0, 50.0, 100.0"},
                },
                "required": ["ms"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_camera_auto_exposure",
            "description": "카메라 자동 노출을 활성화(true) 또는 비활성화(false)한다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean", "description": "true = 자동 노출 ON, false = 수동 노출"},
                },
                "required": ["enabled"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "capture_camera_frame",
            "description": (
                "카메라에서 최신 프레임 1장을 캡처하여 형태(shape), 강도 통계, "
                "선명도 점수(sharpness_score, 라플라시안 분산)를 반환한다. "
                "스트리밍이 활성화된 상태여야 한다. "
                "오토포커스 시 Z 위치별 선명도 비교에 활용할 수 있다."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_microscope_image",
            "description": (
                "TuCam 광학현미경 카메라의 현재 화면을 캡처해 이미지로 전달한다. "
                "샘플 위치 확인, 타겟 식별, 이물질 탐지 등 시각적 판단이 필요할 때 사용해라. "
                "스트리밍이 활성화된 상태여야 한다."
                "반환 시 픽셀 좌표도 함께 반환하게 되며, 이는 스테이지의 좌표가 아니므로 해당 위치로 이동이 필요할 시 move_to_pixel 도구로 변환해 이동해야 한다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "이미지에서 확인하고 싶은 내용 (선택). 예: '샘플의 위치와 밝기를 설명해줘'",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_to_pixel",
            "description": (
                "카메라 이미지 내 픽셀 좌표(pixel_x, pixel_y)를 스테이지 mm 좌표로 변환해 이동한다. "
                "이미지 중심이 현재 스테이지 위치에 대응한다. "
                "analyze_microscope_image로 타겟 픽셀 좌표를 확인한 뒤 이 tool로 이동하세요."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pixel_x": {
                        "type": "integer",
                        "description": "이미지 X 픽셀 좌표 (0 ~ 1060)",
                    },
                    "pixel_y": {
                        "type": "integer",
                        "description": "이미지 Y 픽셀 좌표 (0 ~ 800)",
                    },
                },
                "required": ["pixel_x", "pixel_y"],
            },
        },
    },
    # ── 오토포커스 ────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "run_autofocus",
            "description": (
                "가이드빔 레이저 스팟 면적 최소화 기반 힐클라이밍 오토포커스. "
                "레이저 OFF/ON 차분 이미지에서 Otsu threshold로 스팟 픽셀 수(면적)를 계산하고, "
                "면적이 최소인 Z 위치(레이저 스팟이 가장 날카로운 위치)로 스테이지를 이동한다. "
                "적응형 힐클라이밍으로 보폭을 자동 조절하며 역대 최솟값 위치로 최종 귀환한다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "initial_z": {
                        "type": "number",
                        "description": "탐색 시작 Z 위치 (mm). 생략 시 현재 Z 유지",
                    },
                    "step_size": {
                        "type": "number",
                        "description": "초기 Z 이동 보폭 (mm). 기본 0.030 (30µm)",
                    },
                    "min_step": {
                        "type": "number",
                        "description": "최소 보폭 (mm) — 이 이하면 탐색 종료. 기본 0.001 (1µm)",
                    },
                    "max_steps": {
                        "type": "integer",
                        "description": "최대 스텝 수 — 초과 시 강제 종료. 기본 100",
                    },
                },
                "required": [],
            },
        },
    },
    # ── 데이터 저장 / 로드 ────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "save_spectrum",
            "description": (
                "스펙트럼 강도 배열을 CSV 파일로 저장한다. "
                "raman_shift를 함께 전달하면 캘리브레이션 정보도 포함된다. "
                "파일은 프로젝트 루트/data/ 디렉토리에 저장된다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "강도(intensity) 배열",
                    },
                    "filename": {
                        "type": "string",
                        "description": "저장 파일명 (.csv 확장자 없어도 됨). 예: 'polystyrene_01'",
                    },
                    "raman_shift": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "라만 시프트 축 [cm⁻¹]. 생략 가능",
                    },
                    "wavelength_nm": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "파장 축 [nm]. 생략 가능",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "추가 메타데이터 (노출 시간, 출력 등). 같은 이름의 .json으로 저장",
                    },
                },
                "required": ["data", "filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_spectrum",
            "description": (
                "저장된 스펙트럼 CSV 파일을 로드한다. "
                "data/ 디렉토리 기준 상대 경로 또는 절대 경로 모두 허용."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "파일명 또는 경로. 예: 'polystyrene_01' 또는 'polystyrene_01.csv'",
                    }
                },
                "required": ["filename"],
            },
        },
    },
    # ── 세션 관리 ──────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "create_session",
            "description": (
                "새 실험 세션 디렉토리를 생성하고 메타데이터를 초기화한다. "
                "이후 save_point_data()로 포인트별 스펙트럼·위치 데이터를 저장할 수 있다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "세션 식별자. 예: 'EXP_001'",
                    }
                },
                "required": ["session_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_point_data",
            "description": (
                "실험 세션의 특정 측정 포인트에 스펙트럼·위치 데이터를 저장한다. "
                "create_session()으로 세션을 먼저 생성해야 한다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "세션 ID. 예: 'EXP_001'",
                    },
                    "point_id": {
                        "type": "string",
                        "description": "포인트 식별자. 예: 'P001'",
                    },
                    "spectrum_data": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "강도 배열. 생략 가능",
                    },
                    "raman_shift": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "라만 시프트 축 [cm⁻¹]. 생략 가능",
                    },
                    "position": {
                        "type": "object",
                        "description": "스테이지 위치 {'x': ..., 'y': ..., 'z': ...}. 생략 가능",
                    },
                },
                "required": ["session_id", "point_id"],
            },
        },
    },
    # ── 배경 제거 (IPBSA) ────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "apply_background_subtraction",
            "description": (
                "IPBSA(반복 다항식 배경 제거 알고리즘)로 라만 스펙트럼의 형광 배경을 제거한다. "
                "가장 최근에 수집한 스펙트럼(source='last') 또는 저장된 파일 경로를 소스로 사용한다. "
                "결과는 version_label로 저장되며 list_bg_versions()로 여러 버전을 비교할 수 있다. "
                "poly_order를 높이면 더 복잡한 배경을 제거할 수 있지만 과적합 위험이 있다. "
                "사용자가 다항식 차수를 지정하지 않으면 기본값 5를 사용하라."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "poly_order": {
                        "type": "integer",
                        "description": "다항식 차수 (2~10). 기본값 5. 낮을수록 부드러운 배경, 높을수록 복잡한 배경 추정.",
                        "minimum": 2,
                        "maximum": 10,
                    },
                    "max_iterations": {
                        "type": "integer",
                        "description": "최대 반복 횟수 (10~500). 기본값 100.",
                        "minimum": 10,
                        "maximum": 500,
                    },
                    "threshold": {
                        "type": "number",
                        "description": "수렴 기준 — 반복 간 배경 곡선 상대 L2 변화량 (0.001~1.0). 기본값 0.001. 작을수록 엄격한 수렴.",
                        "minimum": 0.001,
                        "maximum": 1.0,
                    },
                    "source": {
                        "type": "string",
                        "description": (
                            "배경 제거할 스펙트럼 소스. "
                            "'last': 가장 최근 acquire_spectrum() 결과 사용 (기본). "
                            "그 외: 파일 경로 (JSON 또는 CSV, data/ 기준 상대경로 허용)."
                        ),
                    },
                    "version_label": {
                        "type": "string",
                        "description": (
                            "이 결과에 붙일 버전 이름. 예: 'v1_poly5', 'v2_poly7'. "
                            "같은 이름으로 다시 호출하면 덮어쓴다. 기본값 'default'."
                        ),
                    },
                    "save_result": {
                        "type": "boolean",
                        "description": "True이면 보정 스펙트럼을 data/ 디렉토리에 CSV로 저장한다. 기본값 false.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_bg_versions",
            "description": (
                "저장된 모든 배경 제거 결과 버전의 목록과 파라미터, 주요 통계를 반환한다. "
                "데이터 배열은 포함되지 않는다. 전체 데이터는 get_bg_version()을 사용하라. "
                "apply_background_subtraction()을 여러 번 호출해 비교할 때 사용한다."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_bg_version",
            "description": (
                "특정 버전의 배경 제거 결과 전체 데이터(보정 스펙트럼, 배경 곡선, 라만 시프트 축)를 반환한다. "
                "version_label은 list_bg_versions()에서 확인한다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "version_label": {
                        "type": "string",
                        "description": "조회할 버전 이름. 예: 'v1_poly5'",
                    },
                },
                "required": ["version_label"],
            },
        },
    },
]
