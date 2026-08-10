"""
LLM에게 전달할 tool 스키마 정의 (Ollama tool calling 포맷)

[겹치는 도구를 어떻게 구분해 적었는가 — 2026-07-30]
이 툴셋에는 '같은 일을 하는 길이 둘'인 자리가 여럿 있다(측정 조건을 미리 걸기 vs
acquire_spectrum 인자로 넘기기, 화면을 찍는 세 도구, 저장물을 나열하는 네 도구 등).
구현은 공용 경로로 합쳤지만, 모델은 코드를 못 보고 이 설명만 읽는다. 그래서 겹치는
도구마다 description 에 다음 세 가지를 반드시 적는다:
  1) 이 도구가 하는 일         2) 겹치는 다른 도구의 이름
  3) 어느 쪽을 언제 쓰는가(+ 둘 다 부르지 말라는 지시)
설명을 줄일 때 3)을 먼저 지우지 말 것 — 도구를 중복 호출하거나 엉뚱한 쪽을 골라
실패하는 경우의 대부분이 3)의 부재에서 온다.

[스키마에 적는 수치는 config 에서 가져온다]
스테이지 가동범위 같은 값을 문자열에 직접 적어 두면 Config.ini 가 바뀌어도 설명만
옛 값으로 남는다(실제로 그랬다: 스키마 0-75.3 vs 실제 75.7431). 아래 _X/_Y/_Z 참고.
"""
from backend.config import (
    STAGE_MAX_X as _MAX_X, STAGE_MAX_Y as _MAX_Y,
    STAGE_MIN_Z as _MIN_Z, STAGE_MAX_Z as _MAX_Z,
    CAMERA_WIDTH as _CAM_W, CAMERA_HEIGHT as _CAM_H,
)

RAMAN_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "move_stage",
            "description": (
                "Move the stage to an absolute position (mm). Out-of-range targets are REJECTED "
                "with an error rather than clipped, so the reported position is always where the "
                "stage actually is. Related: move_stage_relative (same limits, but you give a "
                "displacement), move_to_pixel (you give a point on the camera image)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": f"X-axis position (mm, 0-{_MAX_X})",
                          "minimum": 0, "maximum": _MAX_X},
                    "y": {"type": "number", "description": f"Y-axis position (mm, 0-{_MAX_Y})",
                          "minimum": 0, "maximum": _MAX_Y},
                    "z": {"type": "number",
                          "description": (f"Z-axis position (mm, {_MIN_Z}-{_MAX_Z}). "
                                          "Optional - omit to keep the current Z."),
                          "minimum": _MIN_Z, "maximum": _MAX_Z},
                },
                "required": ["x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stage_position",
            "description": (
                "Read the current stage X, Y, Z position (mm). You do NOT need this right after a "
                "move - move_stage / move_stage_relative / move_to_pixel already return the "
                "resulting position. get_hardware_status also reports it while diagnosing "
                "connection problems."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_stage_relative",
            "description": (
                "Move the stage by a displacement from its current position (mm). The resulting "
                "target is range-checked exactly like move_stage, so a displacement that would "
                "leave the travel range is rejected (the error tells you the computed target). "
                "Use move_stage when you know the absolute coordinate."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dx": {"type": "number", "description": "Displacement in X (mm)"},
                    "dy": {"type": "number", "description": "Displacement in Y (mm)"},
                    "dz": {"type": "number", "description": "Displacement in Z (mm)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "laser_on",
            "description": ("Turn the laser on. Which beam actually comes out depends on whether the "
                            "power is currently applied to the optics: if it is, the MEASUREMENT beam "
                            "fires; otherwise only the GUIDE beam does (the ND filter is still "
                            "blocking). The response tells you which one via the 'beam' field - check "
                            "it, because a guide-beam 'ON' produces no Raman signal. "
                            "This tool CANNOT arm the measurement beam - use set_laser_power(percent) to "
                            "arm it without firing, or acquire_spectrum(power=...) which arms it and "
                            "handles on -> acquire -> off atomically. Use "
                            "laser_on by itself only for alignment with the guide beam."),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "laser_off",
            "description": ("Stop the laser from firing. The ND filter and the beam splitter stay "
                            "where they are, so turning it on again emits the same measurement beam - "
                            "but the camera also stays blind until you call set_guide_beam_mode. "
                            "You do not need to call this after acquire_spectrum - that tool always "
                            "turns the laser off (even when it fails) and restores the camera view."),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    # [복원 — set_laser_power, 2026-07-31]
    # 한동안 스키마·디스패치 양쪽에서 내렸다가 되살렸다. 내렸을 때의 문제:
    #   · "레이저 파워를 13%로 맞춰 줘"처럼 측정을 요구하지 않는 요청에 답할 수단이
    #     사라졌다. 유일한 대안인 acquire_spectrum(power=) 은 레이저를 실제로 쏘므로,
    #     사용자가 요청하지도 않은 조사를 시료에 넣게 된다.
    #   · laser_on / get_laser_status 의 안내 문구가 이 도구를 부르라고 지시하고 있어서
    #     모델이 "Unknown tool" 을 받는 경로가 남아 있었다.
    # 되살리는 쪽이 맞다. 다만 스키마와 TOOL_DISPATCH 는 **항상 함께** 바꿔야 한다 —
    # 한쪽만 건드리면 모델에게는 보이는데 호출은 실패하는(또는 그 반대) 불일치가 된다.
    {
        "type": "function",
        "function": {
            "name": "set_laser_power",
            "description": (
                "Set the laser power (ND filter transmission, 0.004-100 %) WITHOUT firing the laser. "
                "This only moves the ND filter to arm the measurement beam - it does not turn the laser on. "
                "Use it when the user asks you to set or change the power by itself, or to arm the beam "
                "before alignment. "
                "For an actual measurement prefer acquire_spectrum(power=...), which applies the power, "
                "fires, measures and turns the laser off atomically - chaining "
                "set_laser_power + laser_on + laser_off leaves the beam on the sample during your own "
                "reasoning time, which can photobleach or damage a biological sample. "
                "Out-of-range values are REJECTED, not silently clamped."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "percent": {
                        "type": "number",
                        "description": "Laser power as ND filter transmission in percent (0.004-100).",
                    },
                },
                "required": ["percent"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_laser_status",
            "description": ("Query the current laser state: whether it is firing (is_on), the applied "
                            "power (power_percent, %), and power_armed - whether that power is actually "
                            "applied to the optics right now. If power_armed is false the ND filter is "
                            "at the blocking position, so laser_on() would emit only the guide beam; "
                            "arm it with set_laser_power(percent), or measure with "
                            "acquire_spectrum(power=...), which applies the power itself. "
                            "Always read power_armed, not just power_percent - power_percent is the "
                            "last value that was requested and survives a switch to guide-beam mode, "
                            "so on its own it will make you think the beam is ready when it is not."),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "acquire_spectrum",
            "description": (
                "Acquire a Raman spectrum at the current position. "
                "Supports three acquisition modes: Single (one shot) / Accumulate (averaged, high SNR) / Kinetic (continuous time series). "
                "Automatically handles the laser ON -> power stabilization -> CCD acquisition -> laser OFF flow. "
                "IMPORTANT: every parameter is optional and OMITTING one means 'keep whatever the "
                "instrument is already set to' (shutter works the same way, except that it opens to "
                "'auto' when nobody has ever set it - the CCD boots with the shutter closed). "
                "So set_ccd_exposure / set_ccd_acquisition_mode / set_ccd_read_mode / "
                "set_ccd_trigger_mode are respected - you may either configure first and then call "
                "this with no arguments, or pass the values directly here. Both go through the same "
                "code and are equivalent; do not do both 'just in case'. "
                "This is also the ONLY way to fire the measurement beam: it applies the power, turns "
                "the laser on, acquires, and turns it off again even if the acquisition fails. "
                "It also puts the optics back into the guide-beam/camera position afterwards, so the "
                "microscope camera can see the sample again - you do NOT need to call "
                "set_guide_beam_mode after measuring. "
                "The returned exposure_time / laser_power_pct / num_accumulations are read back from "
                "the hardware, so they tell you what was ACTUALLY used. "
                "When a calibrator is connected, the raman_shift_cm-1, wavelength_nm, laser_nm fields are included. "
                "Kinetic mode returns per-frame data in a frames array."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "exposure": {
                        "type": "number",
                        "description": ("CCD exposure time (seconds). Omit to keep the CCD's current "
                                        "exposure (e.g. one set earlier by set_ccd_exposure)."),
                    },
                    "power": {
                        "type": "number",
                        "description": ("Laser power (transmittance %), a real value in 0.004-100. "
                                        "Omit ONLY to reuse a power you already set earlier in this "
                                        "session - if none has ever been set the call is REFUSED rather "
                                        "than defaulted, because choosing a dose for an unknown sample "
                                        "is your decision, not the tool's. Higher power gives more "
                                        "signal but photobleaches or burns fragile samples; when the "
                                        "sample's tolerance is unknown, start low and raise it after "
                                        "looking at the result."),
                        "minimum": 0.004,
                        "maximum": 100,
                    },
                    "stabilize_sec": {
                        "type": "number",
                        "description": "Wait time for power stabilization after laser ON (seconds). Default 0.5",
                    },
                    "acq_mode": {
                        "type": "string",
                        "enum": ["single", "accumulate", "kinetic"],
                        "description": (
                            "CCD acquisition mode. Omit to keep the CCD's current mode. "
                            "'single': single shot. "
                            "'accumulate': sum num_accumulations shots -> high-SNR spectrum. "
                            "'kinetic': acquire kinetic_count frames continuously -> time-series analysis."
                        ),
                    },
                    "num_accumulations": {
                        "type": "integer",
                        "description": ("Accumulations per frame in accumulate/kinetic mode. "
                                        "Omit to keep the current value."),
                        "minimum": 1,
                    },
                    "kinetic_count": {
                        "type": "integer",
                        "description": ("Total number of frames to acquire in kinetic mode. "
                                        "Omit to keep the current value."),
                        "minimum": 1,
                    },
                    "kinetic_cycle_time": {
                        "type": "number",
                        "description": "Frame interval in kinetic mode (seconds). If omitted, the SDK auto-computes the minimum.",
                    },
                    "read_mode": {
                        "type": "string",
                        "enum": ["fvb", "single_track"],
                        "description": (
                            "CCD readout mode. Omit to keep the CCD's current read mode. "
                            "'fvb': Full Vertical Binning - sum all rows, 1D spectrum. "
                            "'single_track': read only a specific track - single_track_center required."
                        ),
                    },
                    "hbin": {
                        "type": "integer",
                        "description": "Horizontal binning pixel count. Omit to keep the current value.",
                        "minimum": 1,
                    },
                    "single_track_center": {
                        "type": "integer",
                        "description": ("Center pixel row number when read_mode='single_track'. "
                                        "Omit to reuse the currently configured track."),
                    },
                    "single_track_width": {
                        "type": "integer",
                        "description": ("Track width (pixels) when read_mode='single_track'. "
                                        "Omit to keep the current value."),
                        "minimum": 1,
                    },
                    "trigger_mode": {
                        "type": "string",
                        "enum": ["internal", "external", "external_start", "external_exposure", "external_fvb_em", "software"],
                        "description": "CCD trigger mode. Omit to keep the CCD's current trigger mode.",
                    },
                    "shutter": {
                        "type": "string",
                        "enum": ["auto", "open", "close"],
                        "description": (
                            "Shutter mode for this acquisition. Omit to KEEP the current setting - "
                            "including one you set earlier with set_ccd_shutter. If nobody has ever set "
                            "it, it opens to 'auto' (the CCD boots closed, so keeping that would "
                            "silently hand you a dark frame). "
                            "Use 'close' to acquire a DARK / background frame with no light reaching "
                            "the detector - that is the supported way to measure a dark reference."
                        ),
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
            "description": (
                "Start real-time camera streaming. Streaming must be ON before "
                "analyze_microscope_image, capture_scene, preview_grid_scan or run_autofocus can "
                "get a frame. The response field already_streaming tells you whether it was ALREADY "
                "running before your call - if it was true, someone else (the user's live view) owns "
                "the stream, so do not stop it afterwards."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop_camera_stream",
            "description": (
                "Stop real-time camera streaming. Only call this if YOU started the stream - i.e. "
                "start_camera_stream returned already_streaming=false. Stopping a stream the user "
                "was watching blanks their live view, and every image tool stops working until it is "
                "restarted. When in doubt, leave it running."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    # ── CCD 파라미터 설정 ────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_ccd_info",
            "description": (
                "Query all current CCD settings and status. "
                "Returns temperature, cooling status, exposure time, acquisition mode, readout mode, gain, shift speeds, "
                "pixel count, etc. in one call. Use it before and after changing parameters to verify the current state."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_ccd_exposure",
            "description": ("Set the CCD exposure time (seconds). Larger values give stronger signal "
                            "but longer measurement time. The value persists - a later "
                            "acquire_spectrum keeps it unless you pass exposure there. "
                            "OVERLAP: acquire_spectrum(exposure=...) sets exactly the same thing "
                            "through the same code. Use this tool when you set the exposure once and "
                            "then measure several times; pass it to acquire_spectrum when it is a "
                            "one-off. Never do both for the same measurement."),
            "parameters": {
                "type": "object",
                "properties": {
                    "exposure_time": {
                        "type": "number",
                        "description": "Exposure time [seconds]. e.g. 0.1, 0.5, 1.0",
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
                "Set the CCD acquisition mode. "
                "'single': single shot. "
                "'accumulate': sum after num_accumulations shots. "
                "'kinetic': acquire num_kinetics frames continuously. "
                "'run_till_abort': acquire continuously until an abort command (acquire_spectrum "
                "cannot use this one - set single/accumulate/kinetic before measuring). "
                "The mode and counts persist - acquire_spectrum keeps them unless you pass "
                "acq_mode / num_accumulations there. "
                "OVERLAP: acquire_spectrum(acq_mode=, num_accumulations=, kinetic_count=) sets the "
                "same thing through the same code; use one or the other, not both. This tool is the "
                "only way to select 'run_till_abort'. The response reports the counts actually in "
                "effect on the hardware, not what you asked for."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "description": "Acquisition mode",
                        "enum": ["single", "accumulate", "kinetic", "run_till_abort"],
                    },
                    # minimum 을 acquire_spectrum 쪽과 맞춘다(2026-08-05) — 도구 계층이
                    # 1 미만을 거부하므로 그 제약을 스키마에도 적어 둔다. 선언하지 않으면
                    # 모델은 0 을 '기능 끄기'로 보낼 수 있고, 무엇이 잘못됐는지는 실행해
                    # 봐야만 알게 된다.
                    "num_accumulations": {
                        "type": "integer",
                        "minimum": 1,
                        "description": ("Number of accumulations (used in accumulate/kinetic mode). "
                                        "Omit to keep the value already on the CCD - 0 is not a way "
                                        "to switch accumulation off and is rejected. Note that "
                                        "accumulate mode with 1 accumulation is just a single "
                                        "shot - set this deliberately when you want averaging."),
                    },
                    "num_kinetics": {
                        "type": "integer",
                        "minimum": 1,
                        "description": ("Total number of frames to acquire (used in kinetic mode). "
                                        "Omit to keep the value already on the CCD."),
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
                "Set the CCD trigger mode. "
                "'internal': start acquisition via software (default). "
                "'external': start acquisition on an external TTL signal. "
                "'external_start': start on external trigger then internal timing. "
                "'external_exposure': expose while the external TTL is HIGH. "
                "'external_fvb_em': external trigger in FVB/EM readout. "
                "'software': use SendSoftwareTrigger. "
                "The mode persists - acquire_spectrum keeps it unless you pass trigger_mode there. "
                "OVERLAP: acquire_spectrum(trigger_mode=...) accepts exactly the same values through "
                "the same code. Use one or the other, not both."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "description": "Trigger mode",
                        "enum": ["internal", "external", "external_start", "external_exposure",
                                 "external_fvb_em", "software"],
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
                "Set the CCD readout mode. "
                "'fvb': Full Vertical Binning - sum all rows -> 1D spectrum (used for Raman). "
                "'single_track': read only a specific vertical row. single_track_center is required. "
                "'image': full 2D image or ROI (acquire_spectrum cannot build a 1D spectrum from this - "
                "switch back to 'fvb' before measuring). "
                "The mode persists - acquire_spectrum keeps it unless you pass read_mode there. "
                "OVERLAP: acquire_spectrum(read_mode=, hbin=, single_track_center=, "
                "single_track_width=) sets the same thing through the same code, using the same "
                "parameter names. Use one or the other, not both. This tool is the only way to select "
                "'image' mode."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "description": "Readout mode",
                        "enum": ["fvb", "single_track", "image"],
                    },
                    "hbin": {
                        "type": "integer",
                        "description": "Horizontal binning factor. Omit to keep the current value.",
                    },
                    "single_track_center": {
                        "type": "integer",
                        "description": "Center row number to read in single_track mode (1-based), e.g. 256. Required the first time you use single_track; omit to reuse the configured track.",
                    },
                    "single_track_width": {
                        "type": "integer",
                        "description": "Number of rows to read in single_track mode. Omit to keep the current value.",
                    },
                },
                "required": ["mode"],
            },
        },
    },
    # [제거됨 — set_mcp_gain / get_mcp_gain_range, 2026-07-30]
    # 이 카메라는 MCP 이득을 지원하지 않는다(SDK 가 DRV_NOT_SUPPORTED 20991 반환).
    # 노출해 두면 "게인을 낮춰 포화를 해결" 류 과제에서 반드시 실패하는 경로로 유인된다.
    # 대체 수단: set_ccd_preamp_gain + get_ccd_info().preamp_gains_available
    {
        "type": "function",
        "function": {
            "name": "set_ccd_preamp_gain",
            "description": (
                "Set the pre-amplifier gain index. "
                "See preamp_gains_available in get_ccd_info() for the available gain list. "
                "A larger index gives higher gain, which helps measure weak signals but also increases noise."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": "Pre-amplifier gain index (0-based). Typically in the range 0-2",
                    }
                },
                "required": ["index"],
            },
        },
    },
    # [스키마에서 제거 — set_ccd_em_gain / set_ccd_output_amp, 2026-07-31]
    # 이 카메라는 EM CCD 가 아니라 iDus 다(Config.ini: CCDType 0 = IDUS). 두 툴은 각각
    # "항상 에러만 반환" / "선택지가 하나뿐" 이라 모델을 막다른 길로 유인하기만 했다 —
    # set_mcp_gain 을 내린 것과 같은 사유다. 자세한 근거는 raman_tools.py 의 같은 자리 주석.
    # 이 장비에서 이득을 조절하는 유일한 수단은 set_ccd_preamp_gain 이다.
    {
        "type": "function",
        "function": {
            "name": "set_ccd_shift_speeds",
            "description": (
                "Set the vertical (VS) and horizontal (HS) pixel shift speeds. "
                "VS: a larger index is slower but reduces charge-transfer noise. "
                "HS: a larger index is slower but reduces readout noise. "
                "You may specify only one of them. See get_ccd_info() for the available speed list."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "vs_index": {
                        "type": "integer",
                        "description": "Vertical shift speed index",
                    },
                    "hs_index": {
                        "type": "integer",
                        "description": "Horizontal shift speed index",
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
                "Set the CCD cooling target temperature (°C). "
                "Lower temperature reduces dark-current noise. "
                "It may take several minutes to reach; check progress with get_ccd_info(). "
                "Typical range: -80 to 20°C."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "temp": {
                        "type": "integer",
                        "description": "Target temperature [°C]. e.g. -40, -60, -80",
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
            "description": "Turn the CCD Peltier cooler on (true) or off (false). It must be turned off before shutdown.",
            "parameters": {
                "type": "object",
                "properties": {
                    "on": {
                        "type": "boolean",
                        "description": "true = cooler ON, false = cooler OFF",
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
                "Set the CCD shutter mode. "
                "'auto'  - open and close automatically during acquisition (normal measurement). "
                "'open'  - force open. "
                "'close' - force closed. "
                "Like the other CCD settings this one PERSISTS: a later acquire_spectrum keeps it "
                "unless you pass its own shutter argument. So for several dark / background frames "
                "set 'close' once here and measure repeatedly; for a single dark frame "
                "acquire_spectrum(shutter='close') is enough. Set it back to 'auto' before normal "
                "measurements."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "description": "Shutter mode",
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
            "description": ("Set horizontal/vertical flip of the acquired 2D image. Only valid when "
                            "the CCD read mode is 'image' - in the 1D spectrum modes (fvb / "
                            "single_track) flipping would misalign the intensity array against the "
                            "calibrated wavelength axis, and the correct orientation is already set "
                            "at startup, so the call is rejected there. You almost never need this "
                            "for Raman measurements."),
            "parameters": {
                "type": "object",
                "properties": {
                    "hflip": {
                        "type": "boolean",
                        "description": "true = flip horizontally (left-right)",
                    },
                    "vflip": {
                        "type": "boolean",
                        "description": "true = flip vertically (up-down)",
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
            "description": "Query the current stage movement speed (mm/s). Returns the x_speed_mm_s, y_speed_mm_s, z_speed_mm_s fields.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_stage_speed",
            "description": ("Set the stage movement speed. X/Y max 5.0 mm/s, Z max 0.1 mm/s. If a "
                            "specific axis speed is omitted, its current speed is maintained. Values "
                            "above the limit are clipped to it; the response reports the speeds that "
                            "will ACTUALLY be used and lists any clipped axes under 'clipped'."),
            "parameters": {
                "type": "object",
                "properties": {
                    "x_speed_mm_s": {"type": "number", 
                                     "description": "X-axis movement speed (mm/s, max 5.0). Optional."},
                    "y_speed_mm_s": {"type": "number", 
                                     "description": "Y-axis movement speed (mm/s, max 5.0). Optional."},
                    "z_speed_mm_s": {"type": "number", 
                                     "description": "Z-axis movement speed (mm/s, max 0.1). Optional."},
                },
                "required": [],
            },
        },
    },
    # ── 하드웨어 연결 관리 ────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_hardware_status",
            "description": (
                "Report which hardware components (stage, laser, ccd, camera) are currently CONNECTED, "
                "and for the connected ones whether they actually respond. Read `summary` first. "
                "Call this FIRST whenever a hardware tool fails, before trying to fix anything - it tells "
                "you whether one device is down or several, which decides what is even worth attempting. "
                "It touches nothing and fires no laser, so it is always safe to call. "
                "SCOPE: this answers 'is it there and alive'. For the SETTINGS of a working device use the "
                "per-device tools instead - get_ccd_info (exposure, mode, temperature), get_laser_status "
                "(power and whether it is armed), get_stage_position, get_stage_speed."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reconnect_hardware",
            "description": (
                "Release and re-initialize a hardware component that is unresponsive or stuck. "
                "component: 'stage' | 'ccd' | 'camera' | 'laser' | 'all'. "
                "Call get_hardware_status first to see what is actually down. "
                "Read the returned `errors` text carefully - it distinguishes two very different cases. "
                "(a) 'resource is still held by this process': a process-level lock that NO tool can clear. "
                "Calling this tool again will not help; the server must be restarted by a human. Do not retry - "
                "carry on without that component and say so in your final answer. "
                "(b) 're-initialization failed' after a successful release: a device-side problem (power, cable, "
                "driver, or another program holding the device). Retrying once is reasonable; beyond that, proceed "
                "without the component and state the limitation. "
                "Never call this repeatedly in a loop - it cannot fix either case by repetition. "
                "WARNING: reconnecting the 'ccd' re-runs cooling and can block for minutes until -40 C stabilizes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "component": {
                        "type": "string",
                        "enum": ["stage", "ccd", "camera", "laser", "all"],
                        "description": "Which component to reconnect. Default 'all'.",
                    }
                },
                "required": [],
            },
        },
    },
    # ── 레이저 — 가이드빔 ─────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "set_guide_beam_mode",
            "description": (
                "Switch the laser to guide-beam standby state. "
                "Moves the beam splitter to the standby position and the ND filter to the main-beam blocking position. "
                "Use it for sample alignment and focus checking."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    # ── 카메라 확장 ───────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "set_camera_exposure",
            "description": "Set the camera (TUCam) exposure time (ms).",
            "parameters": {
                "type": "object",
                "properties": {
                    "ms": {"type": "number", "description": "Exposure time [ms]. e.g. 10.0, 50.0, 100.0"},
                },
                "required": ["ms"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_camera_auto_exposure",
            "description": "Enable (true) or disable (false) camera auto exposure.",
            "parameters": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean", "description": "true = auto exposure ON, false = manual exposure"},
                },
                "required": ["enabled"],
            },
        },
    },
    # [제거됨 — capture_camera_frame, 2026-07-30] analyze_microscope_image 와 같은 일을 하면서
    # uint16 프레임을 정규화하지 않아 통계 스케일이 어긋났고, sharpness_score 를 오토포커스
    # 지표로 안내했지만 run_autofocus 는 '스팟 면적'을 써서 서로 무관했다.
    # 통계·선명도는 analyze_microscope_image 반환에 병합했다.
    {
        "type": "function",
        "function": {
            "name": "analyze_microscope_image",
            "description": (
                "Capture the current view of the TuCam optical microscope camera and pass it to YOU as an image "
                "to look at. Use it when visual judgment is needed, e.g. checking sample position, identifying "
                "a target, or detecting debris. Streaming must be active. "
                "The tool returns the IMAGE ITSELF - it does NOT return any coordinates. YOU must look "
                "at the image, find the target in it, and read off its pixel coordinates yourself; do "
                "not search the JSON response for an x/y field, there is none. Those pixel coordinates "
                "are not stage coordinates, so to move there pass them to move_to_pixel. "
                "It also returns brightness statistics (min/max/mean_intensity) and a relative "
                "sharpness_score for the same image - use those to check exposure or compare frames. "
                "Do NOT use sharpness_score to focus manually: run_autofocus optimises a different "
                "metric (guide-beam spot area) and would settle at a different Z. "
                "OVERLAP - three tools capture the camera, pick by PURPOSE: "
                "analyze_microscope_image = you look at it now (returns the image to you, saves nothing); "
                "capture_scene = save the view as a file so run_analysis can draw a peak map on top of it "
                "(returns no image for you to inspect); "
                "preview_grid_scan = show where a planned grid would land. "
                "All three return the same pixel coordinate system, so a coordinate read from any of them "
                "can be passed to move_to_pixel. "
                "This tool saves NOTHING: it does not become 'the image at this point' for "
                "save_measurement_point. If you want this view bundled into a measurement-point record, "
                "call capture_scene as well."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "What you want to check in the image (optional). e.g. 'Describe the sample position and brightness'",
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
                "Convert pixel coordinates (pixel_x, pixel_y) within the camera image to stage mm coordinates and move there. "
                "The image center corresponds to the current stage position. "
                "After checking the target pixel coordinates with analyze_microscope_image, move with this tool."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pixel_x": {
                        "type": "integer",
                        "description": f"Image X pixel coordinate (0 - {_CAM_W})",
                    },
                    "pixel_y": {
                        "type": "integer",
                        "description": f"Image Y pixel coordinate (0 - {_CAM_H})",
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
                "Hill-climbing autofocus based on minimizing the guide-beam laser spot area. "
                "Computes the spot pixel count (area) via Otsu thresholding on the laser OFF/ON difference image, "
                "and moves the stage to the Z position with minimum area (where the laser spot is sharpest). "
                "Adaptive hill-climbing auto-adjusts the step size and finally returns to the historical minimum position. "
                "It moves Z only, uses the GUIDE beam (not the measurement beam), and leaves the laser off. "
                "The search is clipped to the Z travel range; if the response contains z_limit_hits the focus "
                "may be physically out of reach, and repeating the call will NOT help - report that instead. "
                "This is the only focusing tool: do not try to focus by comparing sharpness_score from "
                "analyze_microscope_image, which is a different metric and converges elsewhere."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "initial_z": {
                        "type": "number",
                        "description": "Starting Z position for the search (mm). If omitted, keep the current Z",
                    },
                    "step_size": {
                        "type": "number",
                        "description": "Initial Z step size (mm). Default 0.030 (30 um)",
                    },
                    "min_step": {
                        "type": "number",
                        "description": "Minimum step size (mm) - the search ends below this. Default 0.001 (1 um)",
                    },
                    "max_steps": {
                        "type": "integer",
                        "description": "Maximum number of steps - forced stop when exceeded. Default 100",
                    },
                },
                "required": [],
            },
        },
    },
    # ── 그리드 매핑(미리보기 + 실행) ──────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "preview_grid_scan",
            "description": (
                "Preview a rows x cols grid mapping WITHOUT moving the stage or firing the laser. "
                "Overlays the planned scan points as circles on the current camera view and returns that "
                "image so the layout can be visually verified before committing. "
                "ORIENTATION (do not confuse the two): rows = number of points stacked VERTICALLY = grid "
                "HEIGHT (stage Y axis); cols = number of points side-by-side HORIZONTALLY = grid WIDTH "
                "(stage X axis). So rows=3, cols=2 is a TALL grid (3 high x 2 wide) and rows=2, cols=3 is a "
                "WIDE grid (2 high x 3 wide) - these are DIFFERENT layouts, never swap them. When the user "
                "asks for a grid like 'A x B', decide deliberately which number is the horizontal count "
                "(width -> cols) and which is the vertical count (height -> rows), then use this preview "
                "image to confirm the drawn orientation matches what they asked. "
                "MANDATORY HUMAN APPROVAL: always preview FIRST, then STOP - show this preview image to the "
                "user, end your turn, and WAIT. Do NOT call run_grid_scan in the same turn as this preview; "
                "only call it in a later turn after the user has explicitly approved this exact layout. "
                "If center_x/center_y are omitted, the current stage position is used as the grid center, "
                "and that resolved center is what gets approved - a later run_grid_scan with the centre "
                "omitted scans THAT position even if the stage has moved since. "
                "Whatever you pass here you must pass IDENTICALLY to run_grid_scan: the approval is "
                "matched on the arguments themselves, so omitting the centre here and spelling it out "
                "there (or vice versa) is rejected as a mismatch even when both mean the same place. "
                "SIZE LIMIT: rows * cols must be <= 400 points; a larger grid is refused here, so agree "
                "a smaller size with the user before promising a map. "
                "The camera field of view is small, so with wide spacing some points may fall outside "
                "the frame; they are still measured, and the response reports how many are in view "
                "(n_in_view) along with the exact view size in fov_mm - plan spacing from that returned "
                "value rather than from a remembered number."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "rows": {"type": "integer", "description": "Number of grid points stacked VERTICALLY = grid HEIGHT (stage Y axis), integer >= 1. e.g. rows=3 -> 3 points tall."},
                    "cols": {"type": "integer", "description": "Number of grid points side-by-side HORIZONTALLY = grid WIDTH (stage X axis), integer >= 1. e.g. cols=2 -> 2 points wide."},
                    "spacing_mm": {"type": "number", "description": "Distance between adjacent points (mm), > 0"},
                    "center_x": {"type": "number", "description": "Grid center X (mm). Optional; defaults to current stage X"},
                    "center_y": {"type": "number", "description": "Grid center Y (mm). Optional; defaults to current stage Y"},
                },
                "required": ["rows", "cols", "spacing_mm"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_grid_scan",
            "description": (
                "Execute a rows x cols grid mapping: for each point it moves the stage, optionally autofocuses, "
                "acquires one spectrum, and auto-saves it (position-tagged). Returns a single compact summary "
                "(counts, intensity min/max/mean, and per-point data when 32 points or fewer) instead of one tool "
                "message per point - this is the token-efficient way to run a map. "
                "ORIENTATION: rows = vertical count (height, stage Y), cols = horizontal count (width, stage X); "
                "rows=3,cols=2 is a tall 3x2 grid, rows=2,cols=3 is a wide 2x3 grid - do not swap them. "
                "PASS THE APPROVED ARGUMENTS VERBATIM: the approval gate compares the arguments you "
                "give, not the physical positions they work out to. If you OMITTED center_x/center_y in "
                "the preview you must omit them here too - filling in the numeric coordinates of that "
                "same spot is rejected as an 'Approval mismatch'. (An omitted centre runs at the "
                "position the approved preview was drawn at, so the scan lands where the user saw it "
                "even if the stage has moved in between.) "
                "SIZE LIMIT: rows * cols must be <= 400 points. "
                "REQUIRES PRIOR HUMAN APPROVAL: do NOT call this in the same turn as preview_grid_scan. Call it "
                "ONLY after (1) you showed the user a preview_grid_scan image, (2) you ended that turn, and "
                "(3) the user EXPLICITLY approved that exact layout in a later message. If the user has not "
                "explicitly approved the previewed grid, do not call this - preview first and wait. "
                "The laser is fired at every point, so the "
                "estimated cumulative dose is checked up front and the scan is refused if it exceeds the safety limit. "
                "READ THE RESULT BEFORE REPORTING SUCCESS: n_measured can be lower than n_points, and with "
                "autofocus='each' the response may carry n_autofocus_failed (those points were measured at "
                "whatever Z the stage was at, so a weak signal there is an artefact, not a property of the "
                "sample) or n_autofocus_z_limit (the focus is physically out of reach - re-running will not "
                "help). If autofocus fails several times in a row the scan STOPS EARLY and comes back with "
                "ok=false plus an 'aborted' field; the points measured up to then are still saved, but the "
                "grid is incomplete and you must say so rather than reporting the map as done."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "rows": {"type": "integer", "description": "Number of grid points stacked VERTICALLY = grid HEIGHT (stage Y axis), integer >= 1. e.g. rows=3 -> 3 points tall. Must match the approved preview."},
                    "cols": {"type": "integer", "description": "Number of grid points side-by-side HORIZONTALLY = grid WIDTH (stage X axis), integer >= 1. e.g. cols=2 -> 2 points wide. Must match the approved preview."},
                    "spacing_mm": {"type": "number", "description": "Distance between adjacent points (mm), > 0"},
                    "center_x": {"type": "number", "description": "Grid center X (mm). Optional; defaults to current stage X"},
                    "center_y": {"type": "number", "description": "Grid center Y (mm). Optional; defaults to current stage Y"},
                    "autofocus": {
                        "type": "string",
                        "enum": ["each", "center", "none"],
                        "description": (
                            "Autofocus strategy. 'each' = autofocus at every point (most accurate, slowest); "
                            "'center' = autofocus once at the grid center then reuse that Z (fast, for flat samples); "
                            "'none' = no autofocus, keep current Z. "
                            "REQUIRED - this is a real trade-off, not a formality: 'each' costs an extra "
                            "Z sweep (and guide-beam exposure) at every point, which on a large grid "
                            "dominates the run time, while 'center' or 'none' will drift out of focus on a "
                            "tilted or uneven sample and quietly return weak spectra. Decide from what you "
                            "know about the sample's flatness."
                        ),
                    },
                    "exposure": {
                        "type": "number",
                        "description": (
                            "Exposure time per point (s) - REQUIRED. Together with power this sets how "
                            "much light each point receives, so choose it for THIS sample rather than "
                            "reusing a number: too short buries the peaks in read noise, too long "
                            "saturates the detector and multiplies the total run time by the point count."
                        ),
                    },
                    "power": {
                        "type": "number",
                        "description": (
                            "Laser power (%) per point - REQUIRED. This is the dose decision, and it is "
                            "applied at EVERY point, so the sample sees it rows*cols times. Higher power "
                            "raises signal but photobleaches or burns fragile samples (biological, "
                            "polymer, thin film); if you are unsure of the sample's tolerance, start low "
                            "and check one point before committing the whole grid. The cumulative dose is "
                            "estimated up front and the scan is refused outright if it exceeds the limit."
                        ),
                    },
                },
                # exposure / power / autofocus 를 필수로 둔다 — 이 세 값이 '실험 조건' 자체이고,
                # 격자 전체에 rows*cols 번 반복 적용된다. 선택으로 두면 모델이 생략하고 코드
                # 기본값(0.2s / 40% / each)이 조사량을 대신 결정한다(2026-07-31).
                "required": ["rows", "cols", "spacing_mm", "exposure", "power", "autofocus"],
            },
        },
    },
    # ── 데이터 저장 / 로드 ────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "load_spectrum",
            "description": (
                "Load ONE spectrum CSV that this system produced - an auto-saved measurement, a "
                "processed spectrum you wrote with save_result inside run_analysis, or a "
                "background-subtracted result. Accepts a path relative to data/ or an absolute path; "
                "the data/-relative path returned by list_session_artifacts or by save_result can be "
                "passed here verbatim. Returns the intensity array plus any axis columns "
                "(raman_shift_cm-1 / wavelength_nm / background_intensity) and the saved metadata. "
                "1D SPECTRA ONLY (Single / Accumulate). A Kinetic measurement is saved as one row per "
                "frame per pixel; loading it here is REFUSED rather than silently flattening the frames "
                "into one wrong array. Analyse kinetic data with run_analysis, which receives it as a "
                "2D frames array. "
                "PICK THE RIGHT TOOL: for a file the USER attached to the chat use inspect_file "
                "(different store, different format); to compute over MANY saved measurements at once "
                "use run_analysis, which receives them all as `spectra` without any loading; to re-read "
                "a background-subtraction result you made this session use get_bg_version, which needs "
                "no path."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "File name or path. Three forms are accepted: a session artifact path relative to data/ (e.g. 'runs/<session>/spectra/01_corrected.csv'), an absolute path, or a file_id from list_uploaded_files (e.g. '2026-08-07/N05.csv') to load an uploaded input file.",
                    }
                },
                "required": ["filename"],
            },
        },
    },
    # ── 측정점 기록 ────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "save_measurement_point",
            "description": (
                "Group what you just measured at this position into ONE measurement-point record: "
                "the most recent acquire_spectrum result, the most recent capture_scene microscope "
                "image, and the current stage coordinates. "
                "You do NOT pass any arrays - the spectrum and image files are already saved "
                "automatically, and this tool only links them together under a point id. "
                "TIMING IS PART OF THE RECORD: the coordinates are read from the stage AT THE MOMENT "
                "YOU CALL THIS, not from the spectrum's metadata. So call it immediately after "
                "acquiring the spectrum and capturing the view at that position, and BEFORE moving the "
                "stage anywhere else - otherwise the record pairs this point's spectrum with the next "
                "point's coordinates, and nothing later can detect that. "
                "The image must come from capture_scene; analyze_microscope_image does not count. "
                "The response lists anything that was missing (e.g. no image captured yet). "
                "Use one call per position to build a multi-point dataset."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "point_id": {
                        "type": "string",
                        "description": "Short identifier for this point, e.g. 'P001'. Used in the filename.",
                    },
                    "note": {
                        "type": "string",
                        "description": "Optional note about this point (sample region, what you observed).",
                    },
                },
                "required": ["point_id"],
            },
        },
    },
    # ── 배경 제거 (IPBSA) ────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "apply_background_subtraction",
            "description": (
                "Remove the fluorescence background of a Raman spectrum using IPBSA (iterative polynomial "
                "background subtraction). Uses the most recently acquired spectrum (source='last') or a "
                "saved file path as the source. "
                "YOU CHOOSE THE POLYNOMIAL ORDER, and that choice is the substance of this task - there is "
                "no safe default to fall back on. Too low and a curved fluorescence background survives, "
                "tilting the baseline and distorting relative peak heights; too high and the polynomial "
                "starts following the peaks themselves, eating real signal. The right order depends on how "
                "curved THIS spectrum's background is, so look at the data before deciding. "
                "DO NOT settle on the first result: run it at two or three orders, compare them with "
                "list_bg_versions(), and keep the one where the baseline is flat between peaks while peak "
                "heights are unchanged. Say which order you chose and why. "
                "OVERLAP with run_analysis: that sandbox can run any baseline algorithm you write yourself "
                "(asymmetric least squares, rolling ball, wavelet, ...). This tool is IPBSA specifically, "
                "and it keeps parameters comparable across versions and writes the standard CSV format. "
                "Pick whichever the task calls for and state which one you used."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "poly_order": {
                        "type": "integer",
                        "description": (
                            "Polynomial order (2-10) - REQUIRED, decide it yourself from the shape of "
                            "this spectrum's background. A low order (2-3) fits only a gentle slope and "
                            "will leave a curved background behind; a high order (8-10) can bend enough "
                            "to follow the peaks and subtract away real signal. Mid orders (4-6) suit "
                            "the moderate fluorescence curvature that is typical, but confirm it against "
                            "the data rather than assuming."
                        ),
                        "minimum": 2,
                        "maximum": 10,
                    },
                    "max_iterations": {
                        "type": "integer",
                        "description": "Maximum number of iterations (10-500). Default 100.",
                        "minimum": 10,
                        "maximum": 500,
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Convergence criterion - relative L2 change of the background curve between iterations (0.001-1.0). Default 0.001. Smaller means stricter convergence.",
                        "minimum": 0.001,
                        "maximum": 1.0,
                    },
                    "source": {
                        "type": "string",
                        "description": (
                            "Source spectrum to background-subtract. "
                            "'last': use the most recent acquire_spectrum() result (default). "
                            "Otherwise: a file (JSON or CSV) given as a path relative to data/, an "
                            "absolute path, or a file_id from list_uploaded_files (e.g. "
                            "'2026-08-07/N05.csv') to work directly on an uploaded input file. "
                            "TWO LIMITS ON 'last'. (1) It only works on Single/Accumulate spectra - a "
                            "Kinetic measurement has no single intensity array and is REJECTED. "
                            "(2) run_grid_scan acquires internally at every point, so right after a "
                            "grid scan 'last' means ONLY the final point of that grid, not the grid. "
                            "To baseline-correct a whole grid, pass file paths one at a time (get them "
                            "from list_results), or write the loop yourself in run_analysis."
                        ),
                    },
                    "version_label": {
                        "type": "string",
                        "description": (
                            "Version name to attach to this result. e.g. 'v1_poly5', 'v2_poly7'. "
                            "Calling again with the same name overwrites it. Default 'default'."
                        ),
                    },
                    "save_result": {
                        "type": "boolean",
                        "description": ("If True, also write the corrected spectrum to this session's "
                                        "folder as a CSV (standard format: pixel_index, "
                                        "raman_shift_cm-1, intensity, background_intensity) and return "
                                        "its data/-relative path in saved_path, which load_spectrum "
                                        "accepts. Default false - without it the result exists only in "
                                        "memory for this conversation. "
                                        "SET IT TO TRUE IF YOU WILL PLOT OR ANALYSE THE RESULT: "
                                        "run_analysis reads files, not this conversation's memory, so "
                                        "an unsaved version is invisible to it and there is no way to "
                                        "hand the arrays over afterwards. Leave it false only when you "
                                        "are just comparing poly_order settings via list_bg_versions."),
                    },
                },
                # poly_order 만 필수다 — 이 도구에서 '판단'에 해당하는 유일한 인자이고,
                # 선택으로 두면 모델이 생략해 코드 기본값이 대신 결정해 버린다(2026-07-31).
                "required": ["poly_order"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_bg_versions",
            "description": (
                "Return the list of all saved background-subtraction result versions with their parameters and key statistics. "
                "This summary is what you compare versions with - poly_order, iterations_run, converged "
                "and max_corrected_intensity are enough to pick an order. "
                "The spectra themselves are NOT included here, and you cannot read them into the "
                "conversation at all: long arrays are stripped out of every tool result to protect the "
                "context window, so get_bg_version() will not hand you the numbers either. If you need "
                "the actual corrected spectrum, re-run apply_background_subtraction with "
                "save_result=true and work on the saved file in run_analysis. "
                "Use it when calling apply_background_subtraction() several times to compare."
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
                "Re-read the parameters and statistics of one background-subtraction version "
                "(poly_order, iterations_run, converged, max intensities, saved_path if it was saved). "
                "It does NOT put the corrected spectrum in front of you: long arrays are stripped from "
                "tool results, so the corrected_data / background_data arrays never arrive - do not "
                "call this expecting to read the numbers, and do not retry when they are absent. "
                "To work with the actual arrays, save the version (save_result=true) and analyse the "
                "file in run_analysis. Check version_label with list_bg_versions()."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "version_label": {
                        "type": "string",
                        "description": "Version name to query. e.g. 'v1_poly5'",
                    },
                },
                "required": ["version_label"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_results",
            "description": (
                "Query the list of MEASUREMENTS auto-saved by acquire_spectrum. "
                "Returns each item's base (file identifier), session, title, timestamp, and meta (coordinates, etc.). "
                "Get the base to pass to combine_spectra / aggregate_spectra_csv / bundle_results here. "
                "By default this lists only the measurements from YOUR current session - your own work, "
                "not other sessions'. Files live in data/results/<date>/<your session>/. "
                "THERE ARE THREE 'list' TOOLS, one per store - choose by what you are looking for: "
                "list_results = raw measurements you acquired (this one); "
                "list_session_artifacts = files YOU produced (processed spectra from save_result, "
                "measurement-point records, figures), each with a path you can load_spectrum; "
                "list_uploaded_files = data files the USER attached to the chat. "
                "(Background-subtraction versions from this conversation are not files - use "
                "list_bg_versions.)"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date to query 'YYYY-MM-DD'. If omitted, today.",
                    },
                    "scope": {
                        "type": "string", "enum": ["session", "all"],
                        "description": (
                            "Which measurements to consider. 'session' (default) = only the ones "
                            "measured in THIS session, which is almost always what you want. "
                            "'all' = every session saved that day; use it only when the request is "
                            "explicitly about combining work from earlier, separate sessions."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "combine_spectra",
            "description": (
                "Combine several saved measurement spectra into a single grid image and render it. "
                "Each cell title uses the title auto-generated at save time (scan coordinates, power, exposure) as is. "
                "e.g. for a 10x10 scan, arranged by coordinate. If names is omitted, combine everything from that date. "
                "This is the ready-made one-call version - it needs no code. run_analysis can also plot "
                "the same measurements, but only use it when you need a layout or computation this tool "
                "does not give you (overlaid curves, peak maps, custom axes)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Target date 'YYYY-MM-DD'. If omitted, today."},
                    "names": {
                        "type": "array", "items": {"type": "string"},
                        "description": "List of measurement bases to combine (check with list_results). If omitted, the whole date.",
                    },
                    "max_cols": {"type": "integer", "description": "Number of grid columns. Default 4.", "minimum": 1},
                    "scope": {
                        "type": "string", "enum": ["session", "all"],
                        "description": (
                            "Which measurements to consider. 'session' (default) = only the ones "
                            "measured in THIS session, which is almost always what you want. "
                            "'all' = every session saved that day; use it only when the request is "
                            "explicitly about combining work from earlier, separate sessions."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "aggregate_spectra_csv",
            "description": (
                "Build a CSV summarizing several saved measurements, one row per experiment (date, time, title, coordinates, power, exposure, "
                "max intensity, total intensity, peak position). Use it to organize multiple experiments into one table. "
                "One call, no code. It summarizes ONE ROW PER MEASUREMENT - if you need per-point values "
                "computed some other way, or a table of derived quantities, use run_analysis instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Target date 'YYYY-MM-DD'. If omitted, today."},
                    "names": {
                        "type": "array", "items": {"type": "string"},
                        "description": "List of measurement bases to organize. If omitted, all of yours from that date.",
                    },
                    "scope": {
                        "type": "string", "enum": ["session", "all"],
                        "description": (
                            "Which measurements to consider. 'session' (default) = only the ones "
                            "measured in THIS session, which is almost always what you want. "
                            "'all' = every session saved that day; use it only when the request is "
                            "explicitly about combining work from earlier, separate sessions."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "capture_scene",
            "description": (
                "SAVE the current microscope (camera) view as a file, for later use as a background "
                "image. It also computes the stage-coordinate extent of that image (position + "
                "calibrated field of view), so in a later run_analysis you get microscope_image / "
                "image_extent injected and can overlay a peak map on the microscope photo. "
                "Call it once before a scan measurement (camera streaming required). "
                "It does NOT return the image for you to look at - use analyze_microscope_image for "
                "that. It is also what save_measurement_point references as 'the image at this point'."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the external web and fetch the top results (title, URL, summary). Use it to find recent/specialist information "
                "(literature, recommended parameter values, methodology, etc.) that internal knowledge/KB (search_knowledge_base) cannot answer. "
                "It is recommended to first check local knowledge with search_knowledge_base and use this tool for external search when that is insufficient. "
                "If there is no internet it returns a failure, in which case decide from local knowledge."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query. e.g. 'raman baseline correction asymmetric least squares'"},
                    "max_results": {"type": "integer", "description": "Number of results to fetch (1-10). Default 5.", "minimum": 1, "maximum": 10},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_analysis",
            "description": (
                "Run 'computation/visualization' Python code on saved measurement data AND on files the user "
                "attached to the chat, in a safe sandbox. "
                "Use it to handle analyses not provided as tools (baseline correction, peak detection, per-coordinate peak maps/heatmaps, etc.) directly in code. "
                "Already injected into the runtime: "
                "spectra (list[dict] - each item has base, title, x, y, power, exposure, mode, "
                "raman_shift (np.ndarray or None), intensity (np.ndarray)), "
                "np (numpy), plt (matplotlib.pyplot). "
                "Preprocessing helpers are injected too, so you do not have to re-implement them: "
                "ipbsa(y, order=5, max_iterations=100, threshold=0.001) returns "
                "(corrected, background) using the SAME iterative-polynomial routine as the "
                "apply_background_subtraction tool, and poly_baseline(y, order=5, x=None) returns "
                "a single polynomial background fit if you would rather build your own iteration. "
                "Prefer ipbsa(...) over hand-writing a polynomial baseline loop - it is the "
                "single biggest source of over-long code. "
                "A Kinetic measurement carries its time series too: frames is a 2D np.ndarray of "
                "shape (n_frames, n_pixels), so frames.mean(axis=0) is the averaged spectrum and "
                "frames[:, px] is one pixel's intensity over time. For those items intensity is the "
                "frame MEAN (flagged by intensity_is_frame_mean) - use frames, not intensity, when the "
                "question is about change over time. Very long runs are cut to the first 200 frames and "
                "say so in frames_truncated. This is the ONLY way to analyse a kinetic measurement; "
                "load_spectrum refuses those files. "
                "If you pass file_ids, the attached files are parsed and injected as "
                "files (list[dict] - each item has file_id, filename, sheet, columns (list[str]), n_rows, and "
                "table (dict mapping column name -> np.ndarray for numeric columns, list[str] for text columns)). "
                "Inspect a file's structure with inspect_file first, then use the column names you saw as keys of table. "
                "spectra and files can be used together - e.g. overlay an attached reference spectrum on a measured one. "
                "If you called capture_scene first, microscope_image (np.ndarray|None) and "
                "image_extent ([xmin,xmax,ymin,ymax] stage mm|None) are also injected - "
                "after ax.imshow(microscope_image, extent=image_extent), overlaying peaks at the measurement (x,y) "
                "makes a peak map on top of the microscope image. "
                "A figure created with plt is auto-saved and shown in the chat. "
                "Small numeric results are observed if you print() them. "
                "To SAVE a computed spectrum, call the injected hook "
                "save_result(filename, intensity, raman_shift=None, wavelength_nm=None, metadata=None) "
                "inside your code - it writes data/<filename>.csv at full precision and returns the path, "
                "which also comes back in the tool result as saved_files. "
                "This is the correct way to persist a processed spectrum (baseline-corrected, "
                "spike-removed, normalized, smoothed): do it in the same run_analysis call that computes it. "
                "save_result is the ONLY way to write an array - there is no tool that takes an intensity "
                "array as an argument. Do NOT print an array in order to re-type it elsewhere: printing "
                "thousands of numbers overflows the context window and loses precision. Print only a short "
                "summary (how many points, how many spikes removed, where the peaks are). "
                "stdout is truncated past 4000 characters. "
                "Constraints (safety): no hardware (laser/stage/CCD) control, no network access, "
                "no file access other than save_result and plt figures, "
                "imports limited to computation libraries such as numpy/scipy/matplotlib/math. "
                "A 'measurement' like a 3x3 scan is done first with move_stage + acquire_spectrum, not this tool, "
                "and the saved result is analyzed/visualized here. On failure, read error/trace, fix the code, and call again. "
                "WHEN NOT TO USE THIS: a ready-made tool already covers some jobs and needs no code - "
                "combine_spectra (grid of spectra images), aggregate_spectra_csv (one summary row per "
                "measurement), bundle_results (zip for download), apply_background_subtraction (IPBSA "
                "baseline removal). Reach for run_analysis when no such tool fits, not by default."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Python analysis code to run. Use spectra, np, plt directly. "
                            "e.g. compute each spectrum's peak intensity and draw a peak map as an (x,y) scatter. "
                            "If the task asks you to save a computed spectrum, call "
                            "save_result('name', corrected_intensity, raman_shift=x) at the end of this code "
                            "rather than printing the array. "
                            "KEEP EACH CALL SHORT - aim for 40 lines or fewer. This code travels to the "
                            "sandbox as a single JSON string, and a long block (many escaped quotes and "
                            "newlines) is the most common way for a call to be lost in transit: the call "
                            "silently never arrives and the task ends with no answer. Do not write one "
                            "large end-to-end script. Split the work and call this tool several times - "
                            "e.g. (1) load the data and print its shape and column names, (2) do one "
                            "computation step and print a short summary, (3) produce the final numbers. "
                            "Nothing carries over between calls: every call runs in a fresh process, so "
                            "each one must rebuild what it needs from spectra/files, or read back an "
                            "intermediate you wrote earlier with save_result (load_spectrum accepts the "
                            "path it returned)."
                        ),
                    },
                    "date": {"type": "string", "description": "Measurement date to analyze 'YYYY-MM-DD'. If omitted, today."},
                    "names": {
                        "type": "array", "items": {"type": "string"},
                        "description": "List of measurement bases to analyze (check with list_results). If omitted, the whole date.",
                    },
                    "file_ids": {
                        "type": "array", "items": {"type": "string"},
                        "description": (
                            "file_ids of attached files to load into the `files` variable "
                            "(get them from list_uploaded_files). If omitted, no attached file is loaded."
                        ),
                    },
                    "title": {"type": "string", "description": "Title to attach to the result figure."},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bundle_results",
            "description": (
                "Bundle saved measurement files (png/csv/json) into a single zip and provide a download link. "
                "Use it when the user wants to download all the results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Target date 'YYYY-MM-DD'. If omitted, today."},
                    "names": {
                        "type": "array", "items": {"type": "string"},
                        "description": "List of measurement bases to bundle. If omitted, all of yours from that date.",
                    },
                    "scope": {
                        "type": "string", "enum": ["session", "all"],
                        "description": (
                            "Which measurements to bundle. 'session' (default) = only the ones "
                            "measured in THIS session. 'all' = every session saved that day; use it "
                            "only when the request is explicitly about earlier, separate sessions."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
]
