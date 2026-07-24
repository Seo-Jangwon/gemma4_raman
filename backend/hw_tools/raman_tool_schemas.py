"""
LLM에게 전달할 tool 스키마 정의 (Ollama tool calling 포맷)
"""

RAMAN_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "move_stage",
            "description": "Move the stage to an absolute position (mm).",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "X-axis position (mm, 0-75.3)"},
                    "y": {"type": "number", "description": "Y-axis position (mm, 0-50.2)"},
                    "z": {"type": "number", "description": "Z-axis position (mm, -1.0-1.0). Optional"},
                },
                "required": ["x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stage_position",
            "description": "Read the current stage X, Y, Z position (mm).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_stage_relative",
            "description": "Move the stage relative to its current position.",
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
            "description": "Turn the laser on.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "laser_off",
            "description": "Turn the laser off.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_laser_power",
            "description": "Set the laser power (ND filter transmittance). Any real value in the range 0.004-100 %.",
            "parameters": {
                "type": "object",
                "properties": {
                    "percent": {
                        "type": "number",
                        "description": "Power percent (transmittance %). A real value in 0.004-100. e.g. 1, 2.5, 40, 100.",
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
                "Acquire a Raman spectrum at the current position. "
                "Supports three acquisition modes: Single (one shot) / Accumulate (averaged, high SNR) / Kinetic (continuous time series). "
                "Automatically handles the laser ON -> power stabilization -> CCD acquisition -> laser OFF flow. "
                "When a calibrator is connected, the raman_shift_cm-1, wavelength_nm, laser_nm fields are included. "
                "Kinetic mode returns per-frame data in a frames array."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "exposure": {
                        "type": "number",
                        "description": "CCD exposure time (seconds). Default 0.2",
                    },
                    "power": {
                        "type": "number",
                        "description": "Laser power (transmittance %). A real value in 0.004-100. Default 40.",
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
                            "CCD acquisition mode. "
                            "'single': single shot (default). "
                            "'accumulate': sum num_accumulations shots -> high-SNR spectrum. "
                            "'kinetic': acquire kinetic_count frames continuously -> time-series analysis."
                        ),
                    },
                    "num_accumulations": {
                        "type": "integer",
                        "description": "Accumulations per frame in accumulate/kinetic mode. Default 1.",
                        "minimum": 1,
                    },
                    "kinetic_count": {
                        "type": "integer",
                        "description": "Total number of frames to acquire in kinetic mode. Default 1.",
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
                            "CCD readout mode. "
                            "'fvb': Full Vertical Binning - sum all rows, 1D spectrum (default). "
                            "'single_track': read only a specific track - single_track_center required."
                        ),
                    },
                    "hbin": {
                        "type": "integer",
                        "description": "Horizontal binning pixel count. Default 1.",
                        "minimum": 1,
                    },
                    "single_track_center": {
                        "type": "integer",
                        "description": "Center pixel row number when read_mode='single_track'.",
                    },
                    "single_track_width": {
                        "type": "integer",
                        "description": "Track width (pixels) when read_mode='single_track'. Default 1.",
                        "minimum": 1,
                    },
                    "trigger_mode": {
                        "type": "string",
                        "enum": ["internal", "external", "external_start", "external_exposure", "external_fvb_em", "software"],
                        "description": "CCD trigger mode. Default 'internal'.",
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
            "description": "Start real-time camera streaming (preview). Use it to check the sample position or to focus.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop_camera_stream",
            "description": "Stop real-time camera streaming.",
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
            "description": "Set the CCD exposure time (seconds). Larger values give stronger signal but longer measurement time.",
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
                "'run_till_abort': acquire continuously until an abort command."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "description": "Acquisition mode",
                        "enum": ["single", "accumulate", "kinetic", "run_till_abort"],
                    },
                    "num_accumulations": {
                        "type": "integer",
                        "description": "Number of accumulations (used in accumulate/kinetic mode). Default 1",
                    },
                    "num_kinetics": {
                        "type": "integer",
                        "description": "Total number of frames to acquire (used in kinetic mode)",
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
                "'software': use SendSoftwareTrigger."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "description": "Trigger mode",
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
                "Set the CCD readout mode. "
                "'fvb': Full Vertical Binning - sum all rows -> 1D spectrum (default, used for Raman). "
                "'single_track': read only a specific vertical row. The center parameter is required. "
                "'image': full 2D image or ROI."
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
                        "description": "Horizontal binning factor (default 1 = no binning)",
                    },
                    "center": {
                        "type": "integer",
                        "description": "Center row number to read in single_track mode (1-based). Must be specified when using single_track mode. e.g. 256",
                    },
                    "width": {
                        "type": "integer",
                        "description": "Number of rows to read in single_track mode (default 1)",
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
            "description": "Set the MCP (Micro-Channel Plate) gain of the iStar ICCD camera. Check the allowed range with get_mcp_gain_range().",
            "parameters": {
                "type": "object",
                "properties": {
                    "gain": {
                        "type": "integer",
                        "description": "MCP gain value to set"
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
            "description": "Return the allowed MCP gain range (min, max) of the iStar ICCD camera.",
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
    {
        "type": "function",
        "function": {
            "name": "set_ccd_em_gain",
            "description": (
                "Set the EM (Electron Multiplication) gain. EMCCD only. "
                "Higher values amplify the signal more, but too high may saturate the image."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "gain": {
                        "type": "integer",
                        "description": "EM gain value. Within the em_gain_range from get_ccd_info()",
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
                "Select the output amplifier. "
                "0 = EMCCD amp (EM gain enabled). "
                "1 = Conventional amp (EM disabled, low noise)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "amp": {
                        "type": "integer",
                        "description": "Amplifier selection: 0 (EM) or 1 (Conventional)",
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
                "'close' - force closed (for dark-frame / background measurement)."
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
            "description": "Set horizontal/vertical flip of the acquired image.",
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
            "description": "Set the stage movement speed. X/Y max 5.0 mm/s, Z max 0.1 mm/s.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x_speed_mm_s": {"type": "number", 
                                     "description": "X-axis movement speed (mm/s, max 5.0)"},
                    "y_speed_mm_s": {"type": "number", 
                                     "description": "Y-axis movement speed (mm/s, max 5.0)"},
                    "z_speed_mm_s": {"type": "number", 
                                     "description": "Z-axis movement speed (mm/s, max 0.1). Optional"},
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
    {
        "type": "function",
        "function": {
            "name": "capture_camera_frame",
            "description": (
                "Capture the latest single frame from the camera and return its shape, intensity statistics, "
                "and sharpness score (sharpness_score, Laplacian variance). "
                "Streaming must be active. "
                "Useful for comparing sharpness across Z positions during autofocus."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_microscope_image",
            "description": (
                "Capture the current view of the TuCam optical microscope camera and pass it as an image. "
                "Use it when visual judgment is needed, e.g. checking sample position, identifying a target, or detecting debris. "
                "Streaming must be active. "
                "The return also includes pixel coordinates; these are not stage coordinates, so if you need to move to that location, convert and move with the move_to_pixel tool."
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
                        "description": "Image X pixel coordinate (0 - 1060)",
                    },
                    "pixel_y": {
                        "type": "integer",
                        "description": "Image Y pixel coordinate (0 - 800)",
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
                "Adaptive hill-climbing auto-adjusts the step size and finally returns to the historical minimum position."
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
    # ── 데이터 저장 / 로드 ────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "save_spectrum",
            "description": (
                "Save a spectrum intensity array to a CSV file. "
                "If raman_shift is also passed, calibration information is included. "
                "The file is saved to the <project root>/data/ directory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Intensity array",
                    },
                    "filename": {
                        "type": "string",
                        "description": "File name to save (the .csv extension is optional). e.g. 'polystyrene_01'",
                    },
                    "raman_shift": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Raman shift axis [cm-1]. Optional",
                    },
                    "wavelength_nm": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Wavelength axis [nm]. Optional",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Additional metadata (exposure time, power, etc.). Saved as a .json with the same name",
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
                "Load a saved spectrum CSV file. "
                "Both a path relative to the data/ directory and an absolute path are allowed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "File name or path. e.g. 'polystyrene_01' or 'polystyrene_01.csv'",
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
                "Create a new experiment session directory and initialize its metadata. "
                "Afterwards you can save per-point spectrum/position data with save_point_data()."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session identifier. e.g. 'EXP_001'",
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
                "Save spectrum/position data at a specific measurement point of an experiment session. "
                "You must first create a session with create_session()."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID. e.g. 'EXP_001'",
                    },
                    "point_id": {
                        "type": "string",
                        "description": "Point identifier. e.g. 'P001'",
                    },
                    "spectrum_data": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Intensity array. Optional",
                    },
                    "raman_shift": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Raman shift axis [cm-1]. Optional",
                    },
                    "position": {
                        "type": "object",
                        "description": "Stage position {'x': ..., 'y': ..., 'z': ...}. Optional",
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
                "Remove the fluorescence background of a Raman spectrum using IPBSA (iterative polynomial background subtraction). "
                "Uses the most recently acquired spectrum (source='last') or a saved file path as the source. "
                "The result is saved under version_label, and you can compare multiple versions with list_bg_versions(). "
                "Increasing poly_order removes more complex backgrounds but risks overfitting. "
                "If the user does not specify a polynomial order, use the default value 5."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "poly_order": {
                        "type": "integer",
                        "description": "Polynomial order (2-10). Default 5. Lower is a smoother background; higher estimates a more complex background.",
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
                            "Otherwise: a file path (JSON or CSV, a path relative to data/ is allowed)."
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
                        "description": "If True, save the corrected spectrum as a CSV in the data/ directory. Default false.",
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
                "Return the list of all saved background-subtraction result versions with their parameters and key statistics. "
                "Data arrays are not included. Use get_bg_version() for the full data. "
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
                "Return the full data of a specific background-subtraction result version (corrected spectrum, background curve, Raman shift axis). "
                "Check version_label with list_bg_versions()."
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
                "Query the list of measurement results auto-saved by acquire_spectrum. "
                "Returns each item's base (file identifier), title, timestamp, and meta (coordinates, etc.). "
                "Get the base to pass to combine_spectra / aggregate_spectra_csv / bundle_results here."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date to query 'YYYY-MM-DD'. If omitted, today.",
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
                "e.g. for a 10x10 scan, arranged by coordinate. If names is omitted, combine everything from that date."
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
                "max intensity, total intensity, peak position). Use it to organize multiple experiments into one table."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Target date 'YYYY-MM-DD'. If omitted, today."},
                    "names": {
                        "type": "array", "items": {"type": "string"},
                        "description": "List of measurement bases to organize. If omitted, the whole date.",
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
                "Save the current microscope (camera) view. It also computes and saves the stage-coordinate "
                "extent of the image from the stage position and lens FOV, so later in run_analysis you can load it via microscope_image / "
                "image_extent and overlay a peak map on top of the microscope image. "
                "It is good to call this once before a scan measurement (camera streaming required)."
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
                "(literature, recommended parameter values, methodology, etc.) that internal knowledge/KB (search_kb) cannot answer. "
                "It is recommended to first check local knowledge with search_kb and use this tool for external search when that is insufficient. "
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
                "If you pass file_ids, the attached files are parsed and injected as "
                "files (list[dict] - each item has file_id, filename, sheet, columns (list[str]), n_rows, and "
                "table (dict mapping column name -> np.ndarray for numeric columns, list[str] for text columns)). "
                "Inspect a file's structure with inspect_file first, then use the column names you saw as keys of table. "
                "spectra and files can be used together - e.g. overlay an attached reference spectrum on a measured one. "
                "If you called capture_scene first, microscope_image (np.ndarray|None) and "
                "image_extent ([xmin,xmax,ymin,ymax] stage mm|None) are also injected - "
                "after ax.imshow(microscope_image, extent=image_extent), overlaying peaks at the measurement (x,y) "
                "makes a peak map on top of the microscope image. "
                "A figure created with plt is auto-saved and shown in the chat. Numeric results are observed if you print() them. "
                "Constraints (safety): no hardware (laser/stage/CCD) control, no file/network access, "
                "imports limited to computation libraries such as numpy/scipy/matplotlib/math. "
                "A 'measurement' like a 3x3 scan is done first with move_stage + acquire_spectrum, not this tool, "
                "and the saved result is analyzed/visualized here. On failure, read error/trace, fix the code, and call again."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python analysis code to run. Use spectra, np, plt directly. e.g. compute each spectrum's peak intensity and draw a peak map as an (x,y) scatter.",
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
                        "description": "List of measurement bases to bundle. If omitted, the whole date.",
                    },
                },
                "required": [],
            },
        },
    },
]
