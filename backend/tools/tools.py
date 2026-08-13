# -*- coding: utf-8 -*-
"""LLM 에게 주는 도구 전체 — **스키마와 디스패치의 단일 진입점.**

여기 있는 것은 목록 두 개뿐이다.

    ALL_TOOLS      모델에게 보여줄 스키마 (하드웨어 + 파일 + KB)
    TOOL_DISPATCH  이름 → 실행 함수

구현은 역할별 모듈에 있고 이 파일은 그것들을 모으기만 한다:

    hw_tools/stage_tools    스테이지 이동·속도
    hw_tools/laser_tools    레이저 발사·파워·가이드빔
    hw_tools/ccd_tools      CCD 설정
    hw_tools/camera_tools   카메라·캡처·오토포커스·픽셀 이동
    hw_tools/acquire_tools  스펙트럼 수집·측정점 기록·격자 매핑
    hw_tools/system_tools   연결 상태·재연결
    hw_tools/hw_core        핸들·락·세션 상태·공용 검사 (도구 아님)
    tools/data_tools        service 계층 어댑터(목록·병합·분석·검색·KB)
    tools/bg_tools          배경 제거·스펙트럼 CSV 읽기
    tools/file_tools        첨부 파일·세션 산출물 (하드웨어 무관 — 아래 참고)

[왜 description 만 여기 적는가]
인자의 이름·타입·범위·설명은 각 도구 함수의 시그니처에 있고, 스키마는 tool_schema() 가
그 시그니처를 읽어 만든다. 선언이 한 곳뿐이라 어긋날 수가 없다(tools/schema.py 머리말).
시그니처에서 뽑을 수 없는 것은 '언제 이 도구를 쓰는가' 하나뿐이라, 이 파일에는 그것만 적는다.

[description 을 줄일 때]
겹치는 도구(측정 조건을 미리 걸기 vs acquire_spectrum 인자로 넘기기, 화면을 찍는 세 도구,
저장물을 나열하는 네 도구)마다 "어느 쪽을 언제 쓰는가 + 둘 다 부르지 말 것"이 적혀 있다.
그 문장을 먼저 지우지 말 것 — 도구를 중복 호출하거나 엉뚱한 쪽을 골라 실패하는 경우의
대부분이 그 부재에서 온다.

[FILE_DISPATCH 가 왜 이 표에 없는가]
파일 도구는 하드웨어와 무관한데, 이 모듈은 하드웨어를 import 하므로 Config.ini 가 없는
PC 에서는 import 자체가 실패한다. 그때 TOOL_DISPATCH 는 None 이 되지만 파일 분석은
계속 되어야 한다. 그래서 runtime._dispatch 가 FILE_DISPATCH 를 '하드웨어 가드보다 먼저'
따로 본다 — 그 보호를 없애지 말 것(file_tools.py 머리말).

    python backend/tools/tools.py      자체 검사
"""
from __future__ import annotations

from backend.tools.hw_tools.hw_tools import acquire_tools as _acq
from backend.tools.hw_tools.hw_tools import camera_tools as _cam
from backend.tools.hw_tools.hw_tools import ccd_tools as _ccd
from backend.tools.hw_tools.hw_tools import laser_tools as _laser
from backend.tools.hw_tools.hw_tools import stage_tools as _stage
from backend.tools.hw_tools.hw_tools import system_tools as _sys
from backend.tools.hw_tools.hw_tools.hw_core import _cache_and_return
from backend.tools.non_hw_tools import bg_tools as _bg
from backend.tools.non_hw_tools import data_tools as _data
from backend.tools.non_hw_tools.file_tools import FILE_TOOLS
from backend.tools.result import ok
from backend.tools.schema import tool_schema

RAMAN_TOOLS = [
    tool_schema(
        _stage.move_stage,
        'Move the stage to an absolute position (mm). Out-of-range targets are REJECTED with an error rather than clipped, so the reported position is always where the stage actually is. Related: move_stage_relative (same limits, but you give a displacement), move_to_pixel (you give a point on the camera image).',
    ),
    tool_schema(
        _stage.get_stage_position,
        'Read the current stage X, Y, Z position (mm). You do NOT need this right after a move - move_stage / move_stage_relative / move_to_pixel already return the resulting position. get_hardware_status also reports it while diagnosing connection problems.',
    ),
    tool_schema(
        _stage.move_stage_relative,
        'Move the stage by a displacement from its current position (mm). The resulting target is range-checked exactly like move_stage, so a displacement that would leave the travel range is rejected (the error tells you the computed target). Use move_stage when you know the absolute coordinate.',
    ),
    tool_schema(
        _laser.laser_on,
        "Turn the laser on. Which beam actually comes out depends on whether the power is currently applied to the optics: if it is, the MEASUREMENT beam fires; otherwise only the GUIDE beam does (the ND filter is still blocking). The response tells you which one via the 'beam' field - check it, because a guide-beam 'ON' produces no Raman signal. This tool CANNOT arm the measurement beam - use set_laser_power(percent) to arm it without firing, or acquire_spectrum(power=...) which arms it and handles on -> acquire -> off atomically. Use laser_on by itself only for alignment with the guide beam.",
    ),
    tool_schema(
        _laser.laser_off,
        'Stop the laser from firing. The ND filter and the beam splitter stay where they are, so turning it on again emits the same measurement beam - but the camera also stays blind until you call set_guide_beam_mode. You do not need to call this after acquire_spectrum - that tool always turns the laser off (even when it fails) and restores the camera view.',
    ),
    tool_schema(
        _laser.set_laser_power,
        'Set the laser power (ND filter transmission, 0.004-100 %) WITHOUT firing the laser. This only moves the ND filter to arm the measurement beam - it does not turn the laser on. Use it when the user asks you to set or change the power by itself, or to arm the beam before alignment. For an actual measurement prefer acquire_spectrum(power=...), which applies the power, fires, measures and turns the laser off atomically - chaining set_laser_power + laser_on + laser_off leaves the beam on the sample during your own reasoning time, which can photobleach or damage a biological sample. Out-of-range values are REJECTED, not silently clamped.',
    ),
    tool_schema(
        _laser.get_laser_status,
        'Query the current laser state: whether it is firing (is_on), the applied power (power_percent, %), and power_armed - whether that power is actually applied to the optics right now. If power_armed is false the ND filter is at the blocking position, so laser_on() would emit only the guide beam; arm it with set_laser_power(percent), or measure with acquire_spectrum(power=...), which applies the power itself. Always read power_armed, not just power_percent - power_percent is the last value that was requested and survives a switch to guide-beam mode, so on its own it will make you think the beam is ready when it is not.',
    ),
    tool_schema(
        _acq.acquire_spectrum,
        "Acquire a Raman spectrum at the current position. Supports three acquisition modes: Single (one shot) / Accumulate (averaged, high SNR) / Kinetic (continuous time series). Automatically handles the laser ON -> power stabilization -> CCD acquisition -> laser OFF flow. IMPORTANT: every parameter is optional and OMITTING one means 'keep whatever the instrument is already set to' (shutter works the same way, except that it opens to 'auto' when nobody has ever set it - the CCD boots with the shutter closed). So set_ccd_exposure / set_ccd_acquisition_mode / set_ccd_read_mode / set_ccd_trigger_mode are respected - you may either configure first and then call this with no arguments, or pass the values directly here. Both go through the same code and are equivalent; do not do both 'just in case'. This is also the ONLY way to fire the measurement beam: it applies the power, turns the laser on, acquires, and turns it off again even if the acquisition fails. It also puts the optics back into the guide-beam/camera position afterwards, so the microscope camera can see the sample again - you do NOT need to call set_guide_beam_mode after measuring. The returned exposure_time / laser_power_pct / num_accumulations are read back from the hardware, so they tell you what was ACTUALLY used. When a calibrator is connected, the raman_shift_cm-1, wavelength_nm, laser_nm fields are included. Kinetic mode returns per-frame data in a frames array.",
    ),
    tool_schema(
        _cam.start_camera_stream,
        "Start real-time camera streaming. Streaming must be ON before analyze_microscope_image, capture_scene, preview_grid_scan or run_autofocus can get a frame. The response field already_streaming tells you whether it was ALREADY running before your call - if it was true, someone else (the user's live view) owns the stream, so do not stop it afterwards.",
    ),
    tool_schema(
        _cam.stop_camera_stream,
        'Stop real-time camera streaming. Only call this if YOU started the stream - i.e. start_camera_stream returned already_streaming=false. Stopping a stream the user was watching blanks their live view, and every image tool stops working until it is restarted. When in doubt, leave it running.',
    ),
    tool_schema(
        _ccd.get_ccd_info,
        'Query all current CCD settings and status. Returns temperature, cooling status, exposure time, acquisition mode, readout mode, gain, shift speeds, pixel count, etc. in one call. Use it before and after changing parameters to verify the current state.',
    ),
    tool_schema(
        _ccd.set_ccd_exposure,
        'Set the CCD exposure time (seconds). Larger values give stronger signal but longer measurement time. The value persists - a later acquire_spectrum keeps it unless you pass exposure there. OVERLAP: acquire_spectrum(exposure=...) sets exactly the same thing through the same code. Use this tool when you set the exposure once and then measure several times; pass it to acquire_spectrum when it is a one-off. Never do both for the same measurement.',
    ),
    tool_schema(
        _ccd.set_ccd_acquisition_mode,
        "Set the CCD acquisition mode. 'single': single shot. 'accumulate': sum after num_accumulations shots. 'kinetic': acquire num_kinetics frames continuously. 'run_till_abort': acquire continuously until an abort command (acquire_spectrum cannot use this one - set single/accumulate/kinetic before measuring). The mode and counts persist - acquire_spectrum keeps them unless you pass acq_mode / num_accumulations there. OVERLAP: acquire_spectrum(acq_mode=, num_accumulations=, kinetic_count=) sets the same thing through the same code; use one or the other, not both. This tool is the only way to select 'run_till_abort'. The response reports the counts actually in effect on the hardware, not what you asked for.",
    ),
    tool_schema(
        _ccd.set_ccd_trigger_mode,
        "Set the CCD trigger mode. 'internal': start acquisition via software (default). 'external': start acquisition on an external TTL signal. 'external_start': start on external trigger then internal timing. 'external_exposure': expose while the external TTL is HIGH. 'external_fvb_em': external trigger in FVB/EM readout. 'software': use SendSoftwareTrigger. The mode persists - acquire_spectrum keeps it unless you pass trigger_mode there. OVERLAP: acquire_spectrum(trigger_mode=...) accepts exactly the same values through the same code. Use one or the other, not both.",
    ),
    tool_schema(
        _ccd.set_ccd_read_mode,
        "Set the CCD readout mode. 'fvb': Full Vertical Binning - sum all rows -> 1D spectrum (used for Raman). 'single_track': read only a specific vertical row. single_track_center is required. 'image': full 2D image or ROI (acquire_spectrum cannot build a 1D spectrum from this - switch back to 'fvb' before measuring). The mode persists - acquire_spectrum keeps it unless you pass read_mode there. OVERLAP: acquire_spectrum(read_mode=, hbin=, single_track_center=, single_track_width=) sets the same thing through the same code, using the same parameter names. Use one or the other, not both. This tool is the only way to select 'image' mode.",
    ),
    tool_schema(
        _ccd.set_ccd_preamp_gain,
        'Set the pre-amplifier gain index. See preamp_gains_available in get_ccd_info() for the available gain list. A larger index gives higher gain, which helps measure weak signals but also increases noise.',
    ),
    tool_schema(
        _ccd.set_ccd_shift_speeds,
        'Set the vertical (VS) and horizontal (HS) pixel shift speeds. VS: a larger index is slower but reduces charge-transfer noise. HS: a larger index is slower but reduces readout noise. You may specify only one of them. See get_ccd_info() for the available speed list.',
    ),
    tool_schema(
        _ccd.set_ccd_temperature,
        'Set the CCD cooling target temperature (°C). Lower temperature reduces dark-current noise. It may take several minutes to reach; check progress with get_ccd_info(). Typical range: -80 to 20°C.',
    ),
    tool_schema(
        _ccd.set_ccd_cooler,
        'Turn the CCD Peltier cooler on (true) or off (false). It must be turned off before shutdown.',
    ),
    tool_schema(
        _ccd.set_ccd_shutter,
        "Set the CCD shutter mode. 'auto'  - open and close automatically during acquisition (normal measurement). 'open'  - force open. 'close' - force closed. Like the other CCD settings this one PERSISTS: a later acquire_spectrum keeps it unless you pass its own shutter argument. So for several dark / background frames set 'close' once here and measure repeatedly; for a single dark frame acquire_spectrum(shutter='close') is enough. Set it back to 'auto' before normal measurements.",
    ),
    tool_schema(
        _ccd.set_ccd_image_flip,
        "Set horizontal/vertical flip of the acquired 2D image. Only valid when the CCD read mode is 'image' - in the 1D spectrum modes (fvb / single_track) flipping would misalign the intensity array against the calibrated wavelength axis, and the correct orientation is already set at startup, so the call is rejected there. You almost never need this for Raman measurements.",
    ),
    tool_schema(
        _stage.get_stage_speed,
        'Query the current stage movement speed (mm/s). Returns the x_speed_mm_s, y_speed_mm_s, z_speed_mm_s fields.',
    ),
    tool_schema(
        _stage.set_stage_speed,
        "Set the stage movement speed. X/Y max 5.0 mm/s, Z max 0.1 mm/s. If a specific axis speed is omitted, its current speed is maintained. Values above the limit are clipped to it; the response reports the speeds that will ACTUALLY be used and lists any clipped axes under 'clipped'.",
    ),
    tool_schema(
        _sys.get_hardware_status,
        "Report which hardware components (stage, laser, ccd, camera) are currently CONNECTED, and for the connected ones whether they actually respond. Read `summary` first. Call this FIRST whenever a hardware tool fails, before trying to fix anything - it tells you whether one device is down or several, which decides what is even worth attempting. It touches nothing and fires no laser, so it is always safe to call. SCOPE: this answers 'is it there and alive'. For the SETTINGS of a working device use the per-device tools instead - get_ccd_info (exposure, mode, temperature), get_laser_status (power and whether it is armed), get_stage_position, get_stage_speed.",
    ),
    tool_schema(
        _sys.reconnect_hardware,
        "Release and re-initialize a hardware component that is unresponsive or stuck. component: 'stage' | 'ccd' | 'camera' | 'laser' | 'all'. Call get_hardware_status first to see what is actually down. Read the returned `errors` text carefully - it distinguishes two very different cases. (a) 'resource is still held by this process': a process-level lock that NO tool can clear. Calling this tool again will not help; the server must be restarted by a human. Do not retry - carry on without that component and say so in your final answer. (b) 're-initialization failed' after a successful release: a device-side problem (power, cable, driver, or another program holding the device). Retrying once is reasonable; beyond that, proceed without the component and state the limitation. Never call this repeatedly in a loop - it cannot fix either case by repetition. WARNING: reconnecting the 'ccd' re-runs cooling and can block for minutes until -40 C stabilizes.",
    ),
    tool_schema(
        _laser.set_guide_beam_mode,
        'Switch the laser to guide-beam standby state. Moves the beam splitter to the standby position and the ND filter to the main-beam blocking position. Use it for sample alignment and focus checking.',
    ),
    tool_schema(
        _cam.set_camera_exposure,
        'Set the camera (TUCam) exposure time (ms).',
    ),
    tool_schema(
        _cam.set_camera_auto_exposure,
        'Enable (true) or disable (false) camera auto exposure.',
    ),
    tool_schema(
        _cam.analyze_microscope_image,
        "Capture the current view of the TuCam optical microscope camera and pass it to YOU as an image to look at. Use it when visual judgment is needed, e.g. checking sample position, identifying a target, or detecting debris. Streaming must be active. The tool returns the IMAGE ITSELF - it does NOT return any coordinates. YOU must look at the image, find the target in it, and read off its pixel coordinates yourself; do not search the JSON response for an x/y field, there is none. Those pixel coordinates are not stage coordinates, so to move there pass them to move_to_pixel. It also returns brightness statistics (min/max/mean_intensity) and a relative sharpness_score for the same image - use those to check exposure or compare frames. Do NOT use sharpness_score to focus manually: run_autofocus optimises a different metric (guide-beam spot area) and would settle at a different Z. OVERLAP - three tools capture the camera, pick by PURPOSE: analyze_microscope_image = you look at it now (returns the image to you, saves nothing); capture_scene = save the view as a file so run_analysis can draw a peak map on top of it (returns no image for you to inspect); preview_grid_scan = show where a planned grid would land. All three return the same pixel coordinate system, so a coordinate read from any of them can be passed to move_to_pixel. This tool saves NOTHING: it does not become 'the image at this point' for save_measurement_point. If you want this view bundled into a measurement-point record, call capture_scene as well.",
    ),
    tool_schema(
        _cam.move_to_pixel,
        'Convert pixel coordinates (pixel_x, pixel_y) within the camera image to stage mm coordinates and move there. The image center corresponds to the current stage position. After checking the target pixel coordinates with analyze_microscope_image, move with this tool.',
    ),
    tool_schema(
        _cam.run_autofocus,
        'Hill-climbing autofocus based on minimizing the guide-beam laser spot area. Computes the spot pixel count (area) via Otsu thresholding on the laser OFF/ON difference image, and moves the stage to the Z position with minimum area (where the laser spot is sharpest). Adaptive hill-climbing auto-adjusts the step size and finally returns to the historical minimum position. It moves Z only, uses the GUIDE beam (not the measurement beam), and leaves the laser off. The search is clipped to the Z travel range; if the response contains z_limit_hits the focus may be physically out of reach, and repeating the call will NOT help - report that instead. This is the only focusing tool: do not try to focus by comparing sharpness_score from analyze_microscope_image, which is a different metric and converges elsewhere.',
    ),
    tool_schema(
        _acq.preview_grid_scan,
        "Preview a rows x cols grid mapping WITHOUT moving the stage or firing the laser. Overlays the planned scan points as circles on the current camera view and returns that image so the layout can be visually verified before committing. ORIENTATION (do not confuse the two): rows = number of points stacked VERTICALLY = grid HEIGHT (stage Y axis); cols = number of points side-by-side HORIZONTALLY = grid WIDTH (stage X axis). So rows=3, cols=2 is a TALL grid (3 high x 2 wide) and rows=2, cols=3 is a WIDE grid (2 high x 3 wide) - these are DIFFERENT layouts, never swap them. When the user asks for a grid like 'A x B', decide deliberately which number is the horizontal count (width -> cols) and which is the vertical count (height -> rows), then use this preview image to confirm the drawn orientation matches what they asked. MANDATORY HUMAN APPROVAL: always preview FIRST, then STOP - show this preview image to the user, end your turn, and WAIT. Do NOT call run_grid_scan in the same turn as this preview; only call it in a later turn after the user has explicitly approved this exact layout. If center_x/center_y are omitted, the current stage position is used as the grid center, and that resolved center is what gets approved - a later run_grid_scan with the centre omitted scans THAT position even if the stage has moved since. Whatever you pass here you must pass IDENTICALLY to run_grid_scan: the approval is matched on the arguments themselves, so omitting the centre here and spelling it out there (or vice versa) is rejected as a mismatch even when both mean the same place. SIZE LIMIT: rows * cols must be <= 400 points; a larger grid is refused here, so agree a smaller size with the user before promising a map. The camera field of view is small, so with wide spacing some points may fall outside the frame; they are still measured, and the response reports how many are in view (n_in_view) along with the exact view size in fov_mm - plan spacing from that returned value rather than from a remembered number.",
    ),
    tool_schema(
        _acq.run_grid_scan,
        "Execute a rows x cols grid mapping: for each point it moves the stage, optionally autofocuses, acquires one spectrum, and auto-saves it (position-tagged). Returns a single compact summary (counts, intensity min/max/mean, and per-point data when 32 points or fewer) instead of one tool message per point - this is the token-efficient way to run a map. ORIENTATION: rows = vertical count (height, stage Y), cols = horizontal count (width, stage X); rows=3,cols=2 is a tall 3x2 grid, rows=2,cols=3 is a wide 2x3 grid - do not swap them. PASS THE APPROVED ARGUMENTS VERBATIM: the approval gate compares the arguments you give, not the physical positions they work out to. If you OMITTED center_x/center_y in the preview you must omit them here too - filling in the numeric coordinates of that same spot is rejected as an 'Approval mismatch'. (An omitted centre runs at the position the approved preview was drawn at, so the scan lands where the user saw it even if the stage has moved in between.) SIZE LIMIT: rows * cols must be <= 400 points. REQUIRES PRIOR HUMAN APPROVAL: do NOT call this in the same turn as preview_grid_scan. Call it ONLY after (1) you showed the user a preview_grid_scan image, (2) you ended that turn, and (3) the user EXPLICITLY approved that exact layout in a later message. If the user has not explicitly approved the previewed grid, do not call this - preview first and wait. The laser is fired at every point, so the estimated cumulative dose is checked up front and the scan is refused if it exceeds the safety limit. READ THE RESULT BEFORE REPORTING SUCCESS: n_measured can be lower than n_points, and with autofocus='each' the response may carry n_autofocus_failed (those points were measured at whatever Z the stage was at, so a weak signal there is an artefact, not a property of the sample) or n_autofocus_z_limit (the focus is physically out of reach - re-running will not help). If autofocus fails several times in a row the scan STOPS EARLY and comes back with ok=false plus an 'aborted' field; the points measured up to then are still saved, but the grid is incomplete and you must say so rather than reporting the map as done.",
    ),
    # load_spectrum 은 도구 목록에서 뺐다 — 2026-08-12.
    # 모델에게 주던 값이 사실상 없었다: 1024점 intensity 배열을 돌려주는데 tool_slim 의
    # MAX_SCALAR_LIST(32)에 걸려 **모델에게 닿기 전에 잘린다.** 남는 것은 파일명·열 이름·
    # 점 개수뿐이고, 그건 open_file 이 더 많이(통계·head 까지) 준다. 실제 데이터 경로는
    # 처음부터 run_analysis(file_ids=[...]) 였다.
    # 함수 자체는 지우지 않는다 — apply_background_subtraction 이 CSV 리더로 내부 호출한다.
    tool_schema(
        _acq.save_measurement_point,
        "Group what you just measured at this position into ONE measurement-point record: the most recent acquire_spectrum result, the most recent capture_scene microscope image, and the current stage coordinates. You do NOT pass any arrays - the spectrum and image files are already saved automatically, and this tool only links them together under a point id. TIMING IS PART OF THE RECORD: the coordinates are read from the stage AT THE MOMENT YOU CALL THIS, not from the spectrum's metadata. So call it immediately after acquiring the spectrum and capturing the view at that position, and BEFORE moving the stage anywhere else - otherwise the record pairs this point's spectrum with the next point's coordinates, and nothing later can detect that. The image must come from capture_scene; analyze_microscope_image does not count. The response lists anything that was missing (e.g. no image captured yet). Use one call per position to build a multi-point dataset.",
    ),
    tool_schema(
        _bg.apply_background_subtraction,
        "Remove the fluorescence background of a Raman spectrum using IPBSA (iterative polynomial background subtraction). Uses the most recently acquired spectrum (source='last') or a saved file path as the source. YOU CHOOSE THE POLYNOMIAL ORDER, and that choice is the substance of this task - there is no safe default to fall back on. Too low and a curved fluorescence background survives, tilting the baseline and distorting relative peak heights; too high and the polynomial starts following the peaks themselves, eating real signal. The right order depends on how curved THIS spectrum's background is, so look at the data before deciding. DO NOT settle on the first result: run it at two or three orders, compare them with list_bg_versions(), and keep the one where the baseline is flat between peaks while peak heights are unchanged. Say which order you chose and why. OVERLAP with run_analysis: that sandbox can run any baseline algorithm you write yourself (asymmetric least squares, rolling ball, wavelet, ...). This tool is IPBSA specifically, and it keeps parameters comparable across versions and writes the standard CSV format. Pick whichever the task calls for and state which one you used.",
    ),
    tool_schema(
        _bg.list_bg_versions,
        'Return the list of all saved background-subtraction result versions with their parameters and key statistics. This summary is what you compare versions with - poly_order, iterations_run, converged and max_corrected_intensity are enough to pick an order. The spectra themselves are NOT included here, and you cannot read them into the conversation at all: long arrays are stripped out of every tool result to protect the context window, so get_bg_version() will not hand you the numbers either. If you need the actual corrected spectrum, re-run apply_background_subtraction with save_result=true and work on the saved file in run_analysis. Use it when calling apply_background_subtraction() several times to compare.',
    ),
    tool_schema(
        _bg.get_bg_version,
        'Re-read the parameters and statistics of one background-subtraction version (poly_order, iterations_run, converged, max intensities, saved_path if it was saved). It does NOT put the corrected spectrum in front of you: long arrays are stripped from tool results, so the corrected_data / background_data arrays never arrive - do not call this expecting to read the numbers, and do not retry when they are absent. To work with the actual arrays, save the version (save_result=true) and analyse the file in run_analysis. Check version_label with list_bg_versions().',
    ),
    tool_schema(
        _data.list_results,
        "Query the list of MEASUREMENTS auto-saved by acquire_spectrum. Returns each item's base (file identifier), session, title, timestamp, and meta (coordinates, etc.). Get the base to pass to combine_spectra / aggregate_spectra_csv / bundle_results here. By default this lists only the measurements from YOUR current session - your own work, not other sessions'. Files live in data/results/<date>/<your session>/. THERE ARE THREE 'list' TOOLS, one per store - choose by what you are looking for: list_results = raw measurements you acquired (this one); list_session_artifacts = files YOU produced (processed spectra from save_result, measurement-point records, figures), each with a path you can open_file; list_uploaded_files = data files the USER attached to the chat. (Background-subtraction versions from this conversation are not files - use list_bg_versions.)",
    ),
    tool_schema(
        _data.combine_spectra,
        'Combine several saved measurement spectra into a single grid image and render it. Each cell title uses the title auto-generated at save time (scan coordinates, power, exposure) as is. e.g. for a 10x10 scan, arranged by coordinate. If names is omitted, combine everything from that date. This is the ready-made one-call version - it needs no code. run_analysis can also plot the same measurements, but only use it when you need a layout or computation this tool does not give you (overlaid curves, peak maps, custom axes).',
    ),
    tool_schema(
        _data.aggregate_spectra_csv,
        'Build a CSV summarizing several saved measurements, one row per experiment (date, time, title, coordinates, power, exposure, max intensity, total intensity, peak position). Use it to organize multiple experiments into one table. One call, no code. It summarizes ONE ROW PER MEASUREMENT - if you need per-point values computed some other way, or a table of derived quantities, use run_analysis instead.',
    ),
    tool_schema(
        _cam.capture_scene,
        "SAVE the current microscope (camera) view as a file, for later use as a background image. It also computes the stage-coordinate extent of that image (position + calibrated field of view), so in a later run_analysis you get microscope_image / image_extent injected and can overlay a peak map on the microscope photo. Call it once before a scan measurement (camera streaming required). It does NOT return the image for you to look at - use analyze_microscope_image for that. It is also what save_measurement_point references as 'the image at this point'.",
    ),
    tool_schema(
        _data.web_search,
        'Search the external web and fetch the top results (title, URL, summary). Use it to find recent/specialist information (literature, recommended parameter values, methodology, etc.) that internal knowledge/KB (search_knowledge_base) cannot answer. It is recommended to first check local knowledge with search_knowledge_base and use this tool for external search when that is insufficient. If there is no internet it returns a failure, in which case decide from local knowledge.',
    ),
    tool_schema(
        _data.run_analysis,
        "Run 'computation/visualization' Python code on saved measurement data AND on files the user attached to the chat, in a safe sandbox. Use it to handle analyses not provided as tools (baseline correction, peak detection, per-coordinate peak maps/heatmaps, etc.) directly in code. Already injected into the runtime: spectra (list[dict] - each item has base, title, x, y, power, exposure, mode, raman_shift (np.ndarray or None), intensity (np.ndarray)), np (numpy), plt (matplotlib.pyplot). Preprocessing helpers are injected too, so you do not have to re-implement them: ipbsa(y, order=5, max_iterations=100, threshold=0.001) returns (corrected, background) using the SAME iterative-polynomial routine as the apply_background_subtraction tool, and poly_baseline(y, order=5, x=None) returns a single polynomial background fit if you would rather build your own iteration. Prefer ipbsa(...) over hand-writing a polynomial baseline loop - it is the single biggest source of over-long code. A Kinetic measurement carries its time series too: frames is a 2D np.ndarray of shape (n_frames, n_pixels), so frames.mean(axis=0) is the averaged spectrum and frames[:, px] is one pixel's intensity over time. For those items intensity is the frame MEAN (flagged by intensity_is_frame_mean) - use frames, not intensity, when the question is about change over time. Very long runs are cut to the first 200 frames and say so in frames_truncated. This is the ONLY way to analyse a kinetic measurement - open_file only summarises a kinetic file's raw rows and cannot reconstruct the frames. If you pass file_ids, the attached files are parsed and injected as files (list[dict] - each item has file_id, filename, sheet, columns (list[str]), n_rows, and table (dict mapping column name -> np.ndarray for numeric columns, list[str] for text columns)). Inspect a file's structure with open_file first, then use the column names you saw as keys of table. spectra and files can be used together - e.g. overlay an attached reference spectrum on a measured one. If you called capture_scene first, microscope_image (np.ndarray|None) and image_extent ([xmin,xmax,ymin,ymax] stage mm|None) are also injected - after ax.imshow(microscope_image, extent=image_extent), overlaying peaks at the measurement (x,y) makes a peak map on top of the microscope image. A figure created with plt is auto-saved and shown in the chat. Small numeric results are observed if you print() them. To SAVE a computed spectrum, call the injected hook save_result(filename, intensity, raman_shift=None, wavelength_nm=None, metadata=None) inside your code - it writes data/<filename>.csv at full precision and returns the path, which also comes back in the tool result as saved_files. This is the correct way to persist a processed spectrum (baseline-corrected, spike-removed, normalized, smoothed): do it in the same run_analysis call that computes it. save_result is the ONLY way to write an array - there is no tool that takes an intensity array as an argument. Do NOT print an array in order to re-type it elsewhere: printing thousands of numbers overflows the context window and loses precision. Print only a short summary (how many points, how many spikes removed, where the peaks are). stdout is truncated past 4000 characters. Constraints (safety): no hardware (laser/stage/CCD) control, no network access, no file access other than save_result and plt figures, imports limited to computation libraries such as numpy/scipy/matplotlib/math. A 'measurement' like a 3x3 scan is done first with move_stage + acquire_spectrum, not this tool, and the saved result is analyzed/visualized here. On failure, read error/trace, fix the code, and call again. WHEN NOT TO USE THIS: a ready-made tool already covers some jobs and needs no code - combine_spectra (grid of spectra images), aggregate_spectra_csv (one summary row per measurement), bundle_results (zip for download), apply_background_subtraction (IPBSA baseline removal). Reach for run_analysis when no such tool fits, not by default.",
    ),
    tool_schema(
        _data.bundle_results,
        'Bundle saved measurement files (png/csv/json) into a single zip and provide a download link. Use it when the user wants to download all the results.',
    ),
]

#: 아키텍처와 무관한 공통 도구. 두 에이전트가 '같은 리스트'에서 출발하므로 파일 분석
#: 능력이 구조적으로 어긋날 수 없다 — 어긋나면 AILA↔CoALA 비교의 독립변수가
#: 오케스트레이션이 아니라 '어느 쪽에 도구를 더 줬는가'가 되어 실험이 무너진다.
BASE_TOOLS = RAMAN_TOOLS + FILE_TOOLS

# KB 검색은 **구현이 하나, 설명이 둘**이다. 구현(data_tools.search_knowledge_base)을 나누면
# "같은 KB 를 같은 알고리즘으로" 라는 공정성 전제가 깨지지만, 설명문은 아키텍처마다 달라야
# 한다: CoALA 에서는 이 호출이 사이클을 닫지 않는 planning 액션이라는 사실을 모델이 알아야
# 하고(모르면 조회 한 번에 사이클이 닫히는 줄 알고 정보 수집을 아낀다), ReAct 에는 그런
# 개념 자체가 없다. 그래서 스키마만 둘로 둔다.
KB_TOOL = tool_schema(
    _data.search_knowledge_base,
    "Search the Raman measurement protocol and recommended parameters (laser power %, "
    "exposure time in seconds, main peak positions and assignments) by sample type "
    "(graphene, cell, exosome, silicon, CNT, etc.). "
    "Call it before deciding measurement parameters - do not guess; base them on this result. "
    "It does not turn on the laser, so it is harmless to the sample, and may be called multiple times.",
)

KB_TOOL_COALA = tool_schema(
    _data.search_knowledge_base,
    "[semantic memory read - planning] Search the Raman measurement protocol and recommended "
    "parameters (laser power %, exposure time in seconds, main peak positions) by sample type "
    "(graphene, cell, exosome, silicon, CNT, etc.). Call it before deciding measurement parameters. "
    "It does not turn on the laser, so it is harmless, and this call does not end the cycle "
    "(keep planning after gathering information).",
)

#: ReAct(AILA)에 바인딩되는 도구 전체. CoALA 는 KB 설명을 바꾸고 장기기억을 더한다.
ALL_TOOLS = BASE_TOOLS + [KB_TOOL]


TOOL_DISPATCH = {
    # ── 스테이지 ─────────────────────────────────────────────────────────────
    "move_stage":               lambda a: _stage.move_stage(**a),
    "get_stage_position":       lambda a: _stage.get_stage_position(),
    "move_stage_relative":      lambda a: _stage.move_stage_relative(**a),
    "get_stage_speed":          lambda a: _stage.get_stage_speed(),
    "set_stage_speed":          lambda a: _stage.set_stage_speed(**a),
    # ── 하드웨어 연결 관리 ────────────────────────────────────────────────────
    "reconnect_hardware":       lambda a: _sys.reconnect_hardware(**a),
    "get_hardware_status":      lambda a: _sys.get_hardware_status(),
    # ── 레이저 ──────────────────────────────────────────────────────────────
    "laser_on":                 lambda a: _laser.laser_on(),
    "laser_off":                lambda a: _laser.laser_off(),
    "set_laser_power":          lambda a: _laser.set_laser_power(**a),
    "get_laser_status":         lambda a: _laser.get_laser_status(),
    "set_guide_beam_mode":      lambda a: _laser.set_guide_beam_mode(),
    # ── 스펙트럼 수집 ────────────────────────────────────────────────────────
    "acquire_spectrum":         lambda a: _cache_and_return(_acq.acquire_spectrum(**a)),
    # ── 카메라 ──────────────────────────────────────────────────────────────
    "start_camera_stream":      lambda a: _cam.start_camera_stream(),
    "stop_camera_stream":       lambda a: _cam.stop_camera_stream(),
    "set_camera_exposure":      lambda a: _cam.set_camera_exposure(**a),
    "set_camera_auto_exposure": lambda a: _cam.set_camera_auto_exposure(**a),
    "analyze_microscope_image": lambda a: _cam.analyze_microscope_image(**a),
    "move_to_pixel":            lambda a: _cam.move_to_pixel(**a),
    "capture_scene":            lambda a: _cam.capture_scene(),
    # ── 오토포커스 ───────────────────────────────────────────────────────────
    "run_autofocus":            lambda a: _cam.run_autofocus(**a),
    # ── 그리드 매핑(미리보기 + 실행) ─────────────────────────────────────────
    "preview_grid_scan":        lambda a: _acq.preview_grid_scan(**a),
    "run_grid_scan":            lambda a: _acq.run_grid_scan(**a),
    # ── CCD 설정 ─────────────────────────────────────────────────────────────
    "get_ccd_info":             lambda a: _ccd.get_ccd_info(),
    "set_ccd_exposure":         lambda a: _ccd.set_ccd_exposure(**a),
    "set_ccd_acquisition_mode": lambda a: _ccd.set_ccd_acquisition_mode(**a),
    "set_ccd_trigger_mode":     lambda a: _ccd.set_ccd_trigger_mode(**a),
    "set_ccd_read_mode":        lambda a: _ccd.set_ccd_read_mode(**a),
    "set_ccd_preamp_gain":      lambda a: _ccd.set_ccd_preamp_gain(**a),
    # (set_ccd_em_gain / set_ccd_output_amp 제거 — 이 카메라는 EM CCD 가 아니다. 위 주석 참고)
    "set_ccd_shift_speeds":     lambda a: _ccd.set_ccd_shift_speeds(**a),
    "set_ccd_temperature":      lambda a: _ccd.set_ccd_temperature(**a),
    "set_ccd_cooler":           lambda a: _ccd.set_ccd_cooler(**a),
    "set_ccd_shutter":          lambda a: _ccd.set_ccd_shutter(**a),
    "set_ccd_image_flip":       lambda a: _ccd.set_ccd_image_flip(**a),
    # ── 데이터 로드 ──────────────────────────────────────────────────────────
    # (save_spectrum 은 제거했다 — 위 주석 참고. 저장은 자동저장/save_result/
    #  save_measurement_point 가 담당한다.)
    "load_spectrum":            lambda a: _bg.load_spectrum(**a),
    # ── 측정 결과 정리(자동 저장분 대상) ─────────────────────────────────────
    "list_results":             lambda a: ok(items=[
        {k: it[k] for k in ("base", "session", "date", "title", "timestamp", "meta")}
        for it in _data.list_results(**a)]),
    "combine_spectra":          lambda a: _data.combine_spectra(**a),
    "aggregate_spectra_csv":    lambda a: _data.aggregate_spectra_csv(**a),
    "bundle_results":           lambda a: _data.bundle_results(**a),
    # ── 분석 전용 코드 샌드박스(하드웨어 미접근) ─────────────────────────────
    "run_analysis":             lambda a: _data.run_analysis(**a),
    # ── 외부 웹 검색(내부 지식에 없을 때) ────────────────────────────────────
    "web_search":               lambda a: _data.web_search(**a),
    # ── 측정점 기록 ──────────────────────────────────────────────────────────
    "save_measurement_point":   lambda a: _acq.save_measurement_point(**a),
    # ── 배경 제거 (IPBSA) ────────────────────────────────────────────────────
    "apply_background_subtraction": lambda a: _bg.apply_background_subtraction(**a),
    "list_bg_versions":             lambda a: _bg.list_bg_versions(),
    "get_bg_version":               lambda a: _bg.get_bg_version(**a),
}


# ──────────────────────────────────────────────────────────────────────────────
# 자체 점검:  python backend/tools/tools.py
#   스키마와 디스패치는 서로를 모르는 두 목록이라 조용히 갈라질 수 있다. 한쪽에만 있는
#   이름은 '모델이 부를 수 없는 구현' 이거나 '구현 없는 도구'(호출 즉시 실패)다.
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import inspect

    # 실행 표는 셋으로 갈라져 있다. 하드웨어 유무에 묶이면 안 되는 것들이 TOOL_DISPATCH
    # 밖에 있어야 하기 때문이다(위 머리말). 점검은 셋을 다 봐야 의미가 있다.
    from backend.agents.runtime.runtime import RUNTIME_DISPATCH
    from backend.tools.non_hw_tools.file_tools import FILE_DISPATCH

    declared = [t["function"]["name"] for t in ALL_TOOLS]
    assert len(declared) == len(set(declared)), \
        f"스키마에 중복된 이름: {sorted(n for n in declared if declared.count(n) > 1)}"

    runnable = set(TOOL_DISPATCH) | set(FILE_DISPATCH) | set(RUNTIME_DISPATCH)
    #: 스키마 없이 디스패치에만 남겨 두는 이름 — 의도된 예외.
    #: load_spectrum: 2026-08-12 에 도구 목록에서 뺐다. 1024점 intensity 배열을 돌려주는데
    #:   tool_slim 의 MAX_SCALAR_LIST(32)에 걸려 모델에게 닿기 전에 잘렸고, 남는 정보는
    #:   open_file 이 더 많이 준다. 디스패치 항목은 남긴다 — 모델이 이름을 환각해 부를 때
    #:   "Unknown tool" 대신 실제 동작(파일 읽기)이 나가는 편이 낫고, bg_tools 가
    #:   CSV 리더로 내부 호출하기도 한다.
    _DISPATCH_ONLY = {"load_spectrum"}

    orphan_schema = sorted(set(declared) - runnable)
    orphan_impl = sorted(runnable - set(declared) - _DISPATCH_ONLY)
    assert not orphan_schema, f"구현 없는 도구(호출하면 실패한다): {orphan_schema}"
    # run_analysis 는 선언(data_tools)과 실행 가로채기(file_tools)가 갈라져 있어 양쪽에 있다.
    assert not orphan_impl, f"모델이 부를 수 없는 구현: {orphan_impl}"

    # 스키마의 인자는 함수 시그니처에서 파생된다 — 실제로 그런 인자가 있는지 되짚는다.
    from backend.tools.non_hw_tools import file_tools as _file
    checked = 0
    for t in ALL_TOOLS:
        name = t["function"]["name"]
        fn = next((getattr(m, name)
                   for m in (_stage, _laser, _ccd, _cam, _acq, _sys, _data, _bg, _file)
                   if hasattr(m, name)), None)
        assert fn is not None, f"{name}: 어느 도구 모듈에도 같은 이름의 함수가 없다"
        params = set(inspect.signature(fn).parameters)
        declared_args = (set(t["function"]["parameters"]["properties"])
                         | set(t["function"]["parameters"]["required"]))
        assert declared_args <= params, \
            f"{name}: 스키마에만 있는 인자 {sorted(declared_args - params)}"
        checked += 1

    # KB 는 설명만 다른 두 벌이다. 이름과 인자까지 갈라지면 '같은 도구'가 아니게 된다.
    a, b = KB_TOOL["function"], KB_TOOL_COALA["function"]
    assert a["name"] == b["name"], "KB 도구 이름이 갈라졌다"
    assert a["parameters"] == b["parameters"], "KB 도구 인자가 갈라졌다(설명만 달라야 한다)"
    assert a["description"] != b["description"], "CoALA 용 KB 설명이 사라졌다(사이클 표기)"

    print(f"통과: 스키마 {len(declared)}개(인자 정합 {checked}개) · 하드웨어 디스패치 "
          f"{len(TOOL_DISPATCH)}개 · 파일 {len(FILE_DISPATCH)}개 · 런타임 "
          f"{len(RUNTIME_DISPATCH)}개 · 고아 없음")
