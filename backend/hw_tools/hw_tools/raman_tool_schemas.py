# -*- coding: utf-8 -*-
"""LLM 에게 전달할 도구 목록 — **여기에는 '언제 쓰는가'만 적는다.**

인자의 이름·타입·범위·설명은 raman_tools.py 의 함수 시그니처에 있고, 스키마는
tool_schema() 가 그 시그니처를 읽어 만든다. 선언이 한 곳뿐이라 어긋날 수가 없다
(그렇게 바꾼 경위는 tools/schema.py 머리말 참고).

[description 을 줄일 때]
겹치는 도구(측정 조건을 미리 걸기 vs acquire_spectrum 인자로 넘기기, 화면을 찍는 세 도구,
저장물을 나열하는 네 도구)마다 "어느 쪽을 언제 쓰는가 + 둘 다 부르지 말 것"이 적혀 있다.
그 문장을 먼저 지우지 말 것 — 도구를 중복 호출하거나 엉뚱한 쪽을 골라 실패하는 경우의
대부분이 그 부재에서 온다.
"""
from backend.hw_tools.hw_tools import raman_tools as T
from backend.tools.schema import tool_schema

RAMAN_TOOLS = [
    tool_schema(
        T.move_stage,
        'Move the stage to an absolute position (mm). Out-of-range targets are REJECTED with an error rather than clipped, so the reported position is always where the stage actually is. Related: move_stage_relative (same limits, but you give a displacement), move_to_pixel (you give a point on the camera image).',
    ),
    tool_schema(
        T.get_stage_position,
        'Read the current stage X, Y, Z position (mm). You do NOT need this right after a move - move_stage / move_stage_relative / move_to_pixel already return the resulting position. get_hardware_status also reports it while diagnosing connection problems.',
    ),
    tool_schema(
        T.move_stage_relative,
        'Move the stage by a displacement from its current position (mm). The resulting target is range-checked exactly like move_stage, so a displacement that would leave the travel range is rejected (the error tells you the computed target). Use move_stage when you know the absolute coordinate.',
    ),
    tool_schema(
        T.laser_on,
        "Turn the laser on. Which beam actually comes out depends on whether the power is currently applied to the optics: if it is, the MEASUREMENT beam fires; otherwise only the GUIDE beam does (the ND filter is still blocking). The response tells you which one via the 'beam' field - check it, because a guide-beam 'ON' produces no Raman signal. This tool CANNOT arm the measurement beam - use set_laser_power(percent) to arm it without firing, or acquire_spectrum(power=...) which arms it and handles on -> acquire -> off atomically. Use laser_on by itself only for alignment with the guide beam.",
    ),
    tool_schema(
        T.laser_off,
        'Stop the laser from firing. The ND filter and the beam splitter stay where they are, so turning it on again emits the same measurement beam - but the camera also stays blind until you call set_guide_beam_mode. You do not need to call this after acquire_spectrum - that tool always turns the laser off (even when it fails) and restores the camera view.',
    ),
    tool_schema(
        T.set_laser_power,
        'Set the laser power (ND filter transmission, 0.004-100 %) WITHOUT firing the laser. This only moves the ND filter to arm the measurement beam - it does not turn the laser on. Use it when the user asks you to set or change the power by itself, or to arm the beam before alignment. For an actual measurement prefer acquire_spectrum(power=...), which applies the power, fires, measures and turns the laser off atomically - chaining set_laser_power + laser_on + laser_off leaves the beam on the sample during your own reasoning time, which can photobleach or damage a biological sample. Out-of-range values are REJECTED, not silently clamped.',
    ),
    tool_schema(
        T.get_laser_status,
        'Query the current laser state: whether it is firing (is_on), the applied power (power_percent, %), and power_armed - whether that power is actually applied to the optics right now. If power_armed is false the ND filter is at the blocking position, so laser_on() would emit only the guide beam; arm it with set_laser_power(percent), or measure with acquire_spectrum(power=...), which applies the power itself. Always read power_armed, not just power_percent - power_percent is the last value that was requested and survives a switch to guide-beam mode, so on its own it will make you think the beam is ready when it is not.',
    ),
    tool_schema(
        T.acquire_spectrum,
        "Acquire a Raman spectrum at the current position. Supports three acquisition modes: Single (one shot) / Accumulate (averaged, high SNR) / Kinetic (continuous time series). Automatically handles the laser ON -> power stabilization -> CCD acquisition -> laser OFF flow. IMPORTANT: every parameter is optional and OMITTING one means 'keep whatever the instrument is already set to' (shutter works the same way, except that it opens to 'auto' when nobody has ever set it - the CCD boots with the shutter closed). So set_ccd_exposure / set_ccd_acquisition_mode / set_ccd_read_mode / set_ccd_trigger_mode are respected - you may either configure first and then call this with no arguments, or pass the values directly here. Both go through the same code and are equivalent; do not do both 'just in case'. This is also the ONLY way to fire the measurement beam: it applies the power, turns the laser on, acquires, and turns it off again even if the acquisition fails. It also puts the optics back into the guide-beam/camera position afterwards, so the microscope camera can see the sample again - you do NOT need to call set_guide_beam_mode after measuring. The returned exposure_time / laser_power_pct / num_accumulations are read back from the hardware, so they tell you what was ACTUALLY used. When a calibrator is connected, the raman_shift_cm-1, wavelength_nm, laser_nm fields are included. Kinetic mode returns per-frame data in a frames array.",
    ),
    tool_schema(
        T.start_camera_stream,
        "Start real-time camera streaming. Streaming must be ON before analyze_microscope_image, capture_scene, preview_grid_scan or run_autofocus can get a frame. The response field already_streaming tells you whether it was ALREADY running before your call - if it was true, someone else (the user's live view) owns the stream, so do not stop it afterwards.",
    ),
    tool_schema(
        T.stop_camera_stream,
        'Stop real-time camera streaming. Only call this if YOU started the stream - i.e. start_camera_stream returned already_streaming=false. Stopping a stream the user was watching blanks their live view, and every image tool stops working until it is restarted. When in doubt, leave it running.',
    ),
    tool_schema(
        T.get_ccd_info,
        'Query all current CCD settings and status. Returns temperature, cooling status, exposure time, acquisition mode, readout mode, gain, shift speeds, pixel count, etc. in one call. Use it before and after changing parameters to verify the current state.',
    ),
    tool_schema(
        T.set_ccd_exposure,
        'Set the CCD exposure time (seconds). Larger values give stronger signal but longer measurement time. The value persists - a later acquire_spectrum keeps it unless you pass exposure there. OVERLAP: acquire_spectrum(exposure=...) sets exactly the same thing through the same code. Use this tool when you set the exposure once and then measure several times; pass it to acquire_spectrum when it is a one-off. Never do both for the same measurement.',
    ),
    tool_schema(
        T.set_ccd_acquisition_mode,
        "Set the CCD acquisition mode. 'single': single shot. 'accumulate': sum after num_accumulations shots. 'kinetic': acquire num_kinetics frames continuously. 'run_till_abort': acquire continuously until an abort command (acquire_spectrum cannot use this one - set single/accumulate/kinetic before measuring). The mode and counts persist - acquire_spectrum keeps them unless you pass acq_mode / num_accumulations there. OVERLAP: acquire_spectrum(acq_mode=, num_accumulations=, kinetic_count=) sets the same thing through the same code; use one or the other, not both. This tool is the only way to select 'run_till_abort'. The response reports the counts actually in effect on the hardware, not what you asked for.",
    ),
    tool_schema(
        T.set_ccd_trigger_mode,
        "Set the CCD trigger mode. 'internal': start acquisition via software (default). 'external': start acquisition on an external TTL signal. 'external_start': start on external trigger then internal timing. 'external_exposure': expose while the external TTL is HIGH. 'external_fvb_em': external trigger in FVB/EM readout. 'software': use SendSoftwareTrigger. The mode persists - acquire_spectrum keeps it unless you pass trigger_mode there. OVERLAP: acquire_spectrum(trigger_mode=...) accepts exactly the same values through the same code. Use one or the other, not both.",
    ),
    tool_schema(
        T.set_ccd_read_mode,
        "Set the CCD readout mode. 'fvb': Full Vertical Binning - sum all rows -> 1D spectrum (used for Raman). 'single_track': read only a specific vertical row. single_track_center is required. 'image': full 2D image or ROI (acquire_spectrum cannot build a 1D spectrum from this - switch back to 'fvb' before measuring). The mode persists - acquire_spectrum keeps it unless you pass read_mode there. OVERLAP: acquire_spectrum(read_mode=, hbin=, single_track_center=, single_track_width=) sets the same thing through the same code, using the same parameter names. Use one or the other, not both. This tool is the only way to select 'image' mode.",
    ),
    tool_schema(
        T.set_ccd_preamp_gain,
        'Set the pre-amplifier gain index. See preamp_gains_available in get_ccd_info() for the available gain list. A larger index gives higher gain, which helps measure weak signals but also increases noise.',
    ),
    tool_schema(
        T.set_ccd_shift_speeds,
        'Set the vertical (VS) and horizontal (HS) pixel shift speeds. VS: a larger index is slower but reduces charge-transfer noise. HS: a larger index is slower but reduces readout noise. You may specify only one of them. See get_ccd_info() for the available speed list.',
    ),
    tool_schema(
        T.set_ccd_temperature,
        'Set the CCD cooling target temperature (°C). Lower temperature reduces dark-current noise. It may take several minutes to reach; check progress with get_ccd_info(). Typical range: -80 to 20°C.',
    ),
    tool_schema(
        T.set_ccd_cooler,
        'Turn the CCD Peltier cooler on (true) or off (false). It must be turned off before shutdown.',
    ),
    tool_schema(
        T.set_ccd_shutter,
        "Set the CCD shutter mode. 'auto'  - open and close automatically during acquisition (normal measurement). 'open'  - force open. 'close' - force closed. Like the other CCD settings this one PERSISTS: a later acquire_spectrum keeps it unless you pass its own shutter argument. So for several dark / background frames set 'close' once here and measure repeatedly; for a single dark frame acquire_spectrum(shutter='close') is enough. Set it back to 'auto' before normal measurements.",
    ),
    tool_schema(
        T.set_ccd_image_flip,
        "Set horizontal/vertical flip of the acquired 2D image. Only valid when the CCD read mode is 'image' - in the 1D spectrum modes (fvb / single_track) flipping would misalign the intensity array against the calibrated wavelength axis, and the correct orientation is already set at startup, so the call is rejected there. You almost never need this for Raman measurements.",
    ),
    tool_schema(
        T.get_stage_speed,
        'Query the current stage movement speed (mm/s). Returns the x_speed_mm_s, y_speed_mm_s, z_speed_mm_s fields.',
    ),
    tool_schema(
        T.set_stage_speed,
        "Set the stage movement speed. X/Y max 5.0 mm/s, Z max 0.1 mm/s. If a specific axis speed is omitted, its current speed is maintained. Values above the limit are clipped to it; the response reports the speeds that will ACTUALLY be used and lists any clipped axes under 'clipped'.",
    ),
    tool_schema(
        T.get_hardware_status,
        "Report which hardware components (stage, laser, ccd, camera) are currently CONNECTED, and for the connected ones whether they actually respond. Read `summary` first. Call this FIRST whenever a hardware tool fails, before trying to fix anything - it tells you whether one device is down or several, which decides what is even worth attempting. It touches nothing and fires no laser, so it is always safe to call. SCOPE: this answers 'is it there and alive'. For the SETTINGS of a working device use the per-device tools instead - get_ccd_info (exposure, mode, temperature), get_laser_status (power and whether it is armed), get_stage_position, get_stage_speed.",
    ),
    tool_schema(
        T.reconnect_hardware,
        "Release and re-initialize a hardware component that is unresponsive or stuck. component: 'stage' | 'ccd' | 'camera' | 'laser' | 'all'. Call get_hardware_status first to see what is actually down. Read the returned `errors` text carefully - it distinguishes two very different cases. (a) 'resource is still held by this process': a process-level lock that NO tool can clear. Calling this tool again will not help; the server must be restarted by a human. Do not retry - carry on without that component and say so in your final answer. (b) 're-initialization failed' after a successful release: a device-side problem (power, cable, driver, or another program holding the device). Retrying once is reasonable; beyond that, proceed without the component and state the limitation. Never call this repeatedly in a loop - it cannot fix either case by repetition. WARNING: reconnecting the 'ccd' re-runs cooling and can block for minutes until -40 C stabilizes.",
    ),
    tool_schema(
        T.set_guide_beam_mode,
        'Switch the laser to guide-beam standby state. Moves the beam splitter to the standby position and the ND filter to the main-beam blocking position. Use it for sample alignment and focus checking.',
    ),
    tool_schema(
        T.set_camera_exposure,
        'Set the camera (TUCam) exposure time (ms).',
    ),
    tool_schema(
        T.set_camera_auto_exposure,
        'Enable (true) or disable (false) camera auto exposure.',
    ),
    tool_schema(
        T.analyze_microscope_image,
        "Capture the current view of the TuCam optical microscope camera and pass it to YOU as an image to look at. Use it when visual judgment is needed, e.g. checking sample position, identifying a target, or detecting debris. Streaming must be active. The tool returns the IMAGE ITSELF - it does NOT return any coordinates. YOU must look at the image, find the target in it, and read off its pixel coordinates yourself; do not search the JSON response for an x/y field, there is none. Those pixel coordinates are not stage coordinates, so to move there pass them to move_to_pixel. It also returns brightness statistics (min/max/mean_intensity) and a relative sharpness_score for the same image - use those to check exposure or compare frames. Do NOT use sharpness_score to focus manually: run_autofocus optimises a different metric (guide-beam spot area) and would settle at a different Z. OVERLAP - three tools capture the camera, pick by PURPOSE: analyze_microscope_image = you look at it now (returns the image to you, saves nothing); capture_scene = save the view as a file so run_analysis can draw a peak map on top of it (returns no image for you to inspect); preview_grid_scan = show where a planned grid would land. All three return the same pixel coordinate system, so a coordinate read from any of them can be passed to move_to_pixel. This tool saves NOTHING: it does not become 'the image at this point' for save_measurement_point. If you want this view bundled into a measurement-point record, call capture_scene as well.",
    ),
    tool_schema(
        T.move_to_pixel,
        'Convert pixel coordinates (pixel_x, pixel_y) within the camera image to stage mm coordinates and move there. The image center corresponds to the current stage position. After checking the target pixel coordinates with analyze_microscope_image, move with this tool.',
    ),
    tool_schema(
        T.run_autofocus,
        'Hill-climbing autofocus based on minimizing the guide-beam laser spot area. Computes the spot pixel count (area) via Otsu thresholding on the laser OFF/ON difference image, and moves the stage to the Z position with minimum area (where the laser spot is sharpest). Adaptive hill-climbing auto-adjusts the step size and finally returns to the historical minimum position. It moves Z only, uses the GUIDE beam (not the measurement beam), and leaves the laser off. The search is clipped to the Z travel range; if the response contains z_limit_hits the focus may be physically out of reach, and repeating the call will NOT help - report that instead. This is the only focusing tool: do not try to focus by comparing sharpness_score from analyze_microscope_image, which is a different metric and converges elsewhere.',
    ),
    tool_schema(
        T.preview_grid_scan,
        "Preview a rows x cols grid mapping WITHOUT moving the stage or firing the laser. Overlays the planned scan points as circles on the current camera view and returns that image so the layout can be visually verified before committing. ORIENTATION (do not confuse the two): rows = number of points stacked VERTICALLY = grid HEIGHT (stage Y axis); cols = number of points side-by-side HORIZONTALLY = grid WIDTH (stage X axis). So rows=3, cols=2 is a TALL grid (3 high x 2 wide) and rows=2, cols=3 is a WIDE grid (2 high x 3 wide) - these are DIFFERENT layouts, never swap them. When the user asks for a grid like 'A x B', decide deliberately which number is the horizontal count (width -> cols) and which is the vertical count (height -> rows), then use this preview image to confirm the drawn orientation matches what they asked. MANDATORY HUMAN APPROVAL: always preview FIRST, then STOP - show this preview image to the user, end your turn, and WAIT. Do NOT call run_grid_scan in the same turn as this preview; only call it in a later turn after the user has explicitly approved this exact layout. If center_x/center_y are omitted, the current stage position is used as the grid center, and that resolved center is what gets approved - a later run_grid_scan with the centre omitted scans THAT position even if the stage has moved since. Whatever you pass here you must pass IDENTICALLY to run_grid_scan: the approval is matched on the arguments themselves, so omitting the centre here and spelling it out there (or vice versa) is rejected as a mismatch even when both mean the same place. SIZE LIMIT: rows * cols must be <= 400 points; a larger grid is refused here, so agree a smaller size with the user before promising a map. The camera field of view is small, so with wide spacing some points may fall outside the frame; they are still measured, and the response reports how many are in view (n_in_view) along with the exact view size in fov_mm - plan spacing from that returned value rather than from a remembered number.",
    ),
    tool_schema(
        T.run_grid_scan,
        "Execute a rows x cols grid mapping: for each point it moves the stage, optionally autofocuses, acquires one spectrum, and auto-saves it (position-tagged). Returns a single compact summary (counts, intensity min/max/mean, and per-point data when 32 points or fewer) instead of one tool message per point - this is the token-efficient way to run a map. ORIENTATION: rows = vertical count (height, stage Y), cols = horizontal count (width, stage X); rows=3,cols=2 is a tall 3x2 grid, rows=2,cols=3 is a wide 2x3 grid - do not swap them. PASS THE APPROVED ARGUMENTS VERBATIM: the approval gate compares the arguments you give, not the physical positions they work out to. If you OMITTED center_x/center_y in the preview you must omit them here too - filling in the numeric coordinates of that same spot is rejected as an 'Approval mismatch'. (An omitted centre runs at the position the approved preview was drawn at, so the scan lands where the user saw it even if the stage has moved in between.) SIZE LIMIT: rows * cols must be <= 400 points. REQUIRES PRIOR HUMAN APPROVAL: do NOT call this in the same turn as preview_grid_scan. Call it ONLY after (1) you showed the user a preview_grid_scan image, (2) you ended that turn, and (3) the user EXPLICITLY approved that exact layout in a later message. If the user has not explicitly approved the previewed grid, do not call this - preview first and wait. The laser is fired at every point, so the estimated cumulative dose is checked up front and the scan is refused if it exceeds the safety limit. READ THE RESULT BEFORE REPORTING SUCCESS: n_measured can be lower than n_points, and with autofocus='each' the response may carry n_autofocus_failed (those points were measured at whatever Z the stage was at, so a weak signal there is an artefact, not a property of the sample) or n_autofocus_z_limit (the focus is physically out of reach - re-running will not help). If autofocus fails several times in a row the scan STOPS EARLY and comes back with ok=false plus an 'aborted' field; the points measured up to then are still saved, but the grid is incomplete and you must say so rather than reporting the map as done.",
    ),
    tool_schema(
        T.load_spectrum,
        'Load ONE spectrum CSV that this system produced - an auto-saved measurement, a processed spectrum you wrote with save_result inside run_analysis, or a background-subtracted result. Accepts a path relative to data/ or an absolute path; the data/-relative path returned by list_session_artifacts or by save_result can be passed here verbatim. Returns the intensity array plus any axis columns (raman_shift_cm-1 / wavelength_nm / background_intensity) and the saved metadata. 1D SPECTRA ONLY (Single / Accumulate). A Kinetic measurement is saved as one row per frame per pixel; loading it here is REFUSED rather than silently flattening the frames into one wrong array. Analyse kinetic data with run_analysis, which receives it as a 2D frames array. PICK THE RIGHT TOOL: for a file the USER attached to the chat use inspect_file (different store, different format); to compute over MANY saved measurements at once use run_analysis, which receives them all as `spectra` without any loading; to re-read a background-subtraction result you made this session use get_bg_version, which needs no path.',
    ),
    tool_schema(
        T.save_measurement_point,
        "Group what you just measured at this position into ONE measurement-point record: the most recent acquire_spectrum result, the most recent capture_scene microscope image, and the current stage coordinates. You do NOT pass any arrays - the spectrum and image files are already saved automatically, and this tool only links them together under a point id. TIMING IS PART OF THE RECORD: the coordinates are read from the stage AT THE MOMENT YOU CALL THIS, not from the spectrum's metadata. So call it immediately after acquiring the spectrum and capturing the view at that position, and BEFORE moving the stage anywhere else - otherwise the record pairs this point's spectrum with the next point's coordinates, and nothing later can detect that. The image must come from capture_scene; analyze_microscope_image does not count. The response lists anything that was missing (e.g. no image captured yet). Use one call per position to build a multi-point dataset.",
    ),
    tool_schema(
        T.apply_background_subtraction,
        "Remove the fluorescence background of a Raman spectrum using IPBSA (iterative polynomial background subtraction). Uses the most recently acquired spectrum (source='last') or a saved file path as the source. YOU CHOOSE THE POLYNOMIAL ORDER, and that choice is the substance of this task - there is no safe default to fall back on. Too low and a curved fluorescence background survives, tilting the baseline and distorting relative peak heights; too high and the polynomial starts following the peaks themselves, eating real signal. The right order depends on how curved THIS spectrum's background is, so look at the data before deciding. DO NOT settle on the first result: run it at two or three orders, compare them with list_bg_versions(), and keep the one where the baseline is flat between peaks while peak heights are unchanged. Say which order you chose and why. OVERLAP with run_analysis: that sandbox can run any baseline algorithm you write yourself (asymmetric least squares, rolling ball, wavelet, ...). This tool is IPBSA specifically, and it keeps parameters comparable across versions and writes the standard CSV format. Pick whichever the task calls for and state which one you used.",
    ),
    tool_schema(
        T.list_bg_versions,
        'Return the list of all saved background-subtraction result versions with their parameters and key statistics. This summary is what you compare versions with - poly_order, iterations_run, converged and max_corrected_intensity are enough to pick an order. The spectra themselves are NOT included here, and you cannot read them into the conversation at all: long arrays are stripped out of every tool result to protect the context window, so get_bg_version() will not hand you the numbers either. If you need the actual corrected spectrum, re-run apply_background_subtraction with save_result=true and work on the saved file in run_analysis. Use it when calling apply_background_subtraction() several times to compare.',
    ),
    tool_schema(
        T.get_bg_version,
        'Re-read the parameters and statistics of one background-subtraction version (poly_order, iterations_run, converged, max intensities, saved_path if it was saved). It does NOT put the corrected spectrum in front of you: long arrays are stripped from tool results, so the corrected_data / background_data arrays never arrive - do not call this expecting to read the numbers, and do not retry when they are absent. To work with the actual arrays, save the version (save_result=true) and analyse the file in run_analysis. Check version_label with list_bg_versions().',
    ),
    tool_schema(
        T.list_results,
        "Query the list of MEASUREMENTS auto-saved by acquire_spectrum. Returns each item's base (file identifier), session, title, timestamp, and meta (coordinates, etc.). Get the base to pass to combine_spectra / aggregate_spectra_csv / bundle_results here. By default this lists only the measurements from YOUR current session - your own work, not other sessions'. Files live in data/results/<date>/<your session>/. THERE ARE THREE 'list' TOOLS, one per store - choose by what you are looking for: list_results = raw measurements you acquired (this one); list_session_artifacts = files YOU produced (processed spectra from save_result, measurement-point records, figures), each with a path you can load_spectrum; list_uploaded_files = data files the USER attached to the chat. (Background-subtraction versions from this conversation are not files - use list_bg_versions.)",
    ),
    tool_schema(
        T.combine_spectra,
        'Combine several saved measurement spectra into a single grid image and render it. Each cell title uses the title auto-generated at save time (scan coordinates, power, exposure) as is. e.g. for a 10x10 scan, arranged by coordinate. If names is omitted, combine everything from that date. This is the ready-made one-call version - it needs no code. run_analysis can also plot the same measurements, but only use it when you need a layout or computation this tool does not give you (overlaid curves, peak maps, custom axes).',
    ),
    tool_schema(
        T.aggregate_spectra_csv,
        'Build a CSV summarizing several saved measurements, one row per experiment (date, time, title, coordinates, power, exposure, max intensity, total intensity, peak position). Use it to organize multiple experiments into one table. One call, no code. It summarizes ONE ROW PER MEASUREMENT - if you need per-point values computed some other way, or a table of derived quantities, use run_analysis instead.',
    ),
    tool_schema(
        T.capture_scene,
        "SAVE the current microscope (camera) view as a file, for later use as a background image. It also computes the stage-coordinate extent of that image (position + calibrated field of view), so in a later run_analysis you get microscope_image / image_extent injected and can overlay a peak map on the microscope photo. Call it once before a scan measurement (camera streaming required). It does NOT return the image for you to look at - use analyze_microscope_image for that. It is also what save_measurement_point references as 'the image at this point'.",
    ),
    tool_schema(
        T.web_search,
        'Search the external web and fetch the top results (title, URL, summary). Use it to find recent/specialist information (literature, recommended parameter values, methodology, etc.) that internal knowledge/KB (search_knowledge_base) cannot answer. It is recommended to first check local knowledge with search_knowledge_base and use this tool for external search when that is insufficient. If there is no internet it returns a failure, in which case decide from local knowledge.',
    ),
    tool_schema(
        T.run_analysis,
        "Run 'computation/visualization' Python code on saved measurement data AND on files the user attached to the chat, in a safe sandbox. Use it to handle analyses not provided as tools (baseline correction, peak detection, per-coordinate peak maps/heatmaps, etc.) directly in code. Already injected into the runtime: spectra (list[dict] - each item has base, title, x, y, power, exposure, mode, raman_shift (np.ndarray or None), intensity (np.ndarray)), np (numpy), plt (matplotlib.pyplot). Preprocessing helpers are injected too, so you do not have to re-implement them: ipbsa(y, order=5, max_iterations=100, threshold=0.001) returns (corrected, background) using the SAME iterative-polynomial routine as the apply_background_subtraction tool, and poly_baseline(y, order=5, x=None) returns a single polynomial background fit if you would rather build your own iteration. Prefer ipbsa(...) over hand-writing a polynomial baseline loop - it is the single biggest source of over-long code. A Kinetic measurement carries its time series too: frames is a 2D np.ndarray of shape (n_frames, n_pixels), so frames.mean(axis=0) is the averaged spectrum and frames[:, px] is one pixel's intensity over time. For those items intensity is the frame MEAN (flagged by intensity_is_frame_mean) - use frames, not intensity, when the question is about change over time. Very long runs are cut to the first 200 frames and say so in frames_truncated. This is the ONLY way to analyse a kinetic measurement; load_spectrum refuses those files. If you pass file_ids, the attached files are parsed and injected as files (list[dict] - each item has file_id, filename, sheet, columns (list[str]), n_rows, and table (dict mapping column name -> np.ndarray for numeric columns, list[str] for text columns)). Inspect a file's structure with inspect_file first, then use the column names you saw as keys of table. spectra and files can be used together - e.g. overlay an attached reference spectrum on a measured one. If you called capture_scene first, microscope_image (np.ndarray|None) and image_extent ([xmin,xmax,ymin,ymax] stage mm|None) are also injected - after ax.imshow(microscope_image, extent=image_extent), overlaying peaks at the measurement (x,y) makes a peak map on top of the microscope image. A figure created with plt is auto-saved and shown in the chat. Small numeric results are observed if you print() them. To SAVE a computed spectrum, call the injected hook save_result(filename, intensity, raman_shift=None, wavelength_nm=None, metadata=None) inside your code - it writes data/<filename>.csv at full precision and returns the path, which also comes back in the tool result as saved_files. This is the correct way to persist a processed spectrum (baseline-corrected, spike-removed, normalized, smoothed): do it in the same run_analysis call that computes it. save_result is the ONLY way to write an array - there is no tool that takes an intensity array as an argument. Do NOT print an array in order to re-type it elsewhere: printing thousands of numbers overflows the context window and loses precision. Print only a short summary (how many points, how many spikes removed, where the peaks are). stdout is truncated past 4000 characters. Constraints (safety): no hardware (laser/stage/CCD) control, no network access, no file access other than save_result and plt figures, imports limited to computation libraries such as numpy/scipy/matplotlib/math. A 'measurement' like a 3x3 scan is done first with move_stage + acquire_spectrum, not this tool, and the saved result is analyzed/visualized here. On failure, read error/trace, fix the code, and call again. WHEN NOT TO USE THIS: a ready-made tool already covers some jobs and needs no code - combine_spectra (grid of spectra images), aggregate_spectra_csv (one summary row per measurement), bundle_results (zip for download), apply_background_subtraction (IPBSA baseline removal). Reach for run_analysis when no such tool fits, not by default.",
    ),
    tool_schema(
        T.bundle_results,
        'Bundle saved measurement files (png/csv/json) into a single zip and provide a download link. Use it when the user wants to download all the results.',
    ),
]
