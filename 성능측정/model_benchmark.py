# coding=utf-8
"""
Ollama 모델 벤치마크 스크립트
- 여러 모델에 동일한 질문을 던지고 응답 품질/속도 비교
- 스트리밍 + 이중 타임아웃(토큰 무응답 감지 + 절대 최대시간)
"""

import sys
import json
import time
import threading
from datetime import datetime
import ollama

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ── 설정 ──────────────────────────────────────────────────────────────────────

OLLAMA_HOST = "http://192.168.1.15:11434"

MODELS = [
    "gemma4:31b",
    "llama3.1:8b",
    "qwen3-vl:30b",
    "llama3.1:70b",
    "qwen3:30b",
    "deepseek-r1:8b",
]

# 타임아웃 설정
TOKEN_IDLE_TIMEOUT = 120    # 초 — 이 시간 동안 새 토큰이 안 오면 강제종료
ABSOLUTE_TIMEOUT   = 300   # 초 — 응답이 느려도 이 시간이 지나면 무조건 종료

# 결과 저장 경로
OUTPUT_FILE = "benchmark_results.json"

# ── 질문 목록 (29개) ────────────────────────────────────────────────────────

QUESTIONS = [
    # Section 1: Fundamental Physics & Spectroscopy
    "When using a 532 nm laser, explain the formula for converting Raman shift from wavelength units to wavenumber (cm⁻¹).",
    "How do you distinguish between Rayleigh peaks and Raman peaks in experimental data?",
    "Why do Raman peaks get buried in samples with strong background fluorescence?",
    "What are the differences in spectral resolution and detecting range when using 600 gr/mm vs 1800 gr/mm gratings?",
    "Why does the spectrum differ when measuring the same sample with 532 nm vs 785 nm?",

    # Section 2: Experimental Design & Optimization
    "When you get a Raman spectrum with low SNR, which should you adjust first among exposure time, accumulation, and laser power? Explain your reasoning.",
    "If you cool the CCD to -55 °C instead of -40 °C, what noise is reduced and what limitations exist?",
    "What strategies are needed to secure peaks while reducing laser damage in biological samples?",
    "What trade-offs arise in interpreting mapping results when using Full Vertical Binning (FVB)?",
    "When is it appropriate to use Accumulation mode vs Kinetic mode?",
    "Explain how the data flow differs between single acquisition and mapping acquisition.",

    # Section 3: Signal Processing & Data Analysis
    "What problems arise if you set the polynomial order too high in polynomial background subtraction?",
    "When using Savitzky-Golay smoothing, how does increasing the window size affect peak analysis?",
    "What methods are available for removing cosmic rays from single-shot data without accumulation?",
    "Among peak height, peak area, and FWHM, which metric is more suitable for concentration quantification and why?",
    "What's the difference between averaging multiple spectra obtained at the same location vs using the median?",
    "Why do stitching artifacts occur in wide-range spectra obtained through grating scanning (step & stitch)?",

    # Section 4: Spatial Analysis & Mapping
    "What does it physically mean when peak intensity is high in a specific ROI on a mapping image?",
    "What information can you better visualize using ROI ratio maps (ROI1/ROI2)?",
    "If peak intensity decreases with depth in a Z-scan, what causes should you suspect?",
    "When the optical image and Raman map appear misaligned in the same area, what should you check first?",

    # Section 5: Troubleshooting & Diagnostics
    "If all Raman peak intensities suddenly drop, describe the troubleshooting sequence.",
    "What are possible causes when the Rayleigh peak position appears to have drifted?",
    "What hardware issues should you suspect when only specific rows have intensity spikes during mapping?",
    "Why does the spectrum baseline vary day-to-day for the same sample?",
    "When the laser is normal but CCD count is close to 0, what's the most likely problem?",
    "What artifacts occur if you don't provide sufficient settling time after stage movement?",

    # Section 6: System-Level Understanding
    "What problems occur if you don't separate the CCD acquisition thread from the UI thread?",
    "What reproducibility issues arise when metadata is missing when exporting raw data to CSV?",
]

# ── 핵심 로직 ─────────────────────────────────────────────────────────────────

def query_with_timeout(model: str, prompt: str) -> dict:
    """
    스트리밍으로 모델 응답을 수집하고 이중 타임아웃을 적용한다.

    Returns:
        {
            "response": str,          # 수집된 응답 텍스트 (잘렸을 수도 있음)
            "elapsed": float,         # 소요 시간(초)
            "tokens": int,            # 생성된 토큰 수
            "status": "ok" | "token_timeout" | "absolute_timeout" | "error",
            "error": str | None,
        }
    """
    client = ollama.Client(host=OLLAMA_HOST)

    collected        = []
    token_count      = [0]
    last_token       = [time.time()]
    first_token_time = [None]   # TTFT 기록용
    done_event       = threading.Event()
    error_box        = [None]

    def _stream():
        try:
            for chunk in client.generate(model=model, prompt=prompt, stream=True):
                token = chunk.get("response", "")
                now = time.time()
                if token_count[0] == 0:
                    first_token_time[0] = now   # 첫 토큰 도착 시각
                collected.append(token)
                token_count[0] += 1
                last_token[0] = now
                print(token, end="", flush=True)
                if chunk.get("done"):
                    break
        except Exception as e:
            error_box[0] = str(e)
        finally:
            done_event.set()

    t = threading.Thread(target=_stream, daemon=True)
    start = time.time()
    t.start()

    # ── 타임아웃 감시 루프 ────────────────────────────────────────────────
    status = "ok"
    while not done_event.is_set():
        elapsed = time.time() - start
        idle    = time.time() - last_token[0]

        if idle >= TOKEN_IDLE_TIMEOUT:
            status = "token_timeout"
            break
        if elapsed >= ABSOLUTE_TIMEOUT:
            status = "absolute_timeout"
            break

        done_event.wait(timeout=1.0)

    elapsed_total = time.time() - start
    ttft    = round(first_token_time[0] - start, 2) if first_token_time[0] else None
    gen     = round(elapsed_total - ttft, 2)        if ttft is not None else None

    if error_box[0]:
        status = "error"

    return {
        "response": "".join(collected),
        "elapsed":  round(elapsed_total, 2),   # 질문→완료
        "ttft":     ttft,                      # 질문→첫토큰
        "gen":      gen,                       # 첫토큰→완료
        "tokens":   token_count[0],
        "status":   status,
        "error":    error_box[0],
    }


# ── 실행 ──────────────────────────────────────────────────────────────────────

def run_benchmark():
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "token_idle_timeout": TOKEN_IDLE_TIMEOUT,
            "absolute_timeout":   ABSOLUTE_TIMEOUT,
        },
        "models": {},
    }

    for model in MODELS:
        print(f"\n{'='*60}")
        print(f"  모델: {model}")
        print(f"{'='*60}")
        results["models"][model] = []

        for qi, question in enumerate(QUESTIONS, start=1):
            if not question.strip():
                print(f"\n[Q{qi}] (질문 없음, 건너뜀)")
                results["models"][model].append({"q_index": qi, "skipped": True})
                continue

            print(f"\n[Q{qi}] {question}")
            print("-" * 40)

            result = query_with_timeout(model, question)

            if result["status"] != "ok":
                print(f"\n[!] 종료 사유: {result['status']}")
            if result["error"]:
                print(f"[!] 에러: {result['error']}")

            print(f"\n[{result['elapsed']}s | {result['tokens']} tokens | {result['status']}]")

            results["models"][model].append({
                "q_index":  qi,
                "question": question,
                **result,
            })

    # 결과 저장
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n\n결과 저장 완료: {OUTPUT_FILE}")
    _print_summary(results)
    save_pdf(results, OUTPUT_FILE.replace(".json", ".pdf"))


def _val_or_inf(q: dict, key: str) -> float:
    """타임아웃/에러 → 999999, 정상 → 실제 값, 스킵/없음 → None"""
    if q.get("skipped"):
        return None
    if q.get("status") in ("token_timeout", "absolute_timeout", "error"):
        return 999999
    v = q.get(key)
    return v if v is not None else 999999


def _print_table(title: str, results: dict, models: list, valid_qs: list, key: str):
    COL = 12
    model_col = max(len(m) for m in models) + 2

    header = f"{'모델':<{model_col}}" + "".join(f"  Q{qi:<{COL-2}}" for qi in valid_qs)
    sep    = "-" * len(header)

    print(f"\n\n{'='*len(header)}")
    print(f"  {title}   ※ 타임아웃/에러 = 999999")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)

    for model in models:
        qs_map = {q["q_index"]: q for q in results["models"][model]}
        row = f"{model:<{model_col}}"
        for qi in valid_qs:
            val = _val_or_inf(qs_map.get(qi, {"skipped": True}), key)
            cell = f"{val:.1f}" if val is not None else "-"
            row += f"  {cell:<{COL-2}}"
        print(row)

    print(sep)


def _print_summary(results: dict):
    valid_qs = [i for i, q in enumerate(QUESTIONS, start=1) if q.strip()]
    if not valid_qs:
        print("\n(질문이 없어 표를 생성할 수 없습니다)")
        return

    models = list(results["models"].keys())

    _print_table("① 질문→완료  총 시간 (초)",     results, models, valid_qs, "elapsed")
    _print_table("② 질문→첫토큰  TTFT (초)",      results, models, valid_qs, "ttft")
    _print_table("③ 첫토큰→완료  생성 시간 (초)", results, models, valid_qs, "gen")

    # ── 모델별 요약 한 줄 ───────────────────────────────────────────────────
    model_col = max(len(m) for m in models) + 2
    print()
    for model in models:
        qs_data  = results["models"][model]
        answered = [q for q in qs_data if not q.get("skipped") and q.get("status") == "ok"]
        timeouts = [q for q in qs_data if q.get("status") in ("token_timeout", "absolute_timeout")]
        errors   = [q for q in qs_data if q.get("status") == "error"]
        avg_time = (sum(q["elapsed"] for q in answered) / len(answered)) if answered else 0
        print(f"  {model:<{model_col}}  정상:{len(answered)}  타임아웃:{len(timeouts)}  에러:{len(errors)}  평균:{avg_time:.1f}s")


def _build_pdf_table(title: str, results: dict, models: list, valid_qs: list, key: str, styles):
    """표 하나(제목 + Table 객체) 반환"""
    TIMEOUT_MARK = "999999"

    header_row = ["Model"] + [f"Q{qi}" for qi in valid_qs]
    rows = [header_row]

    for model in models:
        qs_map = {q["q_index"]: q for q in results["models"][model]}
        row = [model]
        for qi in valid_qs:
            val = _val_or_inf(qs_map.get(qi, {"skipped": True}), key)
            row.append(TIMEOUT_MARK if val is None else f"{val:.1f}")
        rows.append(row)

    col_w = [90] + [38] * len(valid_qs)
    tbl = Table(rows, colWidths=col_w, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  colors.HexColor("#2c3e50")),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0),  8),
        ("FONTNAME",     (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",     (0, 1), (-1, -1), 7.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
        ("GRID",         (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("ALIGN",        (1, 0), (-1, -1), "CENTER"),
        ("ALIGN",        (0, 0), (0, -1),  "LEFT"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        # 999999 셀 강조
        *[("BACKGROUND", (ci, ri), (ci, ri), colors.HexColor("#fadbd8"))
          for ri, row in enumerate(rows[1:], start=1)
          for ci, cell in enumerate(row)
          if cell == TIMEOUT_MARK],
    ]))

    heading = Paragraph(title, styles["title"])
    return [heading, Spacer(1, 4), tbl, Spacer(1, 18)]


def save_pdf(results: dict, path: str = "benchmark_results.pdf"):
    valid_qs = [i for i, q in enumerate(QUESTIONS, start=1) if q.strip()]
    models   = list(results["models"].keys())

    doc = SimpleDocTemplate(
        path,
        pagesize=landscape(A4),
        leftMargin=15*mm, rightMargin=15*mm,
        topMargin=15*mm,  bottomMargin=15*mm,
    )

    styles = {
        "h1":    ParagraphStyle("h1",    fontName="Helvetica-Bold", fontSize=14, spaceAfter=6),
        "sub":   ParagraphStyle("sub",   fontName="Helvetica",      fontSize=9,  spaceAfter=10, textColor=colors.HexColor("#555555")),
        "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=10, spaceAfter=4,  textColor=colors.HexColor("#2c3e50")),
        "body":  ParagraphStyle("body",  fontName="Helvetica",      fontSize=8,  spaceAfter=3),
    }

    elements = []

    # ── 표지 헤더 ────────────────────────────────────────────────────────────
    elements.append(Paragraph("Ollama Model Benchmark Results", styles["h1"]))
    elements.append(Paragraph(
        f"Generated: {results['timestamp']}  |  "
        f"Token idle timeout: {results['config']['token_idle_timeout']}s  |  "
        f"Absolute timeout: {results['config']['absolute_timeout']}s  |  "
        f"※ Red cell = timeout / error (999999)",
        styles["sub"]
    ))
    elements.append(Spacer(1, 6))

    # ── 시간 측정 표 3개 ─────────────────────────────────────────────────────
    elements += _build_pdf_table("① Total Time: Question → Complete (sec)",       results, models, valid_qs, "elapsed", styles)
    elements += _build_pdf_table("② TTFT: Question → First Token (sec)",          results, models, valid_qs, "ttft",    styles)
    elements += _build_pdf_table("③ Generation Time: First Token → Complete (sec)",results, models, valid_qs, "gen",     styles)

    # ── 모델별 요약 표 ───────────────────────────────────────────────────────
    elements.append(Paragraph("Model Summary", styles["title"]))
    elements.append(Spacer(1, 4))

    summary_rows = [["Model", "OK", "Timeout", "Error", "Avg Time (s)"]]
    for model in models:
        qs_data  = results["models"][model]
        answered = [q for q in qs_data if not q.get("skipped") and q.get("status") == "ok"]
        timeouts = [q for q in qs_data if q.get("status") in ("token_timeout", "absolute_timeout")]
        errors   = [q for q in qs_data if q.get("status") == "error"]
        avg      = (sum(q["elapsed"] for q in answered) / len(answered)) if answered else 0
        summary_rows.append([model, len(answered), len(timeouts), len(errors), f"{avg:.1f}"])

    stbl = Table(summary_rows, colWidths=[120, 50, 60, 50, 80], repeatRows=1)
    stbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  colors.HexColor("#2c3e50")),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 8),
        ("FONTNAME",     (0, 1), (-1, -1), "Helvetica"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
        ("GRID",         (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("ALIGN",        (1, 0), (-1, -1), "CENTER"),
        ("ALIGN",        (0, 0), (0, -1),  "LEFT"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
    ]))
    elements.append(stbl)

    doc.build(elements)
    print(f"\nPDF 저장 완료: {path}")


if __name__ == "__main__":
    run_benchmark()
