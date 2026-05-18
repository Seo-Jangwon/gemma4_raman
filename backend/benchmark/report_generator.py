"""
벤치마크 결과를 JSON + HTML로 저장.
"""

from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime


def save_json(results: list[dict], output_dir: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"benchmark_results_{ts}.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[JSON] {path}")
    return path


def save_checkpoint(result: dict, checkpoint_path: Path) -> None:
    """단일 태스크 결과를 JSONL 파일에 원자적으로 append."""
    line = json.dumps(result, ensure_ascii=False) + "\n"
    with checkpoint_path.open("a", encoding="utf-8") as f:
        f.write(line)


def load_checkpoint(checkpoint_path: Path) -> list[dict]:
    """JSONL 파일에서 완전한 줄만 읽어 결과 리스트 반환 (불완전 마지막 줄 무시)."""
    results = []
    for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            results.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # 크래시로 잘린 마지막 줄
    return results


def find_checkpoint_files(output_dir: Path) -> list[Path]:
    """수정시간 기준 최신순 정렬된 checkpoint_*.jsonl 파일 목록."""
    return sorted(
        output_dir.glob("checkpoint_*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def generate_html(results: list[dict], output_dir: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"benchmark_report_{ts}.html"
    html = _build_html(results, report_filename=path.name)
    path.write_text(html, encoding="utf-8")
    print(f"[HTML] {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────

def _auto_score(result: dict) -> str:
    if result.get("skipped"):
        return "SKIP"
    verdicts = result.get("auto_results", [])
    if not verdicts:
        return "N/A"
    non_human = [v for v in verdicts if not v.get("is_human_only")]
    if not non_human:
        return "HUMAN"
    if all(v["passed"] for v in non_human):
        return "PASS"
    return "FAIL"


def _score_css(score: str) -> str:
    return {
        "PASS": "pass",
        "FAIL": "fail",
        "SKIP": "skip",
        "HUMAN": "human",
        "N/A": "na",
    }.get(score, "na")


def _fmt_args(args: dict) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in args.items()) if args else "()"


def _build_summary_rows(results: list[dict]) -> str:
    rows = []
    for r in results:
        task = r["task"]
        score = _auto_score(r)
        css = _score_css(score)
        elapsed = f"{r.get('elapsed_sec', 0):.1f}s" if not r.get("skipped") else "—"
        human_flag = "⚠️ 인간채점" if task.get("requires_human_review") else ""
        rows.append(
            f"<tr>"
            f"<td><a href='#{task['id']}'>{task['id']}</a></td>"
            f"<td>{task['category']}</td>"
            f"<td>{task['complexity']}</td>"
            f"<td>{task['skill']}</td>"
            f"<td class='score {css}'>{score}</td>"
            f"<td>{elapsed}</td>"
            f"<td>{human_flag}</td>"
            f"<td class='human-score-cell'><input type='number' min='0' max='10' placeholder='—' "
            f"class='score-input' data-task-id='{task['id']}' data-field='score'> / 10</td>"
            f"</tr>"
        )
    return "\n".join(rows)


def _build_detail_sections(results: list[dict]) -> str:
    sections = []
    for r in results:
        task = r["task"]
        score = _auto_score(r)
        css = _score_css(score)

        # --- skip ---
        if r.get("skipped"):
            sections.append(
                f"<section id='{task['id']}' class='task-section skip'>"
                f"<h2>{task['id']} <span class='badge skip'>SKIP</span></h2>"
                f"<p class='task-desc'>{task['description']}</p>"
                f"<p class='skip-reason'>스킵 사유: {task.get('skip_reason', '')}</p>"
                f"</section>"
            )
            continue

        # --- tool trace ---
        trace_rows = []
        for t in r.get("tool_trace", []):
            result_str = json.dumps(t.get("result", {}), ensure_ascii=False)
            if len(result_str) > 200:
                result_str = result_str[:200] + "…"
            args_str = _fmt_args(t.get("args", {}))
            trace_rows.append(
                f"<tr>"
                f"<td>{t['step']}</td>"
                f"<td><code>{t['tool']}</code></td>"
                f"<td><code>{args_str}</code></td>"
                f"<td>{t.get('duration_ms', '?')}ms</td>"
                f"<td class='result-cell'><code>{result_str}</code></td>"
                f"</tr>"
            )
        trace_html = (
            "<table class='trace-table'>"
            "<thead><tr><th>Step</th><th>Tool</th><th>Args</th><th>시간</th><th>결과</th></tr></thead>"
            "<tbody>" + "\n".join(trace_rows) + "</tbody></table>"
            if trace_rows else "<p class='no-data'>tool 호출 없음</p>"
        )

        # --- before/after state ---
        def state_table(state: dict | None) -> str:
            if not state:
                return "<p class='no-data'>상태 없음</p>"
            rows_ = []
            stage = state.get("stage") or {}
            rows_.append(f"<tr><td>Stage X</td><td>{stage.get('x', '?')}</td></tr>")
            rows_.append(f"<tr><td>Stage Y</td><td>{stage.get('y', '?')}</td></tr>")
            rows_.append(f"<tr><td>Stage Z</td><td>{stage.get('z', '?')}</td></tr>")
            vel = stage.get("velocity", {})
            rows_.append(f"<tr><td>Speed X</td><td>{vel.get('x', '?')} mm/s</td></tr>")
            ccd = state.get("ccd") or {}
            rows_.append(f"<tr><td>CCD 온도</td><td>{ccd.get('temperature', '?')}°C</td></tr>")
            rows_.append(f"<tr><td>CCD 노출</td><td>{ccd.get('exposure_time', '?')}s</td></tr>")
            rows_.append(f"<tr><td>CCD 모드</td><td>{ccd.get('ro_mode', '?')}</td></tr>")
            laser = state.get("laser") or {}
            rows_.append(f"<tr><td>레이저</td><td>{'ON' if laser.get('is_on') else 'OFF'} / {laser.get('power_pct', '?')}%</td></tr>")
            return "<table class='state-table'><tbody>" + "\n".join(rows_) + "</tbody></table>"

        state_html = (
            "<div class='state-grid'>"
            "<div><h4>Before</h4>" + state_table(r.get("pre_state")) + "</div>"
            "<div><h4>After</h4>" + state_table(r.get("post_state")) + "</div>"
            "</div>"
        )

        # --- auto verifier results ---
        verify_rows = []
        for v in r.get("auto_results", []):
            v_css = "pass" if v["passed"] else ("human" if v.get("is_human_only") else "fail")
            icon = "✓" if v["passed"] else ("?" if v.get("is_human_only") else "✗")
            verify_rows.append(
                f"<tr class='{v_css}'>"
                f"<td>{icon}</td>"
                f"<td><code>{v['verifier_type']}</code></td>"
                f"<td>{v['detail']}</td>"
                f"</tr>"
            )
        verify_html = (
            "<table class='verify-table'>"
            "<thead><tr><th></th><th>검증 타입</th><th>결과</th></tr></thead>"
            "<tbody>" + "\n".join(verify_rows) + "</tbody></table>"
            if verify_rows else "<p class='no-data'>검증 없음</p>"
        )

        # --- llm response ---
        llm_resp = r.get("llm_response", "")

        sections.append(
            f"<section id='{task['id']}' class='task-section {css}'>"
            f"<h2>{task['id']} — {task['skill']} "
            f"<span class='badge {css}'>{score}</span>"
            f"{'<span class=\"badge human\">인간채점필요</span>' if task.get('requires_human_review') else ''}"
            f"</h2>"
            f"<p class='task-desc'><strong>태스크:</strong> {task['description']}</p>"
            f"<details open><summary>Tool 호출 트레이스 ({len(r.get('tool_trace', []))}회)</summary>{trace_html}</details>"
            f"<details><summary>하드웨어 상태 Before / After</summary>{state_html}</details>"
            f"<details open><summary>자동 검증 결과</summary>{verify_html}</details>"
            f"<details><summary>LLM 응답</summary><pre class='llm-resp'>{llm_resp}</pre></details>"
            f"<div class='human-score-box'>"
            f"<label>인간 채점: <input type='number' min='0' max='10' placeholder='0~10' "
            f"data-task-id='{task['id']}' data-field='score'> / 10점</label>"
            f"<textarea data-task-id='{task['id']}' data-field='note' "
            f"placeholder='메모 (선택사항)'></textarea>"
            f"</div>"
            f"</section>"
        )
    return "\n".join(sections)


def _stats(results: list[dict]) -> dict:
    total = len(results)
    skipped = sum(1 for r in results if r.get("skipped"))
    run = total - skipped
    passed = sum(1 for r in results if not r.get("skipped") and _auto_score(r) == "PASS")
    failed = sum(1 for r in results if not r.get("skipped") and _auto_score(r) == "FAIL")
    human = sum(1 for r in results if not r.get("skipped") and _auto_score(r) == "HUMAN")
    return {"total": total, "run": run, "passed": passed, "failed": failed, "human": human, "skipped": skipped}


def _build_html(results: list[dict], report_filename: str = "") -> str:
    stats = _stats(results)
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_rows = _build_summary_rows(results)
    detail_sections = _build_detail_sections(results)
    results_json = json.dumps(results, ensure_ascii=False)
    report_ns = report_filename.replace(".", "_").replace("-", "_")

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>라만 분광기 벤치마크 결과</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', sans-serif; font-size: 14px; color: #222; background: #f5f5f5; }}
  h1 {{ background: #1a237e; color: #fff; padding: 16px 24px; }}
  h2 {{ padding: 12px 0 6px; font-size: 1.1em; border-bottom: 2px solid #ddd; margin-bottom: 8px; }}
  h4 {{ font-size: 0.9em; color: #555; margin-bottom: 4px; }}
  .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
  .stats {{ display: flex; gap: 12px; margin: 16px 0; flex-wrap: wrap; }}
  .stat-card {{ background: #fff; border-radius: 6px; padding: 12px 20px; text-align: center; min-width: 100px; box-shadow: 0 1px 3px rgba(0,0,0,.15); }}
  .stat-card .num {{ font-size: 1.8em; font-weight: bold; }}
  .stat-card .label {{ font-size: 0.8em; color: #666; }}
  .num.pass {{ color: #2e7d32; }} .num.fail {{ color: #c62828; }} .num.skip {{ color: #616161; }} .num.human {{ color: #e65100; }}
  table {{ width: 100%; border-collapse: collapse; margin: 8px 0; }}
  th, td {{ padding: 6px 10px; text-align: left; border: 1px solid #ddd; vertical-align: top; }}
  th {{ background: #e8eaf6; font-weight: 600; }}
  tr:nth-child(even) {{ background: #fafafa; }}
  .score {{ font-weight: bold; text-align: center; }}
  .score.pass {{ color: #2e7d32; background: #e8f5e9; }}
  .score.fail {{ color: #c62828; background: #ffebee; }}
  .score.skip {{ color: #616161; background: #f5f5f5; }}
  .score.human {{ color: #e65100; background: #fff3e0; }}
  .score.na {{ color: #888; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 0.75em; margin-left: 6px; font-weight: bold; }}
  .badge.pass {{ background: #c8e6c9; color: #1b5e20; }}
  .badge.fail {{ background: #ffcdd2; color: #b71c1c; }}
  .badge.skip {{ background: #eeeeee; color: #424242; }}
  .badge.human {{ background: #ffe0b2; color: #bf360c; }}
  .task-section {{ background: #fff; border-radius: 8px; padding: 16px; margin: 16px 0; box-shadow: 0 1px 4px rgba(0,0,0,.12); }}
  .task-section.pass {{ border-left: 4px solid #4caf50; }}
  .task-section.fail {{ border-left: 4px solid #f44336; }}
  .task-section.skip {{ border-left: 4px solid #9e9e9e; opacity: 0.7; }}
  .task-section.human {{ border-left: 4px solid #ff9800; }}
  .task-desc {{ margin: 8px 0 12px; color: #333; background: #f8f8f8; padding: 8px; border-radius: 4px; }}
  .skip-reason {{ color: #888; font-style: italic; margin: 8px 0; }}
  details {{ margin: 8px 0; }}
  summary {{ cursor: pointer; font-weight: 600; padding: 6px 0; color: #444; }}
  summary:hover {{ color: #1a237e; }}
  .trace-table td, .state-table td, .verify-table td {{ font-size: 0.85em; }}
  .result-cell {{ max-width: 300px; word-break: break-all; }}
  .state-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .verify-table tr.pass {{ background: #f1f8e9; }}
  .verify-table tr.fail {{ background: #ffebee; }}
  .verify-table tr.human {{ background: #fff8e1; }}
  .human-score-box {{ margin-top: 12px; padding: 12px; background: #fff9e6; border: 1px dashed #fbc02d; border-radius: 6px; }}
  .human-score-box label {{ font-weight: 600; display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }}
  .human-score-box input[type=number] {{ width: 70px; padding: 4px; border: 1px solid #ccc; border-radius: 4px; font-size: 1em; }}
  .human-score-cell input {{ width: 60px; padding: 2px 4px; border: 1px solid #ccc; border-radius: 4px; }}
  textarea {{ width: 100%; height: 48px; padding: 6px; border: 1px solid #ccc; border-radius: 4px; font-size: 0.9em; resize: vertical; }}
  pre.llm-resp {{ white-space: pre-wrap; background: #263238; color: #cfd8dc; padding: 12px; border-radius: 4px; font-size: 0.85em; max-height: 200px; overflow-y: auto; }}
  code {{ background: #eff; padding: 1px 4px; border-radius: 3px; font-size: 0.9em; }}
  .no-data {{ color: #aaa; font-style: italic; padding: 6px 0; }}
  .summary-section {{ background: #fff; border-radius: 8px; padding: 16px; margin: 16px 0; box-shadow: 0 1px 4px rgba(0,0,0,.12); }}
  a {{ color: #1a237e; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  @media print {{
    .human-score-box textarea {{ height: 60px; }}
    details {{ display: block; }}
    details[open] summary::before {{ content: ''; }}
  }}
</style>
</head>
<body>
<h1>라만 분광기 에이전트 벤치마크 결과</h1>
<div class="container">
  <p style="color:#555; margin:8px 0;">실행 시각: {run_time}</p>

  <div class="stats">
    <div class="stat-card"><div class="num">{stats['total']}</div><div class="label">전체</div></div>
    <div class="stat-card"><div class="num">{stats['run']}</div><div class="label">실행됨</div></div>
    <div class="stat-card"><div class="num pass">{stats['passed']}</div><div class="label">PASS</div></div>
    <div class="stat-card"><div class="num fail">{stats['failed']}</div><div class="label">FAIL</div></div>
    <div class="stat-card"><div class="num human">{stats['human']}</div><div class="label">HUMAN</div></div>
    <div class="stat-card"><div class="num skip">{stats['skipped']}</div><div class="label">SKIP</div></div>
  </div>

  <div class="summary-section">
    <h2>요약 테이블</h2>
    <table>
      <thead>
        <tr>
          <th>문항</th><th>분류</th><th>난이도</th><th>요구역량</th>
          <th>자동채점</th><th>소요시간</th><th>비고</th><th>인간채점</th>
        </tr>
      </thead>
      <tbody>
        {summary_rows}
      </tbody>
    </table>
  </div>

  {detail_sections}
</div>
<script type="application/json" id="results-data">{results_json}</script>
<button onclick="exportScores()" style="position:fixed;bottom:20px;right:20px;z-index:999;background:#1a237e;color:#fff;border:none;border-radius:6px;padding:10px 18px;font-size:14px;cursor:pointer;box-shadow:0 2px 6px rgba(0,0,0,.3)">채점 내보내기</button>
<script>
(function(){{
  var NS='bm_scores__{report_ns}';
  var results=JSON.parse(document.getElementById('results-data').textContent);

  function loadScores(){{
    var saved=JSON.parse(localStorage.getItem(NS)||'{{}}');
    document.querySelectorAll('[data-task-id]').forEach(function(el){{
      var key=el.dataset.taskId+'.'+el.dataset.field;
      if(saved[key]!==undefined) el.value=saved[key];
    }});
  }}

  function saveScore(e){{
    var el=e.target;
    if(!el.dataset||!el.dataset.taskId) return;
    var key=el.dataset.taskId+'.'+el.dataset.field;
    var saved=JSON.parse(localStorage.getItem(NS)||'{{}}');
    saved[key]=el.value;
    localStorage.setItem(NS,JSON.stringify(saved));
    document.querySelectorAll('[data-task-id="'+el.dataset.taskId+'"][data-field="'+el.dataset.field+'"]').forEach(function(o){{
      if(o!==el) o.value=el.value;
    }});
  }}

  function autoScore(r){{
    if(r.skipped) return 'SKIP';
    var v=r.auto_results||[];
    if(!v.length) return 'N/A';
    var nh=v.filter(function(x){{return !x.is_human_only;}});
    if(!nh.length) return 'HUMAN';
    return nh.every(function(x){{return x.passed;}})?'PASS':'FAIL';
  }}

  window.exportScores=function(){{
    var saved=JSON.parse(localStorage.getItem(NS)||'{{}}');
    var data=results.map(function(r){{
      var t=r.task;
      var scoreVal=saved[t.id+'.score'];
      return {{
        task_id:t.id, category:t.category, complexity:t.complexity,
        skill:t.skill, description:t.description,
        auto_score:autoScore(r),
        tool_trace_summary:(r.tool_trace||[]).map(function(x){{return {{step:x.step,tool:x.tool,duration_ms:x.duration_ms}};}}),
        human_score:(scoreVal!==undefined&&scoreVal!=='')?Number(scoreVal):null,
        note:saved[t.id+'.note']||''
      }};
    }});
    var ts=new Date().toISOString().replace(/[:.]/g,'-').slice(0,19);
    var blob=new Blob([JSON.stringify(data,null,2)],{{type:'application/json'}});
    var a=document.createElement('a');
    a.href=URL.createObjectURL(blob);
    a.download='scores_'+ts+'.json';
    a.click();
    URL.revokeObjectURL(a.href);
  }};

  document.addEventListener('DOMContentLoaded',function(){{
    loadScores();
    document.addEventListener('input',saveScore);
    document.addEventListener('change',saveScore);
  }});
}})();
</script>
</body>
</html>"""
