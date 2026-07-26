# -*- coding: utf-8 -*-
"""
수동채점용 상세 HTML 리포트 생성기.

grade.py 가 만든 graded_*.json 을 받아, 문항마다 AILA/CoALA 를 '나란히' 놓고
  · 보낸 프롬프트 / 채점기준(프로즈) / 수동채점 가이드
  · 각 에이전트의 전체 툴 트레이스(이름·인자·결과), 최종 답변, planning(CoALA),
    dose·소요시간·응답유형, 자동 검증기 결과
를 한 화면에 보여준다. 각 (문항×에이전트)마다 수동채점 위젯(pass/fail/partial + 메모)이
있고 브라우저 localStorage 에 자동 저장된다. 상단 'Export grades' 로 수동채점을 JSON 으로
내려받는다(자동채점 결과와 합쳐 최종 집계에 쓴다).

상단에는 grade.py 요약(에이전트별 auto 통계 + 변형별 되묻기율)을 그대로 싣는다.

실행:
  python -m backend.benchmark.report --graded backend/benchmark/results/graded_XXXX.json
  → results/report_XXXX.html  (브라우저로 열어 채점)
"""
from __future__ import annotations

import argparse
import html
import json
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
AGENTS = ("AILA", "CoALA")


def esc(s) -> str:
    return html.escape("" if s is None else str(s))


def _compact_args(args: dict) -> str:
    if not isinstance(args, dict) or not args:
        return ""
    parts = [f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in list(args.items())[:5]]
    return ", ".join(parts)


def _compact_result(r) -> str:
    """툴 결과를 한 줄 요약 — ok/에러 + 정보성 스칼라 몇 개."""
    if not isinstance(r, dict):
        return esc(str(r)[:120])
    if r.get("_truncated"):
        return "(잘림)"
    if r.get("ok") is False:
        return "✗ " + esc(str(r.get("error", ""))[:160])
    keys = ("max_intensity", "position", "temperature", "shutter", "ro_mode",
            "preamp_gain_i", "n_measured", "n_points", "power_pct", "z", "saved")
    bits = []
    for k in keys:
        if k in r and r[k] is not None:
            v = r[k]
            bits.append(f"{k}={json.dumps(v, ensure_ascii=False)[:60]}")
    return "✓ " + esc(", ".join(bits)) if bits else "✓"


def render_tools(rec: dict) -> str:
    calls = rec.get("tool_calls") or []
    if not calls:
        return '<div class="empty">— 툴 호출 없음 —</div>'
    rows = []
    for c in calls:
        ok = c.get("ok", True)
        cls = "tool" if ok else "tool err"
        name = esc(c.get("name"))
        args = esc(_compact_args(c.get("args") or {}))
        res = _compact_result(c.get("result"))
        rows.append(
            f'<div class="{cls}"><span class="tstep">#{c.get("step","?")}</span>'
            f'<span class="tname">{name}</span>(<span class="targs">{args}</span>) '
            f'<span class="tres">→ {res}</span></div>'
        )
    return "".join(rows)


def render_planning(rec: dict) -> str:
    pl = rec.get("planning") or []
    if not pl:
        return ""
    rows = []
    for p in pl:
        phase = esc(p.get("phase", ""))
        msg = esc(p.get("message", ""))
        extra = ""
        if p.get("scores") is not None:
            extra += f' <span class="pscore">scores={esc(p.get("scores"))}</span>'
        if p.get("chosen"):
            extra += f' <span class="pchosen">→ {esc(p.get("chosen"))}</span>'
        rows.append(f'<div class="pl"><b>{phase}</b> {msg}{extra}</div>')
    return f'<details class="planning"><summary>planning ({len(pl)})</summary>{"".join(rows)}</details>'


def render_verifiers(rec: dict) -> str:
    vrs = rec.get("verifier_results") or []
    if not vrs:
        return '<div class="empty">— 검증기 없음 —</div>'
    rows = []
    for v in vrs:
        if v.get("is_human_only"):
            cls, mark = "vr human", "✎"
        elif v.get("passed"):
            cls, mark = "vr pass", "✓"
        else:
            cls, mark = "vr fail", "✗"
        rows.append(f'<div class="{cls}">{mark} <b>{esc(v.get("type"))}</b> {esc(v.get("detail"))}</div>')
    return "".join(rows)


def _verdict_chip(rec: dict) -> str:
    v = rec.get("auto_verdict")
    if v == "pass":
        return '<span class="chip cok">auto ✓</span>'
    if v == "fail":
        return f'<span class="chip cfail">auto ✗ ({rec.get("n_machine_pass")}/{rec.get("n_machine")})</span>'
    return '<span class="chip cman">수동</span>'


def render_agent_cell(rec: dict | None, report_id: str, run_id: str, agent: str) -> str:
    if rec is None:
        return f'<div class="agent none"><div class="agent-head"><b>{agent}</b> <span class="empty">(미실행)</span></div></div>'
    chips = [_verdict_chip(rec), f'<span class="chip">{esc(rec.get("response_type"))}</span>']
    if rec.get("fired_laser"):
        chips.append('<span class="chip cfire">🔴 발사</span>')
    if rec.get("asked_clarification"):
        chips.append('<span class="chip cask">❓ 되물음</span>')
    if rec.get("run_error"):
        chips.append(f'<span class="chip cerr">ERROR</span>')
    meta = f'{rec.get("elapsed_sec","?")}s · dose {rec.get("dose_mj","?")}mJ · tools {len(rec.get("tool_calls") or [])}'
    answer = rec.get("answer") or rec.get("final_report") or ""
    gid = f'{report_id}|{run_id}|{agent}'
    return f'''<div class="agent">
      <div class="agent-head"><b class="aname">{agent}</b> {" ".join(chips)}<span class="ameta">{esc(meta)}</span></div>
      <div class="sect-label">tool trace</div>
      <div class="tools">{render_tools(rec)}</div>
      {render_planning(rec)}
      <details class="answer" open><summary>최종 답변</summary><pre>{esc(answer)}</pre></details>
      <details class="verifiers"><summary>자동 검증기</summary>{render_verifiers(rec)}</details>
      <div class="manual" data-gid="{esc(gid)}">
        <span class="mlabel">수동:</span>
        <label><input type="radio" name="g_{esc(gid)}" value="pass">✓ Pass</label>
        <label><input type="radio" name="g_{esc(gid)}" value="fail">✗ Fail</label>
        <label><input type="radio" name="g_{esc(gid)}" value="partial">~ Partial</label>
        <input class="mnote" type="text" placeholder="메모…" data-gid="{esc(gid)}">
      </div>
    </div>'''


def render_task_card(run_id: str, meta: dict, cells: dict, report_id: str) -> str:
    variant = meta.get("variant", "none")
    amb = meta.get("is_safety_ambiguous")
    # 두 에이전트 자동판정/되묻기 불일치 → 비교 필터용
    verds = {a: (cells.get(a) or {}).get("auto_verdict") for a in AGENTS}
    asks = {a: (cells.get(a) or {}).get("asked_clarification") for a in AGENTS}
    disagree = (verds.get("AILA") != verds.get("CoALA")) or (asks.get("AILA") != asks.get("CoALA"))
    needs_manual = any((cells.get(a) or {}).get("needs_manual") for a in AGENTS)

    amb_badge = '<span class="badge amb">안전-애매</span>' if amb else ''
    var_badge = f'<span class="badge v-{esc(variant)}">{esc(variant)}</span>'
    note = meta.get("manual_note")
    note_html = f'<div class="mnote-guide">📝 {esc(note)}</div>' if note else ''
    cat0 = str(meta.get("category", "")).split(".")[0].strip()

    agent_cells = "".join(render_agent_cell(cells.get(a), report_id, run_id, a) for a in AGENTS)
    return f'''<div class="card" data-cat="{esc(cat0)}" data-variant="{esc(variant)}"
         data-manual="{int(bool(needs_manual))}" data-disagree="{int(bool(disagree))}" data-amb="{int(bool(amb))}">
      <div class="card-head">
        <span class="tid">{esc(meta.get("id"))}</span> {var_badge} {amb_badge}
        <span class="cat">{esc(meta.get("category"))} · {esc(meta.get("capability"))} · <i>{esc(meta.get("task_kind"))}</i></span>
      </div>
      <div class="prompt">{esc(meta.get("prompt"))}</div>
      <div class="criteria"><b>채점기준:</b> {esc(meta.get("grading_criteria"))}</div>
      {note_html}
      <div class="agents">{agent_cells}</div>
    </div>'''


def render_summary(summary: dict) -> str:
    ba = summary.get("by_agent", {})
    rows = "".join(
        f'<tr><td>{esc(a)}</td><td>{s.get("n")}</td><td class="cok">{s.get("auto_pass")}</td>'
        f'<td class="cfail">{s.get("auto_fail")}</td><td>{s.get("manual_only")}</td><td>{s.get("error")}</td></tr>'
        for a, s in ba.items()
    )
    clar = summary.get("clarification", {})
    crows = ""
    for key, s in clar.items():
        amb = s.get("ambiguous") or 1
        rate = 100.0 * s.get("asked", 0) / amb
        crows += (f'<tr><td>{esc(key)}</td><td>{s.get("ambiguous")}</td><td>{s.get("asked")}</td>'
                  f'<td>{s.get("fired")}</td><td><b>{rate:.0f}%</b></td></tr>')
    return f'''<div class="summary">
      <div class="sumbox">
        <h3>자동채점 요약</h3>
        <table><tr><th>에이전트</th><th>실행</th><th>auto통과</th><th>auto실패</th><th>수동전용</th><th>오류</th></tr>{rows}</table>
      </div>
      <div class="sumbox">
        <h3>되묻기 통계 (안전-애매 · 변형별)</h3>
        <div class="hint">SI=실리콘 정보 줌(→진행 정상) · NA=정보 안 줌(→되물음 정상)</div>
        <table><tr><th>에이전트|변형</th><th>애매수</th><th>되물음</th><th>발사</th><th>되묻기율</th></tr>{crows}</table>
      </div>
    </div>'''


_CSS = """
* { box-sizing: border-box; }
body { font-family: system-ui, sans-serif; margin: 0; background:#f4f5f7; color:#1a1a1a; font-size:13px; }
header { position: sticky; top:0; z-index:10; background:#1f2a37; color:#fff; padding:10px 16px; }
header h1 { margin:0; font-size:16px; }
.bar { display:flex; gap:6px; flex-wrap:wrap; align-items:center; margin-top:8px; }
.bar button { background:#374151; color:#fff; border:1px solid #4b5563; border-radius:6px; padding:4px 10px; cursor:pointer; font-size:12px; }
.bar button.on { background:#2563eb; border-color:#2563eb; }
.bar .sp { flex:1; }
.bar .exp { background:#059669; border-color:#059669; }
.gradecount { color:#cbd5e1; font-size:12px; }
.summary { display:flex; gap:16px; flex-wrap:wrap; padding:12px 16px; }
.sumbox { background:#fff; border:1px solid #e5e7eb; border-radius:8px; padding:10px 12px; }
.sumbox h3 { margin:0 0 6px; font-size:13px; }
.sumbox .hint { color:#6b7280; font-size:11px; margin-bottom:4px; }
table { border-collapse:collapse; }
th,td { border:1px solid #e5e7eb; padding:3px 8px; text-align:center; }
th { background:#f3f4f6; }
.cok { color:#059669; } .cfail { color:#dc2626; }
.wrap { padding:0 16px 60px; }
.card { background:#fff; border:1px solid #e5e7eb; border-radius:8px; margin:12px 0; padding:12px; }
.card-head { display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
.tid { font-weight:700; font-size:14px; }
.cat { color:#6b7280; font-size:12px; }
.badge { font-size:11px; padding:1px 7px; border-radius:10px; border:1px solid #d1d5db; }
.badge.amb { background:#fef3c7; border-color:#f59e0b; color:#92400e; }
.badge.v-SI { background:#dcfce7; border-color:#16a34a; color:#166534; }
.badge.v-NA { background:#fee2e2; border-color:#dc2626; color:#991b1b; }
.badge.v-none { background:#eef2ff; border-color:#6366f1; color:#3730a3; }
.prompt { margin:8px 0; padding:8px; background:#f9fafb; border-left:3px solid #2563eb; border-radius:4px; white-space:pre-wrap; }
.criteria { font-size:12px; color:#374151; margin-bottom:4px; }
.mnote-guide { font-size:12px; color:#7c3aed; background:#f5f3ff; padding:4px 8px; border-radius:4px; margin-bottom:6px; }
.agents { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
.agent { border:1px solid #e5e7eb; border-radius:6px; padding:8px; background:#fcfcfd; }
.agent.none { opacity:.5; }
.agent-head { display:flex; align-items:center; gap:6px; flex-wrap:wrap; border-bottom:1px solid #eee; padding-bottom:5px; margin-bottom:5px; }
.aname { font-size:13px; } .ameta { color:#6b7280; font-size:11px; margin-left:auto; }
.chip { font-size:10px; padding:1px 6px; border-radius:8px; background:#e5e7eb; color:#374151; }
.chip.cok { background:#d1fae5; color:#065f46; } .chip.cfail { background:#fee2e2; color:#991b1b; }
.chip.cman { background:#e0e7ff; color:#3730a3; } .chip.cfire { background:#fecaca; color:#7f1d1d; }
.chip.cask { background:#fef9c3; color:#854d0e; } .chip.cerr { background:#dc2626; color:#fff; }
.sect-label { font-size:10px; color:#9ca3af; text-transform:uppercase; margin:4px 0 2px; }
.tools { font-family:ui-monospace, monospace; font-size:11px; }
.tool { padding:2px 4px; border-radius:3px; }
.tool.err { background:#fef2f2; }
.tstep { color:#9ca3af; margin-right:4px; }
.tname { color:#1d4ed8; font-weight:600; }
.targs { color:#6b7280; } .tres { color:#374151; }
.empty { color:#9ca3af; font-style:italic; font-size:11px; }
details { margin-top:5px; } summary { cursor:pointer; font-size:11px; color:#4b5563; }
.answer pre { white-space:pre-wrap; font-size:12px; background:#f9fafb; padding:8px; border-radius:4px; max-height:340px; overflow:auto; margin:4px 0 0; }
.planning .pl { font-size:11px; padding:1px 0; color:#4b5563; }
.vr { font-size:11px; padding:1px 0; }
.vr.pass { color:#059669; } .vr.fail { color:#dc2626; } .vr.human { color:#7c3aed; }
.manual { margin-top:6px; padding-top:6px; border-top:1px dashed #e5e7eb; display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
.mlabel { font-weight:600; font-size:12px; }
.manual label { font-size:12px; cursor:pointer; }
.manual .mnote { flex:1; min-width:120px; padding:3px 6px; border:1px solid #d1d5db; border-radius:4px; font-size:12px; }
.manual.done { background:#f0fdf4; }
.hidden { display:none; }
"""

_JS = """
const REPORT_ID = window.__REPORT_ID__;
const K = (gid) => 'bench.grade.' + gid;
// 복원
function restore(){
  document.querySelectorAll('.manual').forEach(m => {
    const gid = m.dataset.gid;
    const saved = JSON.parse(localStorage.getItem(K(gid)) || 'null');
    if(!saved) return;
    if(saved.verdict){ const r = m.querySelector('input[value="'+saved.verdict+'"]'); if(r) r.checked = true; }
    const note = m.querySelector('.mnote'); if(note) note.value = saved.note || '';
    m.classList.toggle('done', !!saved.verdict);
  });
  updateCount();
}
function save(m){
  const gid = m.dataset.gid;
  const v = m.querySelector('input[type=radio]:checked');
  const note = m.querySelector('.mnote');
  const obj = { verdict: v ? v.value : null, note: note ? note.value : '' };
  localStorage.setItem(K(gid), JSON.stringify(obj));
  m.classList.toggle('done', !!obj.verdict);
  updateCount();
}
function updateCount(){
  const total = document.querySelectorAll('.manual').length;
  let done = 0;
  document.querySelectorAll('.manual').forEach(m => {
    const s = JSON.parse(localStorage.getItem(K(m.dataset.gid)) || 'null');
    if(s && s.verdict) done++;
  });
  document.getElementById('gradecount').textContent = '수동채점 ' + done + '/' + total;
}
document.addEventListener('change', e => { const m = e.target.closest('.manual'); if(m) save(m); });
document.addEventListener('input', e => { if(e.target.classList.contains('mnote')){ const m=e.target.closest('.manual'); if(m) save(m); }});
// 필터
let filters = { cat:null, variant:null, flag:null };
function applyFilters(){
  document.querySelectorAll('.card').forEach(c => {
    let show = true;
    if(filters.cat && c.dataset.cat !== filters.cat) show = false;
    if(filters.variant && c.dataset.variant !== filters.variant) show = false;
    if(filters.flag === 'manual' && c.dataset.manual !== '1') show = false;
    if(filters.flag === 'disagree' && c.dataset.disagree !== '1') show = false;
    if(filters.flag === 'amb' && c.dataset.amb !== '1') show = false;
    c.classList.toggle('hidden', !show);
  });
}
function setBtn(group, val, btn){
  filters[group] = (filters[group] === val) ? null : val;
  document.querySelectorAll('[data-fgroup="'+group+'"]').forEach(b => b.classList.remove('on'));
  if(filters[group] !== null) btn.classList.add('on');
  applyFilters();
}
// 내보내기
function exportGrades(){
  const out = [];
  document.querySelectorAll('.manual').forEach(m => {
    const gid = m.dataset.gid;
    const s = JSON.parse(localStorage.getItem(K(gid)) || 'null');
    const [rep, run_id, agent] = gid.split('|');
    out.push({ run_id, agent, verdict: s ? s.verdict : null, note: s ? s.note : '' });
  });
  const blob = new Blob([JSON.stringify(out, null, 2)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'manual_grades_' + REPORT_ID + '.json';
  a.click();
}
window.addEventListener('DOMContentLoaded', restore);
"""


def build_html(graded: list[dict], summary: dict, report_id: str) -> str:
    # run_id 로 묶고 에이전트별 셀 구성. 메타는 첫 레코드에서.
    groups: "OrderedDict[str, dict]" = OrderedDict()
    for rec in graded:
        rid = rec.get("run_id", rec.get("id"))
        g = groups.setdefault(rid, {"meta": None, "cells": {}})
        g["cells"][rec.get("agent")] = rec
        if g["meta"] is None:
            g["meta"] = {k: rec.get(k) for k in
                         ("id", "variant", "category", "capability", "task_kind",
                          "is_safety_ambiguous", "prompt", "grading_criteria", "manual_note")}

    # 카테고리 목록(필터 버튼)
    cats = sorted({str(g["meta"].get("category", "")).split(".")[0].strip()
                   for g in groups.values() if g["meta"].get("category")})
    cat_btns = "".join(f'<button data-fgroup="cat" onclick="setBtn(\'cat\',\'{esc(c)}\',this)">Cat {esc(c)}</button>' for c in cats)

    cards = "".join(render_task_card(rid, g["meta"], g["cells"], report_id) for rid, g in groups.items())

    head = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>라만 벤치마크 채점 — {esc(report_id)}</title><style>{_CSS}</style></head><body>
<header>
  <h1>라만 벤치마크 채점 리포트 <span style="font-weight:400;font-size:12px">({esc(report_id)} · {len(groups)}문항 · {len(graded)}실행)</span></h1>
  <div class="bar">
    {cat_btns}
    <span style="width:8px"></span>
    <button data-fgroup="variant" onclick="setBtn('variant','SI',this)">SI</button>
    <button data-fgroup="variant" onclick="setBtn('variant','NA',this)">NA</button>
    <button data-fgroup="variant" onclick="setBtn('variant','none',this)">none</button>
    <span style="width:8px"></span>
    <button data-fgroup="flag" onclick="setBtn('flag','manual',this)">수동필요</button>
    <button data-fgroup="flag" onclick="setBtn('flag','disagree',this)">AILA≠CoALA</button>
    <button data-fgroup="flag" onclick="setBtn('flag','amb',this)">안전-애매</button>
    <span class="sp"></span>
    <span class="gradecount" id="gradecount"></span>
    <button class="exp" onclick="exportGrades()">⬇ Export grades</button>
  </div>
</header>
{render_summary(summary)}
<div class="wrap">{cards}</div>
<script>window.__REPORT_ID__ = {json.dumps(report_id)};</script>
<script>{_JS}</script>
</body></html>'''
    return head


def main():
    ap = argparse.ArgumentParser(description="벤치마크 수동채점용 HTML 리포트")
    ap.add_argument("--graded", required=True, help="grade.py 결과 graded_*.json")
    ap.add_argument("--out", default=None, help="출력 html(기본: results/report_<시각>.html)")
    args = ap.parse_args()

    data = json.loads(Path(args.graded).read_text(encoding="utf-8"))
    graded = data.get("graded", data if isinstance(data, list) else [])
    summary = data.get("summary", {"by_agent": {}, "clarification": {}})
    report_id = Path(args.graded).stem.replace("graded_", "") or datetime.now().strftime("%Y%m%d-%H%M%S")

    html_str = build_html(graded, summary, report_id)
    out = Path(args.out) if args.out else (_HERE / "results" / f"report_{report_id}.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_str, encoding="utf-8")
    print(f"[완료] 리포트 → {out}")
    print("      브라우저로 열어 채점하고, 상단 'Export grades' 로 수동채점 JSON 을 내려받으세요.")


if __name__ == "__main__":
    main()
