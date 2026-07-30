# -*- coding: utf-8 -*-
"""파일처리 48문항 채점 실행 + 리포트 생성.

    python -m filegrade.grade_files --agent AILA

[무엇을 하는가]
raw_runs.jsonl 에서 해당 에이전트의 파일처리 48문항을 뽑아, 문항마다

    부류(A/B) · 정답 정의 · 재계산한 GT · 에이전트 보고값 · 판정 · 왜 그런지

를 만들어 results/file_grading_<AGENT>_<ts>.json 과 .html 로 쓴다.

[점수를 절차와 결과로 나눈다]
지금까지는 "순서를 맞게 했나(절차)"와 "숫자가 맞나(결과)"가 한 pass 에 묶여 있었다.
그러면 'baseline 방법만 다르고 절차는 옳았다' 같은 정보가 사라진다. 그래서

    process  — 지정 파라미터·순서·툴 호출을 지켰는가 (코드/트레이스에서)
    outcome  — 부류 A 는 GT 수치 일치, 부류 B 는 모양새 일치

를 따로 집계하고, 종합 판정은 둘 다 통과해야 pass 로 둔다.

[채점 설정은 tasks.json 에서 다시 당긴다]
raw_runs.jsonl 의 레코드에는 실행 시점의 verifiers 스냅샷이 박혀 있어, 그 뒤에 고친
검증기가 반영되지 않는다. review.py 의 _refresh_config 를 그대로 써서 갱신한다.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from . import BENCH_DIR, RESULTS_DIR
from . import task_class as TC

import diagnostics as D                                    # noqa: E402
from grade import grade_one                                # noqa: E402

# review 는 여기서 최상위 임포트하면 안 된다 — review.py 가 split_verdicts 를 쓰려고
# 이 모듈을 임포트하므로 순환이 된다(review 미완성 상태에서 이름을 못 찾는다).
# 레코드 로딩에만 필요하니 함수 안에서 가져온다.


def _load(agent: str) -> dict[str, dict]:
    import review as RV
    src = BENCH_DIR / "results" / "raw_runs.jsonl"
    rows = RV._load_rows(src)
    RV._refresh_config(rows)                 # 검증기·채점기준을 tasks.json 현재값으로
    rows = RV._dedupe(rows, {agent})         # noqa: F821
    return {(r.get("run_id") or r["id"]): r for r in rows}


def _verdict_of(rows: list[dict], kinds: tuple[str, ...]) -> tuple[str | None, list[dict]]:
    """진단 행 중 해당 종류만 골라 종합 판정."""
    sel = [r for r in rows if r["name"] in kinds or any(k in r["name"] for k in kinds)]
    if not sel:
        return None, []
    vs = [r["verdict"] for r in sel]
    if D.FAIL in vs:
        return "fail", sel
    if D.PASS in vs:
        return "pass", sel
    return None, sel


# 진단 행 이름 중 '절차'에 해당하는 것들. 나머지 pass/fail 행은 결과로 본다.
_PROCESS_NAMES = ("지정 차수", "지정 파라미터", "지정 전처리", "4단계 순서",
                  "① 파일을 찾아봤는가", "③ 파일 저장", "④ 장비를", "⑤ 오류로")


def split_verdicts(drows: list[dict]) -> dict:
    """진단 행들을 절차/결과로 나눠 판정하고, 왜 그런지 한 줄을 만든다.

    review.py 의 채점 콘솔과 이 CLI 가 <b>같은 판정</b>을 내야 하므로 여기 한 곳에만 둔다.
    """
    proc = [r for r in drows if any(n in r["name"] for n in _PROCESS_NAMES)]
    outc = [r for r in drows if r not in proc and r["verdict"] in (D.PASS, D.FAIL)]

    def agg(rows):
        vs = [r["verdict"] for r in rows]
        if not vs:
            return None
        return "fail" if D.FAIL in vs else ("pass" if D.PASS in vs else None)

    p_v, o_v = agg(proc), agg(outc)
    overall = ("fail" if "fail" in (p_v, o_v) else
               "pass" if "pass" in (p_v, o_v) else None)

    fails = [r for r in drows if r["verdict"] == D.FAIL]
    passes = [r for r in drows if r["verdict"] == D.PASS]
    if fails:
        why = " / ".join(f"{r['name']}: {r['note'] or r['measured']}" for r in fails[:3])
    elif passes:
        why = " / ".join(f"{r['name']}: {r['measured']}" for r in passes[:3])
    else:
        why = "자동 판정 가능한 항목이 없다 — 사람이 볼 것"

    return {"process_verdict": p_v, "outcome_verdict": o_v, "verdict": overall,
            "why": _strip(why), "decisive": proc + outc}


def grade_task(tid: str, rec: dict, task: dict, tf: dict) -> dict:
    entry = TC.get(tid) or {}
    drows = D.run(rec, task, tf)
    sv = split_verdicts(drows)
    p_v, o_v, overall, why = (sv["process_verdict"], sv["outcome_verdict"],
                              sv["verdict"], sv["why"])

    g = grade_one(rec)
    return {
        "id": tid,
        "class": entry.get("class"),
        "capability": task.get("capability"),
        "category": task.get("category"),
        "gt_rule": entry.get("gt_rule"),
        "free_params": entry.get("free_params") or [],
        "why_class": entry.get("why"),
        "grading_criteria": task.get("grading_criteria"),
        "process_verdict": p_v,
        "outcome_verdict": o_v,
        "verdict": overall,
        "why": why,
        "diagnostics": drows,
        "verifier_results": g.get("verifier_results"),
        "auto_verdict_legacy": g.get("auto_verdict"),
        "answer": rec.get("answer") or rec.get("final_report") or "",
        "tool_call_order": rec.get("tool_call_order") or [],
        "elapsed_sec": rec.get("elapsed_sec"),
    }


def _strip(s: str) -> str:
    import re
    return re.sub(r"<[^>]+>", "", s or "").replace("&nbsp;", " ").strip()


def run(agent: str = "AILA") -> dict:
    recs = _load(agent)
    tasks = {t["id"]: t for t in json.loads((BENCH_DIR / "tasks.json").read_text(encoding="utf-8"))}
    tfs = json.loads((BENCH_DIR / "task_files.json").read_text(encoding="utf-8"))

    out, missing = [], []
    for tid in TC.FILE_TASKS:
        rec = recs.get(tid)
        if rec is None:
            missing.append(tid)
            continue
        out.append(grade_task(tid, rec, tasks[tid], tfs.get(tid, {})))

    def count(key, val):
        return sum(1 for r in out if r[key] == val)

    return {
        "agent": agent,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_tasks": len(out),
        "missing_runs": missing,
        "summary": {
            "verdict": {v: count("verdict", v) for v in ("pass", "fail", None)},
            "process": {v: count("process_verdict", v) for v in ("pass", "fail", None)},
            "outcome": {v: count("outcome_verdict", v) for v in ("pass", "fail", None)},
            "by_class": {
                c: {v: sum(1 for r in out if r["class"] == c and r["verdict"] == v)
                    for v in ("pass", "fail", None)}
                for c in ("A", "B")},
        },
        "results": out,
    }


# ── 리포트 ───────────────────────────────────────────────────────────────────

_CSS = """
:root{--bg:#fff;--fg:#1a1a1a;--mut:#666;--line:#e3e3e3;--card:#fafafa;
      --pass:#0a7c3f;--fail:#c0392b;--info:#7a6a00;--a:#0b5cad;--b:#8a4b00}
@media(prefers-color-scheme:dark){:root{--bg:#15171a;--fg:#e8e8e8;--mut:#9aa0a6;
      --line:#2c3035;--card:#1c1f23;--pass:#4ec97f;--fail:#ff7b6b;--info:#e0c65a;
      --a:#7fb6f0;--b:#e0a56a}}
*{box-sizing:border-box}
body{margin:0;padding:24px;background:var(--bg);color:var(--fg);
     font:14px/1.6 -apple-system,"Segoe UI",Roboto,"Malgun Gothic",sans-serif}
h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:0}
.sub{color:var(--mut);font-size:13px;margin-bottom:18px}
.wrap{max-width:1100px;margin:0 auto}
.sum{display:flex;flex-wrap:wrap;gap:10px;margin:14px 0 24px}
.kpi{border:1px solid var(--line);border-radius:8px;padding:10px 14px;background:var(--card);min-width:120px}
.kpi b{display:block;font-size:22px;line-height:1.2}
.kpi span{color:var(--mut);font-size:12px}
.card{border:1px solid var(--line);border-radius:10px;margin:0 0 14px;background:var(--card);overflow:hidden}
.hd{display:flex;align-items:center;gap:10px;padding:11px 14px;border-bottom:1px solid var(--line);
    cursor:pointer;user-select:none}
.hd:hover{background:rgba(128,128,128,.07)}
.tag{font-size:11px;font-weight:700;padding:2px 7px;border-radius:4px;border:1px solid currentColor}
.cA{color:var(--a)}.cB{color:var(--b)}
.v{font-weight:700}.vpass{color:var(--pass)}.vfail{color:var(--fail)}.vnone{color:var(--mut)}
.why{color:var(--mut);font-size:12.5px;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.bd{padding:6px 14px 14px;display:none}.card.open .bd{display:block}
.gt{background:rgba(128,128,128,.08);border-left:3px solid var(--line);padding:9px 12px;
    border-radius:0 6px 6px 0;margin:10px 0;font-size:13px}
table{width:100%;border-collapse:collapse;margin:10px 0;font-size:12.5px}
th,td{border-bottom:1px solid var(--line);padding:7px 8px;text-align:left;vertical-align:top}
th{color:var(--mut);font-weight:600;font-size:11.5px;text-transform:uppercase;letter-spacing:.03em}
td.n{white-space:nowrap;font-weight:600;width:1%}
.p{color:var(--pass);font-weight:700}.f{color:var(--fail);font-weight:700}.i{color:var(--info)}
code{background:rgba(128,128,128,.14);padding:1px 4px;border-radius:3px;font-size:12px}
details{margin:10px 0}summary{cursor:pointer;color:var(--mut);font-size:12.5px}
pre{background:rgba(128,128,128,.09);padding:10px;border-radius:6px;overflow-x:auto;
    white-space:pre-wrap;font-size:12px;max-height:340px}
.scroll{overflow-x:auto}
.bar{display:flex;height:8px;border-radius:4px;overflow:hidden;margin:2px 0 0}
.bar i{display:block}.bar .bp{background:var(--pass)}.bar .bf{background:var(--fail)}
.bar .bn{background:var(--mut);opacity:.4}
"""

_JS = """
document.querySelectorAll('.hd').forEach(h=>h.onclick=()=>h.parentElement.classList.toggle('open'));
document.getElementById('all').onclick=()=>{
  const any=[...document.querySelectorAll('.card')].some(c=>!c.classList.contains('open'));
  document.querySelectorAll('.card').forEach(c=>c.classList.toggle('open',any));};
document.querySelectorAll('[data-filter]').forEach(b=>b.onclick=()=>{
  const f=b.dataset.filter;
  document.querySelectorAll('.card').forEach(c=>{
    c.style.display=(f==='all'||c.dataset.v===f||c.dataset.c===f)?'':'none';});});
"""


def _esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            if s is not None else "")


def _vcls(v):
    return {"pass": "vpass", "fail": "vfail"}.get(v, "vnone")


def _vtxt(v):
    return {"pass": "정답", "fail": "오답"}.get(v, "판정보류")


def build_html(data: dict) -> str:
    s = data["summary"]
    n = data["n_tasks"]
    vp, vf = s["verdict"]["pass"], s["verdict"]["fail"]
    vn = s["verdict"][None]

    def kpi(label, val, extra=""):
        return f'<div class="kpi"><b>{val}</b><span>{label}</span>{extra}</div>'

    bar = (f'<div class="bar"><i class="bp" style="flex:{vp}"></i>'
           f'<i class="bf" style="flex:{vf}"></i><i class="bn" style="flex:{vn}"></i></div>')

    cards = []
    for r in data["results"]:
        cls = r["class"]
        head = (
            f'<div class="hd">'
            f'<span class="tag c{cls}">부류 {cls}</span>'
            f'<h2>{r["id"]}</h2>'
            f'<span class="v {_vcls(r["verdict"])}">{_vtxt(r["verdict"])}</span>'
            f'<span class="why">절차 {_vtxt(r["process_verdict"])} · '
            f'결과 {_vtxt(r["outcome_verdict"])} — {_esc(r["why"])[:150]}</span>'
            f'</div>')

        gt = (f'<div class="gt"><b>정답 정의</b> — {_esc(r["gt_rule"])}'
              + (f'<br><b>자유 파라미터</b>: {_esc(", ".join(r["free_params"]))}'
                 if r["free_params"] else "")
              + f'<br><b>이 부류인 이유</b>: {_esc(r["why_class"])}</div>')

        crit = (f'<details><summary>문항의 원래 채점기준</summary>'
                f'<pre>{_esc(r["grading_criteria"])}</pre></details>')

        trs = []
        for d in r["diagnostics"]:
            v = d["verdict"]
            mark = {"pass": '<span class="p">통과</span>', "fail": '<span class="f">실패</span>'}\
                .get(v, '<span class="i">참고</span>')
            rep = f'<br><span style="color:var(--mut)">보고값: {d["reported"]}</span>' \
                if d["reported"] not in ("", "-", None) else ""
            note = f'<br><span style="color:var(--mut)">{d["note"]}</span>' if d["note"] else ""
            trs.append(f'<tr><td class="n">{_esc(d["name"])}</td>'
                       f'<td>{d["criterion"]}</td>'
                       f'<td>{d["measured"]}{rep}{note}</td>'
                       f'<td class="n">{mark}</td></tr>')
        table = ('<div class="scroll"><table><tr><th>항목</th><th>기준</th>'
                 '<th>재계산 / 보고</th><th>판정</th></tr>' + "".join(trs) + "</table></div>")

        ans = (f'<details><summary>에이전트 답변 · 툴 호출 '
               f'({len(r["tool_call_order"])}회)</summary>'
               f'<pre>{_esc(" → ".join(r["tool_call_order"]))}\n\n{_esc(r["answer"])}</pre>'
               f'</details>')

        cards.append(f'<div class="card" data-v="{r["verdict"]}" data-c="{cls}">'
                     f'{head}<div class="bd">{gt}{table}{crit}{ans}</div></div>')

    btns = ('<div style="margin:0 0 14px;display:flex;gap:8px;flex-wrap:wrap">'
            '<button id="all">전체 펼치기/접기</button>'
            '<button data-filter="all">전체</button>'
            '<button data-filter="fail">오답만</button>'
            '<button data-filter="pass">정답만</button>'
            '<button data-filter="A">부류 A</button>'
            '<button data-filter="B">부류 B</button></div>')

    miss = (f'<p style="color:var(--fail)">실행 기록이 없는 문항: '
            f'{", ".join(data["missing_runs"])}</p>' if data["missing_runs"] else "")

    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>파일처리 채점 — {data['agent']} — {data['generated_at'][:16]}</title>
<style>{_CSS}</style></head><body><div class="wrap">
<h1>파일처리 문항 채점 — {data['agent']}</h1>
<div class="sub">{data['generated_at']} · 48문항 중 {n}문항 채점 ·
정답 기준은 레퍼런스 구현이 아니라 <b>과제 명세</b>가 결정한다
(부류 A = 명세가 답을 유일하게 결정 → GT 엄격 비교 ·
 부류 B = 방법군을 부름 → 모양새 일치)</div>
{miss}
<div class="sum">
{kpi('정답', vp, bar)}{kpi('오답', vf)}{kpi('판정보류', vn)}
{kpi('절차 통과', s['process']['pass'])}{kpi('결과 통과', s['outcome']['pass'])}
{kpi('부류 A 정답', f"{s['by_class']['A']['pass']}/{sum(s['by_class']['A'].values())}")}
{kpi('부류 B 정답', f"{s['by_class']['B']['pass']}/{sum(s['by_class']['B'].values())}")}
</div>
{btns}
{''.join(cards)}
</div><script>{_JS}</script></body></html>"""


def main():
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="파일처리 48문항 채점")
    ap.add_argument("--agent", default="AILA")
    ap.add_argument("--out", default=None, help="출력 경로 접두(기본 results/file_grading_…)")
    a = ap.parse_args()

    data = run(a.agent)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = Path(a.out) if a.out else RESULTS_DIR / f"file_grading_{a.agent}_{ts}"
    jp, hp = base.with_suffix(".json"), base.with_suffix(".html")
    jp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    hp.write_text(build_html(data), encoding="utf-8")

    s = data["summary"]
    print(f"{a.agent} 파일처리 {data['n_tasks']}문항")
    print(f"  종합  정답 {s['verdict']['pass']} · 오답 {s['verdict']['fail']} · "
          f"판정보류 {s['verdict'][None]}")
    print(f"  절차  통과 {s['process']['pass']} · 실패 {s['process']['fail']} · "
          f"해당없음 {s['process'][None]}")
    print(f"  결과  통과 {s['outcome']['pass']} · 실패 {s['outcome']['fail']} · "
          f"판정불가 {s['outcome'][None]}")
    for c in ("A", "B"):
        b = s["by_class"][c]
        print(f"  부류 {c}: 정답 {b['pass']} · 오답 {b['fail']} · 보류 {b[None]}")
    if data["missing_runs"]:
        print(f"  실행 기록 없음: {data['missing_runs']}")
    print(f"\n  {jp}\n  {hp}")


if __name__ == "__main__":
    main()
