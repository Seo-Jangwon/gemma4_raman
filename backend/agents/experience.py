"""
ExperienceStore v2 — 하이브리드 에피소드-절차 기억 (H-EPM 차용).

[출처와 차용 범위]
arXiv:2512.07287 "Experience-Evolving Multi-Turn Tool-Use Agent with Hybrid
Episodic-Procedural Memory"의 기억 구조를 이 프로젝트에 맞게 번안했다:

  차용 O — 절차 기억: 성공 실험의 step 전이 그래프.
           edge 가중치 w' = N + c·Σ(1/nₖ)  (논문 4.2절 수식 그대로:
           N=해당 전이가 등장한 성공 실험 수, nₖ=그 실험의 총 step 수,
           c=효율 항 비중. 논문 ablation에서 c=1이 최적이라 1로 고정)
  차용 O — 에피소드 기억: 각 전이/실험에 "어떤 상황이었나"(컨텍스트)를 주석으로
           붙이고, 새 실험의 컨텍스트와 유사도를 비교해 회수한다.
  차용 O — 적응형 회수: 유사한 에피소드가 있으면 에피소드 우선, 없으면
           절차 가중치로 fallback (논문 Fig.2의 이중 경로).
  차용 O — 제안은 advisory: 회수 결과는 Planner LLM 프롬프트에 "과거 경험"으로
           주입될 뿐 강제하지 않는다 (논문의 "Suggested next tools" 방식).
  차용 X — RL(GRPO) 통합: 학습 파이프라인이 없으므로 제외.
  차용 X — 임베딩 유사도(all-MiniLM): 의존성 추가 대비 이득이 작다. 우리
           컨텍스트는 이미 구조화된 필드(sample_type/substrate/region)라서
           필드 기반 유사도로 충분하다. 데이터가 수백 건을 넘으면 업그레이드.
  차용 X — LLM 요약 툴: 논문은 비구조적 대화를 압축해야 하지만, 우리 상태는
           ExperimentState로 이미 구조화되어 있어 LLM 없이 결정적으로 컨텍스트를
           뽑을 수 있다 (호출 비용 0, 재현성 100%).
  확장 + — 실패 에피소드도 기록: 논문은 성공 궤적만 쓰지만, "이 기판에서 이
           방식은 실패했다"는 부정적 지식이 재계획 시 같은 실수 반복을 막는다.
           단, 그래프(절차 기억)에는 성공만 반영 — 나쁜 절차가 가중치를 얻으면
           안 되기 때문.

[왜 이렇게 세분화하나]
기판[종류/위치]과 시료마다 배경 신호·최적 파워/노출이 전부 다르다.
v1처럼 sample_type 정확 일치만 보면 "같은 그래핀이라도 유리 기판 vs 실리콘
기판"을 구분하지 못한다. 컨텍스트를 (시료, 기판, 타겟 외형, 스테이지 영역)
4개 필드로 잡고 부분 일치 점수를 매기면, 완전히 같은 조건이 없어도
"가장 비슷한 과거"를 회수할 수 있다 — 논문이 말하는 partially overlapping
experience의 재사용이다.

[저장 스키마] (experience_store.json — 프로그램 재시작에도 유지)
{
  "version": 2,
  "next_id": 7,
  "episodes": [                       # 에피소드 기억 (성공+실패 모두)
    {"id": 1, "timestamp": "...", "session_id": "...", "success": true,
     "context": {"sample_type": "graphene", "substrate": "sio2 wafer",
                 "target_description": "어두운 육각 플레이크", "region": "30_20"},
     "outcome": {"power_pct": 12.5, "exposure_s": 0.8, "max_intensity": 34000,
                 "background_max_intensity": 4200, "tuned": true,
                 "roi_mode": "visual_search", "c3_issues": [], "notes": "..."},
     "plan_signature": ["roi_detector:visual_search", "hw_manager:acquire_target", ...],
     "steps": 5}
  ],
  "graph": {                          # 절차 기억 (성공 실험만 반영)
    "edges": {"roi_detector:visual_search->hw_manager:acquire_target":
              {"n": 4, "eff": 0.8, "episode_ids": [1, 3, 5, 6]}}
  }
}
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

_STORE_PATH = Path(__file__).parent / "experience_store.json"

_C_EFF = 1.0          # 효율 항 비중 c — 논문 Table 7에서 c=1이 최적
_MAX_EPISODES = 200   # 에피소드 상한 (초과 시 오래된 것부터 폐기)
_EDGE_EPISODE_CAP = 10  # edge당 에피소드 참조 상한 — 최근 것만 유지
_SIM_THRESHOLD = 3.0  # 에피소드 회수 최소 유사도.
                      # sample_type 일치(+3)가 사실상 필수 조건이 되도록 설정 —
                      # 시료가 다르면 파워/노출 재사용은 위험하다.

_REGION_GRID_MM = 5.0  # 스테이지 위치를 5mm 격자로 양자화해 "기판 위치별" 구분.
                       # 너무 잘게 나누면 같은 기판인데도 매칭이 안 되고,
                       # 너무 크면 위치 구분이 무의미해진다 — 시편 홀더 크기 감안.


# ══════════════════════════════════════════════════════════════════════════════
# 저장소 I/O (v1과 동일한 원자적 쓰기 — 프로세스 중단에도 파일이 깨지지 않음)
# ══════════════════════════════════════════════════════════════════════════════

def _empty_store() -> dict:
    return {"version": 2, "next_id": 1, "episodes": [], "graph": {"edges": {}}}


def _load() -> dict:
    """로드 실패·구버전 스키마는 빈 저장소로 강등 — 기억이 없어도 실험은 진행된다."""
    try:
        if _STORE_PATH.exists():
            with open(_STORE_PATH, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and data.get("version") == 2:
                return data
    except Exception:
        pass
    return _empty_store()


def _save(data: dict) -> None:
    """원자적 저장 (temp → os.replace). 실패는 조용히 무시 — 기록은 best-effort."""
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(dir=str(_STORE_PATH.parent), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, _STORE_PATH)
    except Exception:
        try:
            if tmp:
                os.unlink(tmp)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# 컨텍스트 추출 & 유사도 (에피소드 기억의 핵심)
# ══════════════════════════════════════════════════════════════════════════════

def _region(pos: dict | None) -> str:
    """스테이지 좌표 → 격자 영역 키. 기판 위 '어느 구역'인지를 나타낸다."""
    if not pos or pos.get("x") is None:
        return ""
    gx = round(float(pos.get("x", 0)) / _REGION_GRID_MM) * _REGION_GRID_MM
    gy = round(float(pos.get("y", 0)) / _REGION_GRID_MM) * _REGION_GRID_MM
    return f"{gx:g}_{gy:g}"


def build_context(state: dict) -> dict:
    """
    ExperimentState → 에피소드 컨텍스트. LLM 없이 결정적으로 추출한다
    (논문의 state-summarization tool을 대체 — 우리 상태는 이미 구조화돼 있으므로).
    """
    intent = state.get("intent") or {}
    pos = state.get("next_roi") or state.get("stage_position") or {}
    return {
        "sample_type": (intent.get("sample_type") or "unknown").lower().strip(),
        "substrate": (intent.get("substrate") or "").lower().strip(),
        "target_description": intent.get("target_description") or "",
        "region": _region(pos),
    }


def _tokens(s: str) -> set:
    return {t for t in re.split(r"[\s_\-,;/()]+", (s or "").lower()) if t}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def similarity(c1: dict, c2: dict) -> float:
    """
    구조화 필드 기반 유사도 (임베딩 대체 — 모듈 docstring의 차용 X 항목 참고).
    가중치 근거:
      sample_type 일치 +3  — 시료가 다르면 측정 조건 재사용이 위험 → 지배적 비중
      substrate   최대 +2  — 기판이 배경 신호를 결정 → 두 번째 비중
      target_desc 최대 +2  — 같은 시료라도 타겟 형태가 다르면 신호 크기가 다름
      region 일치     +1  — 같은 기판이라도 구역별 상태(오염·코팅 불균일) 반영
    만점 8. _SIM_THRESHOLD=3 → sample_type이 맞아야 사실상 회수된다.
    """
    score = 0.0
    if c1.get("sample_type") and c1.get("sample_type") not in ("", "unknown") \
            and c1.get("sample_type") == c2.get("sample_type"):
        score += 3.0
    score += 2.0 * _jaccard(_tokens(c1.get("substrate", "")), _tokens(c2.get("substrate", "")))
    score += 2.0 * _jaccard(_tokens(c1.get("target_description", "")),
                            _tokens(c2.get("target_description", "")))
    if c1.get("region") and c1.get("region") == c2.get("region"):
        score += 1.0
    return score


def _rank_episodes(context: dict, episodes: list, success_only: bool) -> list[tuple[float, dict]]:
    """컨텍스트 유사도로 에피소드 정렬 (임계값 이상만)."""
    scored = []
    for ep in episodes:
        if success_only and not ep.get("success"):
            continue
        s = similarity(context, ep.get("context", {}))
        if s >= _SIM_THRESHOLD:
            scored.append((s, ep))
    # 동점이면 최신 우선 (id 큰 것) — 기판/장비 상태는 시간에 따라 변하므로
    scored.sort(key=lambda x: (x[0], x[1].get("id", 0)), reverse=True)
    return scored


# ══════════════════════════════════════════════════════════════════════════════
# 회수 API (에피소드 우선 → 절차 fallback = 논문 Fig.2의 적응형 회수)
# ══════════════════════════════════════════════════════════════════════════════

def recall_params(context: dict) -> dict | None:
    """
    hw_manager 적응형 튜닝의 시작 파라미터 회수.
    성공 + 튜닝 수렴(tuned) 에피소드 중 가장 유사한 것의 확정 조건을 반환.

    안전 조건 2중:
      1. sample_type '정확 일치' 필수 — 기판/외형/영역이 다 같아도 시료가 다르면
         유사도 합계가 임계값을 넘을 수 있는데(최대 5.0 > 3.0), 다른 시료의
         파워/노출을 시작점으로 쓰는 것은 광손상 위험이 있다. 파라미터를
         '직접' 재사용하는 이 경로만은 유사도 점수로 대체할 수 없는 하드 조건.
         (advisory 텍스트인 recall_summary/suggest_next_steps는 임계값만 적용 —
          LLM 참고 자료일 뿐 레이저 조사에 직결되지 않으므로)
      2. 유사도 ≥ 임계값 — 기판/영역까지 어느 정도 겹쳐야 조건 재사용이 유효.
    """
    if not context.get("sample_type") or context["sample_type"] == "unknown":
        return None
    data = _load()
    for score, ep in _rank_episodes(context, data["episodes"], success_only=True):
        if ep.get("context", {}).get("sample_type") != context["sample_type"]:
            continue  # 하드 조건 1
        out = ep.get("outcome", {})
        if out.get("tuned") and out.get("power_pct") is not None:
            return {
                "power_pct": out["power_pct"],
                "exposure_s": out.get("exposure_s"),
                "similarity": round(score, 2),
                "episode_id": ep.get("id"),
            }
    return None


def _edge_weight(edge: dict) -> float:
    """논문 4.2절: w' = N + c·Σ(1/nₖ). N=성공 횟수, eff=Σ(1/nₖ) 누적값."""
    return float(edge.get("n", 0)) + _C_EFF * float(edge.get("eff", 0.0))


def suggest_next_steps(prev_key: str, context: dict, top_k: int = 2) -> list[dict]:
    """
    "prev_key step 다음에 무엇이 왔었나" 제안 — 재계획 시 Planner 프롬프트에 주입.

    적응형 회수 (논문 Fig.2):
      1. prev_key에서 나가는 edge들을 찾는다
      2. edge에 달린 에피소드 중 현재 컨텍스트와 유사한 것이 있으면
         → 에피소드 유사도 순 (상황이 비슷했던 전이가 우선)
      3. 없으면 → 절차 가중치 순 (그냥 자주 성공한 전이)
    top_k=2인 이유: 논문 Table 4(D) — 1개는 과도한 제약, 3개는 노이즈.
    """
    data = _load()
    edges = data["graph"]["edges"]
    ep_by_id = {ep["id"]: ep for ep in data["episodes"]}

    candidates = []
    prefix = prev_key + "->"
    for key, edge in edges.items():
        if not key.startswith(prefix):
            continue
        next_key = key[len(prefix):]
        # 에피소드 유사도: 이 전이가 등장한 실험들의 컨텍스트와 비교
        best_sim = 0.0
        for eid in edge.get("episode_ids", []):
            ep = ep_by_id.get(eid)
            if ep:
                best_sim = max(best_sim, similarity(context, ep.get("context", {})))
        candidates.append({
            "next": next_key,
            "weight": _edge_weight(edge),
            "sim": best_sim,
            "n": edge.get("n", 0),
        })

    if not candidates:
        return []

    episodic = [c for c in candidates if c["sim"] >= _SIM_THRESHOLD]
    if episodic:
        episodic.sort(key=lambda c: c["sim"], reverse=True)
        for c in episodic:
            c["mode"] = "episodic"
        return episodic[:top_k]

    # 절차 fallback — 가중치 정규화(논문의 outgoing normalize)는 순위에 영향이
    # 없으므로 생략하고 원시 가중치로 정렬한다.
    candidates.sort(key=lambda c: c["weight"], reverse=True)
    for c in candidates:
        c["mode"] = "procedural"
    return candidates[:top_k]


def recall_summary(context: dict) -> str:
    """
    Planner 계획 생성 프롬프트용 요약.
    구성: ① 유사 성공 에피소드(조건 재사용) ② 유사 실패 에피소드(회피 경고)
          ③ 자주 성공한 계획 절차 (절차 기억의 계획 수준 표현)
    """
    data = _load()
    lines = []

    # ① 성공 경험 (최대 2건 — 프롬프트 길이 통제)
    successes = _rank_episodes(context, data["episodes"], success_only=True)[:2]
    for score, ep in successes:
        out = ep.get("outcome", {})
        c = ep.get("context", {})
        lines.append(
            f"- [성공, 유사도 {score:.1f}] 시료 {c.get('sample_type')}, "
            f"기판 {c.get('substrate') or '?'}, 영역 {c.get('region') or '?'}: "
            f"레이저 {out.get('power_pct')}%, 노출 {out.get('exposure_s')}s → "
            f"신호 {out.get('max_intensity', '?')} ADU"
            f"{', 배경 ' + str(out.get('background_max_intensity')) + ' ADU' if out.get('background_max_intensity') else ''}. "
            f"{out.get('notes', '')}"
        )

    # ② 실패 경험 — 같은 방식의 실패를 계획 단계에서 회피시키는 부정적 지식
    fails = [(s, ep) for s, ep in _rank_episodes(context, data["episodes"], success_only=False)
             if not ep.get("success")][:1]
    for score, ep in fails:
        out = ep.get("outcome", {})
        lines.append(
            f"- [실패 경험, 유사도 {score:.1f}] 주의: {out.get('notes', '원인 불명')} — "
            "같은 방식을 반복하지 말 것"
        )

    # ③ 자주 성공한 계획 절차 (성공 에피소드의 plan_signature 최빈값)
    sig_count: dict[str, int] = {}
    for ep in data["episodes"]:
        if ep.get("success") and ep.get("plan_signature"):
            sig = " → ".join(ep["plan_signature"])
            sig_count[sig] = sig_count.get(sig, 0) + 1
    if sig_count:
        best_sig, n = max(sig_count.items(), key=lambda kv: kv[1])
        lines.append(f"- 자주 성공한 절차 ({n}회): {best_sig}")

    if not lines:
        return "과거 측정 경험 없음 — 낮은 출력에서 시작하는 적응형 측정 권장."
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 기록 API (orchestrator가 실험 종료 시 1회 호출 — 성공/실패/중단 모두)
# ══════════════════════════════════════════════════════════════════════════════

def action_key(step: dict) -> str:
    """plan step → 그래프 노드 키. 논문의 '툴 노드'에 해당하는 것이 우리
    아키텍처에서는 'agent:task 단위의 plan step'이다 (하드웨어 툴 시퀀스는
    이미 결정적 코드로 고정돼 있어 그래프로 배울 것이 없다)."""
    params = step.get("params", {}) or {}
    task = params.get("task") or params.get("mode") or "default"
    return f"{step.get('agent', '?')}:{task}"


def record_experiment(state: dict) -> None:
    """
    실험 1건의 최종 상태를 에피소드 + 절차 그래프에 기록.
    호출 지점을 orchestrator(그래프 invoke 직후) 단일 지점으로 둔 이유:
    abort로 중단된 실험은 report_generator에 도달하지 않으므로, 거기서 기록하면
    실패 지식이 영영 쌓이지 않는다. invoke 직후는 모든 종료 경로가 지나간다.
    """
    intent = state.get("intent") or {}
    plan = state.get("plan", []) or []
    acq = state.get("acquisition_params") or {}
    bg = state.get("background_reference") or {}
    critic_log = state.get("critic_log", []) or []
    failure_log = state.get("failure_log", []) or []

    context = build_context(state)
    # 무엇인지 모르는 시료의 조건은 재사용 가치가 없어 기록하지 않는다
    if context["sample_type"] in ("", "unknown") and not context["substrate"]:
        return

    # 성공 판정: 중단되지 않았고 + 스펙트럼 획득까지 도달했는가.
    # (분석/보고 LLM이 다소 부실해도 "측정 조건" 지식은 유효하므로
    #  측정 도달 여부를 기준으로 삼는다)
    success = state.get("abort_reason") is None and bool(acq)

    executed = [s for s in plan if s.get("status") in ("done", "skipped")]
    c3_issues = [e["reason"] for e in critic_log
                 if e.get("checkpoint") == "C3" and e.get("verdict") != "APPROVE"]

    notes_parts = []
    if acq:
        notes_parts.append("튜닝 " + ("수렴" if acq.get("tuned") else "미수렴(한계 도달)"))
    if c3_issues:
        notes_parts.append("C3: " + "; ".join(c3_issues[:2]))
    if state.get("abort_reason"):
        notes_parts.append(f"중단: {state['abort_reason']}")
    elif failure_log:
        notes_parts.append(f"실패 {len(failure_log)}건: {failure_log[-1].get('error', '')[:80]}")

    data = _load()
    ep_id = data["next_id"]
    data["next_id"] = ep_id + 1

    episode = {
        "id": ep_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": state.get("session_id", ""),
        "success": success,
        "context": context,
        "outcome": {
            "power_pct": acq.get("power_pct"),
            "exposure_s": acq.get("exposure_s"),
            "tuned": bool(acq.get("tuned")),
            "max_intensity": (acq.get("history") or [{}])[-1].get("max_adu"),
            "background_max_intensity": bg.get("max_intensity"),
            "roi_mode": (state.get("next_roi") or {}).get("mode"),
            "c3_issues": c3_issues[:3],
            "notes": "; ".join(notes_parts) or "특이사항 없음",
        },
        "plan_signature": [action_key(s) for s in executed],
        "steps": len(executed),
    }
    data["episodes"].append(episode)
    if len(data["episodes"]) > _MAX_EPISODES:
        data["episodes"] = data["episodes"][-_MAX_EPISODES:]

    # ── 절차 그래프 갱신: 성공 실험만 ──────────────────────────────────────────
    # 실패 실험의 전이가 가중치를 얻으면 나쁜 절차가 "자주 한 절차"로 둔갑한다.
    # 실패 지식은 에피소드(success=False)로만 남긴다.
    if success and len(episode["plan_signature"]) >= 2:
        n_steps = max(episode["steps"], 1)
        edges = data["graph"]["edges"]
        sig = episode["plan_signature"]
        for a, b in zip(sig, sig[1:]):
            key = f"{a}->{b}"
            edge = edges.get(key) or {"n": 0, "eff": 0.0, "episode_ids": []}
            edge["n"] += 1
            edge["eff"] += 1.0 / n_steps          # 논문의 Σ(1/nₖ) 항 누적
            edge["episode_ids"] = (edge["episode_ids"] + [ep_id])[-_EDGE_EPISODE_CAP:]
            edges[key] = edge

    _save(data)
