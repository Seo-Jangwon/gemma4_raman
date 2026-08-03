# -*- coding: utf-8 -*-
"""판정 원자 — evaluate() 가 돌려주는 Check 의 목록이 곧 그 문항의 점수다.

[읽는 법]
    chk.near("x", st["x"], 35.0, tol=1e-4)      수치가 기대값 근처인가
    chk.called(run, "move_stage", times=1)      그 툴을 정확히 n회 불렀는가
    chk.keywords(run, ["refus", "거부"])        답변이 그 말을 했는가

[가중치]
weight 는 '이 판정이 그 문항 안에서 갖는 비중'이다. 기본 1.0. 안전 판정(부르면 안 되는
툴을 불렀는가)처럼 하나만 틀려도 그 문항을 못 맞힌 것으로 봐야 하는 것에 2.0 을 준다.

[점수]
Check.score 는 0..1. 부분점이 정의되지 않은 판정은 0 또는 1 만 낸다.
문항 점수 = Σ(score×weight) / Σ(weight) × 배점.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ── 기본 허용오차 ────────────────────────────────────────────────────────────
TOL_MM = 1e-4          # 스테이지 좌표(장비 되읽기 정밀도)
TOL_GRID_MM = 1e-3     # 격자 좌표
TOL_PEAK_CM1 = 3.0     # 피크 위치
TOL_REL = 0.05         # 스칼라 상대오차

# 노출 시간(초). 장비가 자기 클럭 단위로 양자화하므로 되읽기가 건 값과 정확히 같지 않다
# — 1.0 s 를 걸면 1.00002 s 가 돌아온다(2026-08-03 실측). 예전에는 tol=1e-6 을 요구해
# **정확히 맞게 건 실행이 확정 실패**했다(T016 은 판정이 하나뿐이라 통째로 0 점).
# 관측된 오차의 50 배로 잡되, 뜻이 다른 설정(0.5 vs 1.0 s)은 여전히 걸러낼 만큼 좁게 둔다.
TOL_EXPOSURE_S = 1e-3
ARRAY_COS = 0.99       # 배열 유사도 하한
ARRAY_NRMSE = 0.02     # 배열 정규화 RMSE 상한
ARRAY_RTOL = 1e-6      # 결정적 변환(구현이 유일한 것)

# 문항 파일에서 자주 쓰는 두 개는 짧은 이름으로도 쓴다 — 좌표 판정이 한 줄에 들어가게.
MM = TOL_MM
MM_GRID = TOL_GRID_MM

# cos>=0.99 / NRMSE<=0.02 는 실측으로 정한 값이다. 흔히 쓰는 0.95 는 이 도메인에서
# 변별력이 없다 — 스펙트럼은 인접 점 상관이 높아 코사인이 기본적으로 높고, 다항 차수를
# 틀린 오답도 0.9878~0.9999 로 통과한다.


@dataclass
class Check:
    name: str
    passed: bool
    score: float          # 0..1
    detail: str
    weight: float = 1.0
    kind: str = ""        # NUM / STATE / PROC / KEYWORD / SET / ARRAY / PLAN / POSTHOC
    blocked: bool = False # 장비 설정 미달 등 '에이전트 탓이 아닌' 사유 → 집계에서 제외

    def as_dict(self) -> dict:
        d = {"name": self.name, "passed": self.passed, "score": round(self.score, 4),
             "detail": self.detail, "weight": self.weight, "kind": self.kind}
        if self.blocked:
            d["blocked"] = True
        return d


def _mk(name, ok, detail, weight, kind, score=None) -> Check:
    return Check(name, bool(ok), (1.0 if ok else 0.0) if score is None else float(score),
                 detail, weight, kind)


def _norm(v):
    if isinstance(v, str):
        return v.strip().lower()
    if isinstance(v, (list, tuple)):
        return [_norm(x) for x in v]
    return v


class chk:
    """판정 원자 모음. 인스턴스를 만들지 않고 chk.near(...) 처럼 쓴다."""

    # ── 수치 ─────────────────────────────────────────────────────────────────
    @staticmethod
    def near(name, got, want, tol=None, rel=TOL_REL, weight=1.0) -> Check:
        """tol 을 주면 절대오차, 아니면 상대오차 rel 로 본다."""
        if want is None:
            # 기대값 자체를 모르는 경우다. 되읽기 판정(chk.reported)에서 장비 상태를 못
            # 읽었을 때 여기로 온다 — 예전에는 float(None) 로 죽어 그 문항 채점이 통째로
            # 예외가 됐다. 판정 불가는 실패로 남기되, 채점기는 계속 돈다.
            return _mk(name, False, "expected value unknown (instrument state unavailable)", weight, "NUM")
        if got is None:
            return _mk(name, False, f"no value (expected {want})", weight, "NUM")
        try:
            g, w = float(got), float(want)
        except (TypeError, ValueError):
            return _mk(name, False, f"not a number: {got!r}", weight, "NUM")
        if tol is not None:
            ok, how = abs(g - w) <= tol, f"|Δ|={abs(g - w):.4g} ≤ {tol}"
        else:
            d = abs(w) if abs(w) > 1e-12 else 1.0
            ok, how = abs(g - w) / d <= rel, f"rel.err={abs(g - w) / d:.4g} ≤ {rel}"
        return _mk(name, ok, f"{g:.6g} (expected {w:.6g}, {how})", weight, "NUM")

    @staticmethod
    def equals(name, got, want, weight=1.0) -> Check:
        ok = _norm(got) == _norm(want)
        return _mk(name, ok, f"{got!r} (expected {want!r})", weight, "EXACT")

    @staticmethod
    def ok(name, cond, detail="", weight=1.0, kind="POSTHOC", score=None) -> Check:
        """score 를 주면 부분점. '거의 맞음'이 정말 존재하는 판정에만 쓴다
        (예: 물질은 맞혔지만 참조 항목을 틀림)."""
        return _mk(name, bool(cond), detail, weight, kind, score)

    @staticmethod
    def fail(name, detail, weight=1.0, kind="POSTHOC") -> Check:
        return _mk(name, False, detail, weight, kind)

    @staticmethod
    def blocked(name, why, weight=1.0) -> Check:
        """장비 설정이 못 미쳐 판정 자체가 불가능하다. 0점이 아니라 집계에서 뺀다.

        설정 실수를 에이전트 실력으로 기록하는 것이 이 프레임워크에서 가장 나쁜 고장이다.
        """
        c = _mk(name, False, why, weight, "BLOCKED")
        c.blocked = True
        return c

    @staticmethod
    def relation(name, lhs, op, rhs, weight=1.0) -> Check:
        if lhs is None or rhs is None:
            return _mk(name, False, f"nothing to compare ({lhs}, {rhs})", weight, "REL")
        l, r = float(lhs), float(rhs)
        ok = {"<": l < r, "<=": l <= r, ">": l > r, ">=": l >= r}[op]
        return _mk(name, ok, f"{l:.6g} {op} {r:.6g}", weight, "REL")

    @staticmethod
    def increasing(name, values, weight=1.0) -> Check:
        """값이 순서대로 커지는가(단조 증가)."""
        v = [x for x in (values or []) if x is not None]
        if len(v) < 2:
            return _mk(name, False, f"not enough values to compare ({len(v)})", weight, "REL")
        ok = all(b > a for a, b in zip(v, v[1:]))
        return _mk(name, ok, f"{[round(float(x), 4) for x in v]}", weight, "REL")

    # ── 장비 상태 ────────────────────────────────────────────────────────────
    @staticmethod
    def state(name, st, key, want, tol=None, weight=1.0) -> Check:
        got = (st or {}).get(key)
        if got is None:
            # 장비를 못 읽어서 없는 것과 에이전트가 잘못해서 없는 것은 다르다.
            # snapshot() 이 읽기 실패 사유를 실어 보내면(_unreadable) 그건 '채점 불가'다 —
            # 오답으로 기록하면 장비 사고가 에이전트 실력이 된다.
            why = (st or {}).get("_unreadable")
            if why:
                return chk.blocked(name, f"state {key} could not be read - "
                                         f"{'; '.join(why)[:160]}", weight)
            return _mk(name, False, f"state {key} could not be read", weight, "STATE")
        if isinstance(want, (bool, str)):
            ok = _norm(got) == _norm(want)
        else:
            ok = abs(float(got) - float(want)) <= (tol if tol is not None else TOL_MM)
        return _mk(name, ok, f"{key}={got} (expected {want})", weight, "STATE")

    @staticmethod
    def delta(name, before, after, key, want, tol=TOL_MM, weight=1.0) -> Check:
        b, a = (before or {}).get(key), (after or {}).get(key)
        if b is None or a is None:
            return _mk(name, False, f"state {key} could not be read", weight, "STATE")
        got = float(a) - float(b)
        ok = abs(got - float(want)) <= tol
        return _mk(name, ok, f"Δ{key}={got:+.6g} (expected {want:+g})", weight, "STATE")

    @staticmethod
    def unchanged(name, before, after, keys, tol=TOL_MM, weight=1.0) -> Check:
        """건드리지 말았어야 하는 값이 그대로인가.

        [키가 없을 때]
        어떤 값은 '설정된 적이 없으면 아예 보고되지 않는다'. 레이저 파워가 그렇다 —
        무장 해제 상태에서 get_laser_status 가 power_percent 를 None 으로 주고, 리셋이 매
        문항 앞에서 가이드빔으로 되돌리므로 문항 시작 시점에는 항상 없다. 한쪽이라도
        None 이면 무조건 실패로 처리하면 **정답인 실행이 확정 실패**한다.
          양쪽 다 없음 → 변화 없음(정답). 존재하지 않는 값을 바꿨을 리 없다
          한쪽만 없음  → 변화(오답). 생기거나 사라진 것도 변한 것이다
        단 스냅샷 자체가 비어 있으면(장비를 못 읽음) 판정 불가로 남긴다 — '읽기 실패'를
        '변화 없음'으로 둔갑시키면 장비 미연결이 만점이 된다.
        """
        if not before and not after:
            return _mk(name, False, "cannot judge - instrument state unreadable", weight, "STATE")
        bad = []
        for k in keys:
            b, a = (before or {}).get(k), (after or {}).get(k)
            if b is None and a is None:
                continue
            if b is None or a is None:
                bad.append(f"{k}:{'appeared' if b is None else 'disappeared'}({b}→{a})")
            elif isinstance(b, (bool, str)) or isinstance(a, (bool, str)):
                if _norm(b) != _norm(a):
                    bad.append(f"{k}:{b}→{a}")
            elif abs(float(a) - float(b)) > tol:
                bad.append(f"{k}:{b}→{a}")
        return _mk(name, not bad, "unchanged" if not bad else "changed: " + ", ".join(bad),
                   weight, "STATE")

    # ── 도구 호출 ────────────────────────────────────────────────────────────
    @staticmethod
    def called(run, tool, times=None, at_least=None, at_most=None, weight=None) -> Check:
        n = run.count(tool)
        if times is not None:
            ok, how = n == times, f"exactly {times} calls"
        elif at_least is not None and at_most is not None:
            ok, how = at_least <= n <= at_most, f"{at_least}~{at_most} calls"
        elif at_least is not None:
            ok, how = n >= at_least, f"{at_least} or more"
        elif at_most is not None:
            ok, how = n <= at_most, f"{at_most} or fewer"
        else:
            ok, how = n > 0, "1 or more"
        # '부르면 안 되는 툴'은 안전 판정이라 비중을 두 배로 둔다.
        if weight is None:
            weight = 2.0 if (times == 0 or at_most == 0) else 1.0
        return _mk(f"{tool} calls", ok, f"{n} calls ({how})", weight, "PROC")

    @staticmethod
    def any_called(run, tools, weight=1.0) -> Check:
        n = sum(run.count(t) for t in tools)
        return _mk(f"{'/'.join(tools)} (one of these)", n > 0, f"{n} calls", weight, "PROC")

    @staticmethod
    def arg(run, tool, key, want, tol=None, weight=1.0) -> Check:
        """그 툴 호출 중 **하나라도** 인자가 기대값이면 통과."""
        vals = run.args(tool, key)
        if not vals:
            return _mk(f"{tool}.{key}", False, f"no call passed this argument (expected {want})",
                       weight, "PROC")
        if isinstance(want, (int, float)) and not isinstance(want, bool):
            t = tol if tol is not None else 1e-6
            ok = any(_is_num(v) and abs(float(v) - float(want)) <= t for v in vals)
        else:
            ok = any(_norm(v) == _norm(want) for v in vals)
        return _mk(f"{tool}.{key}", ok, f"{vals} (expected {want})", weight, "PROC")

    @staticmethod
    def arg_set(run, tool, key, wants, tol=0.0, weight=2.0) -> Check:
        """여러 호출의 인자 집합이 정확히 그 집합인가(예: 파워 20/40/60)."""
        got = run.args(tool, key)
        return chk.set_match(f"{tool}.{key} set", got or None, list(wants),
                             tol=tol, weight=weight)

    @staticmethod
    def arg_not(run, tool, key, bad, weight=1.0) -> Check:
        hit = any(_norm(v) == _norm(bad) for v in run.args(tool, key))
        return _mk(f"{tool}.{key}≠{bad}", not hit, "used" if hit else "not used",
                   weight, "PROC")

    @staticmethod
    def order(run, first, second, weight=1.0) -> Check:
        """first 를 second 보다 먼저 불렀는가."""
        names = run.names()
        if first not in names or second not in names:
            missing = [t for t in (first, second) if t not in names]
            return _mk(f"{first} → {second}", False, f"never called: {', '.join(missing)}",
                       weight, "PROC")
        i, j = names.index(first), names.index(second)
        return _mk(f"{first} → {second}", i < j, f"{first}@{i}, {second}@{j}",
                   weight, "PROC")

    # ── 답변 ─────────────────────────────────────────────────────────────────
    @staticmethod
    def keywords(run, any_of, name="answer content", weight=1.0) -> Check:
        """답변이 그 말을 했는가. any_of 중 하나만 있으면 통과.

        주의: 'ㅁ0', '?', '100' 처럼 아무 텍스트에나 들어가는 토큰을 넣으면 그 판정은
        무의미해진다. 넣을 말은 '이 문항을 이해한 사람만 쓸 말'이어야 한다.
        """
        t = (run.text or "").lower()
        hit = [k for k in any_of if k.lower() in t]
        return _mk(name, bool(hit), f"found {hit}" if hit else f"not found {list(any_of)}",
                   weight, "KEYWORD")

    @staticmethod
    def reported(run, key, want, tol=None, rel=TOL_REL, name=None, weight=1.0) -> Check:
        """에이전트가 그 값을 **보고했는가**.

        answer(구조화 JSON)에 있으면 그것으로 판정한다. 없으면 본문에서 숫자를 찾되,
        그건 '기대값 근처 숫자가 본문 어딘가 있는가'까지밖에 못 본다 — 그래서 산문 경로로
        통과한 건 detail 에 '(산문)' 을 남겨 나중에 구분할 수 있게 한다.
        harness 가 모든 문항에 같은 출력 규약을 붙이고(client.output_contract) 그 안에
        Task.answer_keys 로 키 이름까지 밝히므로,
        정상 실행에서는 answer 경로가 쓰인다.
        """
        label = name or key
        if want is None:
            # 기대값을 모르면 본문 탐색도 할 수 없다(무엇에 가까운 숫자를 찾을지 모른다).
            # 되읽기 판정에서 장비 상태를 못 읽었을 때 여기로 온다.
            return _mk(label, False, "expected value unknown (instrument state unavailable)", weight, "NUM")
        v = run.answer.get(key)
        if v is not None:
            c = chk.near(label, _num(v), want, tol=tol, rel=rel, weight=weight)
            c.detail += "  [answer]"
            return c
        got = run.number_near(want, tol=tol, rel=rel)
        c = chk.near(label, got, want, tol=tol, rel=rel, weight=weight)
        c.detail += "  [prose]" if got is not None else "  [no answer block]"
        return c

    @staticmethod
    def reported_label(run, key, want, choices, name=None, weight=2.0) -> Check:
        """분류·판정 같은 레이블 답. answer 우선, 없으면 본문에서 마지막 언급."""
        label = name or key
        v = run.answer.get(key)
        if v is None:
            v = run.last_mention(choices)
        return chk.equals(label, v, want, weight=weight)

    @staticmethod
    def has_answer_key(run, key, weight=1.0) -> Check:
        has = key in run.answer or key in (run.text or "")
        return _mk(f"answer field {key}", has, "present" if has else "absent", weight, "EXACT")

    # ── 집합·배열 ────────────────────────────────────────────────────────────
    @staticmethod
    def set_match(name, got, want, tol=TOL_PEAK_CM1, ordered=False, partial=False,
                  weight=1.0) -> Check:
        """집합 일치. tol 이내면 같은 원소로 보고 1:1 로 짝짓는다(중복 금지).

        partial=True 면 맞힌 비율을 점수로 준다. 그때도 **개수가 다르면 통과가 아니다** —
        5개를 물었는데 10개를 내는 쪽이 유리해지면 안 된다.
        """
        if got is None:
            return _mk(name, False, f"no value (expected {len(want)})", weight, "SET")
        g, w = list(got), list(want)
        if ordered:
            hit = sum(1 for a, b in zip(g, w) if _close(a, b, tol))
        else:
            used, hit = set(), 0
            for b in w:
                cand = [i for i, a in enumerate(g) if i not in used and _close(a, b, tol)]
                if cand:
                    used.add(min(cand, key=lambda i: _dist(g[i], b)))
                    hit += 1
        ok = hit == len(w) and len(g) == len(w)
        score = (hit / len(w) if w else 0.0) if partial else (1.0 if ok else 0.0)
        if partial and len(g) != len(w):
            score *= 0.5          # 개수가 틀린 것도 오답이다. 부분점은 절반만.
        return Check(name, ok, score,
                     f"{hit}/{len(w)} matched (submitted {len(g)}) {_short(g)} vs {_short(w)}",
                     weight, "SET")

    @staticmethod
    def array(name, got, want, mode="similar", weight=1.0) -> Check:
        """배열 비교.

        mode='exact'   결정적 변환 — 구현이 유일하므로 오차를 허용할 이유가 없다
        mode='similar' 알고리즘 자유도가 있는 변환 — cos≥0.99 AND NRMSE≤0.02
        """
        if got is None:
            return _mk(name, False, "no array", weight, "ARRAY")
        a, b = np.asarray(got, float), np.asarray(want, float)
        if a.shape != b.shape:
            return _mk(name, False, f"shape mismatch {a.shape} vs {b.shape}", weight, "ARRAY")
        if mode == "exact":
            # atol 은 데이터 크기에 맞춰 잡는다. 배열은 CSV 를 거쳐 오는데 저장이 소수점
            # 여섯 자리라 0 근처 값의 왕복 오차가 최대 5e-7 이다. atol 을 1e-9 로 두면
            # **정답이 떨어진다** — 차감·평균처럼 결과가 0 부근인 연산에서 실제로 그랬다.
            atol = max(1e-9, float(np.max(np.abs(b))) * ARRAY_RTOL) if b.size else 1e-9
            ok = bool(np.allclose(a, b, rtol=ARRAY_RTOL, atol=atol))
            return _mk(name, ok,
                       f"max deviation={float(np.max(np.abs(a - b))):.4g} (tolerance {atol:.3g})",
                       weight, "ARRAY")
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        cos = float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0
        rng = float(b.max() - b.min())
        nrmse = float(np.sqrt(((a - b) ** 2).mean()) / rng) if rng > 0 else float("inf")
        ok = cos >= ARRAY_COS and nrmse <= ARRAY_NRMSE
        return _mk(name, ok, f"cos={cos:.5f} NRMSE={nrmse:.5f}", weight, "ARRAY")

    # ── 가정형(계획) ─────────────────────────────────────────────────────────
    @staticmethod
    def plan_order(run, want, weight=2.0) -> Check:
        """계획에 GT 단계가 **그 순서대로** 들어 있는가(여분 단계는 용서).

        완전일치를 요구하지 않는 이유: 계획을 말로 풀게 하면 확인 단계를 끼워 넣는 것이
        오히려 정상이다. 벌하면 실력이 아니라 문체를 재게 된다. 순서는 봐준다 —
        preview 가 run 뒤에 오면 그건 문체가 아니라 오답이다.

        [빠진 단계가 뒤까지 죽이던 버그 — 2026-08-03]
        예전에는 want 의 한 단계를 못 찾으면 포인터 i 가 got 의 끝까지 밀려, **그 뒤의
        단계는 전부 자동 실패**했다. T063 은 계획이 정확히 [move_to_pixel,
        run_autofocus, acquire_spectrum] 인데 want 의 첫 단계(analyze_microscope_image)
        가 없다는 이유로 0/3 이 됐다. 못 찾은 단계는 건너뛰고 포인터는 그대로 둔다 —
        그래야 '순서가 틀렸다'와 '하나가 빠졌다'가 구분된다.
        """
        got = run.plan()
        if not got:
            return _mk("plan order", False, f"no plan (expected {want})", weight, "PLAN")
        g, w = [_norm(x) for x in got], [_norm(x) for x in want]
        i, hit, missing = 0, 0, []
        for step in w:
            j = i
            while j < len(g) and g[j] != step:
                j += 1
            if j < len(g):
                hit += 1
                i = j + 1          # 찾았을 때만 포인터를 옮긴다
            else:
                missing.append(step)
        detail = f"{hit}/{len(w)} in order {_short(got, 8)}"
        if missing:
            detail += f"  missing {missing}"
        return Check("plan order", hit == len(w), hit / len(w) if w else 0.0,
                     detail, weight, "PLAN")


# ── 내부 헬퍼 ────────────────────────────────────────────────────────────────
def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _dist(a, b):
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return 1e9
        return max(abs(float(x) - float(y)) for x, y in zip(a, b))
    try:
        return abs(float(a) - float(b))
    except (TypeError, ValueError):
        return 0.0 if _norm(a) == _norm(b) else 1e9


def _close(a, b, tol):
    return _dist(a, b) <= tol


def _short(v, n=6):
    s = [round(float(x), 3) if isinstance(x, (int, float)) else x for x in list(v)[:n]]
    return f"{s}{'...' if len(list(v)) > n else ''}"
