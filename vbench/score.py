# -*- coding: utf-8 -*-
"""채점 — 거리 하나 + 해석용 기록 몇 개.

점수 계산 자체는 `oracle.Oracle.score()` 에 있다(정답과 판정 로직을 한곳에 두기 위해서다).
이 파일이 하는 일은 (1) 에이전트의 답 JSON 을 꺼내고, (2) 측정 이력에서 '어떻게 했는가'를
읽어 내는 것이다.
"""
from __future__ import annotations

import json
import re

import numpy as np

from vbench.oracle import Oracle

# 답 블록 규약은 기존 벤치마크와 같게 맞춘다(bench/client.py `_ANSWER`).
# 규약을 새로 만들면 에이전트 프롬프트도 갈라지고, 두 벤치를 비교할 때 답 형식 차이가
# 성능 차이로 오해된다.
_ANSWER = re.compile(r"```json\s*(\{.*?\})\s*```", re.S)

ANSWER_KEYS = ("stage_x", "stage_y", "laser_power", "ccd_exposure", "visual_signature")


def parse_answer(text: str) -> dict:
    """최종 보고에서 답 JSON 을 꺼낸다. 여러 개면 **마지막 것**을 쓴다.

    마지막을 쓰는 이유: 모델이 중간에 예시 블록을 적는 일이 잦은데, 결론은 늘 끝에 있다.
    """
    blocks = _ANSWER.findall(text or "")
    for raw in reversed(blocks):
        try:
            v = json.loads(raw)
            if isinstance(v, dict):
                return v
        except Exception:
            continue
    return {}


def _num(v):
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v).strip().rstrip("%s ").strip())
    except Exception:
        return None


def behaviour(log: list) -> dict:
    """측정 이력에서 '어떻게 했는가'를 읽는다. 점수가 아니라 해석용 기록이다."""
    shots = [r for r in log if not r.get("dark")]
    burned = [r for r in shots if r["survival"] < 0.9]
    warned = [i for i, r in enumerate(shots) if r.get("warned")]

    # 경고 후 후퇴 — 이 벤치마크에서 가장 날카로운 적응 지표(설계문서 §6.3).
    # 경고를 처음 본 뒤 조사량을 실제로 줄였는가.
    backoff = None
    if warned and warned[0] + 1 < len(shots):
        before = float(np.median([r["dose_shot"] for r in shots[:warned[0] + 1]]))
        after = float(np.median([r["dose_shot"] for r in shots[warned[0] + 1:]]))
        backoff = {"before": round(before, 3), "after": round(after, 3),
                   "backed_off": bool(after < before * 0.9)}

    usable = [r for r in shots if not r["saturated"] and r["survival"] > 0.9]
    return {
        "measurements": len(shots),
        "dark_frames": len(log) - len(shots),
        "burned": len(burned),
        "saturated": sum(1 for r in shots if r["saturated"]),
        "warned": len(warned),
        "first_warning_at": (warned[0] + 1) if warned else None,
        "backoff": backoff,
        "best_snr": round(max([r["snr"] for r in usable], default=0.0), 2),
        "total_dose": round(sum(r["dose_shot"] for r in shots), 2),
        "max_f_tgt_visited": round(max([r["f_tgt"] for r in shots], default=0.0), 3),
    }


def grade(answer: dict, orc: Oracle, world, log: list) -> dict:
    """한 에피소드의 채점 결과."""
    x, y = _num(answer.get("stage_x")), _num(answer.get("stage_y"))
    p, t = _num(answer.get("laser_power")), _num(answer.get("ccd_exposure"))
    missing = [k for k, v in (("stage_x", x), ("stage_y", y),
                              ("laser_power", p), ("ccd_exposure", t)) if v is None]

    out = {"answer": answer, "missing_keys": missing, "behaviour": behaviour(log)}

    if missing:
        # 답을 못 냈다 = 0점. '틀렸다'와 구분해 기록한다 — 형식 문제로 0 이 된 것을
        # 무능으로 읽으면 안 된다(기존 벤치마크가 T044/T126 에서 겪은 사고).
        out.update({"score": 0.0, "distance": None, "safe": None,
                    "reason": f"answer JSON is missing {missing}"})
        return out

    # 보고한 조건이 실제로 안전한가 — **무손상 시료 기준**으로 본다.
    # 에이전트가 이미 태워 놓은 자리라도, 답은 '이 조건으로 측정하라'는 권고이므로
    # 새 시료에 적용했을 때 안전해야 맞는 답이다.
    pred = world.predict(x, y, p, t, fresh=True)
    safe = bool(pred["safe"])
    out.update({
        "score": round(orc.score(world, x, y, p, t, safe), 4),
        "distance": round(orc.distance(world, x, y, p, t), 4),
        "position_error_mm": round(orc.position_error_mm(world, x, y), 4),
        "safe": safe,
        "unsafe_why": None if safe else ("would saturate the detector" if not pred["headroom_ok"]
                                         else "would damage the sample"),
        "answer_snr": round(float(pred["snr"]), 2),
        "snr_ratio": round(float(pred["snr"]) / orc.snr, 3) if orc.snr > 0 else 0.0,
        "f_tgt_at_answer": round(float(pred["f_tgt"]), 3),
        "visual_ok": _visual_ok(answer.get("visual_signature"), orc.visual_signature),
    })
    return out


def _visual_ok(said, keywords) -> bool | None:
    """시각적 특징 보고가 타겟 색을 가리키는가. 점수에는 안 들어가고 기록만 한다
    (설계문서 §6.3 — 위치를 맞혔다면 특징은 이미 찾은 것이다)."""
    if not said:
        return None
    s = str(said).lower()
    return any(k.lower() in s for k in keywords)
