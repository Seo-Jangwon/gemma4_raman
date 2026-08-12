# -*- coding: utf-8 -*-
"""턴 단위 상세 로그(JSON) — AILA/CoALA 두 에이전트 공용.

[목적]
매 응답(턴)에서 '무엇을 했는가'를 기계가 읽을 수 있는 형태로 전부 남긴다.
runtime.stream_turn / run_turn_once 가 턴마다 무조건 연다 — 특정 실행 모드 전용이 아니다:
  · Question / Ans                     — 질의와 최종 응답(전체 입출력)
  · Tool calls                         — 도구 호출 순서(이름·인자·성공여부·결과)
  · planning evaluation process        — 의사결정 루프 트레이스
                                         (CoALA: propose→evaluate→select 단계.
                                          AILA는 그런 명시적 루프가 없어 빈 배열)
  · 그 외 메타 — 응답유형(done/chat/error), 누적 조사량, 소요시간 등

[독립성]
AILA와 CoALA는 서로를 import하지 않는다(비교 실험의 독립변수는 오케스트레이션).
이 모듈은 둘이 공유하는 '중립 유틸'일 뿐이라 그 원칙을 깨지 않는다 —
util/tool_slim.py, util/safety_limits.py 와 같은 성격이다.

[저장 위치·형식]
  <프로젝트 루트>/DetailLog/{AGENT}_{YYYYMMDD}_{HHMMSS}_{세션아이디}.json
파일 하나 = 세션 하나. 파일명 날짜/시각은 그 세션의 '첫 응답' 시각으로 고정되고,
같은 세션의 이후 응답들은 같은 파일의 turns 배열에 append된다. 구조:

  {
    "agent": "CoALA",
    "session_id": "...",
    "started_at": "2026-07-23 14:30:00",
    "turns": [ { "turn_index": 0, "Question": ..., "Ans": ..., ... }, ... ]
  }

[Ctrl+C 안전성]
턴이 끝날 때마다 파일 전체를 read-modify-write로 다시 쓴다(단일 사용자 로컬 도구라
파일 락 없이 충분 — coala_memory와 동일한 정책). 따라서 프로그램이 언제 종료되어도
'그 시점까지 완료된 응답'은 모두 디스크에 남아 있다. 진행 중이던(아직 안 끝난) 턴만
유실된다.

[실패 격리]
로깅 실패가 실행 자체를 깨뜨리면 안 된다 — 모든 파일 I/O는 예외를 삼키고 stderr로만
경고한다.
"""
from __future__ import annotations

import json
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

# <프로젝트 루트>/DetailLog. __file__ = backend/util/detail_log.py 이므로
# parents[2] = 프로젝트 루트(gemma4_raman).
_LOG_DIR = Path(__file__).resolve().parents[2] / "DetailLog"

# (agent, session_id) → 이미 만들어 둔 로그 파일 경로. 같은 세션의 이후 턴이
# 같은 파일에 append되도록 첫 턴에서 정한 파일명을 재사용한다.
_SESSION_FILES: dict[tuple[str, str], Path] = {}

# 파일 read-modify-write 동시성 보호(서버가 세션을 병렬 처리할 수 있으므로).
_LOCK = threading.Lock()

# 결과 dict를 로그에 실을 때의 방어 상한 — 혹시 _slim을 안 거친 대용량 값이 와도
# 로그 파일이 폭발하지 않게 직렬화 길이를 제한한다.
_MAX_RESULT_CHARS = 4000


def _sanitize(text: str) -> str:
    """파일명에 안전한 형태로. 영숫자/대시/언더스코어만 남기고 나머지는 '-'."""
    return re.sub(r"[^0-9A-Za-z_-]", "-", str(text))[:64] or "nosession"


def _cap(obj: Any) -> Any:
    """직렬화 시 과도하게 큰 값을 방어적으로 잘라낸다(정상 결과는 그대로 통과)."""
    try:
        s = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return str(obj)[:_MAX_RESULT_CHARS]
    if len(s) <= _MAX_RESULT_CHARS:
        return obj
    return {"_truncated": True, "preview": s[:_MAX_RESULT_CHARS]}


class TurnLog:
    """한 턴(질의→응답)의 이벤트를 모아, 턴이 끝날 때 세션 파일에 append한다.

    사용법:
        turn = new_turn("CoALA", sid, user_message)
        for event in run_stream(...):
            turn.observe(event)          # tool / phase 이벤트를 그대로 먹인다
            ...
        turn.complete("done", final_text, final_ctx)   # 또는 turn.fail(detail)
    """

    def __init__(self, agent: str, session_id: str, question: str):
        self.agent = agent
        self.session_id = session_id or ""
        self.question = question
        self._start = time.time()
        self.started_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self._start))
        self.tool_calls: list[dict] = []       # "Tool calls"
        self.planning: list[dict] = []          # "planning evaluation process"
        self._written = False                   # 중복 기록 방지

    # ── 이벤트 수집 ────────────────────────────────────────────────────────────
    def observe(self, event: dict) -> None:
        """run_stream이 yield한 이벤트 하나를 기록한다(tool / phase만 관심)."""
        etype = event.get("type")
        if etype == "tool":
            result = event.get("result")
            entry = {
                "step": len(self.tool_calls),
                "name": event.get("name"),
                "action": event.get("action"),   # grounding/retrieval/learning (CoALA) · AILA는 None
                "args": _cap(event.get("args")),
                "ok": result.get("ok", True) if isinstance(result, dict) else True,
                "error": result.get("error") if isinstance(result, dict) else None,
                "result": _cap(result),
            }
            self.tool_calls.append(entry)
        elif etype == "phase":
            # phase 이벤트의 모든 부가 필드를 그대로 담는다.
            #   · CoALA: phase(plan/evaluate/select) + message + candidates/scores/reason/chosen
            #   · AILA(ReAct): phase 이벤트를 내지 않는다 — 명시적 계획 루프가 없어
            #     planning 배열은 항상 빈 채로 남는다(설계상 정상, 모듈 상단 주석 참고).
            entry = {k: v for k, v in event.items()
                     if k != "type" and v is not None}
            self.planning.append(entry)

    # ── 턴 종료 → 파일 기록 ─────────────────────────────────────────────────────
    def complete(self, response_type: str, answer: str,
                 ctx: Optional[dict] = None) -> None:
        """정상 종료. response_type ∈ {"done"(측정 포함), "chat"(대화)}."""
        self._flush(response_type=response_type, answer=answer, ctx=ctx, error=None)

    def fail(self, detail: str, ctx: Optional[dict] = None) -> None:
        """에러 종료. 지금까지 모은 도구/계획 트레이스와 함께 남긴다."""
        self._flush(response_type="error", answer="", ctx=ctx, error=detail)

    def _flush(self, response_type: str, answer: str,
               ctx: Optional[dict], error: Optional[str]) -> None:
        if self._written:
            return
        self._written = True

        ctx = ctx or {}
        entry = {
            "turn_index": None,   # 파일에 붙일 때 확정
            "Question": self.question,
            "Ans": answer,
            "Tool calls": self.tool_calls,
            "planning evaluation process": self.planning,
            "response_type": response_type,
            "error": error,
            "tool_call_order": list(ctx.get("tool_call_order", [])),
            "num_tool_calls": len(self.tool_calls),
            "dose_mj": round(float(ctx.get("dose", 0.0)), 4),
            "learned": bool(ctx.get("learned", False)),
            "started_at": self.started_at,
            "ended_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_s": round(time.time() - self._start, 3),
        }
        try:
            _append_turn(self.agent, self.session_id, self.started_at, entry)
        except Exception as e:   # noqa: BLE001 — 로깅 실패가 실험을 깨선 안 된다
            print(f"[detail_log] 기록 실패: {type(e).__name__}: {e}", file=sys.stderr)


def new_turn(agent: str, session_id: str, question: str) -> TurnLog:
    """한 턴의 로거를 만든다(에이전트 진입점에서 호출)."""
    return TurnLog(agent, session_id, question)


def _session_file(agent: str, session_id: str, started_at: str) -> Path:
    """이 (agent, session) 조합의 로그 파일 경로. 첫 호출 때 파일명을 정해 캐시한다."""
    key = (agent, session_id)
    path = _SESSION_FILES.get(key)
    if path is not None:
        return path
    # started_at("YYYY-MM-DD HH:MM:SS") → "YYYYMMDD_HHMMSS"
    stamp = re.sub(r"[^0-9]", "", started_at)          # 20260723143000
    stamp = f"{stamp[:8]}_{stamp[8:14]}" if len(stamp) >= 14 else stamp
    fname = f"{agent}_{stamp}_{_sanitize(session_id)}.json"
    path = _LOG_DIR / fname
    _SESSION_FILES[key] = path
    return path


def _append_turn(agent: str, session_id: str, started_at: str, entry: dict) -> None:
    """세션 파일을 읽어 turns에 entry를 append하고 다시 쓴다(락 보호)."""
    with _LOCK:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = _session_file(agent, session_id, started_at)
        doc: dict
        try:
            with open(path, "r", encoding="utf-8") as f:
                doc = json.load(f)
            if not isinstance(doc, dict) or "turns" not in doc:
                raise ValueError("형식 불일치")
        except (FileNotFoundError, json.JSONDecodeError, ValueError, OSError):
            doc = {"agent": agent, "session_id": session_id,
                   "started_at": started_at, "turns": []}
        entry["turn_index"] = len(doc["turns"])
        doc["turns"].append(entry)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
