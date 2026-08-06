# -*- coding: utf-8 -*-
"""추론과정 로그 — 세션(=벤치 문항) 하나당 사람이 읽는 .log 파일 하나.

[detail_log.py 와 무엇이 다른가]
detail_log 는 '채점에 필요한 것'(질문/최종답/도구호출/planning 요약)을 JSON 으로 남긴다.
이 모듈은 '모델이 무슨 생각을 하며 그렇게 했는가'를 시간순 텍스트로 남긴다. 둘은 서로를
대체하지 않는다 — JSON 은 기계가, 이 로그는 사람이 읽는다.

또 하나 결정적인 차이: **벤치 경로에서는 detail_log 가 아예 안 돈다.**
서버의 /api/bench/stream 은 stream_experiment/run_experiment 를 거치지 않고
mod.run_stream() 을 직접 소비하는데(server.py:704 _producer), new_turn() 은 그 두
진입점에만 걸려 있기 때문이다. 그래서 이 모듈은 run_stream 안쪽에 건다.

[무엇을 남길 수 있고 무엇은 못 남기는가 — Ollama 기준]
Ollama HTTP API 가 돌려주는 것은 이게 전부다:
    message.content        생성된 본문 텍스트
    message.tool_calls     도구 호출
    message.thinking       '생각' 토큰 — **think 능력이 있는 모델에서 think=true 일 때만**
    prompt_eval_count / eval_count / *_duration     토큰 수·소요시간
토큰별 logit, attention, hidden state 같은 '진짜 내부 연산'은 API 표면에 없다. 즉
"Ollama 내부 추론과정"으로 남길 수 있는 최대치는 (a) 우리가 보낸 프롬프트, (b) 모델이
낸 텍스트, (c) thinking 토큰(모델이 지원할 때), (d) 토큰수·시간이다.

그중 (b)가 실은 지금까지 통째로 버려지던 부분이다. ReAct 루프는 매 step 의 AIMessage 에
tool_calls 와 **함께** 본문 텍스트를 담아 오는데(= ReAct 의 Thought), 에이전트는 마지막
step 의 텍스트만 쓰고 중간 step 의 텍스트는 messages 에 넣고 끝이었다. 이 로그가 그것을
'Gemma TEXT' 로 남긴다.

(c)는 모델 의존이다. 기동 시 /api/show 로 capabilities 를 조회해 'thinking' 이 있을
때만 think 를 켠다 — 없는 모델에 켜면 Ollama 가 400("does not support thinking")을
돌려주고 그 실행이 통째로 죽는다. 조회 결과는 로그 머리말에 그대로 적어, 'THINKING 줄이
왜 없는가'를 파일 하나만 보고 알 수 있게 한다.

[저장 위치]
벤치 세션(session_id 가 `bench_<run_id>_<agent>_<task>_<hex>`)이면
    results/<run_id>/<task>.log          예: results/2026-08-06_AILA/T001.log
즉 채점 결과 T001.json 바로 옆에 T001.log 가 놓인다. 그 외(프론트 대화 등)는
    DetailLog/reasoning/<agent>_<YYYYMMDD_HHMMSS>_<sid>.log

[환경변수]
    RAMAN_REASON_LOG=0            로깅 끄기(기본 1)
    RAMAN_REASON_LOG_PROMPT=full  매 step 프롬프트 전문 기록(기본 roster = 메시지 목록만)
    RAMAN_REASON_LOG_MAXCHARS=2000  한 필드 최대 길이
    RAMAN_LLM_THINK=auto|1|0      think 토글(기본 auto = capabilities 조회로 결정)

[실패 격리]
로깅이 실험을 깨뜨리면 안 된다 — 파일 I/O 와 능력 조회는 예외를 삼키고 stderr 경고만
낸다. 단 invoke() 자체는 삼키지 않는다(그건 실험의 본체다).
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional

# backend/agents/reason_log.py → parents[2] = 프로젝트 루트
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = _PROJECT_ROOT / "results"
FALLBACK_ROOT = _PROJECT_ROOT / "DetailLog" / "reasoning"

try:
    from backend.hardware_manager import OLLAMA_HOST, OLLAMA_MODEL
except Exception:
    OLLAMA_HOST = "http://192.168.1.15:11434"
    OLLAMA_MODEL = "gemma4:31b"


def _flag(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() not in ("0", "false", "no", "off")


_ENABLED = _flag("RAMAN_REASON_LOG", "1")
_PROMPT_MODE = os.getenv("RAMAN_REASON_LOG_PROMPT", "roster").strip().lower()
_MAX_CHARS = int(os.getenv("RAMAN_REASON_LOG_MAXCHARS", "2000") or 2000)

_LOCK = threading.Lock()
# 이미 이번 프로세스에서 연 (파일 경로) — 첫 열기는 덮어쓰고 이후 턴은 이어 쓴다.
_OPENED: set[str] = set()


# ══════════════════════════════════════════════════════════════════════════════
# think(생각 토큰) 능력 조회
# ══════════════════════════════════════════════════════════════════════════════

_THINK_ENV = os.getenv("RAMAN_LLM_THINK", "auto").strip().lower()
_think_state: Optional[tuple[bool, str]] = None


def _show_capabilities() -> tuple[Optional[list], str]:
    """Ollama /api/show 로 이 모델의 capabilities 를 읽는다. (목록 or None, 설명)."""
    url = OLLAMA_HOST.rstrip("/") + "/api/show"
    body = json.dumps({"model": OLLAMA_MODEL}).encode("utf-8")
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=8.0) as r:
            doc = json.loads(r.read().decode("utf-8", "replace"))
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    caps = doc.get("capabilities")
    if not isinstance(caps, list):
        # 구버전 Ollama 는 capabilities 를 안 준다. 그 경우 판정 불가.
        return None, "응답에 capabilities 필드가 없음(구버전 Ollama)"
    return caps, ""


def think_capability() -> tuple[bool, str]:
    """(think 를 켤 것인가, 사람이 읽을 근거). 프로세스당 1회만 조회하고 캐시한다."""
    global _think_state
    if _think_state is not None:
        return _think_state
    if _THINK_ENV in ("0", "false", "no", "off"):
        _think_state = (False, "RAMAN_LLM_THINK=0 — 강제 비활성")
        return _think_state
    forced = _THINK_ENV in ("1", "true", "yes", "on")
    caps, err = _show_capabilities()
    if caps is None:
        if forced:
            _think_state = (True, f"RAMAN_LLM_THINK=1 강제 — 능력 조회 실패({err})")
        else:
            _think_state = (False, f"능력 조회 실패({err}) → think 미사용")
    else:
        ok = "thinking" in caps
        if ok:
            _think_state = (True, f"{OLLAMA_MODEL} capabilities={caps} → thinking 지원")
        elif forced:
            _think_state = (True, f"RAMAN_LLM_THINK=1 강제 — 단 capabilities={caps} 에 "
                                  f"thinking 이 없어 Ollama 가 400 을 낼 수 있음")
        else:
            _think_state = (False, f"{OLLAMA_MODEL} capabilities={caps} → thinking 미지원. "
                                   f"이 모델은 '생각 토큰'을 따로 내보내지 않는다 — "
                                   f"[Gemma TEXT] 가 모델이 낸 유일한 추론 텍스트다")
    return _think_state


def _disable_think(reason: str) -> None:
    """think 를 켰다가 서버가 거부하면 영구 강등한다(같은 실행을 두 번 죽이지 않게)."""
    global _think_state
    _think_state = (False, f"런타임 강등 — {reason}")


# ══════════════════════════════════════════════════════════════════════════════
# 경로 결정
# ══════════════════════════════════════════════════════════════════════════════

def _sanitize(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z_-]", "-", str(text))[:64] or "nosession"


def parse_session(session_id: str) -> tuple[str, str]:
    """벤치 session_id 에서 (run_id, task) 를 뽑는다. 벤치 형식이 아니면 ("", "").

    형식: bench_<run_id>_<agent>_<task>_<hex6>
      예: bench_2026-08-06_AILA_AILA_T001_79933f → ("2026-08-06_AILA", "T001")
    run_id 안에 '_' 가 들어 있으므로(날짜_에이전트) 뒤에서부터 세 조각을 떼어낸다.
    """
    sid = str(session_id or "")
    if not sid.startswith("bench_"):
        return "", ""
    try:
        head = sid[len("bench_"):].rsplit("_", 1)[0]      # <run_id>_<agent>_<task>
        head, task = head.rsplit("_", 1)                   # <run_id>_<agent>, <task>
        run_id, _agent = head.rsplit("_", 1)
    except ValueError:
        return "", ""
    if not run_id or not task:
        return "", ""
    return run_id, task


def log_path(agent: str, session_id: str) -> Path:
    """이 세션의 로그 파일 경로."""
    run_id, task = parse_session(session_id)
    if run_id and task:
        return RESULTS_ROOT / run_id / f"{_sanitize(task)}.log"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return FALLBACK_ROOT / f"{agent}_{stamp}_{_sanitize(session_id)}.log"


# ══════════════════════════════════════════════════════════════════════════════
# 메시지 → 텍스트
# ══════════════════════════════════════════════════════════════════════════════

def _text(msg) -> str:
    """LangChain 메시지의 content 에서 순수 텍스트만. (에이전트의 _msg_text 와 동일 규칙 —
    이 모듈은 에이전트를 import 하지 않으므로 여기 다시 둔다. 양쪽이 서로를 import 하면
    순환이 되고, AILA/CoALA 독립성 원칙도 깨진다.)"""
    content = getattr(msg, "content", msg)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, str):
                parts.append(b)
            elif isinstance(b, dict):
                if b.get("type") == "text":
                    parts.append(b.get("text", ""))
                elif b.get("type") == "image_url":
                    parts.append("<image>")
        return "".join(parts)
    return str(content or "")


def _role(msg) -> str:
    return {"SystemMessage": "system", "HumanMessage": "human",
            "AIMessage": "ai", "ToolMessage": "tool"}.get(
        type(msg).__name__, type(msg).__name__.replace("Message", "").lower())


def _roster(messages) -> str:
    """프롬프트에 실린 메시지들을 한 줄 요약으로."""
    out = []
    for m in messages:
        piece = f"{_role(m)}({len(_text(m))}ch)"
        tcs = getattr(m, "tool_calls", None)
        if tcs:
            piece += "[" + ",".join(str(c.get("name")) for c in tcs) + "]"
        out.append(piece)
    return " · ".join(out)


def _clip(s: str, limit: Optional[int] = None) -> str:
    lim = _MAX_CHARS if limit is None else limit
    s = str(s or "")
    return s if len(s) <= lim else s[:lim] + f"\n… <{len(s) - lim}자 생략>"


def _jd(obj: Any, limit: Optional[int] = None) -> str:
    try:
        return _clip(json.dumps(obj, ensure_ascii=False, default=str), limit)
    except Exception:
        return _clip(str(obj), limit)


def _ns(v) -> float:
    """Ollama 의 나노초 값 → 초."""
    try:
        return float(v) / 1e9
    except (TypeError, ValueError):
        return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 로거
# ══════════════════════════════════════════════════════════════════════════════

class _Null:
    """로깅이 꺼졌을 때의 무동작 대역. invoke 만 실제로 일한다."""

    path = None
    enabled = False

    def invoke(self, messages, llm=None, stage="", step=0, note=""):
        return llm.invoke(messages)

    def __getattr__(self, _name):
        return lambda *a, **k: None


# 로거를 못 받은 헬퍼가 `rlog = rlog or reason_log.NULL` 로 쓰는 공용 무동작 대역.
NULL = _Null()


class ReasonLog:
    """세션 하나의 추론 로그. 매 기록마다 flush 하므로 Ctrl+C 로 죽어도 남는다."""

    enabled = True

    def __init__(self, agent: str, session_id: str, question: str):
        self.agent = agent
        self.session_id = session_id or ""
        self.path = log_path(agent, self.session_id)
        self._t0 = time.time()
        self._fh = None
        self._llm_calls = 0
        self._tool_calls = 0
        self._closed = False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            key = str(self.path)
            with _LOCK:
                first = key not in _OPENED
                _OPENED.add(key)
            self._fh = open(self.path, "w" if first else "a", encoding="utf-8")
        except Exception as e:
            print(f"[reason_log] 파일 열기 실패({self.path}): {type(e).__name__}: {e}",
                  file=sys.stderr)
            return
        self._header(question)

    # ── 기본 기록 ────────────────────────────────────────────────────────────
    def _w(self, s: str) -> None:
        if self._fh is None:
            return
        try:
            self._fh.write(s)
            self._fh.flush()
        except Exception:
            self._fh = None      # 한 번 깨지면 이후는 조용히 포기

    def rec(self, tag: str, head: str = "", body: str = "") -> None:
        """`[HH:MM:SS.mmm] [tag] head` 한 줄 + 들여쓴 본문."""
        t = time.time()
        stamp = time.strftime("%H:%M:%S", time.localtime(t)) + f".{int(t % 1 * 1000):03d}"
        line = f"[{stamp}] [{tag}]" + (f" {head}" if head else "") + "\n"
        if body:
            line += "".join(f"    {ln}\n" for ln in str(body).splitlines())
        self._w(line)

    def _header(self, question: str) -> None:
        think_on, why = think_capability()
        run_id, task = parse_session(self.session_id)
        bar = "═" * 78
        self._w(
            f"{bar}\n"
            f" {self.agent}"
            + (f" · {task}" if task else "")
            + f" · {OLLAMA_MODEL} @ {OLLAMA_HOST}\n"
            f" session : {self.session_id}\n"
            f" started : {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f" thinking: {'ON' if think_on else 'OFF'} — {why}\n"
            f"{bar}\n\n"
        )
        self.rec("QUESTION", "", _clip(question, 4000))

    # ── LLM 호출 ─────────────────────────────────────────────────────────────
    def invoke(self, messages, llm=None, stage: str = "LLM", step: int = 0, note: str = ""):
        """llm.invoke 를 감싸 프롬프트·응답·토큰통계를 남기고 AIMessage 를 그대로 돌려준다.

        think 가 켜져 있으면 reasoning=True 로 호출한다(생각 토큰을 따로 받기 위해).
        모델이 think 를 지원하지 않으면 Ollama 가 에러를 내므로, 첫 실패에서 think 를
        영구 강등하고 같은 호출을 한 번만 재시도한다 — 실험이 그것 때문에 죽지 않게.
        """
        head = (f"step {step} · " if step else "") + stage
        if _PROMPT_MODE == "full":
            self.rec("Gemma <- PROMPT", head,
                     "\n".join(f"--- {_role(m)} ---\n{_clip(_text(m))}" for m in messages))
        elif _PROMPT_MODE != "off":
            self.rec("Gemma <- PROMPT", f"{head} · {len(messages)} msgs",
                     _roster(messages))

        think_on, _ = think_capability()
        t0 = time.time()
        try:
            ai = llm.invoke(messages, reasoning=True) if think_on else llm.invoke(messages)
        except Exception as e:
            if think_on and "think" in f"{e}".lower():
                _disable_think(f"{type(e).__name__}: {e}")
                self.rec("Gemma THINK-OFF",
                         "Ollama 가 think 를 거부해 이후 호출에서 끕니다", f"{type(e).__name__}: {e}")
                ai = llm.invoke(messages)
            else:
                self.rec("Gemma ERROR", f"{head} · {time.time() - t0:.2f}s",
                         f"{type(e).__name__}: {e}")
                raise
        elapsed = time.time() - t0
        self._llm_calls += 1
        self._response(ai, step, elapsed, note)
        return ai

    def _response(self, ai, step: int, elapsed: float, note: str = "") -> None:
        pre = f"step {step}" if step else ""
        think = ""
        try:
            think = (getattr(ai, "additional_kwargs", None) or {}).get("reasoning_content") or ""
        except Exception:
            pass
        if think:
            self.rec("Gemma THINKING", f"{pre} · {len(think)}ch", _clip(think, 8000))

        text = _text(ai).strip()
        tcs = list(getattr(ai, "tool_calls", None) or [])
        if text:
            self.rec("Gemma TEXT", pre, _clip(text, 8000))
        elif not tcs:
            self.rec("Gemma TEXT", pre + " · (빈 응답 — 컨텍스트 초과 의심)")

        if tcs:
            body = "\n".join(
                f"{i + 1}) {c.get('name')}({_jd(c.get('args') or {}, 800)})"
                for i, c in enumerate(tcs))
            self.rec("Gemma TOOL_CALLS", f"{pre} · {len(tcs)}개", body)

        meta = getattr(ai, "response_metadata", None) or {}
        usage = getattr(ai, "usage_metadata", None) or {}
        pin = usage.get("input_tokens") or meta.get("prompt_eval_count")
        pout = usage.get("output_tokens") or meta.get("eval_count")
        ev = _ns(meta.get("eval_duration"))
        tps = (pout / ev) if (pout and ev > 0) else 0.0
        stats = (f"prompt {pin or '?'} tok · out {pout or '?'} tok · "
                 f"wall {elapsed:.2f}s · eval {ev:.2f}s"
                 + (f" ({tps:.1f} tok/s)" if tps else "")
                 + (f" · load {_ns(meta.get('load_duration')):.2f}s"
                    if meta.get("load_duration") else "")
                 + (f" · done={meta.get('done_reason')}" if meta.get("done_reason") else "")
                 + (f" · {note}" if note else ""))
        self.rec("Gemma STATS", stats)

    # ── ReAct / CoALA 단계 ───────────────────────────────────────────────────
    def reasoning(self, step: int, summary: str, body: str = "") -> None:
        self.rec("ReAct REASONING", f"step {step} · {summary}", body)

    def acting(self, step: int, name: str, args: dict) -> None:
        self._tool_calls += 1
        self.rec("ReAct ACTING", f"step {step} · {name}", _jd(args, 1200))

    def observation(self, step: int, name: str, result, ms: float) -> None:
        ok = result.get("ok", True) if isinstance(result, dict) else True
        self.rec("ReAct OBSERVATION",
                 f"step {step} · {name} · {'ok' if ok else 'FAIL'} · {ms:.0f} ms",
                 _jd(result))

    def phase(self, name: str, head: str, body: str = "") -> None:
        """CoALA 사이클 단계(PLAN / RETRIEVAL / EVALUATE / SELECT / EXECUTE / OBSERVE)."""
        self.rec(f"CoALA {name.upper()}", head, body)

    def executing(self, name: str, args: dict) -> None:
        self._tool_calls += 1
        self.rec("CoALA EXECUTE", name, _jd(args, 1200))

    def observed(self, name: str, result, ms: float, action: str = "") -> None:
        ok = result.get("ok", True) if isinstance(result, dict) else True
        self.rec("CoALA OBSERVE",
                 f"{name} · {'ok' if ok else 'FAIL'} · {ms:.0f} ms"
                 + (f" · {action}" if action else ""),
                 _jd(result))

    # ── 종료 ─────────────────────────────────────────────────────────────────
    def final(self, text: str, ctx: Optional[dict] = None) -> None:
        ctx = ctx or {}
        self.rec("FINAL",
                 f"{time.time() - self._t0:.2f}s · LLM {self._llm_calls}회 · "
                 f"도구 {self._tool_calls}회 · dose {float(ctx.get('dose', 0.0)):.2f} mJ",
                 _clip(text, 8000))
        self.close()

    def failed(self, detail: str) -> None:
        self.rec("ERROR",
                 f"{time.time() - self._t0:.2f}s · LLM {self._llm_calls}회 · "
                 f"도구 {self._tool_calls}회", _clip(detail, 4000))
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._w("═" * 78 + "\n\n")
        try:
            if self._fh is not None:
                self._fh.close()
        except Exception:
            pass
        self._fh = None


def open_turn(agent: str, question: str, session_id: str = ""):
    """이 턴의 로거를 연다. 로깅이 꺼져 있으면 무동작 대역을 돌려준다.

    session_id 를 안 주면 run_store 의 현재(스레드로컬) 세션을 쓴다 — 벤치 경로는
    server.py 가 run_stream 직전에 begin_session 을 부르므로 그것으로 충분하고,
    run_stream 의 시그니처를 건드리지 않아도 된다.
    """
    if not _ENABLED:
        return _Null()
    sid = session_id
    if not sid:
        try:
            from backend.agents import run_store
            sid = run_store.current().get("session_id", "")
        except Exception:
            sid = ""
    try:
        return ReasonLog(agent, sid, question)
    except Exception as e:      # noqa: BLE001 — 로깅이 실험을 깨선 안 된다
        print(f"[reason_log] 로거 생성 실패: {type(e).__name__}: {e}", file=sys.stderr)
        return _Null()


# ══════════════════════════════════════════════════════════════════════════════
# 진단 — `python -m backend.agents.reason_log`
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"Ollama   : {OLLAMA_HOST}")
    print(f"모델      : {OLLAMA_MODEL}")
    caps, err = _show_capabilities()
    if caps is None:
        print(f"capabilities 조회 실패: {err}")
        print("  → 장비 PC(Ollama 가 도는 곳)에서 실행해야 합니다.")
    else:
        print(f"capabilities: {caps}")
        if "thinking" in caps:
            print("  → 이 모델은 '생각 토큰'을 따로 내보냅니다. think=true 로 호출하면")
            print("     Ollama 가 message.thinking 을 채워 주고, 로그의 [Gemma THINKING] 에 찍힙니다.")
        else:
            print("  → 이 모델은 thinking 능력이 없습니다. think=true 를 보내면 Ollama 가")
            print("     400 'does not support thinking' 을 돌려줍니다. 따라서 [Gemma THINKING]")
            print("     줄은 나오지 않고, [Gemma TEXT](= ReAct 의 Thought)가 모델이 내놓는")
            print("     유일한 추론 텍스트입니다. 그 밖의 내부 연산(logit·attention)은")
            print("     Ollama HTTP API 자체에 표면이 없어 어떤 설정으로도 얻을 수 없습니다.")
    on, why = think_capability()
    print(f"\n이번 설정에서 think: {'ON' if on else 'OFF'} — {why}")
    print(f"로그 위치 예시: {log_path('AILA', 'bench_2026-08-06_AILA_AILA_T001_abc123')}")
