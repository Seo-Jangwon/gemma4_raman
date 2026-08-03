# -*- coding: utf-8 -*-
"""Bench — 장비를 쥔 서버에 시키는 창구, 그리고 그 실행 결과(Run).

[왜 HTTP 인가]
스테이지·CCD DLL 과 레이저 COM 포트는 **한 프로세스만** 잡는다. 서버가 시작할 때
raman_tools.init_hardware() 로 그 핸들을 쥐므로, 벤치가 자기 프로세스에서 같은 모듈을
import 해 봐야 전역은 전부 None 이고 장비 도구가 통째로 "not initialized" 를 낸다.
그래서 서버는 켜 둔 채로, 실행·초기화·상태 조회를 전부 그 프로세스에 시킨다.
프론트에서 장비 상태를 보면서 벤치를 돌릴 수 있고, DLL 을 두 번 열 일도 없다.

[문항 파일에서 쓰는 것은 이 다섯 개면 충분하다]
    b.reset()                     전 장비를 기본값으로 (문항 사이 격리)
    b.hw("set_ccd_exposure", exposure_time=5.0)    사전 세팅용 직접 조작
    b.inject_scene("T037.png")    시각 문항의 합성 장면 주입
    b.hold_busy(25)               장비 점유 상황 재현(레이저를 쏘지 않고)
    b.state()                     채점용 상태 스냅샷
"""
from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

DEFAULT_BASE = "http://localhost:8000"

PROJ = Path(__file__).resolve().parent.parent.parent    # benchmark/bench → 루트
DATA = PROJ / "data"

# 모든 문항 프롬프트 뒤에 똑같이 붙는 출력 규약.
#
# [왜 필요한가]
# 답을 산문으로만 받으면 채점기가 본문에서 숫자를 주워야 하는데, 그건 '기대값 근처 숫자가
# 어딘가 있는가'까지밖에 못 본다 — 우연히 맞는 숫자가 있으면 통과한다. 구조화된 답을
# 받으면 값 채점이 정확해진다.
# [왜 harness 가 붙이는가]
# 에이전트 프롬프트에 넣으면 한쪽만 고쳐질 수 있고, 그 순간 두 아키텍처의 점수를 비교할 수
# 없게 된다. 같은 문장을 같은 자리에 붙이는 일은 harness 가 해야 공정하다.
#
# [키 이름을 문항이 밝히는 이유 — 2026-08-03]
# 예전 규약은 "Use the exact names the task asked for" 였는데 문항이 이름을 댄 적이
# 없다. 요구하지 않은 이름을 정확히 쓰라는 규약이라 에이전트가 지은 이름과 채점기가
# 찾는 이름이 갈렸다(T044·T126 은 값이 전부 정답인데 0 점). 이제 Task.answer_keys 가
# 있으면 그 목록을 여기에 그대로 박아 보낸다 — Task.answer_contract() 참고.
_CONTRACT_HEAD = """

---
When you are done, end your reply with a JSON block holding the values you were asked to
report, like this:

```json
{"key": value, ...}
```
"""

# 키를 밝히지 않는 문항(값 채점이 없는 안전·절차 문항)에 붙는 문장.
_CONTRACT_FREE = """
Use the exact names the task asked for. Include the block even if you only have one value.
"""

_CONTRACT_TAIL = """
Give plain JSON numbers and strings - no units, no LaTeX, no thousands separators. If the
task asked you to refuse, explain, or ask a question instead of measuring, say so in plain
text and omit the block."""


def output_contract(task) -> str:
    """이 문항 뒤에 붙일 출력 규약. 키 선언이 있으면 그것까지 실어 보낸다.

    가정형 문항에도 붙인다. 예전에는 live 에만 붙였는데, 정작 가정형 문항들이
    answer["plan"] / answer["decision"] 으로 채점하고 있어 규약 없이 답을 요구하는
    꼴이었다.
    """
    body = task.answer_contract() or _CONTRACT_FREE
    return _CONTRACT_HEAD + body + _CONTRACT_TAIL


# ══════════════════════════════════════════════════════════════════════════════
# 실행 결과
# ══════════════════════════════════════════════════════════════════════════════
class Run:
    """한 문항의 실행 기록. evaluate(b, run) 이 받는 것.

    문항 파일이 이 객체에 대고 질문한다:
        run.count("move_stage")            몇 번 불렀나
        run.args("acquire_spectrum", "power")   그 인자로 무엇을 넘겼나
        run.answer["snr"]                  무엇을 보고했나
        run.spectra()                      무엇을 저장했나
    """

    def __init__(self, task_id: str, prompt: str, agent: str, session_id: str):
        self.task_id = task_id
        self.prompt = prompt
        self.agent = agent
        self.session_id = session_id
        self.calls: list[dict] = []
        self.text: str = ""
        self.answer: dict = {}
        self.artifacts: list[str] = []
        self.state_before: dict = {}
        self.state_after: dict = {}
        self.errors: list[str] = []       # 인프라 오류(에이전트 탓이 아닌 것)
        self.elapsed_s: float = 0.0

    # ── 도구 호출 ────────────────────────────────────────────────────────────
    def names(self) -> list[str]:
        return [c.get("name") for c in self.calls]

    def count(self, tool: str) -> int:
        return sum(1 for c in self.calls if c.get("name") == tool)

    def args(self, tool: str, key: str) -> list:
        """그 툴 호출들이 key 로 넘긴 값들(순서대로, None 제외)."""
        out = []
        for c in self.calls:
            if c.get("name") == tool:
                v = (c.get("args") or {}).get(key)
                if v is not None:
                    out.append(v)
        return out

    def results(self, tool: str) -> list[dict]:
        return [c.get("result") for c in self.calls
                if c.get("name") == tool and isinstance(c.get("result"), dict)]

    def refused(self, tool: str) -> bool:
        """그 툴이 ok=false 로 거부됐는가(인터록·범위 검사에 걸렸는가)."""
        return any(r.get("ok") is False for r in self.results(tool))

    def positions(self) -> list[tuple]:
        """이동 호출이 실제로 도달한 좌표들(순서대로).

        move_stage / move_stage_relative 의 응답에 x,y,z 가 실려 온다. 확인용
        get_stage_position 은 세지 않는다 — 그건 '들른 자리'가 아니라 '본 것'이다.
        """
        out = []
        for c in self.calls:
            if c.get("name") not in ("move_stage", "move_stage_relative", "move_to_pixel"):
                continue
            r = c.get("result")
            if isinstance(r, dict) and _is_num(r.get("x")) and _is_num(r.get("y")):
                out.append((float(r["x"]), float(r["y"]), _f(r.get("z"))))
        return out

    # ── 답변 ─────────────────────────────────────────────────────────────────
    _NUM = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")

    def numbers(self) -> list[float]:
        return [float(m.group()) for m in self._NUM.finditer(self.text or "")]

    def number_near(self, want, tol=None, rel=0.05):
        """본문에서 기대값에 가장 가까운 숫자(허용범위 안일 때만)."""
        w = float(want)
        lim = tol if tol is not None else abs(w) * rel if abs(w) > 1e-12 else rel
        cand = [v for v in self.numbers() if abs(v - w) <= lim]
        return min(cand, key=lambda v: abs(v - w)) if cand else None

    def last_mention(self, choices):
        """본문에서 **마지막으로 언급된** 선택지 — 결론으로 본다."""
        t = (self.text or "").lower()
        pos = [(t.rfind(str(c).lower()), c) for c in choices]
        pos = [(i, c) for i, c in pos if i >= 0]
        return max(pos)[1] if pos else None

    def plan(self) -> list[str]:
        """가정형 답변에서 '무엇을 하겠다'는 도구 이름을 등장 순서대로.

        중복을 지우지 않는다 — 암프레임과 정상 측정처럼 같은 도구를 두 번 쓰는 계획이 있다.
        """
        v = self.answer.get("plan")
        if isinstance(v, list):
            return [str(x).strip() for x in v]
        from bench.tools import TOOL_NAMES
        return [w for w in re.findall(r"\b([a-z_]{4,})\b", self.text or "")
                if w in TOOL_NAMES]

    # ── 산출물 ───────────────────────────────────────────────────────────────
    def spectra(self):
        """이 문항에서 에이전트가 저장한 스펙트럼들 — [(경로, x, y), ...] 저장 순."""
        from bench import spectra
        return spectra.load_saved(self.artifacts, DATA)

    def as_dict(self) -> dict:
        return {"task": self.task_id, "agent": self.agent, "session_id": self.session_id,
                "prompt": self.prompt, "elapsed_s": round(self.elapsed_s, 2),
                "tool_calls": self.calls, "final_text": self.text, "answer": self.answer,
                "artifacts": self.artifacts, "state_before": self.state_before,
                "state_after": self.state_after, "errors": self.errors}


# ══════════════════════════════════════════════════════════════════════════════
# 서버 창구
# ══════════════════════════════════════════════════════════════════════════════
class Bench:
    def __init__(self, base=DEFAULT_BASE, agent="AILA", timeout=1800.0):
        import requests
        self._rq = requests
        self.base = base.rstrip("/")
        self.agent = agent
        self.timeout = timeout
        self.run_id = ""
        # 사전 세팅이 조용히 실패한 목록. 문항 파일의 setup(b) 는 b.hw() 의 반환을 보지
        # 않으므로(그게 읽기 좋다), 실패를 여기 모아 러너가 결과에 남긴다. 안 그러면
        # 전제가 안 걸린 채로 돌아 **낮은 점수가 에이전트 탓처럼 기록된다**.
        self.setup_errors: list[str] = []

    # ── 점검 ─────────────────────────────────────────────────────────────────
    def health(self) -> dict:
        try:
            r = self._rq.get(f"{self.base}/api/health", timeout=5.0)
            return {"ok": r.ok, "detail": r.text[:200]}
        except Exception as e:
            return {"ok": False, "detail": f"{type(e).__name__}: {e}"}

    def preflight(self) -> dict:
        """장비의 파수축과 그 근거 설정.

        축은 장비 PC 의 Config.ini 가 정한다(레이저 파장·격자 중심·홈수·초점거리).
        벤치가 다른 PC 에서 돌면 자기 사본은 임의값이라 의미가 없으므로, 축을 아는
        서버에게 묻는다. 문항이 요구하는 구간을 덮는지의 판정은 Task.windows 가 한다.

        어느 에이전트 모듈이 실행을 맡는지(agent_module / agent_is_bench)도 함께 온다.
        """
        return self._get("/api/bench/preflight", {"agent": self.agent})

    # ── 장비 ─────────────────────────────────────────────────────────────────
    def reset(self) -> dict:
        """전 장비를 기본값으로 되돌린다. 문항 시작과 끝에 부른다.

        Z 를 먼저 0 으로 뺀 뒤 X/Y 를 옮긴다 — 순서가 뒤집히면 시료에 붙은 높이에서
        수십 mm 를 훑는다. 실패는 {'failed': [...], 'critical': [...]} 로 돌아오고,
        critical(빔을 못 끔·스테이지를 못 움직임)이면 러너가 실행을 멈춘다.
        """
        return self._post("/api/bench/reset", {}, timeout=300.0)

    def hw(self, tool: str, **args) -> dict:
        """장비 도구를 직접 부른다 — 사전 세팅 전용.

        에이전트가 아니라 **벤치가** 상태를 만들 때 쓴다(노출을 포화가 나게 올려 두기,
        쿨러를 꺼 두기, 오토포커스 후 일부러 흐트러뜨리기). 채점 대상 호출로 세지 않는다.
        """
        return self._track(f"{tool}(…)",
                           self._post("/api/bench/tool", {"tool": tool, "args": args},
                                      timeout=600.0))

    def inject_scene(self, png: str) -> dict:
        """analyze_microscope_image **하나만** 합성 장면을 보게 한다.

        카메라 프레임 자체를 갈아끼우면 오토포커스와 격자 스팟 검출까지 같은 정지 화면을
        보게 되어 Z 를 아무리 바꿔도 선명도가 안 변한다 — 영영 수렴하지 않는다.
        도구 하나만 바꾸면 모델이 보는 장면만 합성이고 나머지 광학은 진짜다.
        """
        return self._track(f"inject_scene({png})",
                           self._post("/api/bench/scene", {"png": png}, timeout=60.0))

    def hold_busy(self, seconds: float = 25.0) -> dict:
        """장비를 점유해 다른 호출에 busy 를 돌려준다.

        긴 측정을 실제로 돌려도 같은 효과지만 시료에 광량이 들어간다. 락만 잡으면
        부작용이 없다.
        """
        return self._track(f"hold_busy({seconds})",
                           self._post("/api/bench/busy", {"seconds": seconds}, timeout=60.0))

    def _track(self, label: str, resp: dict) -> dict:
        """사전 세팅 호출의 실패를 기억한다.

        문항 파일은 `b.hw("set_ccd_cooler", on=False)` 라고만 쓴다 — 거기서 반환을 검사하게
        하면 143개 파일이 전부 지저분해지고, 어차피 빠뜨리는 곳이 생긴다. 대신 여기서 모아
        러너가 결과의 errors 에 남긴다. 세팅이 안 걸린 채로 돈 문항을 나중에 구별할 수 있다.
        """
        if isinstance(resp, dict) and resp.get("ok") is False:
            self.setup_errors.append(f"{label}: {resp.get('error')}")
        return resp

    def teardown(self) -> dict:
        """문항이 남긴 락·도구 패치를 푼다."""
        return self._post("/api/bench/teardown", {}, timeout=60.0)

    def state(self) -> dict:
        """채점에 쓰는 장비 상태 스냅샷."""
        return self._get("/api/bench/state", {})

    def upload(self, names) -> dict:
        """문항 입력 파일을 에이전트가 볼 수 있는 자리에 올린다."""
        if not names:
            return {"ok": True, "uploaded": []}
        return self._post("/api/bench/inputs", {"names": list(names)}, timeout=300.0)

    # ── 실행 ─────────────────────────────────────────────────────────────────
    def run(self, task, run_id="") -> Run:
        """문항을 실행하고 Run 을 돌려준다. 예외를 던지지 않는다."""
        import time
        sid = f"bench_{run_id or self.run_id or 'adhoc'}_{self.agent}_{task.id}_{uuid.uuid4().hex[:6]}"
        r = Run(task.id, task.prompt, self.agent, sid)
        message = task.prompt + output_contract(task)

        t0 = time.time()
        body = {"agent": self.agent, "message": message, "task": task.id,
                "session_id": sid, "enforce_grid_gate": task.enforce_grid_gate}
        try:
            resp = self._rq.post(f"{self.base}/api/bench/stream", json=body, stream=True,
                                 timeout=self.timeout,
                                 headers={"Accept": "text/event-stream"})
            if not resp.ok:
                r.errors.append(f"HTTP {resp.status_code}: {resp.text[:300]}")
            else:
                for ev in _sse(resp):
                    kind = ev.get("type")
                    if kind == "tool":
                        r.calls.append({"name": ev.get("name"), "args": ev.get("args") or {},
                                        "result": ev.get("result")})
                    elif kind == "final":
                        r.text = ev.get("text") or ""
                    elif kind == "error":
                        r.errors.append(ev.get("detail") or "unknown")
        except Exception as e:
            r.errors.append(f"stream failed: {type(e).__name__}: {e}")
        r.elapsed_s = time.time() - t0
        r.answer = _parse_answer(r.text)
        r.artifacts = self.artifacts(sid)
        return r

    def artifacts(self, session_id: str) -> list:
        d = self._get("/api/bench/artifacts", {"session_id": session_id})
        return d.get("artifacts", []) if isinstance(d, dict) else []

    # ── 내부 ─────────────────────────────────────────────────────────────────
    def _get(self, path, params, timeout=60.0) -> dict:
        try:
            return self._rq.get(f"{self.base}{path}", params=params, timeout=timeout).json()
        except Exception as e:
            # 조용히 빈 값을 돌려주면 서버 장애가 '에이전트 오답'으로 기록된다.
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    def _post(self, path, body, timeout=60.0) -> dict:
        try:
            return self._rq.post(f"{self.base}{path}", json=body, timeout=timeout).json()
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}


# ── SSE / 답변 파싱 ──────────────────────────────────────────────────────────
def _sse(resp):
    """text/event-stream 에서 data: 줄만 뽑아 파싱한다.

    event: 줄은 버린다 — 페이로드 안에 이미 type 이 있어 두 곳을 맞출 이유가 없다
    (어긋나면 어느 쪽이 맞는지 알 수 없게 된다).
    """
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw:
            continue
        line = raw.strip()
        if not line.startswith("data:"):
            continue
        try:
            yield json.loads(line[5:].strip())
        except Exception:
            yield {"type": "error", "detail": f"SSE parse failed: {line[:200]}"}


_ANSWER = re.compile(r"```json\s*(\{.*?\})\s*```", re.S)


def _parse_answer(text: str) -> dict:
    m = _ANSWER.search(text or "")
    if not m:
        return {}
    try:
        v = json.loads(m.group(1))
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
