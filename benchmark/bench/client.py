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
        # errors  : 답을 **받지 못하게 만든** 사유. 하나라도 있으면 그 문항은 채점이
        #           성립하지 않으므로 result="error" 가 되고 해결률 분모에서 빠진다.
        # warnings : 기록해 둘 값어치는 있지만 채점에는 영향이 없는 것(예: 리셋이
        #           비치명적으로 일부 실패). 예전에는 이 둘이 한 목록이라, 리셋 경고 한
        #           줄이 붙었다는 이유로 **모든 판정을 통과한 실행이 error** 가 되어
        #           분모에서 빠졌다 — 해결률이 장비 잡음에 흔들렸다.
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.elapsed_s: float = 0.0
        # 문항별 시간 상한에 걸려 중간에 끊겼는가. 채점은 그대로 한다 — 그때까지 한
        # 일로 판정할 뿐이고, 판정 항목이 늘거나 줄지 않는다(상한 없이 돈 예전 결과와
        # checks_total 이 같아야 비교가 성립한다).
        self.timed_out: bool = False
        # 중단을 요청했는데 유예 안에 **안 멈춘** 경우. timed_out 과 달리 이건
        # 그 문항 하나의 문제가 아니다 — 에이전트가 장비를 쥔 채 계속 돌고 있다는
        # 뜻이라 뒤에 오는 문항이 전부 그 위에서 돈다. 러너가 실행을 세운다.
        self.abandoned: bool = False

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

        확인용 get_stage_position 은 세지 않는다 — 그건 '들른 자리'가 아니라 '본 것'이다.

        [좌표가 어디 실려 오는가 — 2026-08-03]
        raman_tools.move_stage 는 {"ok": True, "position": {"x":…, "y":…, "z":…}} 를
        돌려준다. 여기서는 최상위 r["x"] 만 찾고 있었고, 그래서 이 함수는 **어떤 실행에서도
        빈 목록**이었다. 좌표로 채점하는 T029·T033·T066·T074 는 에이전트가 무엇을 하든
        0 점이 확정돼 있었다(실제로 T029 는 6 개 좌표를 정확히 찍고도 0 점).
        중첩·평면 두 모양을 다 받는다 — 도구가 어느 쪽으로 바뀌어도 안 깨지게.
        """
        out = []
        for c in self.calls:
            if c.get("name") not in ("move_stage", "move_stage_relative", "move_to_pixel"):
                continue
            r = c.get("result")
            if not isinstance(r, dict):
                continue
            p = r.get("position") if isinstance(r.get("position"), dict) else r
            if _is_num(p.get("x")) and _is_num(p.get("y")):
                out.append((float(p["x"]), float(p["y"]), _f(p.get("z"))))
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

    def acquisitions(self) -> list[dict]:
        """측정 하나하나를 **그때의 실효 설정과 함께** — 호출 순서대로.

        [왜 저장 순서로 짝지으면 안 되는가 — 2026-08-06]
        예전에는 문항들이 run.spectra() 의 저장 순서가 곧 측정 순서라고 가정하고
        `snrs[:3]` 을 20/40/60 % 에, `snrs[1]-snrs[0]` 을 2.0 s/0.5 s 에 매핑했다.
        run.spectra() 는 **모든 CSV 산출물**을 저장 순으로 주므로, 중간 산출물(배경보정본·
        합본)을 하나만 저장해도 매핑이 통째로 어긋난다. 순서는 프롬프트가 요구한 바도
        아니라, 2.0 s 를 먼저 잰 정답 실행이 부호가 뒤집혀 떨어졌다.

        [설정은 인자가 아니라 되읽기로 본다]
        acquire_spectrum 은 결과에 그 측정에 **실제로 걸린** exposure_time /
        laser_power_pct 를 싣는다. 인자로 넘겼든 set_ccd_exposure·set_laser_power 로
        미리 걸어 뒀든 같은 값이 나오므로, 되읽기를 보면 도구 선택을 벌하지 않는다
        (문항들의 정답 기준이 원래 "readback" 이라고 적고 있던 그대로다).
        그리고 그 결과에는 저장된 CSV 경로도 함께 실려 있어 (설정 ↔ 스펙트럼) 짝이
        추측 없이 정해진다.
        """
        from bench import spectra
        out = []
        for c in self.calls:
            if c.get("name") != "acquire_spectrum":
                continue
            r = c.get("result")
            if not isinstance(r, dict) or r.get("ok") is False:
                continue
            csv = ((r.get("saved") or {}).get("files") or {}).get("csv")
            xy = spectra.read_xy(Path(csv)) if csv else None
            out.append({"exposure": _f(r.get("exposure_time")),
                        "power": _f(r.get("laser_power_pct")),
                        "shutter": r.get("shutter"),
                        "path": csv,
                        "x": xy[0] if xy else None,
                        "y": xy[1] if xy else None})
        return out

    def as_dict(self) -> dict:
        return {"task": self.task_id, "agent": self.agent, "session_id": self.session_id,
                "prompt": self.prompt, "elapsed_s": round(self.elapsed_s, 2),
                "tool_calls": self.calls, "final_text": self.text, "answer": self.answer,
                "artifacts": self.artifacts, "state_before": self.state_before,
                "state_after": self.state_after, "errors": self.errors,
                "warnings": self.warnings}


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
    def run(self, task, run_id="", timeout_s=None, grace_s=300.0) -> Run:
        """문항을 실행하고 Run 을 돌려준다. 예외를 던지지 않는다.

        timeout_s 를 주면 그 초를 넘긴 실행을 끊는다(None·0 이면 무제한 — 상한이
        없던 예전 실행과 같다). 컷은 세 단계다:

          1. 상한을 넘긴 것을 **이벤트를 받은 자리에서** 안다. 도구 호출 하나가
             진행 중이면 그것이 끝나야 오므로, 실제 컷은 '상한 + 마지막 도구 1회'다.
          2. /api/bench/cancel 로 서버의 에이전트 루프에 중단을 요청한다. 스트림만
             끊으면 에이전트는 executor 스레드에서 계속 돌며 장비를 쥐고 있어
             **다음 문항이 그 잔재 위에서 채점된다** — 이쪽이 훨씬 나쁘다.
          3. grace_s 동안 스트림이 실제로 닫히는지 본다. 안 닫히면 run.abandoned 를
             세우고, 러너가 거기서 실행 전체를 세운다 — 폭주하는 에이전트가 장비를
             쥔 채로 남은 문항이 도는 것보다 멈추는 편이 낫다.

        읽기 타임아웃도 함께 줄인다 — 이벤트가 아예 안 오는(완전히 멈춘) 실행은
        1 번이 영영 안 오기 때문이다.

        [grace_s 를 300 초로 잡은 근거 — 2026-08-04]
        중단은 이벤트 경계에서만 걸리므로, 유예는 **도구 호출 1 회보다 넉넉해야**
        한다. 짧으면 멀쩡한 문항을 '안 멈췄다'고 오판해 실행을 세운다. 그날 AILA
        143 문항에서 관측된 최장 단일 호출은 T088 의 약 101 초였다(1 호출 101 초).
        그 3 배이자 reset() 자체 타임아웃과 같은 값으로 맞췄다.
        """
        import time
        sid = f"bench_{run_id or self.run_id or 'adhoc'}_{self.agent}_{task.id}_{uuid.uuid4().hex[:6]}"
        r = Run(task.id, task.prompt, self.agent, sid)
        message = task.prompt + output_contract(task)

        cap = float(timeout_s) if timeout_s else 0.0
        # 무응답 백스톱. 상한이 있으면 '상한+유예'만큼 기다렸다 읽기 타임아웃으로 끊는다.
        read_to = (cap + grace_s) if cap > 0 else self.timeout

        t0 = time.time()
        cut_at = None                      # 중단을 요청한 시각(아직이면 None)
        body = {"agent": self.agent, "message": message, "task": task.id,
                "session_id": sid, "enforce_grid_gate": task.enforce_grid_gate}
        try:
            resp = self._rq.post(f"{self.base}/api/bench/stream", json=body, stream=True,
                                 timeout=(10.0, read_to),
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

                    now = time.time()
                    if cut_at is None:
                        if cap > 0 and now - t0 > cap:
                            r.timed_out = True
                            cut_at = now
                            self.cancel(sid)
                    elif now - cut_at > grace_s:
                        r.abandoned = True
                        r.warnings.append(
                            f"the agent did not stop within {grace_s:.0f}s of the time-limit cut "
                            f"- it may still be holding the instrument, so every task after this "
                            f"one is suspect")
                        break
                resp.close()
        except Exception as e:
            if cap > 0 and time.time() - t0 > cap:
                # 이벤트가 끊긴 채로 상한을 넘겼다. 중단을 요청은 하지만 **멈췄는지
                # 확인할 방법이 없다**(확인에 쓸 스트림이 이미 죽었다). 확인 못 한
                # 것은 안 멈춘 것으로 친다 — 장비를 쥔 채 도는 에이전트 위에서
                # 남은 문항을 돌리는 쪽이 훨씬 나쁘다.
                r.timed_out = True
                r.abandoned = True
                self.cancel(sid)
                r.errors.append(f"no event for {read_to:.0f}s after the {cap:.0f}s time limit "
                                f"- cut ({type(e).__name__})")
            else:
                r.errors.append(f"stream failed: {type(e).__name__}: {e}")
        r.elapsed_s = time.time() - t0
        r.answer = _parse_answer(r.text)
        r.artifacts = self.artifacts(sid)
        return r

    def cancel(self, session_id: str) -> dict:
        """진행 중인 실행에 중단을 요청한다(협조적 — 다음 이벤트 경계에서 멈춘다)."""
        return self._post("/api/bench/cancel", {"session_id": session_id}, timeout=30.0)

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
