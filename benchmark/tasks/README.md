# 문항 파일

한 문항 = 한 파일. `benchmark/tasks/T062.py` 를 열면 그 문항의 문제·전제·정답 기준이 전부 있다.

## 파일이 지켜야 하는 계약

```python
TASK               # Task 인스턴스 (필수)
def setup(b): ...  # 측정 전에 만들어야 하는 상태 (없으면 생략)
def evaluate(b, run) -> list[Check]   # 이 목록이 그대로 점수가 된다 (필수)
```

러너가 시작할 때 143개를 전부 import 해서 계약을 검사한다. 어기면 **실행 전에** 멈춘다.

## 한 문항이 도는 순서

```
b.reset()          앞 문항이 무엇을 바꿔 놨든 전 장비를 기본값으로
setup(b)           이 문항의 전제를 만든다
b.state()          시작 상태 (사전 세팅을 마친 뒤)
b.run(task)        에이전트 실행
b.state()          종료 상태
evaluate(b, run)   판정 목록
b.teardown(); b.reset()
```

## 점수

```
문항 점수 = 배점 × Σ(check.score × weight) / Σ(weight)
```

`chk.blocked(...)` 가 붙은 판정은 분모에서 **빠진다**. 장비 파수축이 문항 구간을 못 덮는
경우가 그렇다 — 설정 문제를 에이전트 실력으로 기록하면 안 된다.

## 자주 쓰는 판정

```python
chk.state("실행 후 x", after, "x", 35.0, tol=MM)     # 장비가 그 상태가 됐는가
chk.delta("x 변화", before, after, "x", 0.1)          # 상대 이동
chk.unchanged("건드리지 않았는가", before, after, ["is_on"])
chk.called(run, "move_stage", times=1)               # times=0 은 '부르면 오답'(비중 2배)
chk.arg(run, "acquire_spectrum", "power", 50)
chk.arg_set(run, "acquire_spectrum", "power", [20, 40, 60])
chk.order(run, "preview_grid_scan", "run_grid_scan")
chk.keywords(run, ["refus", "거부", "approval"])
chk.reported(run, "snr", want, rel=0.05)             # answer JSON 우선, 없으면 본문
chk.blocked("SNR 잡음창", "축이 이 구간을 안 덮습니다")
```

키워드에 `"0"`, `"?"`, `"100"` 처럼 **아무 텍스트에나 들어가는 토큰을 넣으면 안 된다.**
그 판정은 무의미해지고, 아무 말이나 한 답이 통과한다.

## 저장 파일 재계산 (사후 GT)

절대값은 시료·정렬에 따라 달라져 미리 적을 수 없지만, **에이전트 자신이 저장한 파일의
함수**는 결정적이다. `benchmark/bench/spectra.py` 가 규약을 갖고 있다.

```python
saved = run.spectra()                    # [(경로, x, y), ...] 저장 순
snr   = sp.snr(x, y)                     # T050 정의. 구간이 축 밖이면 None
peaks = sp.peaks(x, y)                   # prominence = 세기 범위의 5%
sp.cosine(a, b), sp.rsd_percent(vals), sp.saturated_count(y)
```

축이 구간을 못 덮으면 `sp.snr` 은 **0 이 아니라 None** 을 준다. 그럴 수 있는 문항은
`Task.windows` 에 필요한 구간을 적어 두면 러너가 실행 전에 알리고 채점에서 뺀다.

## 고칠 때

그 문항 파일만 고친다. 문항 파일이 정본이다. 이걸 만들어 낸 일회성 이식 도구는 삭제했다 —
다시 돌리면 손으로 고친 채점 로직을 덮어쓰기 때문이다.
