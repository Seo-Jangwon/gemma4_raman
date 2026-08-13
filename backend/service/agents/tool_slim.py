# -*- coding: utf-8 -*-
"""
[역할] 도구 결과 중 **무엇을 모델 컨텍스트에 싣는가**의 단일 출처.

원본 숫자 배열(스펙트럼 세기, 파장축 등)은 이미 디스크에 저장되고 run_analysis 가
파일에서 직접 읽으므로, 대화에 다시 실을 이유가 없다. 여기서는 판단에 필요한
요약(길이·최대·합계·경로)만 남기고 배열을 버린다.

[왜 별도 모듈인가 — 2026-08-09]
같은 _slim() 과 _SLIM_KEEP_KEYS 가 네 파일에 한 글자도 다르지 않게 복사돼 있었다:

    single_agent_AILA.py:399          single_agent_AILA_bench.py:399
    single_agent_CoALA.py:1004        single_agent_CoALA_bench.py:1004

CoALA 쪽 주석이 이미 "AILA의 _SLIM_KEEP_KEYS와 동일해야 비교가 공정하다"고 적고
있었다 — 즉 네 사본이 갈라지면 안 된다는 걸 알면서 사본으로 두고 있었다. 갈라지는
대상이 하필 '두 에이전트가 같은 관측을 받는가'라서, 어긋나면 AILA↔CoALA 비교의
독립변수가 오케스트레이션 하나가 아니게 된다.

[왜 에이전트 서로를 import 하지 않는가 — 그 원칙과의 관계]
AILA/CoALA 는 서로를 import 하지 않는다(각 파일 머리말). 이 모듈은 두 에이전트의
공통 상위 의존이지 서로에 대한 결합이 아니므로 그 원칙을 깨지 않는다 —
backend/util/safety_limits.py 가 조사량 상한에 대해 같은 논리로 만들어졌다.
backend.config(Config.ini)에도 장비 SDK 에도 의존하지 않으므로 어디서든 import 된다.

[왜 재귀가 필요한가 — 2026-08-09 N07 실패]
옛 _slim 은 **최상위 키 한 겹만** 보고 '길이 32 초과 리스트'를 버렸다. Single 모드는
원본이 최상위 data/raman_shift 라서 잘 걸렸지만(40,948자 → 157자), Kinetic 모드는
구조가 다르다:

    {"mode": "kinetic", "frames": [ {"intensity": [2000점], ...}, ... ]}
                         └ 길이 5. 32 이하라 통과 → 안쪽 배열은 손도 안 댐

그 결과 5프레임 측정 하나가 224,078자(≈22만 토큰)로 대화에 실렸다. num_ctx 100,000
을 2.2배 넘겨 Ollama 가 프롬프트를 조용히 잘라냈고(경고도 done_reason 표시도 없다),
모델은 자기가 방금 측정한 데이터를 못 본 채로 답해야 했다. 출력이 불안정해지면서
본문도 도구 호출도 없는 빈 응답이 확률적으로 나왔다("The model returned an empty
reply" 의 실제 원인 — 재현 12회 중 1회, 깨끗한 맥락에서는 24회 중 0회).

반대로 프레임이 33개를 넘으면 이번엔 frames 리스트 자체가 길이 필터에 걸려 **통째로**
사라졌다(모델은 num_frames 만 받고 데이터는 한 점도 못 받는다). 즉 옛 필터는 프레임
수를 재면서 정작 무거운 축인 픽셀 수를 못 봐서, 프레임이 적으면 폭발하고 많으면
소실하는 양쪽 고장을 갖고 있었다.

[고친 규칙]
dict 들의 리스트(frames, files 같은 '레코드 묶음')는 버리지 않고 원소마다 재귀한다.
길이 필터는 **스칼라 리스트**에만 적용한다 — 그게 원래 막으려던 원본 배열이다.
acquire_spectrum 이 프레임마다 length/max_intensity/sum_intensity 를 이미 계산해
넣어 두므로(acquire_tools.py 의 결과 조립부), 배열만 빼면 요약은 저절로 남는다.
새로 만들 요약도, 새로 정할 형식도 없다.

    5프레임    204,764자 → 784자
    40프레임   (옛 규칙은 239자로 데이터 소실) → 4,561자
"""
from __future__ import annotations

from typing import Any

# 길이 필터를 면제하는 키.
# files(list_uploaded_files)를 버리면 모델은 count만 받고 file_id를 얻을 길이 없어
# "ok인데 목록이 없다"며 같은 도구를 수십 번 재호출한다(실측: 업로드 62개일 때 25회).
# 항목당 4필드짜리 짧은 dict라 통째로 실어도 토큰 부담은 작다 —
# 이 필터가 원래 막으려던 건 수천 점짜리 숫자 배열이다.
# artifacts/saved_files: list_session_artifacts 와 run_analysis 의 산출물 목록.
# 항목당 몇 필드짜리 짧은 dict 이고, 버리면 모델이 "저장은 됐다는데 경로가 없다"며
# 같은 저장을 반복한다(files 를 예외로 둔 것과 정확히 같은 이유).
KEEP_KEYS = {"files", "artifacts", "saved_files"}

# 스칼라 리스트를 버리기 시작하는 길이. KB 검색 결과(최대 3개)와 짧은 좌표 목록은
# 그대로 통과한다.
MAX_SCALAR_LIST = 32

# 레코드 묶음에서 요약을 남길 원소 수 상한. 넘으면 앞뒤로 잘라 싣고 몇 개를 생략했는지
# 알린다 — 200프레임 kinetic 이어도 요약만이라 ~22KB 지만, 상한이 없으면 프레임 수에
# 비례해 무한정 늘어난다. analysis_sandbox._MAX_KINETIC_FRAMES(=200, 샌드박스 주입
# 상한)와 목적이 다르다: 저쪽은 '분석 코드가 받는 원본', 이쪽은 '모델이 읽는 요약'이다.
MAX_RECORDS = 60

# 재귀 깊이 상한
_MAX_DEPTH = 4


def _is_record_list(v: Any) -> bool:
    # 리스트 타입이면서, 길이가 0보다 크고, 모든 원소가 dict 타입인지 검사하여 반환
    return isinstance(v, list) and len(v) > 0 and all(isinstance(e, dict) for e in v)


def slim(result: Any, _depth: int = 0) -> Any:
    """도구 결과에서 원본 배열을 걷어내고 요약만 남기도록 재귀적으로 처리한다
    재귀 깊이가 상한에 도달했거나 결과가 dict가 아니면 원본 그대로 반환한다
    """
    if _depth >= _MAX_DEPTH or not isinstance(result, dict):
        return result

    out: dict = {}
    # 딕셔너리의 모든 키-값 쌍을 순회한다
    for k, v in result.items():
        # 예외 처리 대상 키인 경우 길이 필터 없이 그대로 저장한다
        if k in KEEP_KEYS:
            out[k] = v
        # 값(v)이 레코드 묶음(dict들의 리스트)인 경우 원소마다 재귀 처리한다
        elif _is_record_list(v):
            # 레코드 수가 상한을 초과하면 앞부분과 뒷부분으로 나누어 요약한다
            if len(v) > MAX_RECORDS:
                # 앞/뒤로 자를 개수를 계산한다
                head, tail = MAX_RECORDS // 2, MAX_RECORDS - MAX_RECORDS // 2
                # 앞부분과 뒷부분에 대해 각각 재귀적으로 slim을 호출해 요약본을 만든다
                kept = [slim(e, _depth + 1) for e in v[:head]]
                kept += [slim(e, _depth + 1) for e in v[-tail:]]
                # 잘라낸 요약본 리스트를 저장한다
                out[k] = kept
                # 생략된 레코드 수와 전체 레코드 수를 부가적인 키로 저장하여 모델에 알린다
                out[f"{k}_omitted"] = len(v) - MAX_RECORDS
                out[f"{k}_total"] = len(v)
            else:
                # 상한을 넘지 않으면 전체 원소에 대해 재귀적으로 slim을 호출한다
                out[k] = [slim(e, _depth + 1) for e in v]
        # 값(v)이 스칼라 리스트이고 설정된 최대 길이를 초과하면 원본 숫자 배열로 간주하고 버린다(continue)
        elif isinstance(v, list) and len(v) > MAX_SCALAR_LIST:
            continue
        # 값(v)이 또 다른 dict인 경우 깊이를 1 증가시키고 재귀적으로 처리한다
        elif isinstance(v, dict):
            out[k] = slim(v, _depth + 1)
        # 위의 어떤 조건에도 해당하지 않는 스칼라 값들은 그대로 저장한다
        else:
            out[k] = v
    # 모든 처리가 끝난 요약 딕셔너리를 반환한다
    return out


if __name__ == "__main__":
    # 옛 규칙이 냈던 두 가지 고장이 다시 나지 않는지만 본다.
    #   python backend/util/tool_slim.py
    def _kinetic(n_frames: int) -> dict:
        return {"mode": "kinetic",
                "frames": [{"intensity": list(range(2000)), "length": 2000,
                            "max_intensity": 1999} for _ in range(n_frames)]}

    # ① 프레임이 적을 때: 예전엔 frames 길이(5)가 필터를 통과해 안쪽 2000점이 그대로 실렸다.
    few = slim(_kinetic(5))
    assert len(few["frames"]) == 5
    assert all("intensity" not in f for f in few["frames"]), "안쪽 원본 배열이 남았다"
    assert all(f["length"] == 2000 for f in few["frames"]), "요약까지 지워졌다"

    # ② 프레임이 많을 때: 예전엔 frames 리스트 자체가 길이 필터에 걸려 통째로 사라졌다.
    many = slim(_kinetic(200))
    assert len(many["frames"]) == MAX_RECORDS, "레코드 묶음이 통째로 사라졌다"
    assert many["frames_omitted"] == 200 - MAX_RECORDS and many["frames_total"] == 200

    # ③ 최상위 원본 배열은 여전히 버린다(원래 막으려던 것). 요약 스칼라는 남는다.
    single = slim({"data": list(range(2000)), "length": 2000, "max_intensity": 7})
    assert "data" not in single and single["length"] == 2000

    # ④ 면제 키는 길이와 무관하게 통과 — 버리면 모델이 file_id 를 못 얻어 재호출을 반복한다.
    assert len(slim({"files": [{"file_id": i} for i in range(62)]})["files"]) == 62

    # ⑤ dict 가 아닌 결과는 그대로 (호출부가 타입을 가리지 않아도 되게).
    assert slim("ok") == "ok" and slim([1, 2, 3]) == [1, 2, 3]

    print("통과: kinetic 소량/대량, 최상위 배열, 면제 키, 비-dict")
