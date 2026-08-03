# -*- coding: utf-8 -*-
"""생성된 데이터셋이 실제로 문항을 성립시키는지 확인한다.

생성만 하고 넘어가면 'GT 는 있는데 풀 수 없는 문항'이 남는다. 여기서 보는 것은
'파일이 있는가'가 아니라 '이 파일로 그 문항이 의도한 판별을 할 수 있는가'다.
하나라도 실패하면 0 이 아닌 코드로 끝나 다음 단계로 못 넘어간다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent
GT = ROOT / "gt"
IN = ROOT / "inputs"

M = json.loads((GT / "manifest.json").read_text(encoding="utf-8"))
fails, warns = [], []


def check(name, cond, detail=""):
    (fails if not cond else []).append(f"{name}: {detail}") if not cond else None
    print(f"  {'OK  ' if cond else 'FAIL'} {name}" + (f"  ({detail})" if detail else ""))


def warn(name, cond, detail=""):
    if not cond:
        warns.append(f"{name}: {detail}")
    print(f"  {'OK  ' if cond else 'WARN'} {name}" + (f"  ({detail})" if detail else ""))


print("[입력 파일 존재]")
missing = []
for t, d in M.items():
    for f in d["inputs"]:
        if not (IN / f).exists():
            missing.append(f"{t}:{f}")
check("모든 입력 파일 존재", not missing, ", ".join(missing[:5]))

print("\n[전처리·피크 문항이 실제로 풀리는가]")
check("T043 피크 7개 확보", len(M["T043"]["peaks"]) == 7,
      f"검출 {M['T043']['n_detected']}개 중 상위 7")
p44 = [p["position"] for p in M["T044"]["peaks"]]
check("T044 상위3 서로 다름", len(set(p44)) == 3, str([round(v, 1) for v in p44]))
check("T045 FWHM 유효", 3.0 < M["T045"]["fwhm_cm1"] < 30.0, f"{M['T045']['fwhm_cm1']:.2f} cm-1")
check("T046 일치쌍 존재", len(M["T046"]["matched_pairs"]) >= 5,
      f"{len(M['T046']['matched_pairs'])}/{M['T046']['n_ref_peaks']} = {M['T046']['match_ratio']:.2f}")
check("T050 SNR 유한·유의", 5.0 < M["T050"]["snr"] < 1e4, f"{M['T050']['snr']:.1f}")
check("T052 5cm-1 이상 쌍 존재", len(M["T052"]["pairs"]) >= 3, f"{len(M['T052']['pairs'])}쌍")
check("T053 면적 양수", M["T053"]["area"] > 0, f"{M['T053']['area']:.1f}")
check("T054 비율 유의", 1.0 < M["T054"]["ratio"] < 4.0, f"{M['T054']['ratio']:.3f}")
check("T040 스파이크 검출됨", len(M["T040"]["spike_indices"]) > 0,
      f"주입 {len(M['T040']['injected_indices'])} → 검출 {len(M['T040']['spike_indices'])}")
check("T057 스파이크 판정 True", M["T057"]["has_spikes"] is True)
sp = set(M["T057"]["spike_indices"])
import csv as _csv
xs = [float(r["raman_shift_cm-1"]) for r in
      _csv.DictReader(open(IN / "T057.csv", encoding="utf-8-sig"))]
sp_pos = {xs[i] for i in sp if i < len(xs)}
check("T057 피크에 스파이크 미포함",
      all(min(abs(p - s) for s in sp_pos) > 3.0 for p in M["T057"]["peaks"]) if sp_pos else True,
      "스파이크가 피크로 보고되면 문항이 무의미")

print("\n[맵·세션 문항]")
check("T051 9좌표", len(M["T051"]["values"]) == 9)
check("T066 저SNR 지점 존재", 1 <= len(M["T066"]["low_snr_positions"]) <= 8,
      f"{len(M['T066']['low_snr_positions'])}/9 지점")
evr = M["T072"]["explained_variance_ratio"]
check("T072 PC 내림차순·합<=1", evr == sorted(evr, reverse=True) and sum(evr) <= 1.0001,
      f"{[round(v,3) for v in evr]}")
g = [a["group"] for a in M["T073"]["assignments"]]
c = [a["cluster"] for a in M["T073"]["assignments"]]
agree = max(sum(int(x == y) for x, y in zip(g, c)),
            sum(int(x != y) for x, y in zip(g, c))) / len(g)
check("T073 클러스터가 실제 그룹과 일치", agree == 1.0, f"순열보정 후 일치율 {agree:.0%}")
check("T074 이상지점 존재", 1 <= len(M["T074"]["anomalies"]) <= 4,
      f"{len(M['T074']['anomalies'])}/9 지점")
check("T075 4수치 산출", all(k in M["T075"] for k in
      ("peak_position_diff", "rsd_a_pct", "rsd_b_pct", "cosine_of_means")),
      f"위치차 {M['T075']['peak_position_diff']:.1f} / 유사도 {M['T075']['cosine_of_means']:.4f}")

print("\n[트러블슈팅]")
check("T098 형광 보정 후 피크 확보", len(M["T098"]["peaks"]) >= 5, f"{len(M['T098']['peaks'])}개")
check("T101 프레임별 스파이크 위치가 다름",
      len({tuple(v) for v in M["T101"]["spike_indices_per_frame"]}) == 3,
      "같으면 '위치가 매번 다르다'는 원인 판정 근거가 사라진다")
check("T103 시프트 부호·크기", M["T103"]["shift_cm1"] == 1.0)
check("T105 신호↓ 배경↑", M["T105"]["signal_slope"] < 0 < M["T105"]["background_slope"],
      f"신호 {M['T105']['signal_slope']:.1f} / 배경 {M['T105']['background_slope']:.1f}")
check("T106 레이블이 undecidable 아님", M["T106"]["label"] != "undecidable",
      f"{M['T106']['label']} (FWHM {M['T106']['fwhm_cm1']:.1f})")
check("T108 드리프트 방향 일치", M["T108"]["drift_slope"] > 0,
      f"기울기 {M['T108']['drift_slope']:.2f} (주입 {M['T108']['injected_slope']})")
check("T112 보정으로 피크가 늘어남", M["T112"]["n_newly_visible"] > 0,
      f"{M['T112']['n_peaks_before']} → {M['T112']['n_peaks_after']}")

print("\n[매칭 — 의도한 정답이 실제로 1위인가]")
INTENT = {"T113": "polystyrene", "T114": "PET", "T117": "PET", "T119": "polystyrene",
          "T121": "polystyrene", "T123": "silicon", "T124": "aragonite",
          "T128": "calcite", "T129": "polystyrene"}
for t, want in INTENT.items():
    got = M[t].get("top1", {}).get("material") or M[t].get("material")
    sc = M[t].get("top1", {}).get("score")
    check(f"{t} top1 = {want}", got == want,
          f"실제 {got}" + (f" (score {sc:.4f})" if sc else ""))
check("T115 전처리가 결과를 바꾼다", M["T115"]["preprocessing_matters"] is True,
      f"보정 전 {M['T115']['without_correction_material']} → 후 {M['T115']['material']}")
check("T116 OOD 판정", M["T116"]["reliable_match"] is False,
      f"최고 {M['T116']['best_score']:.4f} < 0.75")
check("T118 우세성분 = polystyrene", M["T118"]["dominant"] == "polystyrene",
      str({k: round(v, 4) for k, v in M["T118"]["scores"].items()}))
check("T120 피크매칭 물질 = PET", M["T120"]["material"] == "PET")
check("T122 항목까지 식별 가능", M["T122"]["top1"]["spectrum_id"] == "PMMA_02",
      f"1위 {M['T122']['top1']['spectrum_id']} / 2위 {M['T122']['ranking'][1]['spectrum_id']} "
      f"(차 {M['T122']['top1']['score']-M['T122']['ranking'][1]['score']:.4f})")
check("T125 근거 피크 2개 이상", len(M["T125"]["discriminating_peaks"]) >= 2,
      f"{M['T125']['material']} vs {M['T125']['second_candidate']}: {M['T125']['discriminating_peaks']}")
check("T126 5개 전부 의도대로", M["T126"]["materials"] == M["T126"]["intended"],
      str(M["T126"]["materials"]))
check("T127 불일치 판정", M["T127"]["matches"] is False,
      f"주장 {M['T127']['claimed_material']} / 실제 {M['T127']['actual_material']}")
check("T128 동점이 실제로 발생", M["T128"]["tie_actually_occurs"] is True,
      f"동점 후보 {M['T128']['tied_candidates']} → 규칙대로 {M['T128']['top1']['spectrum_id']} 선택")
hits = [q["hit_rate"] for q in M["T130"]["per_query"]]
check("T130 적중률이 자명하지 않음", 0.0 < M["T130"]["mean_hit_rate"] <= 1.0,
      f"쿼리별 {hits} 평균 {M['T130']['mean_hit_rate']:.3f}")
warn("T130 적중률에 변별 여지", len(set(hits)) > 1,
     f"전부 {hits[0]:.2f} 면 평균이 상수라 변별력이 약하다")

print("\n[합성 영상]")
for t in ("T037", "T063", "T076"):
    n = len(M[t]["targets"])
    check(f"{t} 타깃 {n}개", n == (4 if t == "T037" else 1),
          str([(d["pixel_x"], d["pixel_y"]) for d in M[t]["targets"]]))

print("\n" + "=" * 60)
if warns:
    print(f"경고 {len(warns)}건 (문항은 성립하나 변별력 점검 필요):")
    for w in warns:
        print("  -", w)
if fails:
    print(f"\n실패 {len(fails)}건:")
    for f in fails:
        print("  -", f)
    raise SystemExit(1)
print("검증 통과 — 모든 문항이 생성된 데이터로 풀린다.")
