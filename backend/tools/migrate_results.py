# -*- coding: utf-8 -*-
"""
data/results/<날짜>/ 에 평평하게 쌓인 구버전 측정 파일을 '세션 폴더'로 정리한다.

[왜]
예전 spectrum_store 는 개별 측정을 날짜 폴더에 그대로 쏟았다:
    data/results/2026-07-29/170901_769_x37.876_y25.248.{png,csv,json}
하루에 수십~수백 개가 쌓이는데 파일명만으로는 어느 문항·어느 에이전트의 측정인지 알
수 없었다. 지금은 data/results/<날짜>/<세션>/ 아래로 저장하지만, 이미 쌓인 파일은
그대로 남아 있다(list_results 가 하위호환으로 읽어 주므로 동작은 한다).

[귀속을 어떻게 알아내나 — 추측하지 않는다]
DetailLog/<AGENT>_<시각>_<sid>.json 의 툴 호출 결과에는 저장된 파일의 URL 이 그대로
들어 있다(acquire_spectrum → result.saved.image_url / csv_url / json_url). 그 URL 의
파일 stem 을 세션 라벨에 대응시키면 '이 파일은 이 세션이 만들었다'가 정확히 나온다.
시각 근접 추정 같은 건 쓰지 않는다 — 틀리면 채점 근거가 오염되기 때문이다.
소유자를 못 찾은 파일은 건드리지 않고 그대로 둔다(원하면 --unassigned 로 모은다).

[안전]
· 기본은 dry-run 이다. 실제로 옮기려면 --apply 를 준다.
· 서버/벤치가 도는 중에는 쓰지 말 것 — 쓰고 있는 파일을 옮기면 경합이 난다.
· 대상 경로에 같은 이름이 이미 있으면 건너뛴다(덮어쓰지 않는다).
· json 안에 "session" 키를 채워 넣어, 파일을 다시 옮겨도 귀속이 남게 한다.

실행:
    python -m backend.tools.migrate_results                    # 오늘, dry-run
    python -m backend.tools.migrate_results --date 2026-07-29 --apply
    python -m backend.tools.migrate_results --all-dates --apply
    python -m backend.tools.migrate_results --apply --unassigned   # 미귀속도 _unassigned 로
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RESULTS = _PROJECT_ROOT / "data" / "results"
_DETAIL_LOG = _PROJECT_ROOT / "DetailLog"

# 측정 3종 + 혹시 남아 있는 구버전 run_analysis 그림(<세션>__fig<stamp>_<i>.png)
_EXTS = (".png", ".csv", ".json")
_OLD_FIG = re.compile(r"^(?P<label>.+?)__fig\d{6}_\d{3}_\d+$")

# '_' 접두 생성물. 원래는 '세션을 넘는 집계물'이라 제외했지만, list_results 가
# scope="session" 기본이 된 뒤로는 이것들도 대부분 한 세션의 산출물이다:
#   _summary_*/_combined_*/_bundle_*  한 세션의 측정만 묶은 결과
#   _grid_preview_*                   그 세션 그리드 스캔의 미리보기
# 그래서 --include-generated 로 함께 정리할 수 있게 한다(npz/zip 확장자 포함).
_GEN_EXTS = (".png", ".csv", ".json", ".npz", ".zip")
_SCENE = re.compile(r"^_scene_\d{6}_\d{3}$")


def _sanitize(text: str) -> str:
    """run_store._sanitize / detail_log._sanitize 와 동일 규칙."""
    return re.sub(r"[^0-9A-Za-z_-]", "-", str(text))[:64] or "nosession"


def _walk(obj):
    """중첩 dict/list 를 전부 훑는다 — 저장 URL 이 어디에 박혀 있어도 찾도록."""
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)


def build_owner_map() -> dict[str, str]:
    """{파일 stem: 세션 라벨}. DetailLog 의 저장 URL 에서만 뽑는다(추측 없음)."""
    owner: dict[str, str] = {}
    if not _DETAIL_LOG.exists():
        return owner
    for lp in sorted(_DETAIL_LOG.glob("*.json")):
        try:
            doc = json.loads(lp.read_text(encoding="utf-8"))
        except Exception:
            continue
        # 라벨은 파일명(<AGENT>_<시각>_<sid>.json)보다 문서의 session_id 를 신뢰한다.
        sid = doc.get("session_id") or doc.get("session") or ""
        label = _sanitize(sid) if sid else ""
        if not label:
            m = re.match(r"^[^_]+_\d{6}_(?P<sid>.+)$", lp.stem)
            label = m.group("sid") if m else ""
        if not label:
            continue
        for node in _walk(doc):
            # zip_url: bundle_results / scene_npz: capture_scene 는 URL 이 아니라
            # 파일 경로를 돌려주므로 별도로 봐야 한다.
            for key in ("image_url", "csv_url", "json_url", "zip_url", "scene_npz"):
                v = node.get(key)
                if not isinstance(v, str) or not v:
                    continue
                if "/api/results/" not in v and "results" not in v.replace("\\", "/"):
                    continue
                # 이미 세션 폴더 아래인 URL(신버전)은 정리할 필요가 없다
                owner.setdefault(Path(v).stem, label)
    return owner


def migrate_date(date: str, owner: dict[str, str], apply: bool, unassigned: bool,
                 include_generated: bool = False,
                 include_scenes: bool = False) -> dict:
    day_dir = _RESULTS / date
    if not day_dir.exists():
        return {"date": date, "error": "no such date dir"}

    exts = _GEN_EXTS if include_generated else _EXTS
    # 날짜 폴더 '직속' 파일만 대상. 하위 세션 폴더는 이미 정리된 것이다.
    flat = [p for p in day_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    groups: dict[str, list[Path]] = defaultdict(list)
    held_scenes: list[str] = []
    for p in flat:
        if p.stem.startswith("_"):
            if not include_generated:
                continue                 # 기본: 집계물은 손대지 않는다
            if _SCENE.match(p.stem) and not include_scenes:
                # latest_scene() 이 '날짜 폴더 직속'을 glob 해서 run_analysis 의
                # microscope_image 로 주입한다. 서버가 도는 중에 옮기면 진행 중인
                # 문항이 현미경 이미지를 잃는다 — 명시적으로 허용해야 옮긴다.
                if p.stem not in held_scenes:
                    held_scenes.append(p.stem)
                continue
        groups[p.stem].append(p)

    plan: list[tuple[str, str, list[Path]]] = []   # (stem, 목적지 세션, 파일들)
    unowned: list[str] = []
    for stem, paths in sorted(groups.items()):
        label = owner.get(stem)
        if not label:
            m = _OLD_FIG.match(stem)     # 구버전 그림은 파일명 자체에 라벨이 있다
            if m:
                label = m.group("label")
        if not label:
            unowned.append(stem)
            if not unassigned:
                continue
            label = "_unassigned"
        plan.append((stem, label, paths))

    moved = skipped = 0
    for stem, label, paths in plan:
        dest_dir = day_dir / label
        if apply:
            dest_dir.mkdir(parents=True, exist_ok=True)
        for p in paths:
            dest = dest_dir / p.name
            if dest.exists():
                skipped += 1
                continue
            if apply:
                # json 안에도 귀속을 남긴다 — 파일이 또 움직여도 세션을 알 수 있게.
                if p.suffix.lower() == ".json":
                    try:
                        doc = json.loads(p.read_text(encoding="utf-8"))
                        if isinstance(doc, dict):
                            doc.setdefault("session", label)
                            p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
                    except Exception:
                        pass
                p.rename(dest)
            moved += 1

    return {"date": date, "groups": len(groups), "planned": len(plan),
            "files": moved, "skipped_exists": skipped, "unowned": unowned,
            "held_scenes": held_scenes,
            "moved_detail": [(s, l) for s, l, _ in plan]}


def main():
    ap = argparse.ArgumentParser(description="구버전 평평한 측정 파일을 세션 폴더로 정리")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (기본: 오늘)")
    ap.add_argument("--all-dates", action="store_true", help="data/results 의 모든 날짜")
    ap.add_argument("--apply", action="store_true", help="실제로 옮긴다(기본은 dry-run)")
    ap.add_argument("--unassigned", action="store_true",
                    help="소유자를 못 찾은 파일도 _unassigned/ 로 모은다")
    ap.add_argument("--include-generated", action="store_true",
                    help="'_' 접두 생성물(_summary_/_combined_/_bundle_/_grid_preview_)도 "
                         "세션 폴더로 옮긴다. _scene_ 은 제외(--include-scenes 필요)")
    ap.add_argument("--include-scenes", action="store_true",
                    help="_scene_*.npz/.png 까지 옮긴다. latest_scene() 이 날짜 폴더를 "
                         "glob 하므로 서버/벤치가 도는 중에는 절대 쓰지 말 것")
    args = ap.parse_args()

    owner = build_owner_map()
    print(f"DetailLog 에서 찾은 파일-세션 매핑: {len(owner)}건")

    if args.all_dates:
        dates = sorted(p.name for p in _RESULTS.iterdir()
                       if p.is_dir() and re.match(r"^\d{4}-\d{2}-\d{2}$", p.name))
    else:
        dates = [args.date or datetime.now().strftime("%Y-%m-%d")]

    if not args.apply:
        print("*** dry-run - 아무것도 옮기지 않는다. 실제 실행은 --apply ***")
    if args.include_scenes and not args.include_generated:
        print("[오류] --include-scenes 는 --include-generated 와 함께 줘야 한다"); raise SystemExit(1)

    for d in dates:
        r = migrate_date(d, owner, args.apply, args.unassigned,
                         args.include_generated, args.include_scenes)
        if r.get("error"):
            print(f"[{d}] {r['error']}")
            continue
        print(f"[{d}] 묶음 {r['groups']}개 중 {r['planned']}개 귀속 → "
              f"파일 {r['files']}개 {'이동' if args.apply else '이동 예정'}"
              + (f", 이미 존재해 건너뜀 {r['skipped_exists']}" if r["skipped_exists"] else ""))
        for stem, lab in r.get("moved_detail", []):
            print(f"      {stem}  ->  {lab}/")
        if r.get("held_scenes"):
            print(f"      _scene_ {len(r['held_scenes'])}개 보류(latest_scene 이 읽는 파일): "
                  + ", ".join(r["held_scenes"]))
            print("      -> 벤치/서버를 멈춘 뒤 --include-scenes 로 옮길 것")
        if r["unowned"]:
            print(f"      귀속 실패 {len(r['unowned'])}개(그대로 둠): "
                  + ", ".join(r["unowned"][:6]) + ("…" if len(r["unowned"]) > 6 else ""))
            if not args.unassigned:
                print("      -> 이것들도 모으려면 --unassigned")


if __name__ == "__main__":
    main()
