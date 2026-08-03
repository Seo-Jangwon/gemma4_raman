# -*- coding: utf-8 -*-
"""benchmark/inputs → data/uploads/<날짜>/ 복사.

[왜 복사인가]
upload_store 는 data/uploads/<YYYY-MM-DD>/<파일명> 을 보고 file_id 를 <날짜>/<파일명> 으로
만든다. 별도 인덱스 파일이 없어서 폴더에 두기만 하면 list_uploaded_files 가 인식한다.
다만 인자를 생략한 list_uploaded_files 는 '오늘'을 보므로, 실행일 폴더에 있어야 한다.

canonical 은 benchmark/inputs 이고 data/uploads 는 런타임 사본이다. 반대로 두면
(uploads 를 원본으로 삼으면) 날짜 폴더가 늘어날 때마다 어느 것이 진짜인지 알 수 없게 된다.
"""
from __future__ import annotations

import shutil
import sys
from datetime import date
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "inputs"
DST_ROOT = ROOT.parent / "data" / "uploads"


def main(day: str | None = None, clean: bool = False):
    day = day or date.today().isoformat()
    dst = DST_ROOT / day
    if clean and dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(SRC.iterdir()):
        if p.is_file():
            shutil.copy2(p, dst / p.name)
            n += 1
    print(f"{n}개 파일 → {dst}")
    print(f"file_id 예: {day}/T038.csv")
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    raise SystemExit(main(args[0] if args else None, "--clean" in sys.argv))
