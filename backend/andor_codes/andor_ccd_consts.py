"""
andor_ccd_consts.py — Andor CCD SDK 상수 정의 (pyAndorSDK2 재수출)
===================================================================

pyAndorSDK2.atmcd_errors.Error_Codes 에서 모든 에러/상태 코드를 가져와 재수출.
기존 코드(hardware_manager.py 등)와의 하위 호환성을 위해 동일한 이름으로 노출.

[GetCapabilities 비트마스크]
  AC_ACQMODE_*, AC_READMODE_*, AC_TRIGGERMODE_* 는
  GetCapabilities() 의 비트마스크 결과 해석용 상수이며,
  SetAcquisitionMode() 같은 직접 제어 함수의 인수로 쓰는 값과 다름.
  직접 제어 함수에는 andor_ccd_consts.Acquisition_Mode, Read_Mode, Trigger_Mode 열거형 사용.
"""

import sys
import os

# pyAndorSDK2 패키지 경로 설정 (pip install 안 된 경우 대비)
_SDK_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', 'Andor SDK', 'Python', 'pyAndorSDK2')
)
if _SDK_PATH not in sys.path:
    sys.path.insert(0, _SDK_PATH)

from pyAndorSDK2.atmcd_errors import Error_Codes as _ec
from pyAndorSDK2 import atmcd_codes

MAX_PATH = 260  # Windows MAX_PATH: 파일 경로 최대 길이

# ══════════════════════════════════════════════════════
# pyAndorSDK2 Error_Codes 전체 재수출 (int 변환 → 타입 호환)
# hardware_manager.py 등에서 직접 상수 이름으로 비교할 때 사용
# ══════════════════════════════════════════════════════
for _name, _member in _ec.__members__.items():
    globals()[_name] = int(_member)

# ══════════════════════════════════════════════════════
# GetCapabilities 비트마스크 — 취득 모드 지원 여부 조회용
# SetAcquisitionMode() 의 인수값이 아님에 주의.
# ══════════════════════════════════════════════════════
AC_ACQMODE_SINGLE        = 1    # 단일 프레임 수집 지원
AC_ACQMODE_VIDEO         = 2    # 비디오(Run till abort) 지원
AC_ACQMODE_ACCUMULATE    = 4    # Accumulate 모드 지원
AC_ACQMODE_KINETIC       = 8    # Kinetic series 지원
AC_ACQMODE_FRAMETRANSFER = 16   # Frame transfer 지원
AC_ACQMODE_FASTKINETICS  = 32   # Fast kinetics 지원
AC_ACQMODE_OVERLAP       = 64   # Overlap 모드 지원

# ══════════════════════════════════════════════════════
# GetCapabilities 비트마스크 — 읽기 모드 지원 여부 조회용
# SetReadMode() 의 인수값(0~4)과 다름에 주의.
# ══════════════════════════════════════════════════════
AC_READMODE_FULLIMAGE      = 1   # 전체 이미지 읽기 지원
AC_READMODE_SUBIMAGE       = 2   # 서브이미지(ROI) 지원
AC_READMODE_SINGLETRACK    = 4   # 단일 트랙 지원
AC_READMODE_FVB            = 8   # Full Vertical Binning 지원
AC_READMODE_MULTITRACK     = 16  # 멀티트랙 지원
AC_READMODE_RANDOMTRACK    = 32  # 랜덤트랙 지원
AC_READMODE_MULTITRACKSCAN = 64  # 멀티트랙 스캔 지원

# ══════════════════════════════════════════════════════
# GetCapabilities 비트마스크 — 트리거 모드 지원 여부 조회용
# SetTriggerMode() 의 인수값(0, 1, 6, 7, 9, 10, 12)과 다름에 주의.
# ══════════════════════════════════════════════════════
AC_TRIGGERMODE_INTERNAL              = 1
AC_TRIGGERMODE_EXTERNAL              = 2
AC_TRIGGERMODE_EXTERNAL_FVB_EM       = 4
AC_TRIGGERMODE_CONTINUOUS            = 8
AC_TRIGGERMODE_EXTERNALSTART         = 16
AC_TRIGGERMODE_EXTERNALEXPOSURE      = 32
AC_TRIGGERMODE_INVERTED              = 0x40
AC_TRIGGERMODE_EXTERNAL_CHARGESHIFTING = 0x80

# ══════════════════════════════════════════════════════
# 모드 열거형 (새 코드에서 직접 사용)
# ══════════════════════════════════════════════════════
Acquisition_Mode = atmcd_codes.Acquisition_Mode   # SINGLE_SCAN=1, ACCUMULATE=2, KINETICS=3, ...
Read_Mode        = atmcd_codes.Read_Mode           # FULL_VERTICAL_BINNING=0, ..., IMAGE=4
Trigger_Mode     = atmcd_codes.Trigger_Mode        # INTERNAL=0, EXTERNAL=1, SOFTWARE_TRIGGER=10, ...
Spool_Mode       = atmcd_codes.Spool_Mode          # SPOOL_TO_16_BIT_FITS=5, SPOOL_TO_SIF=6, ...
Gate_Mode        = atmcd_codes.Gate_Mode           # GATE_USING_DDG=5, ...
Shutter_Mode     = atmcd_codes.Shutter_Mode        # FULLY_AUTO=0, PERMANENTLY_OPEN=1, ...

# ══════════════════════════════════════════════════════
# 숫자 → 상수 이름 역방향 조회 딕셔너리
# 에러 코드를 사람이 읽을 수 있는 이름으로 변환.
# 예: consts_by_num[20002] → 'DRV_SUCCESS'
# ══════════════════════════════════════════════════════
consts_by_num = {}
for _name, _member in _ec.__members__.items():
    consts_by_num[int(_member)] = _name
for _name in ('AC_ACQMODE_SINGLE', 'AC_ACQMODE_VIDEO', 'AC_ACQMODE_ACCUMULATE',
              'AC_ACQMODE_KINETIC', 'AC_ACQMODE_FRAMETRANSFER', 'AC_ACQMODE_FASTKINETICS',
              'AC_ACQMODE_OVERLAP', 'AC_READMODE_FULLIMAGE', 'AC_READMODE_SUBIMAGE',
              'AC_READMODE_SINGLETRACK', 'AC_READMODE_FVB', 'AC_READMODE_MULTITRACK',
              'AC_READMODE_RANDOMTRACK', 'AC_READMODE_MULTITRACKSCAN',
              'AC_TRIGGERMODE_INTERNAL', 'AC_TRIGGERMODE_EXTERNAL'):
    consts_by_num[globals()[_name]] = _name
