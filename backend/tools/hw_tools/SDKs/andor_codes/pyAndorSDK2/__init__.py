__title__ = 'pyAndorSDK2'

__authors__ = 'Andor SDK2 team'
__email__ = "row_productsupport@andor.com"

__license__ = 'Andor internal'
__copyright__ = 'Copyright 2017 Andor'

import os
# libs 디렉터리를 PATH에 추가하여 atmcd64d.dll 등을 자동 탐색하도록 설정
_path = os.path.join(os.path.dirname(__file__), 'libs') + ';' + os.environ['PATH']
os.environ['PATH'] = _path

from backend.tools.hw_tools.SDKs.andor_codes.pyAndorSDK2._version import __version__, __version_info__
from backend.tools.hw_tools.SDKs.andor_codes.pyAndorSDK2.atmcd import atmcd  # 카메라 제어 핵심 래퍼 클래스

# 패키지 외부로 공개할 심볼 목록
__all__ = [
    'atmcd',
    '__title__', '__authors__', '__email__', '__license__',
    '__copyright__', '__version__', '__version_info__',
]
