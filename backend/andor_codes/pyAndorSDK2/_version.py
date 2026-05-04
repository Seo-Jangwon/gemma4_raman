# major: 절대 변경하지 않음
# minor: 코드 변경 시 version_updater.py --minor 로 업데이트
# build: DLL 업데이트 시 version_updater.py --build 로 업데이트

major = 1
minor = 2
build = 0
__version_info__ = (major, minor, build)  # (메이저, 마이너, 빌드) 버전 튜플
__version__ = '.'.join(map(str, __version_info__))  # "1.2.0" 형태의 버전 문자열
