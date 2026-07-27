import serial
import time
import sys
import math


class LaserController:
    """
    Fastech EziMOTION Plus-R 미문서화 ASCII (@...$ 래퍼) 기반 레이저 모터 컨트롤러 v3.

    개선 사항 (v2 → v3):
      - [추가] WCHS/WCNS 속도 파라미터 설정 (Config.txt 값 기반)
      - [추가] GMMS 폴링 기반 모션 완료 대기 (Rays-ON.exe 핵심 패턴)
      - [추가] ercd 에러 자동 감지 → ERCL 리셋 → 재시도
      - [추가] GMCP 위치 검증
      - [수정] SMSE 인자 없이 전송 (ASCII 래퍼는 이진 프로토콜과 달리 ARG 불필요)
      - [수정] 초기화 순서를 Rays-ON.exe 캡처와 정확히 동기화

    v3 → v3.1:
      - [수정] set_power를 ND 필터 캘리브레이션(config [ND_FILTER] Mode=1) 기반
               연속 제어로 교체. 임의 투과율(%)을 log-선형 보간하여 펄스 위치로 변환.

    근거 자료:
      - 20260421패킷분석.xlsx: Device Monitoring Studio COM4 캡처 (10055행)
      - MOTION_EziSERVO_DEFINE.h: EZISERVO_AXISSTATUS (FFLAG 32-bit 비트필드)
      - FM_EZISERVO_PARAM: FAS_SetParameter 파라미터 번호 enum
      - CommInterface.c/.h: FAS_* API 구현 (이진 프로토콜)
      - Config.txt: 모터 매핑 및 속도값
      - Config [ND_FILTER] Mode=1: 투과율(%) ↔ 펄스 위치 캘리브레이션 곡선
    """

    # ── 모터 축 정의 (Config.txt에서 확인) ──
    # MotorNumber = 시리얼 패킷의 target_id
    AXES = {
        "02": {"name": "ND_FILTER",     "homing_vel": 150000, "moving_vel": 400000},
        "01": {"name": "LASER_FILTER",  "homing_vel": 150000, "moving_vel": 400000},
        "05": {"name": "GRATING",       "homing_vel": 150000, "moving_vel": 200000},
        "04": {"name": "BEAM_SPLITTER", "homing_vel": 150000, "moving_vel": 400000},
    }
    AXIS_ORDER = ["02", "01", "05", "04"]  # Rays-ON.exe 초기화 순서

    # ── FFLAG 비트 마스크 (MOTION_EziSERVO_DEFINE.h → EZISERVO_AXISSTATUS) ──
    FFLAG_ERRORALL        = 0x00000001
    FFLAG_HWPOSILMT       = 0x00000002
    FFLAG_HWNEGALMT       = 0x00000004
    FFLAG_ERRPOSOVERFLOW   = 0x00000080
    FFLAG_ERROVERCURRENT   = 0x00000100
    FFLAG_EMGSTOP         = 0x00010000  # ← ercd 0x10000 의 정체
    FFLAG_SLOWSTOP        = 0x00020000
    FFLAG_ORIGINRETURNING = 0x00040000
    FFLAG_INPOSITION      = 0x00080000
    FFLAG_SERVOON         = 0x00100000
    FFLAG_ALARMRESET      = 0x00200000
    FFLAG_ORIGINRETOK     = 0x02000000
    FFLAG_MOTIONING       = 0x08000000
    FFLAG_MOTIONPAUSE     = 0x10000000
    FFLAG_MOTIONACCEL     = 0x20000000
    FFLAG_MOTIONDECEL     = 0x40000000
    FFLAG_MOTIONCONST     = 0x80000000

    # ── FAS_SetParameter 파라미터 번호 (FM_EZISERVO_PARAM enum) ──
    # WCNS → param 1  (SERVO_AXISMAXSPEED = MovingVelocity)
    # WCHS → param 17 (SERVO_ORGSPEED     = HomingVelocity)
    PARAM_AXISMAXSPEED = 1
    PARAM_ORGSPEED     = 17

    # ── GMMS 응답 비트 마스크 (패킷 캡처 실증) ──
    # gmms는 dwAxisStatus의 6-hex 압축 뷰로, FFLAG와 비트 위치가 다름
    # 관측:
    #   @02gmms0010007$ → 001000 = 원점 복귀 중
    #   @02gmms0100007$ → 010000 = 원점 복귀 완료
    #   @02gmms0020008$ → 002000 = 이동 중
    #   @01gmms0000005$ → 000000 = 정지
    GMMS_HOMING    = 0x001000
    GMMS_ORIGIN_OK = 0x010000
    GMMS_MOVING    = 0x002000
    GMMS_BUSY      = 0x003000  # HOMING | MOVING

    # ── ND 필터 투과율 캘리브레이션 (config [ND_FILTER] Mode=1) ──
    #    (transmittance_percent, pulse)  — % 내림차순 / 위치 오름차순
    #    위치는 30000 펄스 균등 간격, 마지막 602895만 빔 차단점.
    ND_CAL = [
        (100.0, 47944), (46.272, 77944), (21.93, 107944), (11.206, 137944),
        (6.316, 167944), (3.509, 197944), (1.943, 227944), (1.118, 257944),
        (0.596, 287944), (0.371, 317944), (0.219, 347944), (0.125, 377944),
        (0.07, 407944), (0.042, 437944), (0.026, 467944), (0.018, 497944),
        (0.013, 527944), (0.009, 557944), (0.007, 587944), (0.004, 602895),
    ]
    ND_MIN_PCT, ND_MAX_PCT = 0.004, 100.0

    def __init__(self, port='COM4', baud=115200):
        self.port = port
        self.baud = baud
        self.ser = None
        self._power_set = False
        self.is_on = False       # SSPW(레이저 발사) 상태 추적 — 하드웨어 상태 조회/채점용
        self.power_pct = None    # 마지막으로 설정한 파워(%) 추적
        self._connect()
        time.sleep(0.5)
        # self._full_initialization()

    def _connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
            print(f"✅ [{self.port}] 레이저 컨트롤러 연결 성공!")
        except Exception as e:
            print(f"❌ [{self.port}] 연결 실패: {e}")
            print("포트가 사용 중이거나 연결되지 않았습니다.")

    # ==============================================================
    # 패킷 통신 핵심 함수 (v3 개선)
    # ==============================================================
    def _make_packet(self, target_id, cmd, arg=""):
        """
        체크섬 자동 계산 및 패킷 조립.

        포맷: @{target_id:2}{CMD:4}{ARG}{CSUM:2hex}$
        체크섬: sum(body의 각 ASCII 바이트) % 256 → 2-hex 소문자
        """
        body = f"{target_id}{cmd}{arg}"
        ascii_sum = sum(ord(c) for c in body)
        checksum = f"{ascii_sum % 256:02x}"
        return f"@{body}{checksum}$".encode('utf-8')

    def _flush_rx(self):
        """수신 버퍼 비우기"""
        if self.ser and self.ser.in_waiting:
            self.ser.read(self.ser.in_waiting)

    def _read_frames(self, timeout=2.0):
        """
        @...$ 프레임을 읽어서 리스트로 반환.

        짧은 무응답 구간이 생기면 수집 종료 (Rays-ON.exe 관찰: 응답 즉시 도착).
        """
        start = time.time()
        buffer = ""
        frames = []

        while (time.time() - start) < timeout:
            if self.ser.in_waiting:
                chunk = self.ser.read(self.ser.in_waiting).decode(errors='ignore')
                buffer += chunk

                # 완성된 프레임 추출
                while '@' in buffer and '$' in buffer:
                    at = buffer.index('@')
                    dollar = buffer.index('$', at)
                    frames.append(buffer[at:dollar + 1])
                    buffer = buffer[dollar + 1:]

                # 프레임이 있고 추가 데이터가 없으면 종료
                if frames:
                    time.sleep(0.03)
                    if not self.ser.in_waiting:
                        return frames
            else:
                if frames:
                    return frames
                time.sleep(0.01)

        return frames

    def _send_command(self, target_id, cmd, arg="", timeout=5.0, retries=2):
        """
        패킷 전송 + ACK 응답 확인.

        ercd 응답 감지 시 → ERCL 알람 리셋 → 재시도.
        v2의 _execute_command를 대체하는 저수준 통신 함수.

        Returns:
            (success: bool, response_data: str or None)
        """
        if not (self.ser and self.ser.is_open):
            print("⚠️ 포트 미연결")
            return False, None

        expected_key = f"{target_id}{cmd.lower()}"

        for attempt in range(retries + 1):
            self._flush_rx()
            packet = self._make_packet(target_id, cmd, arg)
            print(f"   [TX] {packet.decode()}")
            self.ser.write(packet)

            frames = self._read_frames(timeout=timeout)
            if not frames:
                print(f"   ⚠️ 응답 없음 [{attempt + 1}/{retries + 1}]")
                if attempt < retries:
                    time.sleep(0.3)
                continue

            for f in frames:
                print(f"   [RX] {f}")

            # 응답 프레임 분석
            got_ercd = False
            for f in frames:
                inner = f[1:-1]  # @ $ 제거

                # ercd 에러 감지
                if "ercd" in inner:
                    ercd_idx = inner.index("ercd") + 4
                    ercd_raw = inner[ercd_idx:]
                    # 체크섬(마지막 2자리) 제거
                    ercd_data = ercd_raw[:8] if len(ercd_raw) >= 8 else ercd_raw[:-2]
                    print(f"   ❌ ercd 에러: 0x{ercd_data} [{attempt + 1}/{retries + 1}]")
                    got_ercd = True
                    break

                # 정상 ACK (소문자 cmd echo)
                inner_lower = inner.lower()
                if expected_key in inner_lower:
                    data_start = inner_lower.index(expected_key) + len(expected_key)
                    # 원본 inner에서 데이터를 추출하여 대소문자 변형 방지
                    data_raw = inner[data_start:]
                    return True, data_raw

            if got_ercd and attempt < retries:
                # 알람 리셋 후 재시도
                self._alarm_reset_quick(target_id)
                time.sleep(0.2)
                continue

            if not got_ercd and attempt < retries:
                time.sleep(0.3)

        print(f"   ⚠️ 최종 실패: {cmd}")
        return False, None

    def _alarm_reset_quick(self, target_id):
        """알람 리셋 (재시도 로직 내부용, 로그 최소화)

        ercd 응답이 항상 "00"(브로드캐스트)에서 오므로,
        "00" + 해당 축 모두에 ERCL 전송.
        """
        for tid in ["00", target_id]:
            pkt = self._make_packet(tid, "ERCL", "")
            self._flush_rx()
            self.ser.write(pkt)
            time.sleep(0.1)
        self._flush_rx()

    # ==============================================================
    # GMMS 상태 폴링 (v3 핵심 추가)
    #
    # Rays-ON.exe 관찰: 모든 SMMH/SMMA 후 ~120ms 간격으로
    # 4축 순환 GMMS 폴링. 완료 확인 없이 다음 명령 전송 시
    # ercd 0x10000 (FFLAG_EMGSTOP) 발생.
    # ==============================================================
    def _query_gmms(self, axis_id):
        """
        GMMS 상태 조회.

        응답 예시: @02gmms0010007$
          - 데이터: 6-hex (001000)
          - 체크섬: 1-hex (7)

        Returns:
            int: 상태값 (0x001000=호밍중, 0x010000=원점완료, 0x002000=이동중)
            None: 통신 실패
        """
        self._flush_rx()
        pkt = self._make_packet(axis_id, "GMMS", "")
        self.ser.write(pkt)

        frames = self._read_frames(timeout=1.0)
        if not frames:
            return None

        key = f"{axis_id}gmms"
        for f in frames:
            inner = f[1:-1]
            if key in inner:
                data_start = inner.index(key) + len(key)
                data_raw = inner[data_start:]
                # 6-hex 데이터 + 1~2-hex 체크섬
                if len(data_raw) >= 7:
                    try:
                        return int(data_raw[:6], 16)
                    except ValueError:
                        pass
        return None

    def _wait_motion_complete(self, axis_id, timeout=30.0, poll_interval=0.12,
                              check_origin=False):
        """
        GMMS 폴링으로 모션 완료 대기.

        Rays-ON.exe 핵심 로직:
          - SMMH 후: GMMS_HOMING(001000) 소멸 + GMMS_ORIGIN_OK(010000) 확인
          - SMMA 후: GMMS_MOVING(002000) 소멸 확인

        Args:
            check_origin: True면 원점 복귀 완료(ORIGIN_OK)까지 확인 (SMMH용)
            poll_interval: 폴링 간격 (Rays-ON.exe 관찰: ~120ms)
        """
        start = time.time()
        last_status = None

        while (time.time() - start) < timeout:
            status = self._query_gmms(axis_id)

            if status is not None:
                last_status = status
                is_busy = (status & self.GMMS_BUSY) != 0

                if not is_busy:
                    if check_origin:
                        if (status & self.GMMS_ORIGIN_OK) != 0:
                            return True
                        # 원점 복귀 끝났지만 ORIGIN_OK 미설정 → 조금 더 대기
                    else:
                        return True

            time.sleep(poll_interval)

        if last_status is not None:
            print(f"   ⚠️ 모션 완료 대기 타임아웃 (axis={axis_id}, "
                  f"last=0x{last_status:06x})")
        else:
            print(f"   ⚠️ 모션 완료 대기 타임아웃 (axis={axis_id}, 응답 없음)")
        return False

    # ==============================================================
    # GMCP 위치 확인 (v3 추가)
    # ==============================================================
    def _query_position(self, axis_id):
        """
        GMCP로 현재 지령 위치 조회.

        응답 예시: @02gmcp00000000xx$
          - 데이터: 8-hex (two's complement signed 32-bit)
          - 체크섬: 2-hex

        Returns:
            int: 부호 있는 위치값 (pulse 단위)
            None: 통신 실패
        """
        self._flush_rx()
        pkt = self._make_packet(axis_id, "GMCP", "")
        self.ser.write(pkt)

        frames = self._read_frames(timeout=1.0)
        if not frames:
            return None

        key = f"{axis_id}gmcp"
        for f in frames:
            inner = f[1:-1]
            if key in inner:
                data_start = inner.index(key) + len(key)
                data_raw = inner[data_start:]
                # 부호+7자리 10진수 위치값 + 2자리 체크섬 (e.g. "-0602895" + "2b")
                if len(data_raw) >= 10:
                    try:
                        return int(data_raw[:8])
                    except ValueError:
                        pass
        return None

    # ==============================================================
    # 속도 파라미터 설정 — WCHS/WCNS (v3 추가)
    #
    # 주의: 이 ASCII 래퍼가 WCHS/WCNS를 지원하지 않을 수 있음 (ercd 03).
    # 실패해도 EEPROM에 저장된 기본값(Rays-ON.exe가 이전에 설정한 값)이
    # 사용되므로 치명적이지 않음. 단, 실패 시 잔여 에러를 반드시 클리어해야
    # 후속 명령(SMMH 등)이 영향받지 않음.
    # ==============================================================
    def _set_speed_param(self, axis_id, param_cmd, value):
        """
        WCHS 또는 WCNS로 속도 파라미터 설정 시도.

        Args:
            param_cmd: "WCHS" (HomingVelocity) 또는 "WCNS" (MovingVelocity)
            value: 속도값 (pps, unsigned 32-bit → 8-hex 인코딩)
        """
        arg = str(value)
        ok, _ = self._send_command(axis_id, param_cmd, arg, timeout=3.0, retries=0)
        return ok

    def _set_all_speeds(self):
        """4축 WCHS(HomingVelocity) + WCNS(MovingVelocity) 설정 시도."""
        print("\n   [속도 설정] WCHS + WCNS 시도 (Config.txt 값)")

        any_failed = False
        for axis_id in self.AXIS_ORDER:
            cfg = self.AXES[axis_id]
            name = cfg["name"]
            h_vel = cfg["homing_vel"]
            m_vel = cfg["moving_vel"]

            ok1 = self._set_speed_param(axis_id, "WCHS", h_vel)
            ok2 = self._set_speed_param(axis_id, "WCNS", m_vel)

            if ok1 and ok2:
                print(f"   ✅ [{axis_id} {name}] WCHS={h_vel}, WCNS={m_vel}")
            else:
                any_failed = True
            time.sleep(0.05)

        if any_failed:
            print("   ⚠️ WCHS/WCNS 미지원 래퍼 → EEPROM 기본값 사용")
            # 잔여 에러 상태 클리어 (후속 SMMH 보호)
            print("   🔄 잔여 에러 클리어 중... (ERCL)")
            self._send_command("00", "ERCL", "", timeout=3.0, retries=0)
            time.sleep(0.1)
            self._send_command("00", "ERCL", "", timeout=3.0, retries=0)
            time.sleep(0.1)

    # ==============================================================
    # 모션 명령 래퍼 (ACK + GMMS 폴링 + GMCP 검증)
    # ==============================================================
    def _execute_command(self, target_id, cmd, arg, timeout=15.0, retries=2):
        """
        v2 호환 인터페이스.

        모션 명령(SMMA/SMMI)인 경우:
          1. 패킷 전송 → ACK 확인
          2. GMMS 폴링 → 실제 모션 완료 대기
          3. GMCP → 위치 검증
        비모션 명령은 ACK 확인만.
        """
        is_motion = cmd.upper() in ("SMMA", "SMMI")
        # 1) ACK 확인
        ok, _ = self._send_command(target_id, cmd, arg,
                                   timeout=5.0, retries=retries)

        if ok and is_motion:
            # 2) GMMS 폴링으로 모션 완료 대기
            print(f"   ⏳ GMMS 폴링 대기... (axis={target_id})")
            done = self._wait_motion_complete(
                target_id,
                timeout=timeout,
                check_origin=False
            )
            if done:
                # 3) 위치 검증
                pos = self._query_position(target_id)
                if pos is not None:
                    print(f"   📍 위치 확인: {pos}")
                return True
            else:
                print(f"   ❌ 모션 미완료 (axis={target_id})")
                return False

        return ok

    # ==============================================================
    # 초기화 시퀀스 (Rays-ON.exe 패킷 캡처 기반, 체크섬 검증 완료)
    #
    # 실증된 순서:
    #   1. ERCL ×2           — 알람 리셋
    #   2. SMSE ×4축         — 서보 활성화 (02→01→05→04)
    #   3. SSPW 0            — 레이저 OFF (안전)
    #   3.5. WCHS/WCNS ×4축  — 속도 설정 (10진수 문자열 형식)
    #   4. home_search()     — SMMHF(모드설정) → SMMA(사전위치) → SIPW(트리거)
    #   5. SMMA              — 셔터/가이드빔 위치 이동
    # ==============================================================
    def _full_initialization(self):
        if not (self.ser and self.ser.is_open):
            return

        print("\n" + "=" * 55)
        print("   🔧 장비 초기화 시퀀스 시작 (v3)")
        print("=" * 55)

        # ── Step 1: 알람 리셋 (ERCL × 2회) ──
        print("\n[Step 1/5] 알람 리셋 (ERCL × 2회)")
        self._send_command("00", "ERCL", "", timeout=3.0, retries=0)
        time.sleep(0.1)
        self._send_command("00", "ERCL", "", timeout=3.0, retries=0)
        time.sleep(0.1)

        # ── Step 2: 4축 서보 활성화 (SMSE) ──
        print("\n[Step 2/5] 4축 서보 활성화 (SMSE)")
        for axis in self.AXIS_ORDER:
            self._send_command(axis, "SMSE", "", timeout=5.0, retries=1)
            time.sleep(0.05)

        # ── Step 3: 레이저 OFF (안전) ──
        print("\n[Step 3/5] 레이저 OFF 확인 (SSPW 0)")
        self._send_command("00", "SSPW", "0", timeout=3.0, retries=1)
        time.sleep(0.05)

        # ── Step 3.5: 속도 파라미터 설정 (WCHS + WCNS) ──
        self._set_all_speeds()

        # ── Step 4: 원점 복귀 (SMMHF 모드설정 → SMMA 사전위치 → SIPW 트리거) ──
        self.home_search()

        # ── Step 5: 가이드빔 대기 상태 ──
        print("\n[Step 5/5] 가이드빔 대기 상태 전환")
        self.set_guide_beam()

        print("\n" + "=" * 55)
        print("   ✅ 초기화 시퀀스 완료!")
        print("=" * 55 + "\n")

    # ==============================================================
    # 원점 복귀 (사용자 수동 실행 전용)
    # ==============================================================
    def home_search(self):
        """
        4축 원점 복귀.

        패킷 캡처(Rays-ON.exe) 검증된 실제 순서:
          1. SMMHF → 즉시 홈 복귀 실행 (축 02, 05, 04 순서, 각 축 ORIGIN_OK 대기)
          2. SMMA  — 홈 복귀 완료 후 운용 위치로 이동
          3. SIPW 1 — 축01 트리거
          4. SMMA 축01 → -0070600
        """
        print("\n🏠 4축 원점 복귀 시작")

        # [1] 축별 홈 복귀: SMMHF 전송 → 즉시 물리적 홈 복귀 시작 → ORIGIN_OK 대기
        #     축01(LASER_FILTER)은 SMMHF 없음 — SIPW로 별도 처리
        home_axes = [
            ("02", 30.0),  # ND_FILTER
            ("05", 30.0),  # GRATING (가장 오래 걸림)
            ("04", 30.0),  # BEAM_SPLITTER
            ("01", 30.0),
        ]
        for axis, timeout in home_axes:
            name = self.AXES[axis]["name"]
            print(f"\n   [{axis} {name}] 홈 복귀 중...")
            ok, _ = self._send_command(axis, "SMMHF", "", timeout=3.0, retries=1)
            if not ok:
                print(f"   ❌ [{axis} {name}] SMMHF 실패")
                return False
            done = self._wait_motion_complete(axis, timeout=timeout, check_origin=True)
            if done:
                print(f"   ✅ [{axis} {name}] 홈 복귀 완료")
            else:
                print(f"   ❌ [{axis} {name}] 홈 복귀 타임아웃")
                return False

        # [2] 홈 복귀 후 운용 위치로 이동 (패킷 캡처 순서: 04→05→02)
        print("\n   [2/4] 홈 복귀 후 운용 위치 이동")
        post_positions = [
            ("04", "-0010000"),  # BEAM_SPLITTER 운용 대기
            ("05", "-0570000"),  # GRATING 운용 대기
            ("02", "-0602895"),  # ND_FILTER → 가이드빔(메인빔 차단) 위치
        ]
        for axis, pos in post_positions:
            name = self.AXES[axis]["name"]
            print(f"   [{axis} {name}] → {pos}")
            self._execute_command(axis, "SMMA", pos, timeout=15.0)

        # [3] SIPW — 축01 홈 복귀 트리거
        print("\n   [3/4] SIPW — 축01 트리거")
        ok, _ = self._send_command("00", "SIPW", "1", timeout=5.0, retries=1)
        if not ok:
            print("   ❌ SIPW 전송 실패")
            return False

        # [4] 축01 운용 위치 이동 (패킷 캡처 Row 114)
        print("\n   [4/4] 축01(LASER_FILTER) 운용 위치 이동")
        self._execute_command("01", "SMMA", "-0070600", timeout=20.0)

        print("\n🏠 4축 원점 복귀 전체 완료!")
        return True

    # ==============================================================
    # 레이저 제어 기능
    # ==============================================================
    def laser_on(self):
        """
        레이저 발사 (SSPW 1).

        주의: 먼저 set_power()로 파워를 설정해야 합니다.
        가이드빔 상태(ND 필터 극한 위치)에서는 빔이 차단됩니다.
        """
        if self._power_set:
            print("⚡ 레이저 발사 준비 (축04 측정 위치 이동 중...)")
            self._execute_command("04", "SMMA", "-0303764", timeout=10.0)
        else:
            print("🔦 가이드빔 ON (축04 이동 없음)")
        print("⚡ 레이저 ON (SSPW 1)")
        self._send_command("00", "SSPW", "1", timeout=3.0)
        self.is_on = True

    def laser_off(self):
        print("🛑 레이저 정지 (OFF)")
        self._send_command("00", "SSPW", "0", timeout=3.0)
        self.is_on = False
        self._power_set = False
        self.set_guide_beam()

    # ==============================================================
    # ND 필터 투과율 ↔ 펄스 위치 변환 (config [ND_FILTER] Mode=1 보간)
    # ==============================================================
    def _percent_to_pulse(self, percent):
        """
        목표 투과율(%) → ND 필터 펄스 위치.
        log10(%) 기준 선형 보간 (OD가 위치에 대해 거의 선형이라 정확도 높음).
        """
        percent = max(self.ND_MIN_PCT, min(self.ND_MAX_PCT, float(percent)))
        lt = math.log10(percent)
        cal = self.ND_CAL  # % 내림차순 → log10(%)도 내림차순
        for i in range(len(cal) - 1):
            t_hi, p_hi = cal[i]       # 더 밝은(투과율 높은) 점
            t_lo, p_lo = cal[i + 1]   # 더 어두운(투과율 낮은) 점
            lhi, llo = math.log10(t_hi), math.log10(t_lo)
            if llo <= lt <= lhi:
                frac = (lt - lhi) / (llo - lhi)
                return int(round(p_hi + frac * (p_lo - p_hi)))
        # 표 범위 밖(클램프 후엔 도달 안 함) → 경계값
        return cal[0][1] if lt >= math.log10(cal[0][0]) else cal[-1][1]

    def pulse_to_percent(self, pulse):
        """펄스 위치 → 예상 투과율(%). 진단/로깅용 역변환."""
        cal = self.ND_CAL
        for i in range(len(cal) - 1):
            p_hi, p_lo = cal[i][1], cal[i + 1][1]
            if p_hi <= pulse <= p_lo:
                lhi, llo = math.log10(cal[i][0]), math.log10(cal[i + 1][0])
                frac = (pulse - p_hi) / (p_lo - p_hi)
                return 10 ** (lhi + frac * (llo - lhi))
        return cal[0][0] if pulse <= cal[0][1] else cal[-1][0]

    def set_power(self, percent):
        """
        메인 레이저 출력 연속 조절 (ND 필터, 축02 SMMA).

        percent: 투과율 % (0.004 ~ 100, 실수 허용). 예) 1, 2.5, 37, 100.

        config 캘리브레이션(ND_CAL)을 log-선형 보간하여 임의 %를 펄스
        위치로 변환한다. 기존 이산 power_map(20/40/60/80/100)을 대체하며,
        해당 값들도 그대로 동작한다(하위호환).
        """
        if not isinstance(percent, (int, float)):
            print("⚠️ percent는 숫자여야 합니다.")
            return False

        clamped = max(self.ND_MIN_PCT, min(self.ND_MAX_PCT, float(percent)))
        if clamped != float(percent):
            print(f"⚠️ {percent}% → 허용 범위로 보정: {clamped}% "
                  f"(허용 {self.ND_MIN_PCT}~{self.ND_MAX_PCT})")

        pulse = self._percent_to_pulse(clamped)
        target_pos = f"-{pulse:07d}"

        print(f"⚙️ 메인 레이저 파워 {clamped}% 설정 중... "
              f"(ND 위치 {target_pos}, 모터 이동 대기)")
        success = self._execute_command("02", "SMMA", target_pos, timeout=15.0)
        if success:
            self._power_set = True
            self.power_pct = clamped
            time.sleep(0.1)  # 모터 안정화
            print(f"   -> 파워 설정 완료! (≈{clamped}% 투과)")
        else:
            print("   -> 파워 설정 실패 (응답 없음)")
        return success

    def set_guide_beam(self):
        """
        가이드빔 대기 상태로 전환.
        - 축04(BEAM_SPLITTER) → -0612828 (대기 위치)
        - 축02(ND_FILTER)    → -0602895 (메인 빔 차단)
        """
        self._power_set = False
        print("🔦 가이드빔 모드로 전환 중...")
        self._execute_command("04", "SMMA", "-0612828", timeout=10.0)
        self._execute_command("02", "SMMA", "-0602895", timeout=15.0)
        time.sleep(0.1)  # 모터 안정화
        print("   -> 🔦 가이드빔 전환 완료!")

    # ==============================================================
    # 종료 시퀀스
    # ==============================================================
    def close(self):
        """
        안전 종료 시퀀스.
          1. SSPW 0 — 레이저 OFF
          2. set_guide_beam() — 축04 대기 위치, 축02 빔 차단
        """
        if self.ser and self.ser.is_open:
            print("\n🔌 안전 종료 시퀀스 시작...")

            # 1. 레이저 OFF
            print("   [1/2] 레이저 OFF")
            self._send_command("00", "SSPW", "0", timeout=5.0, retries=1)
            self.is_on = False
            time.sleep(0.1)

            # # 2. 가이드빔 대기 상태 (축04 → -0612828, 축02 → -0602895)
            # print("   [2/2] 가이드빔 대기 위치로 복귀")
            # self.set_guide_beam()
            time.sleep(0.1)

            self.ser.close()
            print("🔌 장비 연결 해제 완료.")


def main():
    print("\n" + "=" * 55)
    print("   Raman Laser Control Terminal")
    print("      Rays-ON.exe 패킷 캡처 + Fastech SDK 헤더 기반")
    print("      GMMS 폴링 / ercd 자동감지 / GMCP 위치검증")
    print("      ND 필터 연속 파워 제어 (0.004 ~ 100 %)")
    print("=" * 55)

    port_input = input("연결할 COM 포트를 입력하세요 (기본값: COM4): ").strip()
    port = port_input.upper() if port_input else 'COM4'

    laser = LaserController(port=port)
    if not laser.ser:
        sys.exit()

    while True:
        print("\n" + "-" * 50)
        print("-" * 50)
        print("1. ⚡ 레이저 켜기 (ON)")
        print("2. 🛑 레이저 끄기 (OFF)")
        print("3. ⚙️ 파워 설정 (0.004 ~ 100 %, 실수 허용)")
        print("4. 🔦 가이드빔 켜기")
        print("Q. 🚪 프로그램 종료")
        print("-" * 50)

        choice = input("명령 선택: ").strip().upper()

        if choice == '1':
            laser.laser_on()
        elif choice == '2':
            laser.laser_off()
        elif choice == '3':
            val = input("파워(%)를 입력하세요 (0.004 ~ 100, 예: 1, 2.5, 40): ").strip()
            try:
                laser.set_power(float(val))
            except ValueError:
                print("⚠️ 숫자로 입력해주세요.")
        elif choice == '4':
            laser.set_guide_beam()
        elif choice == 'Q':
            laser.close()
            print("터미널을 종료합니다. 수고하셨습니다!")
            break
        else:
            print("⚠️ 잘못된 입력입니다. 다시 선택해주세요.")


if __name__ == "__main__":
    main()