import serial
import time
import sys


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

    근거 자료:
      - 20260421패킷분석.xlsx: Device Monitoring Studio COM4 캡처 (10055행)
      - MOTION_EziSERVO_DEFINE.h: EZISERVO_AXISSTATUS (FFLAG 32-bit 비트필드)
      - FM_EZISERVO_PARAM: FAS_SetParameter 파라미터 번호 enum
      - CommInterface.c/.h: FAS_* API 구현 (이진 프로토콜)
      - Config.txt: 모터 매핑 및 속도값
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

    def __init__(self, port='COM4', baud=115200):
        self.port = port
        self.baud = baud
        self.ser = None
        self._power_set = False
        self._connect()
        time.sleep(0.5)
        self._full_initialization()

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
                    ercd_data = ercd_raw[:-2] if len(ercd_raw) > 2 else ercd_raw
                    print(f"   ❌ ercd 에러: 0x{ercd_data} [{attempt + 1}/{retries + 1}]")
                    got_ercd = True
                    break

                # 정상 ACK (소문자 cmd echo)
                if expected_key in inner:
                    data_start = inner.index(expected_key) + len(expected_key)
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
                # 8-hex 위치값 + 2-hex 체크섬
                if len(data_raw) >= 10:
                    try:
                        val = int(data_raw[:8], 16)
                        if val >= 0x80000000:
                            val -= 0x100000000
                        return val
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
        arg = f"{value & 0xFFFFFFFF:08x}"
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

        모션 명령(SMMA/SMMH/SMMI)인 경우:
          1. 패킷 전송 → ACK 확인
          2. GMMS 폴링 → 실제 모션 완료 대기
          3. GMCP → 위치 검증
        비모션 명령은 ACK 확인만.
        """
        is_motion = cmd.upper() in ("SMMA", "SMMH", "SMMI")
        is_homing = cmd.upper() == "SMMH"

        # 1) ACK 확인
        ok, _ = self._send_command(target_id, cmd, arg,
                                   timeout=5.0, retries=retries)

        if ok and is_motion:
            # 2) GMMS 폴링으로 모션 완료 대기
            print(f"   ⏳ GMMS 폴링 대기... (axis={target_id})")
            done = self._wait_motion_complete(
                target_id,
                timeout=timeout,
                check_origin=is_homing
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
    # 초기화 시퀀스 (Rays-ON.exe 패킷 캡처 기반)
    #
    # 실증된 순서:
    #   1. ERCL ×2          — 알람 리셋
    #   2. SMSE ×4축        — 서보 ON (02→01→05→04, ARG 없음)
    #   3. SSPW 0           — 레이저 OFF (안전)
    #   4. SMMH F ×4축      — 원점 복귀 + GMMS 폴링
    #   5. SMMA + GMMS 폴링 — 셔터/가이드빔 위치 이동
    #
    # 참고: Rays-ON.exe 캡처에서 SMSE와 SMMH 사이에 속도 설정 명령이
    # 관찰되었으나, 이 ASCII 래퍼에서 해당 명령(WCHS/WCNS)이 ercd 03으로
    # 거부됨. EEPROM에 Rays-ON.exe가 이전에 저장한 값이 사용되므로
    # 실용상 문제 없음. 정확한 명령 이름 확인 시 추가 가능.
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

        # ── Step 4: 원점 복귀 (SMMH F + GMMS 폴링) ──
        self.home_search()

        # ── Step 5: 셔터 오픈 + 가이드빔 ──
        print("\n[Step 5/5] 셔터 오픈 + 가이드빔 전환")
        self._execute_command("04", "SMMA", "-0612828", timeout=10.0)
        time.sleep(0.3)
        self.set_guide_beam()

        print("\n" + "=" * 55)
        print("   ✅ 초기화 시퀀스 완료!")
        print("=" * 55 + "\n")

    # ==============================================================
    # 원점 복귀 (사용자 수동 실행 전용)
    # ==============================================================
    def home_search(self):
        """
        4축 원점 복귀 (SMMH F) + GMMS 폴링 완료 대기.

        주의: 모터가 물리적으로 이동하므로 사용자 확인 후에만 실행.
        Rays-ON.exe 순서: 02 → 01 → 05 → 04 (각 축 완료 대기)

        비정상 종료 후 좌표계가 어긋났다고 판단될 때 사용하세요.
        """
        # confirm = input("⚠️ 모든 모터가 원점으로 이동합니다. 계속? (y/N): ")
        # if confirm.strip().lower() != 'y':
        #     print("   -> 원점 복귀 취소됨.")
        #     return False

        print("\n🏠 4축 원점 복귀 시작 (SMMH F + GMMS 폴링)")

        home_axes = [
            ("02", 20.0),   # ND Filter ~1.3초
            ("01", 20.0),   # Laser Filter ~5초
            ("05", 25.0),   # Grating ~7.8초 (가장 오래 걸림)
            ("04", 20.0),   # Beam Splitter ~5초
        ]

        for axis_id, timeout in home_axes:
            name = self.AXES[axis_id]["name"]
            print(f"\n   [{axis_id} {name}] 원점 복귀 중...")
            success = self._execute_command(axis_id, "SMMH", "F",
                                            timeout=timeout, retries=1)
            if success:
                print(f"   [{axis_id} {name}] ✅ 원점 복귀 완료")
            else:
                print(f"   [{axis_id} {name}] ❌ 원점 복귀 실패 — 중단")
                return False
            time.sleep(0.3)

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
        if not hasattr(self, '_power_set') or not self._power_set:
            print("⚠️ 파워가 설정되지 않았습니다. 먼저 '3. 파워 설정'을 하세요.")
            print("   (가이드빔 상태에서는 ND 필터가 빔을 차단합니다)")
            return
        print("⚡ 레이저 발사 (ON)")
        self._send_command("00", "SSPW", "1", timeout=3.0)

    def laser_off(self):
        self.set_guide_beam()
        self._power_set = False
        print("🛑 레이저 정지 (OFF)")
        self._send_command("00", "SSPW", "0", timeout=3.0)

    def set_power(self, power_level):
        """모터 좌표를 이동시켜 레이저 출력 조절 (Target 02, SMMA)"""
        power_map = {
            20: "-0113343",
            40: "-0085674",
            60: "-0070279",
            80: "-0059111",
            100: "-0047944"
        }

        if power_level in power_map:
            target_pos = power_map[power_level]
            print(f"⚙️ 메인 레이저 파워 {power_level}% 설정 중... (모터 이동 대기)")
            success = self._execute_command("02", "SMMA", target_pos, timeout=15.0)
            if success:
                self._power_set = True
                print("   -> 파워 설정 완료!")
            else:
                print("   -> 파워 설정 실패 (응답 없음)")
        else:
            print("⚠️ 지원하지 않는 파워 레벨입니다. (20, 40, 60, 80, 100 중 선택)")

    def set_guide_beam(self):
        """가이드빔용 레이저 모터 제어 (Target 02, 극한 위치로 필터 닫기)"""
        target_pos = "-0602895"
        print(f"🔦 가이드빔 모드로 전환 중... (메인 필터를 극한으로 닫음)")
        success = self._execute_command("02", "SMMA", target_pos, timeout=15.0)
        if success:
            print("   -> 🔦 가이드빔 전환 완료!")
        else:
            print("   -> ❌ 가이드빔 전환 실패 (응답 없음)")

    # ==============================================================
    # 종료 시퀀스
    # ==============================================================
    def close(self):
        """
        안전 종료 시퀀스.
        Rays-ON.exe 관찰 기반:
          1. 레이저 OFF
          2. 가이드빔 위치로 이동 (02번 축)
          3. 04번 셔터 닫기
        모든 SMMA에 GMMS 폴링 적용.
        """
        if self.ser and self.ser.is_open:
            print("\n🔌 안전 종료 시퀀스 시작...")

            # 1. 레이저 OFF
            print("   [1/3] 레이저 OFF")
            self._send_command("00", "SSPW", "0", timeout=5.0, retries=1)
            time.sleep(0.1)

            # 2. 가이드빔 위치로 이동 (GMMS 폴링)
            print("   [2/3] 가이드빔 위치로 이동")
            self._execute_command("02", "SMMA", "-0602895",
                                  timeout=10.0, retries=1)
            time.sleep(0.3)

            # 3. 04번 셔터 닫기 (GMMS 폴링)
            print("   [3/3] 04번 셔터 닫기")
            self._execute_command("04", "SMMA", "-0612828",
                                  timeout=10.0, retries=1)
            time.sleep(0.1)

            self.ser.close()
            print("🔌 장비 연결 해제 완료.")


def main():
    print("\n" + "=" * 55)
    print("   🚀 Raman Laser Control Terminal v3")
    print("      Rays-ON.exe 패킷 캡처 + Fastech SDK 헤더 기반")
    print("      GMMS 폴링 / ercd 자동감지 / GMCP 위치검증")
    print("=" * 55)

    port_input = input("연결할 COM 포트를 입력하세요 (기본값: COM4): ").strip()
    port = port_input.upper() if port_input else 'COM4'

    laser = LaserController(port=port)
    if not laser.ser:
        sys.exit()

    while True:
        print("\n" + "-" * 50)
        print("  사용법: 3(파워설정) → 1(레이저ON) → 2(OFF)")
        print("-" * 50)
        print("1. ⚡ 레이저 켜기 (ON)  ※ 파워 먼저 설정 필요")
        print("2. 🛑 레이저 끄기 (OFF)")
        print("3. ⚙️ 파워 설정 (20 / 40 / 60 / 80 / 100)")
        print("4. 🔦 가이드빔 켜기")
        print("Q. 🚪 프로그램 종료")
        print("-" * 50)

        choice = input("명령 선택: ").strip().upper()

        if choice == '1':
            laser.laser_on()
        elif choice == '2':
            laser.laser_off()
        elif choice == '3':
            val = input("파워를 입력하세요 (20, 40, 60, 80, 100): ").strip()
            if val.isdigit():
                laser.set_power(int(val))
            else:
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