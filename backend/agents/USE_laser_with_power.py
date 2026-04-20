import serial
import time
import sys

class LaserController:
    def __init__(self, port='COM4', baud=115200):
        self.port = port
        self.baud = baud
        self.ser = None
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
    # 패킷 통신 핵심 함수
    # ==============================================================
    def _make_packet(self, target_id, cmd, arg):
        """체크섬 자동 계산 및 패킷 조립"""
        body = f"{target_id}{cmd}{arg}"
        ascii_sum = sum(ord(c) for c in body)
        checksum = f"{ascii_sum % 256:02x}"
        packet = f"@{body}{checksum}$"
        return packet.encode('utf-8')

    def _execute_command(self, target_id, cmd, arg, timeout=15.0, retries=2):
        """
        패킷을 전송하고, 장비로부터 '소문자 변환된 완료 응답'이 올 때까지 대기합니다.
        ercd(장비 알람) 응답 시 즉시 감지하고 최대 retries회 재시도합니다.
        """
        if not (self.ser and self.ser.is_open):
            print("⚠️ 포트가 연결되어 있지 않습니다.")
            return False

        expected_body = f"{target_id}{cmd.lower()}{arg}"

        for attempt in range(retries + 1):
            packet = self._make_packet(target_id, cmd, arg)

            # 전송 전에 버퍼에 남아있는 쓰레기 응답들 비우기
            if self.ser.in_waiting:
                self.ser.read(self.ser.in_waiting)

            print(f"   [데이터 송신] {packet.decode()}")
            self.ser.write(packet)

            start_time = time.time()
            buffer = ""

            while (time.time() - start_time) < timeout:
                if self.ser.in_waiting:
                    chunk = self.ser.read(self.ser.in_waiting).decode(errors='ignore')
                    buffer += chunk

                    if '$' in buffer:
                        print(f"   [장비 수신 로그] {buffer.strip()}")

                        if expected_body in buffer:
                            print(f"   ✅ [목표 도달 확인] {expected_body} 일치!")
                            return True
                        elif "ercd" in buffer:
                            print(f"   ❌ 장비 에러(ercd) [{attempt + 1}/{retries + 1}회]")
                            break
                        else:
                            buffer = ""
                else:
                    time.sleep(0.01)
            else:
                print(f"⚠️ 시간 초과 [{attempt + 1}/{retries + 1}회] (명령: {cmd}, 기다린응답: {expected_body})")

            if attempt < retries:
                print(f"   ⏳ 300ms 대기 후 재시도...")
                time.sleep(0.3)
            else:
                break

        print(f"⚠️ 최종 실패: {cmd} 명령이 {retries + 1}회 모두 실패")
        return False

    # ==============================================================
    # 초기화 시퀀스 (Fastech DLL 문서 기반)
    # ==============================================================
    def _full_initialization(self):
        """
        Rays-ON.exe 패킷 분석 + Fastech DLL 문서 기반 초기화 시퀀스.
        
        순서:
          1. ERCL  (FAS_ServoAlarmReset) — 알람 상태 리셋
          2. SMSE  (FAS_ServoEnable)     — 4축 서보 활성화
          3. SSPW0                       — 레이저 OFF 확인 (안전)
          4. 04번 셔터 오픈 + 가이드빔 전환
        """
        if not (self.ser and self.ser.is_open):
            return

        print("\n" + "=" * 50)
        print("   🔧 장비 초기화 시퀀스 시작")
        print("=" * 50)

        # ── Step 1: 알람 리셋 (ERCL × 2회) ──
        # Fastech DLL: FAS_ServoAlarmReset
        # Rays-ON.exe가 부팅 시 반드시 2회 전송.
        # 이전 세션의 알람을 제거하지 않으면 이후 명령에 ercd 발생.
        print("\n[Step 1/4] 알람 리셋 (ERCL × 2회) — FAS_ServoAlarmReset")
        self._execute_command("00", "ERCL", "", timeout=5.0, retries=0)
        time.sleep(0.1)
        self._execute_command("00", "ERCL", "", timeout=5.0, retries=0)
        time.sleep(0.1)

        # ── Step 2: 4축 서보 활성화 (SMSE) ──
        # Fastech DLL: FAS_ServoEnable (arg: 1=ON)
        # 서보가 비활성 상태면 SMMA 모터 이동 명령이 동작하지 않음.
        # Rays-ON.exe 순서: 02 → 01 → 05 → 04
        print("\n[Step 2/4] 4축 서보 활성화 (SMSE) — FAS_ServoEnable")
        for axis in ["02", "01", "05", "04"]:
            self._execute_command(axis, "SMSE", "", timeout=5.0, retries=1)
            time.sleep(0.05)

        # ── Step 3: 레이저 OFF 확인 (안전) ──
        print("\n[Step 3/4] 레이저 OFF 확인 (SSPW 0)")
        self._execute_command("00", "SSPW", "0", timeout=5.0, retries=1)
        time.sleep(0.05)

        # ── Step 4: 셔터 오픈 + 가이드빔 전환 ──
        print("\n[Step 4/4] 셔터 오픈 + 가이드빔 전환")
        self._execute_command("04", "SMMA", "-0612828", timeout=10.0)
        time.sleep(0.3)
        self.set_guide_beam()

        print("\n" + "=" * 50)
        print("   ✅ 초기화 시퀀스 완료!")
        print("=" * 50 + "\n")

    # ==============================================================
    # 원점 복귀 (사용자 수동 실행 전용)
    # ==============================================================
    def home_search(self):
        """
        4축 원점 복귀 (SMMH) — Fastech DLL: FAS_MoveOriginSingleAxis
        
        주의: 모터가 물리적으로 이동하므로 사용자 확인 후에만 실행.
        Rays-ON.exe 순서: 02 → 01 → 05 → 04 (각 축 완료 대기)
        
        비정상 종료 후 좌표계가 어긋났다고 판단될 때 사용하세요.
        """
        confirm = input("⚠️ 모든 모터가 원점으로 이동합니다. 계속하시겠습니까? (y/N): ").strip().lower()
        if confirm != 'y':
            print("   -> 원점 복귀 취소됨.")
            return

        print("\n🏠 4축 원점 복귀 시작 (SMMH) — FAS_MoveOriginSingleAxis")

        # Rays-ON.exe 관찰: 각 축에 SMMH 전송 → GMMS 폴링으로 완료 대기 → 다음 축
        # 02축: ~1.3초, 01축: ~5초, 05축: ~7.8초, 04축: ~5초 소요
        home_axes = [
            ("02", "F", 15.0),   # ARG "F", Rays-ON.exe 응답: @02smmhF5d$
            ("01", "F", 15.0),   # ARG "F", Rays-ON.exe 응답: @01smmhF5c$
            ("05", "F", 20.0),   # ARG "F", 가장 오래 걸림
            ("04", "F", 15.0),   # ARG "F", Rays-ON.exe 응답: @04smmhFdf$
        ]

        for axis, arg, timeout in home_axes:
            print(f"\n   [{axis}번 축] 원점 복귀 중...")
            success = self._execute_command(axis, "SMMH", arg, timeout=timeout, retries=1)
            if success:
                print(f"   [{axis}번 축] ✅ 원점 복귀 완료")
            else:
                print(f"   [{axis}번 축] ❌ 원점 복귀 실패 — 중단합니다.")
                return
            time.sleep(0.3)

        print("\n🏠 4축 원점 복귀 전체 완료!")

    # ==============================================================
    # 레이저 제어 기능
    # ==============================================================
    def laser_on(self):
        print("⚡ 레이저 발사 (ON)")
        self._execute_command("00", "SSPW", "1")

    def laser_off(self):
        self.set_guide_beam()
        print("🛑 레이저 정지 (OFF)")
        self._execute_command("00", "SSPW", "0")

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
        """
        if self.ser and self.ser.is_open:
            print("\n🔌 안전 종료 시퀀스 시작...")

            # 1. 레이저 OFF
            print("   [1/3] 레이저 OFF")
            self._execute_command("00", "SSPW", "0", timeout=5.0, retries=1)
            time.sleep(0.1)

            # 2. 가이드빔 위치로 이동
            print("   [2/3] 가이드빔 위치로 이동")
            self._execute_command("02", "SMMA", "-0602895", timeout=10.0, retries=1)
            time.sleep(0.3)

            # 3. 04번 셔터 닫기 (Rays-ON.exe 종료 시 관찰됨)
            print("   [3/3] 04번 셔터 닫기")
            self._execute_command("04", "SMMA", "-0612828", timeout=10.0, retries=1)
            time.sleep(0.1)

            self.ser.close()
            print("🔌 장비 연결 해제 완료.")


def main():
    print("\n" + "=" * 50)
    print("   🚀 Raman Laser Control Terminal v2")
    print("      (Rays-ON.exe 패킷 분석 + Fastech 문서 기반)")
    print("=" * 50)

    port_input = input("연결할 COM 포트를 입력하세요 (기본값: COM4): ").strip()
    port = port_input.upper() if port_input else 'COM4'

    laser = LaserController(port=port)
    if not laser.ser:
        sys.exit()

    while True:
        print("\n" + "-" * 50)
        print("1. ⚡ 레이저 켜기 (ON)")
        print("2. 🛑 레이저 끄기 (OFF)")
        print("3. ⚙️ 파워 설정 (20 / 40 / 60 / 80 / 100)")
        print("4. 🔦 가이드빔 켜기")
        # print("5. 🏠 원점 복귀 (SMMH — 모터 물리적 이동 주의)")
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
        # elif choice == '5':
        #     laser.home_search()
        elif choice == 'Q':
            laser.close()
            print("터미널을 종료합니다. 수고하셨습니다!")
            break
        else:
            print("⚠️ 잘못된 입력입니다. 다시 선택해주세요.")

if __name__ == "__main__":
    main()