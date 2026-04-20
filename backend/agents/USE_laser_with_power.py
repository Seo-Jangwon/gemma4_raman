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
        self.set_guide_beam()

    def _connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
            print(f"✅ [{self.port}] 레이저 컨트롤러 연결 성공!")
            
            # [추가됨] 장비 연결 직후, 04번 셔터를 한 번 열어주는 초기화 과정 (로그 1번 줄 반영)
            print("  ")
            self._execute_command("04", "SMMA", "-0612828", timeout=10.0)
            
        except Exception as e:
            print(f"❌ [{self.port}] 연결 실패: {e}")
            print("포트가 사용 중이거나 연결되지 않았습니다.")

    def _make_packet(self, target_id, cmd, arg):
        """체크섬 자동 계산 및 패킷 조립"""
        body = f"{target_id}{cmd}{arg}"
        ascii_sum = sum(ord(c) for c in body)
        checksum = f"{ascii_sum % 256:02x}"
        packet = f"@{body}{checksum}$"
        return packet.encode('utf-8')

    def _execute_command(self, target_id, cmd, arg, timeout=15.0, retries=2):
        """
        패킷을 전송하고, 장비로부터 '소문자 변환된 완료 응답'이 올 때까지 스마트하게 대기합니다.
        ercd(장비 알람) 응답 시 즉시 감지하고 최대 retries회 재시도합니다.
        """
        if not (self.ser and self.ser.is_open):
            print("⚠️ 포트가 연결되어 있지 않습니다.")
            return False

        expected_body = f"{target_id}{cmd.lower()}{arg}"

        for attempt in range(retries + 1):
            # 1. 패킷 생성 및 전송
            packet = self._make_packet(target_id, cmd, arg)

            # 전송 전에 버퍼에 남아있는 쓰레기 응답들 싹 비우기
            if self.ser.in_waiting:
                self.ser.read(self.ser.in_waiting)

            print(f"   [데이터 송신] {packet.decode()}")
            self.ser.write(packet)

            start_time = time.time()
            buffer = ""

            # 2. 응답 대기 루프
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
                            # 장비 알람 상태 — 타임아웃 기다리지 않고 즉시 감지
                            print(f"   ❌ 장비 에러(ercd) [{attempt + 1}/{retries + 1}회]")
                            break
                        else:
                            buffer = ""
                else:
                    time.sleep(0.01)
            else:
                # 타임아웃
                print(f"⚠️ 시간 초과 [{attempt + 1}/{retries + 1}회] (명령: {cmd}, 기다린응답: {expected_body})")

            # 재시도 여부 판단
            if attempt < retries:
                print(f"   ⏳ 300ms 대기 후 재시도...")
                time.sleep(0.3)
            else:
                break

        print(f"⚠️ 최종 실패: {cmd} 명령이 {retries + 1}회 모두 실패")
        return False

    # ==========================================
    # 레이저 제어 기능
    # ==========================================
    def laser_on(self):
        print("⚡ 레이저 발사 (ON)")
        # 만약 0.x초만 켜지고 꺼진다면 "1"을 "100"이나 "1000"으로 수정하세요
        self._execute_command("00", "SSPW", "1")

    def laser_off(self):
        self.set_guide_beam()
        print("🛑 레이저 정지 (OFF)")
        self._execute_command("00", "SSPW", "0")

    def set_power(self, power_level):

        """모터 좌표를 이동시켜 레이저 출력 조절"""
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
                print("   -> 파워 설정 완벽하게 끝남!")
            else:
                print("   -> 파워 설정 실패 (응답 없음)")
        else:
            print("⚠️ 지원하지 않는 파워 레벨입니다. (20, 40, 60, 80, 100 중 선택)")

    def set_guide_beam(self):
        """가이드빔용 레이저 모터 제어 (Target: 02 극한 위치)"""
        # 로그에서 찾아낸 진짜 가이드빔 좌표
        target_pos = "-0602895" 
        print(f"🔦 가이드빔 모드로 전환 중... (메인 필터를 극한으로 닫음)")
        
        success = self._execute_command("02", "SMMA", target_pos, timeout=15.0)
        
        if success:
            print("   -> 🔦 가이드빔 전환 완료!")
        else:
            print("   -> ❌ 가이드빔 전환 실패 (응답 없음)")

    def close(self):
        if self.ser and self.ser.is_open:
            self.laser_off() # 안전을 위해 끄고 종료
            self.ser.close()
            print("🔌 장비 연결 해제 완료.")


def main():
    print("\n" + "="*40)
    print("   🚀 Raman Laser Control Terminal (최종 완성본)")
    print("="*40)
    
    port_input = input("연결할 COM 포트를 입력하세요 (기본값: COM4): ").strip()
    port = port_input.upper() if port_input else 'COM4'
    
    laser = LaserController(port=port)
    if not laser.ser:
        sys.exit()

    while True:
        print("\n" + "-"*40)
        print("1. ⚡ 레이저 켜기 (ON)")
        print("2. 🛑 레이저 끄기 (OFF)")
        print("3. ⚙️ 파워 설정 (20 / 40 / 60 / 80 / 100)")
        print("4. 🔦 가이드빔 켜기 (안전 시퀀스 적용)")
        print("Q. 🚪 프로그램 종료")
        print("-" * 40)
        
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