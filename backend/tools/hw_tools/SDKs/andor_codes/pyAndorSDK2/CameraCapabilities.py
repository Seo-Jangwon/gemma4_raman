from backend.tools.hw_tools.SDKs.andor_codes.pyAndorSDK2 import atmcd_capabilities
from pyAndorSDK2 import atmcd


class CapabilityHelper:
    """연결된 카메라의 상세 기능 정보를 조회하고 출력하는 헬퍼 클래스.

    This class provides several methods for extracting information about the current device.

    Attributes
    ----------
        param1 : atmcd
            활성화된 atmcd/sdk 카메라 객체
    """

    def __init__(self, sdk: atmcd) -> None:
        """CapabilityHelper를 초기화하고 카메라의 기능 정보를 조회한다."""
        (_, caps) = sdk.GetCapabilities()
        self.caps = caps
        self.sdk = sdk

    def print_all(self):
        """카메라가 지원하는 모든 모드와 기능을 콘솔에 출력한다."""
        self.print_acquisition_modes()
        print("")
        self.print_read_modes()
        print("")
        self.print_trigger_modes()
        print("")
        self.print_camera_types()
        print("")
        self.print_pixel_modes()
        print("")
        self.print_set_functions()
        print("")
        self.print_get_functions()
        print("")
        self.print_features()
        print("")
        self.print_pci_card()
        print("")
        self.print_emgain_compatilibity()
        print("")
        self.print_FTRead_modes()
        print("")
        self.print_features2()
        print("")

    def print_acquisition_modes(self):
        """카메라가 지원하는 획득 모드(단일/누적/키네틱 등)를 출력한다."""
        mode = atmcd_capabilities.acquistionModes
        val = self.caps.ulAcqModes
        print("Available Acquisition modes")
        for i in range(0, len(bin(val))):
            self.__get_bit(val, i, mode)

    def print_read_modes(self):
        """카메라가 지원하는 읽기 모드(FVB/이미지/트랙 등)를 출력한다."""
        mode = atmcd_capabilities.readmodes
        val = self.caps.ulReadModes
        print("Available Read modes")
        for i in range(0, len(bin(val))):
            self.__get_bit(val, i, mode)

    def print_trigger_modes(self):
        """카메라가 지원하는 픽셀 모드를 출력한다."""
        mode = atmcd_capabilities.PixelModes
        val = self.caps.ulPixelMode

        print("Available Pixel Modes")
        for i in range(0, len(bin(val))):
            self.__get_bit(val, i, mode)

    def print_camera_types(self):
        """카메라 모델 타입을 출력한다."""
        mode = atmcd_capabilities.cameratype
        val = self.caps.ulCameraType

        print("Camera Type")
        for i in range(0, len(bin(val))):
            self.__get_bit(val, i, mode)

    def print_pixel_modes(self):
        """카메라가 지원하는 픽셀 비트 깊이 및 색상 모드를 출력한다."""
        mode = atmcd_capabilities.PixelModes
        val = self.caps.ulPixelMode

        print("Available Pixel Modes")
        for i in range(0, len(bin(val))):
            self.__get_bit(val, i, mode)

    def print_set_functions(self):
        """카메라에서 설정 가능한 기능 목록을 출력한다."""
        mode = atmcd_capabilities.SetFunctions
        val = self.caps.ulSetFunctions

        print("Available Set Functions")
        for i in range(0, len(bin(val))):
            self.__get_bit(val, i, mode)

    def print_get_functions(self):
        """카메라에서 조회 가능한 기능 목록을 출력한다."""
        mode = atmcd_capabilities.GetFunctions
        val = self.caps.ulGetFunctions

        print("Available get Functions")
        for i in range(0, len(bin(val))):
            self.__get_bit(val, i, mode)

    def print_features(self):
        """카메라가 지원하는 부가 기능(스풀링, 셔터, 팬 등) 목록을 출력한다."""
        mode = atmcd_capabilities.Features
        val = self.caps.ulFeatures

        print("Available Features")
        for i in range(0, len(bin(val))):
            self.__get_bit(val, i, mode)

    def print_pci_card(self):
        """PCI 카드의 최대 전송 속도(Hz)를 출력한다."""
        val = self.caps.ulPCICard

        print("Pci max speed in Hz ")
        print("- {}".format(val))

    def print_step_modes(self):
        """키네틱 시리즈의 스텝 간격 변화 방식을 출력한다."""
        mode = atmcd_capabilities.stepmodes
        val = self.caps.ulstepmodes

        print("Available Functions")
        for i in range(0, len(bin(val))):
            self.__get_bit(val, i, mode)

    def print_emgain_compatilibity(self):
        """카메라가 지원하는 EM 이득 모드(8비트/12비트 등)를 출력한다."""
        mode = atmcd_capabilities.EmGainModes
        val = self.caps.ulEMGainCapability

        print("Available Emgain compatibility Modes")
        for i in range(0, len(bin(val))):
            self.__get_bit(val, i, mode)

    def print_FTRead_modes(self):
        """프레임 전송(Frame Transfer) 모드와 호환되는 읽기 모드 목록을 출력한다."""
        mode = atmcd_capabilities.readmodes
        val = self.caps.ulFeatures2

        print("Available FT compatible Read Modes")
        for i in range(0, len(bin(val))):
            self.__get_bit(val, i, mode)

    def print_features2(self):
        """카메라가 지원하는 추가 기능(ulFeatures2) 목록을 출력한다."""
        mode = atmcd_capabilities.Features
        val = self.caps.ulFeatures

        print("Available Features")
        for i in range(0, len(bin(val))):
            self.__get_bit(val, i, mode)

    def __get_bit(self, number, bitNumber, mode):
        """정수 값에서 특정 비트를 추출하여 해당 enum 값을 조회한다."""
        mask = 1 << bitNumber
        bit = (number & mask) >> bitNumber
        if bit != 1:
            pass
        else:
            self.__iterate_through_enum(int(mask), mode)

    def __iterate_through_enum(self, num, mode):
        """주어진 정수 값과 일치하는 enum 멤버를 찾아 이름을 출력한다."""
        for x in mode:
            if x.value == num:
                temp = x.name.split('_', -1)
                print("- {},".format(temp[2]))
