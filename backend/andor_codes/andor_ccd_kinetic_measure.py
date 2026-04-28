from ScopeFoundry import Measurement
from ScopeFoundry import h5_io
from ScopeFoundry.helper_funcs import load_qt_ui_file, sibling_path
import pyqtgraph as pg
import numpy as np
import time

class AndorCCDKineticMeasure(Measurement): # Measurement 클래스를 상속받아 'AndorCCDKineticMeasure'라는 새로운 측정 모듈 생성
    
    name = 'andor_ccd_kinetic' # 이 측정 모듈의 고유 식별자 지정.

    def setup(self): # 모듈이 로드될 때 설정 변수(Settings)들을 초기화하는 함수
        pass # 여기서는 추가로 설정할 고유 변수가 없으므로 비워둠
    
    def setup_figure(self): # GUI 구성
        
        # 현재 파이썬 스크립트와 같은 폴더에 있는 'andor_ccd_readout.ui' 파일을 읽어서 화면 창을 띄웁니다.
        ui = self.ui = load_qt_ui_file(sibling_path(__file__, 'andor_ccd_readout.ui'))
        
        ## ui connection (하드웨어 설정값과 화면 UI 위젯을 서로 연결해줍니다)
        andor = self.app.hardware['andor_ccd'] # ScopeFoundry 하드웨어 목록에서 Andor CCD 장치 객체를 가져옵니다.
        
        # 카메라의 각종 설정값(노출시간, EM 증폭, 온도 등)을 화면의 스핀박스(숫자입력칸) 및 텍스트와 연동합니다. 
        # 이렇게 연결하면 화면에서 숫자를 바꾸면 장치 설정이 바뀌고, 장치 설정이 바뀌면 화면 숫자도 바뀝니다.
        andor.settings.exposure_time.connect_to_widget(ui.andor_ccd_int_time_doubleSpinBox)
        andor.settings.em_gain.connect_to_widget(ui.andor_ccd_emgain_doubleSpinBox)
        andor.settings.temperature.connect_to_widget(ui.andor_ccd_temp_doubleSpinBox)
        andor.settings.ccd_status.connect_to_widget(ui.andor_ccd_status_label) # 현재 카메라 상태(대기중, 촬영중 등) 표시
        andor.settings.shutter_open.connect_to_widget(ui.andor_ccd_shutter_open_checkBox) # 셔터 열림/닫힘 체크박스
        
        # 백그라운드 빼기 기능을 체크박스에 연결하려던 흔적
        self.settings.bg_subtract.connect_to_widget(ui.andor_ccd_bgsub_checkBox)
        
        # '연속 촬영(Continuous)' 체크박스에 체크/해제 할 때마다 측정을 시작/중지하는 함수(self.start_stop)가 실행되도록 연결합니다.
        ui.andor_ccd_acquire_cont_checkBox.stateChanged.connect(self.start_stop)
        
        # (주석 처리됨) 버튼 클릭 시 백그라운드 촬영이나 단일 촬영을 시작하게 하려던 흔적입니다.
        #ui.andor_ccd_acq_bg_pushButton.clicked.connect(self.acquire_bg_start)
        #ui.andor_ccd_read_single_pushButton.clicked.connect(self.acquire_single_start)

        #### Plot window (데이터를 보여줄 그래프 창 설정 영역)
        self.graph_layout = pg.GraphicsLayoutWidget() # 여러 그래프를 바둑판처럼 배치할 수 있는 도화지(레이아웃)를 만듭니다.
        self.ui.plot_groupBox.layout().addWidget(self.graph_layout) # UI 파일에서 만들어둔 그룹박스 영역 안에 이 도화지를 집어넣습니다.
        
        # 1. 1D 스펙트럼 그래프 설정
        self.spec_plot = self.graph_layout.addPlot() # 도화지에 그래프 공간을 하나 추가합니다.
        self.spec_plot_line = self.spec_plot.plot([1,3,2,4,3,5]) # 그래프에 임시 데이터 뼈대를 그립니다. (나중에 실제 데이터로 업데이트됨)
        self.spec_plot.enableAutoRange() # 데이터 크기에 맞춰서 그래프의 축(X/Y) 범위가 자동으로 늘어나고 줄어들게 설정합니다.
        
        self.graph_layout.nextRow() # 다음 추가될 그래프는 아래 줄로 넘어가서 배치되도록 줄바꿈을 합니다.
        
        # 2. 2D 이미지 뷰어 설정
        self.img_plot = self.graph_layout.addPlot() # 도화지 아래 줄에 두 번째 그래프 공간(카메라 이미지용)을 추가합니다.
        
        # (주석 처리됨) 이미지 뷰어의 축 한계 범위를 수동으로 고정하려던 흔적입니다.
        #self.img_plot.getViewBox().setLimits(minXRange=-10, maxXRange=100, minYRange=-10, maxYRange=100)
        self.img_plot.showGrid(x=True, y=True) # 이미지 그래프 영역에 격자(그리드)를 표시합니다.
        self.img_plot.setAspectLocked(lock=True, ratio=1) # 1:1 비율을 고정해서 이미지가 납작하게 찌그러지지 않게 방지합니다.
        self.img_item = pg.ImageItem() # 실제 픽셀 데이터를 넣을 '이미지 아이템' 빈 틀을 만듭니다.
        self.img_plot.addItem(self.img_item) # 생성한 이미지 아이템을 공간에 추가합니다.

        # 3. 색상/대비 조절 막대(히스토그램) 설정
        self.hist_lut = pg.HistogramLUTItem() # 이미지의 대비(Contrast)와 밝기를 사용자가 시각적으로 조절할 수 있는 히스토그램 위젯을 만듭니다.
        self.hist_lut.autoHistogramRange() # 히스토그램의 범위도 데이터에 맞게 자동으로 조정되도록 합니다.
        self.hist_lut.setImageItem(self.img_item) # 이 조절 막대가 방금 만든 카메라 2D 이미지(img_item)에 적용되도록 연결합니다.
        self.graph_layout.addItem(self.hist_lut) # 이 조절 막대를 그래프 레이아웃 우측에 추가합니다.


    def run(self): # 실제로 측정을 시작(Start)하면 별도의 스레드에서 돌아가는 핵심 동작 로직입니다.
        ccd_hw = self.app.hardware['andor_ccd'] # ScopeFoundry의 하드웨어 객체를 가져옵니다.
        ccd_dev = ccd_hw.ccd_dev # 카메라 제조사(Andor)의 SDK와 직접 통신하는 하위 레벨 디바이스 객체를 가져옵니다.

        # 센서 정보 읽어오기
        N = ccd_dev.get_num_kinetics() # 키네틱 모드에서 총 몇 장의 이미지를 연속으로 찍을지 개수를 읽어옵니다.
        width_px = ccd_dev.Nx_ro # 센서 영역의 가로 픽셀 수를 읽어옵니다.
        height_px = ccd_dev.Ny_ro # 센서 영역의 세로 픽셀 수를 읽어옵니다.
        
        try:
            ccd_dev.start_acquisition() # 실제 카메라 하드웨어에 '촬영 시작!' 명령을 내립니다.

            # 카메라의 현재 상태를 읽어옵니다 (예: 'IDLE', 'ACQUIRING' 등)
            stat = ccd_hw.settings.ccd_status.read_from_hardware()
            
            # 카메라가 데이터를 찍고 있는 동안('ACQUIRING') 계속 반복해서 검사합니다 (Polling 방식)
            while stat == 'ACQUIRING': 
                # 콘솔창에 현재 상태와, 지금까지 카메라 센서가 캡처한 이미지의 총 장수를 출력해 모니터링합니다.
                print(stat, "GetTotalNumberImagesAcquired",
                      ccd_dev.get_total_number_images_acquired())
                
                time.sleep(0.1) # 컴퓨터 CPU가 100% 돌아가는 것을 막기 위해 0.1초 동안 멈췄다가 다시 상태를 체크합니다.
                stat = ccd_hw.settings.ccd_status.read_from_hardware() # 상태 최신화 (촬영이 끝나면 이 값이 바뀌어 루프를 탈출합니다)
                
                # 측정 도중에 사용자가 GUI에서 '중지(Stop/Interrupt)' 버튼을 눌렀는지 체크합니다.
                if self.interrupt_measurement_called: 
                    ccd_hw.interrupt_acquisition() # 사용자가 중지를 눌렀다면 하드웨어에 즉시 촬영 중단 명령을 보냅니다.
                    break # while 무한 루프를 강제로 탈출합니다.
                
        finally: # 촬영이 무사히 끝났든, 에러가 났든, 사용자가 강제 중지했든 상관없이 항상 마지막에 실행되는 구문입니다.
            print("done") # 콘솔에 "done"이라고 출력하여 측정 프로세스가 끝났음을 알립니다.