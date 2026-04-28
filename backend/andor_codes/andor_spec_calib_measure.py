from ScopeFoundry import Measurement, h5_io
from ScopeFoundry.helper_funcs import sibling_path, load_qt_ui_file,\
    replace_spinbox_in_layout

import pyqtgraph as pg
import numpy as np
import time

class AndorSpecCalibMeasure(Measurement):
    
    name = 'andor_spec_calib'
    
    def setup(self):
        # 파장 스윕(Sweep) 범위를 설정하기 위한 'sweep_wls'라는 새로운 범위(Range) 설정 추가
        self.settings.New_Range('sweep_wls', dtype=float)
        
        # 현재 파이썬 파일과 동일한 경로에 있는 UI 파일('andor_spec_calib_measure.ui')의 경로를 가져옴
        self.ui_filename = sibling_path(__file__, 'andor_spec_calib_measure.ui')
        # UI 파일을 로드하여 self.ui 객체에 저장
        self.ui = load_qt_ui_file(self.ui_filename)
        
    def setup_figure(self):

        self.graph_layout=pg.GraphicsLayoutWidget()
        self.ui.plot_widget.layout().addWidget(self.graph_layout)
        
        self.img_plot = self.graph_layout.addPlot()
        self.img_plot.showGrid(x=True, y=True)
        self.img_item = pg.ImageItem()
        self.img_plot.addItem(self.img_item)

        self.hist_lut = pg.HistogramLUTItem()
        self.hist_lut.autoHistogramRange()
        self.hist_lut.setImageItem(self.img_item)
        self.graph_layout.addItem(self.hist_lut)


        self.graph_layout.nextRow()
        
        self.spectrum_plot = self.graph_layout.addPlot(
            title="Spectrum", colspan=2)        
        self.current_spec_plotline = self.spectrum_plot.plot()


        # start stop buttons
        self.ui.start_pushButton.clicked.connect(
            self.start)
        self.ui.interrupt_pushButton.clicked.connect(
            self.interrupt)


        # WL sweep controls
        self.settings.sweep_wls_min.connect_to_widget(
            self.ui.sweep_wls_min_doubleSpinBox)
        self.settings.sweep_wls_max.connect_to_widget(
            self.ui.sweep_wls_max_doubleSpinBox)
        self.settings.sweep_wls_step.connect_to_widget(
            self.ui.sweep_wls_step_doubleSpinBox)
        self.settings.sweep_wls_num.connect_to_widget(
            self.ui.sweep_wls_num_doubleSpinBox)
        
        # Camera settings
        self.andor_ccd = self.app.hardware['andor_ccd']
        self.andor_ccd.settings.em_gain.connect_to_widget(
            self.ui.andor_emgain_doubleSpinBox)
        self.andor_ccd.settings.exposure_time.connect_to_widget(
            self.ui.andor_exp_time_doubleSpinBox)
        
        # Spectrometer settings
        if 'acton_spectrometer' in list(self.app.hardware.keys()):
            self.spec = spec = self.app.hardware['acton_spectrometer']
            spec.settings.entrance_slit.connect_to_widget(self.ui.spec_ent_slit_doubleSpinBox)
        elif 'andor_spec' in list(self.app.hardware.keys()):
            self.spec = spec = self.app.hardware['andor_spec']
            spec.settings.slit_input_side.connect_to_widget(self.ui.spec_ent_slit_doubleSpinBox)
        else:
            raise Exception('No spectrometer!')
        spec.settings.center_wl.connect_to_widget(
            self.ui.spec_center_wl_doubleSpinBox)
        
        spec.settings.grating_id.connect_to_widget(
            self.ui.spec_grating_id_comboBox)


    def run(self):
        """실제 데이터 측정이 수행되는 메인 루프 (별도의 쓰레드에서 실행됨)"""
        
        # 1. 하드웨어 초기 설정
        self.andor_ccd.settings['acq_mode'] = 'single'       # 단일 촬영 모드
        self.andor_ccd.settings['trigger_mode'] = 'internal' # 내부 트리거 사용
        self.andor_ccd.set_readout()                         # 설정값 카메라에 적용
        
        ccd_hw = self.app.hardware['andor_ccd']
        ccd_dev = ccd_hw.ccd_dev
        
        # CCD 센서의 가로, 세로 픽셀 수를 가져옴
        width_px = ccd_dev.Nx_ro
        height_px = ccd_dev.Ny_ro

        try:
            # 2. 데이터를 저장할 HDF5 파일 및 배열 생성
            self.h5_file = h5_io.h5_base_file(app=self.app, measurement=self)
            self.h5m = h5_io.h5_create_measurement_group(measurement=self, h5group=self.h5_file)
            
            # 스윕할 파장 배열을 가져와 HDF5 그룹에 저장
            self.sweep_wls = self.settings.ranges['sweep_wls'].array
            self.h5m['sweep_wls'] = self.sweep_wls
            
            # 메모리에 빈 배열 (파장 스윕 수 x CCD 가로 픽셀) 생성
            self.spectra = np.zeros((len(self.sweep_wls), width_px), dtype=float)
            # HDF5 파일 내에도 동일한 크기의 데이터셋 생성
            self.spectra_h5 = self.h5m.create_dataset('spectra', 
                                                      shape=(len(self.sweep_wls), width_px),
                                                      dtype=float)
            
            # 3. 파장 스윕 측정 루프
            for ii, center_wl in enumerate(self.sweep_wls):
                # 사용자가 중지 버튼을 누르면 루프 탈출
                if self.interrupt_measurement_called:
                    break
                    
                # 분광기를 현재 스텝의 중심 파장(center_wl)으로 이동시킴
                self.spec.settings['center_wl'] = center_wl
                
                # 카메라 촬영 시작
                ccd_dev.start_acquisition()
    
                # 카메라의 촬영 상태 확인
                stat = ccd_hw.settings.ccd_status.read_from_hardware()
                # 'ACQUIRING'(촬영 중) 상태인 동안 대기
                while stat == 'ACQUIRING':
                    if self.interrupt_measurement_called:
                        break
                    time.sleep(0.01) # 0.01초 대기 후 상태 재확인
                    stat = ccd_hw.settings.ccd_status.read_from_hardware()
    
                # 촬영이 완료되어 'IDLE' 상태가 되면 데이터 수집
                if stat == 'IDLE':
                    self.ccd_img = ccd_dev.get_acquired_data() # CCD의 2D 이미지 데이터 가져오기
                    self.spectrum = np.average(self.ccd_img, axis=0) # y축(세로) 방향으로 픽셀을 평균 내어 1D 스펙트럼 추출
                    
                    # 메모리 배열과 HDF5 데이터셋에 각각 스펙트럼 데이터 저장
                    self.spectra[ii,:] = self.spectrum
                    self.spectra_h5[ii,:] = self.spectrum

        finally:
            print(self.name, 'done')
            self.h5_file.close()
        
    def update_display(self):   
        """run 루프가 도는 동안 메인 UI 쓰레드에서 주기적으로 호출되어 화면을 갱신하는 함수"""
        # 누적된 스펙트럼 2D 배열을 전치(.T)하여 이미지로 표시
        self.img_item.setImage(self.spectra.T)
        # 현재 측정된 단일 스펙트럼 데이터를 1D 그래프로 갱신
        self.current_spec_plotline.setData(self.spectrum)
        
# 이 스크립트가 직접 실행될 때(Standalone 테스트용) 작동하는 코드
if __name__ == '__main__':
    import sys
    from ScopeFoundry import BaseMicroscopeApp
    
    # ScopeFoundry의 기본 앱을 상속받아 테스트용 앱 생성
    class TestApp(BaseMicroscopeApp):
        
        def setup(self):
            # 필요한 하드웨어 컴포넌트(카메라, 분광기) 추가
            from ScopeFoundryHW.andor_camera import AndorCCDHW
            self.add_hardware(AndorCCDHW(self))
            from ScopeFoundryHW.acton_spec import ActonSpectrometerHW
            self.add_hardware(ActonSpectrometerHW(self))
            
            # 현재 작성한 측정(Measurement) 모듈 추가
            self.add_measurement(AndorSpecCalibMeasure(self))
            
    # 앱 실행
    app = TestApp(sys.argv)
    app.exec_()