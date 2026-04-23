import pandas as pd
import matplotlib.pyplot as plt

def plot_my_csv(file_path):
    # CSV 파일 읽기 (주석 '#' 처리된 메타데이터 무시)
    df = pd.read_csv(file_path, comment='#')
    
    # x축: 캘리브레이션 축이 있으면 그것으로, 없으면 pixel로 설정
    if 'raman_shift_cm-1' in df.columns:
        x_data = df['raman_shift_cm-1']
        x_label = 'Raman Shift (cm⁻¹)'
    else:
        x_data = df['pixel']
        x_label = 'Pixel'
        
    y_data = df['intensity']
    
    # 그래프 그리기
    plt.figure(figsize=(10, 5))
    plt.plot(x_data, y_data, color='red', linewidth=1.5)
    plt.xlabel(x_label)
    plt.ylabel('Intensity (counts)')
    plt.title(f'My Data: {file_path}')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# 아래 파일명만 바꿔서 실행하세요
plot_my_csv('./spectrum.csv')