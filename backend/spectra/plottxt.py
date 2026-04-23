import matplotlib.pyplot as plt

def plot_commercial_txt(file_path):
    # 파일 읽기
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
        
    # 'X Axis Data' 라인 이후부터 실제 데이터 시작
    start_idx = 0
    for i, line in enumerate(lines):
        if "X Axis Data" in line:
            start_idx = i + 1
            break
            
    # X, Y 데이터 추출
    x_data, y_data = [], []
    for line in lines[start_idx:]:
        # 탭이나 콤마 분리 처리
        parts = line.strip().replace('\t', ',').split(',')
        if len(parts) >= 2:
            try:
                x_data.append(float(parts[0]))
                y_data.append(float(parts[1]))
            except ValueError:
                continue
                
    # 그래프 그리기
    plt.figure(figsize=(10, 5))
    plt.plot(x_data, y_data, color='blue', linewidth=1.5)
    plt.xlabel('Raman Shift (cm⁻¹)')
    plt.ylabel('Intensity (counts)')
    plt.title(f'Commercial Data: {file_path}')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# 아래 파일명만 바꿔서 실행하세요
plot_commercial_txt('SPECTRUM_ans.txt')