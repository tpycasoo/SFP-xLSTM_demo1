import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# 生成模拟振动信号（例如，正弦波叠加噪声）
fs = 1000  # 采样频率
t = np.linspace(0, 10, fs * 10, endpoint=False)  # 时间向量
signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.random.randn(t.size)  # 模拟信号（50Hz 正弦波加噪声）

# 计算短时傅里叶变换（STFT）
f, t_spec, Sxx = spectrogram(signal, fs)

# 绘制频谱图
plt.pcolormesh(t_spec, f, np.log(Sxx))  # 使用对数尺度来更好地显示幅度
plt.ylabel('频率 [Hz]')
plt.xlabel('时间 [sec]')
plt.title('短时傅里叶变换（STFT）频谱图')
plt.colorbar(label='幅度（对数）')
plt.show()
