
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, freqz, lfilter

# 1. Đặt các tham số
fs = 44100  # Tần số lấy mẫu (Hz)
num_taps = 101  # Số lượng taps (số coefficients)

# 2. Tính toán tần số cắt cho từng băng tần (chuẩn hóa theo Nyquist)
nyquist = fs / 2
low_cutoff = [20 / nyquist, 200 / nyquist]  # Low band
mid_cutoff = [200 / nyquist, 2000 / nyquist]  # Mid band
high_cutoff = [2000 / nyquist, 20000 / nyquist]  # High band

# 3. Thiết kế bộ lọc FIR cho từng băng tần
low_band = firwin(num_taps, low_cutoff, pass_zero=False,
                  window='hamming')  # Bộ lọc thông dải Low
mid_band = firwin(num_taps, mid_cutoff, pass_zero=False,
                  window='hamming')  # Bộ lọc thông dải Mid
high_band = firwin(num_taps, high_cutoff, pass_zero=False,
                   window='hamming')  # Bộ lọc thông dải High

# 4. Hàm apply_filter để áp dụng bộ lọc FIR lên tín hiệu đầu vào


def apply_filter(signal, filter_coefficients):
    return lfilter(filter_coefficients, 1.0, signal)

# 5. Kiểm tra đáp ứng tần số của từng bộ lọc


def plot_frequency_response(coefficients, band_name):
    w, h = freqz(coefficients, worN=8000)
    plt.plot((w / np.pi) * nyquist, np.abs(h), label=f'{band_name} Band')
    plt.title(f'Frequency Response - {band_name} Band')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid()
    plt.show()


plot_frequency_response(low_band, 'Low')
plot_frequency_response(mid_band, 'Mid')
plot_frequency_response(high_band, 'High')
