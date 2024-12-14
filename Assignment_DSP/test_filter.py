import unittest
import numpy as np
from scipy.signal import freqz
from fir_filter import fs, low_band, mid_band, high_band, apply_filter  # thay "your_filter_module" bằng tên file của bạn

class TestFIRFilters(unittest.TestCase):
    def setUp(self):
        # Đặt tần số lấy mẫu và tần số Nyquist
        self.fs = fs
        self.nyquist = self.fs / 2
        self.duration = 1.0  # 1 giây tín hiệu
        self.t = np.linspace(0, self.duration, int(self.fs * self.duration), endpoint=False)

        # Tín hiệu thử nghiệm chứa tần số thấp (100 Hz), trung bình (1000 Hz), và cao (5000 Hz)
        self.test_signal = (
            np.sin(2 * np.pi * 100 * self.t) +  # Tần số thấp
            np.sin(2 * np.pi * 1000 * self.t) +  # Tần số trung bình
            np.sin(2 * np.pi * 5000 * self.t)  # Tần số cao
        )

    def test_low_band_filter(self):
        """Kiểm tra bộ lọc băng tần thấp (20-200 Hz)"""
        w, h = freqz(low_band)
        pass_band_gain = np.abs(h[(w * self.fs / (2 * np.pi)) < 200]).max()
        stop_band_gain = np.abs(h[(w * self.fs / (2 * np.pi)) > 200]).max()
        
        # Đảm bảo bộ lọc có gain lớn trong băng thông thấp và suy giảm ngoài băng thông
        self.assertGreater(pass_band_gain, 0.8, "Gain của băng tần thấp quá thấp trong băng thông.")
        self.assertLess(stop_band_gain, 0.2, "Độ suy giảm của băng tần thấp quá cao ngoài băng thông.")

    def test_mid_band_filter(self):
        """Kiểm tra bộ lọc băng tần trung bình (200-2000 Hz)"""
        w, h = freqz(mid_band)
        pass_band_gain = np.abs(h[(w * self.fs / (2 * np.pi) >= 200) & (w * self.fs / (2 * np.pi) <= 2000)]).max()
        stop_band_gain_low = np.abs(h[(w * self.fs / (2 * np.pi) < 200)]).max()
        stop_band_gain_high = np.abs(h[(w * self.fs / (2 * np.pi) > 2000)]).max()
        
        # Đảm bảo bộ lọc có gain lớn trong băng thông trung bình và suy giảm ngoài băng thông
        self.assertGreater(pass_band_gain, 0.8, "Gain của băng tần trung bình quá thấp trong băng thông.")
        self.assertLess(stop_band_gain_low, 0.2, "Độ suy giảm của băng tần trung bình quá cao ngoài băng thông thấp.")
        self.assertLess(stop_band_gain_high, 0.2, "Độ suy giảm của băng tần trung bình quá cao ngoài băng thông cao.")

    def test_high_band_filter(self):
        """Kiểm tra bộ lọc băng tần cao (2000-20000 Hz)"""
        w, h = freqz(high_band)
        pass_band_gain = np.abs(h[(w * self.fs / (2 * np.pi)) > 2000]).max()
        stop_band_gain = np.abs(h[(w * self.fs / (2 * np.pi)) < 2000]).max()
        
        # Đảm bảo bộ lọc có gain lớn trong băng thông cao và suy giảm ngoài băng thông
        self.assertGreater(pass_band_gain, 0.8, "Gain của băng tần cao quá thấp trong băng thông.")
        self.assertLess(stop_band_gain, 0.2, "Độ suy giảm của băng tần cao quá cao ngoài băng thông.")

    def test_apply_filter_output_stability(self):
        """Kiểm tra độ ổn định của bộ lọc (đầu ra không vượt quá dải giá trị của tín hiệu đầu vào)"""
        stable_signal = np.random.uniform(-1, 1, len(self.test_signal))
        output = apply_filter(stable_signal, low_band)
        self.assertTrue(np.all(np.abs(output) <= 1), "Bộ lọc không ổn định, giá trị đầu ra vượt ngoài phạm vi.")

    def test_apply_filter_energy_conservation(self):
        """Kiểm tra bảo toàn năng lượng của tín hiệu trong băng thông sau khi lọc"""
        filtered_signal = apply_filter(self.test_signal, mid_band)
        input_energy = np.sum(self.test_signal**2)
        output_energy = np.sum(filtered_signal**2)
        self.assertAlmostEqual(input_energy, output_energy, delta=0.1 * input_energy,
                               msg="Năng lượng tín hiệu không được bảo toàn trong băng thông sau khi lọc.")

    def test_apply_filter_output_length(self):
        """Kiểm tra độ dài của tín hiệu sau khi lọc không thay đổi"""
        filtered_signal = apply_filter(self.test_signal, low_band)
        self.assertEqual(len(filtered_signal), len(self.test_signal), "Độ dài tín hiệu sau khi lọc không khớp với tín hiệu gốc.")
        
    def test_linear_phase_response(self):
        """Kiểm tra đáp ứng pha tuyến tính của bộ lọc"""
        w, h = freqz(low_band)
        phase = np.unwrap(np.angle(h))
        linearity = np.polyfit(w, phase, 1)[0]  # Hệ số góc của pha
        self.assertAlmostEqual(linearity, 0, places=2, msg="Bộ lọc không có pha tuyến tính.")

    def test_phase_shift(self):
        """Kiểm tra độ lệch pha giữa tín hiệu đầu vào và đầu ra"""
        test_signal = np.sin(2 * np.pi * 100 * self.t)  # Tín hiệu tần số 100 Hz
        filtered_signal = apply_filter(test_signal, low_band)
        phase_shift = np.angle(np.fft.fft(filtered_signal)[1]) - np.angle(np.fft.fft(test_signal)[1])
        
        # Đảm bảo rằng độ lệch pha là một hằng số nhỏ, cho thấy bộ lọc có pha tuyến tính
        self.assertAlmostEqual(phase_shift, 0, delta=0.1, msg="Độ lệch pha không nhỏ như mong đợi.")

    def test_single_frequency_response_within_band(self):
        """Kiểm tra phản hồi với tín hiệu tần số đơn nằm trong băng thông"""
        test_freq = 150  # Tần số nằm trong băng tần thấp
        test_signal = np.sin(2 * np.pi * test_freq * self.t)
        filtered_signal = apply_filter(test_signal, low_band)
        output_energy = np.sum(filtered_signal**2)
        input_energy = np.sum(test_signal**2)
        
        # Kiểm tra rằng năng lượng tín hiệu gần như không thay đổi trong băng tần
        self.assertAlmostEqual(output_energy, input_energy, delta=0.1 * input_energy,
                               msg="Bộ lọc không bảo toàn năng lượng cho tần số trong băng thông.")

    def test_single_frequency_response_outside_band(self):
        """Kiểm tra phản hồi với tín hiệu tần số đơn nằm ngoài băng thông"""
        test_freq = 5000  # Tần số nằm ngoài băng tần thấp
        test_signal = np.sin(2 * np.pi * test_freq * self.t)
        filtered_signal = apply_filter(test_signal, low_band)
        output_energy = np.sum(filtered_signal**2)
        
        # Đảm bảo rằng năng lượng tín hiệu bị suy giảm mạnh ngoài băng thông
        self.assertLess(output_energy, 0.1, "Bộ lọc không suy giảm đủ mạnh tín hiệu ngoài băng thông.")

    def test_mid_band_filter_with_white_noise(self):
        """Kiểm tra bộ lọc băng tần trung với tín hiệu white noise"""
        noise = np.random.normal(0, 1, len(self.t))
        filtered_signal = apply_filter(noise, mid_band)
        
        # Kiểm tra năng lượng của tín hiệu sau khi lọc (năng lượng trong băng tần trung)
        energy_before = np.sum(noise**2)
        energy_after = np.sum(filtered_signal**2)
        self.assertLess(energy_after, energy_before, "Năng lượng sau khi lọc không giảm so với tín hiệu white noise gốc.")

    def test_high_band_filter_sine_wave(self):
        """Kiểm tra bộ lọc băng tần cao với tín hiệu tần số cao"""
        test_freq = 10000  # Tần số nằm trong băng tần cao
        test_signal = np.sin(2 * np.pi * test_freq * self.t)
        filtered_signal = apply_filter(test_signal, high_band)
        output_energy = np.sum(filtered_signal**2)
        input_energy = np.sum(test_signal**2)
        
        # Kiểm tra rằng năng lượng trong băng tần cao gần như không thay đổi
        self.assertAlmostEqual(output_energy, input_energy, delta=0.1 * input_energy,
                               msg="Bộ lọc không bảo toàn năng lượng cho tần số cao trong băng thông.")
if __name__ == '__main__':
    unittest.main()
