import unittest
import numpy as np
from scipy.signal import freqz
from module_filter import ThreeBandEqualizer


class TestThreeBandEqualizer(unittest.TestCase):
    def setUp(self):
        self.equalizer = ThreeBandEqualizer()
        self.sample_rate = self.equalizer.sample_rate
        self.nyquist = self.sample_rate / 2
        self.duration = 1.0
        self.t = np.linspace(0, self.duration, int(
            self.sample_rate * self.duration), endpoint=False)

    def test_band_responses(self):
        """Kiểm tra đáp ứng băng tần của các bộ lọc băng tần thấp, trung và cao"""
        for i, (cutoff, name) in enumerate([(self.equalizer.low_cutoff, 'Low'),
                                            (self.equalizer.mid_cutoff, 'Mid'),
                                            (self.equalizer.high_cutoff, 'High')]):
            w, h = freqz(self.equalizer.filters[i], worN=8000)
            pass_band_gain = np.abs(
                h[(w / np.pi) * self.nyquist >= cutoff[0] * self.nyquist]).max()
            stop_band_gain = np.abs(
                h[(w / np.pi) * self.nyquist < cutoff[0] * self.nyquist]).max()
            self.assertGreater(pass_band_gain, 0.8,
                               f"{name} band filter gain too low in pass band")
            self.assertLess(stop_band_gain, 0.2,
                            f"{name} band filter stop band gain too high")

    def test_energy_conservation(self):
        """Kiểm tra bảo toàn năng lượng của tín hiệu trong băng tần sau khi lọc"""
        test_signal = np.sin(
            2 * np.pi * 500 * self.t)  # Tần số nằm trong băng tần trung
        processed_signal = self.equalizer.process_audio(test_signal)
        input_energy = np.sum(test_signal**2)
        output_energy = np.sum(processed_signal**2)
        self.assertAlmostEqual(input_energy, output_energy, delta=0.1 * input_energy,
                               msg="Năng lượng không được bảo toàn sau khi lọc")

    def test_linear_phase_response(self):
        """Kiểm tra đáp ứng pha tuyến tính của bộ lọc"""
        for i in range(3):
            w, h = freqz(self.equalizer.filters[i], worN=8000)
            phase = np.unwrap(np.angle(h))
            linearity = np.polyfit(w, phase, 1)[0]
            self.assertAlmostEqual(
                linearity, 0, places=2, msg="Bộ lọc không có pha tuyến tính")

    def test_output_stability(self):
        """Kiểm tra độ ổn định của tín hiệu đầu ra"""
        stable_signal = np.random.uniform(-1, 1, len(self.t))
        processed_signal = self.equalizer.process_audio(stable_signal)
        self.assertTrue(np.all(np.abs(processed_signal) <= 1),
                        "Output signal is unstable")

    def test_output_length(self):
        """Kiểm tra độ dài tín hiệu sau khi lọc"""
        processed_signal = self.equalizer.process_audio(self.t)
        self.assertEqual(len(processed_signal), len(self.t),
                         "Output length does not match input length")

    def test_phase_shift(self):
        """Kiểm tra độ lệch pha giữa tín hiệu đầu vào và đầu ra"""
        test_signal = np.sin(2 * np.pi * 100 * self.t)
        processed_signal = self.equalizer.process_audio(test_signal)
        phase_shift = np.angle(np.fft.fft(processed_signal)[
                               1]) - np.angle(np.fft.fft(test_signal)[1])
        self.assertAlmostEqual(phase_shift, 0, delta=0.1,
                               msg="Độ lệch pha không nhỏ như mong đợi")

    def test_response_within_band(self):
        """Kiểm tra phản hồi với tín hiệu tần số đơn nằm trong băng thông"""
        test_freq = 150  # Tần số nằm trong băng tần thấp
        test_signal = np.sin(2 * np.pi * test_freq * self.t)
        processed_signal = self.equalizer.process_audio(test_signal)
        input_energy = np.sum(test_signal**2)
        output_energy = np.sum(processed_signal**2)
        self.assertAlmostEqual(input_energy, output_energy, delta=0.1 * input_energy,
                               msg="Bộ lọc không bảo toàn năng lượng trong băng thông")

    def test_response_outside_band(self):
        """Kiểm tra phản hồi với tín hiệu tần số đơn nằm ngoài băng thông"""
        test_freq = 5000  # Tần số nằm ngoài băng tần thấp
        test_signal = np.sin(2 * np.pi * test_freq * self.t)
        processed_signal = self.equalizer.process_audio(test_signal)
        output_energy = np.sum(processed_signal**2)
        self.assertLess(
            output_energy, 0.1, "Bộ lọc không suy giảm đủ mạnh tín hiệu ngoài băng thông")

    def test_mid_band_with_white_noise(self):
        """Kiểm tra bộ lọc băng tần trung với tín hiệu white noise"""
        white_noise = np.random.normal(0, 1, len(self.t))
        processed_signal = self.equalizer.process_audio(white_noise)
        energy_before = np.sum(white_noise**2)
        energy_after = np.sum(processed_signal**2)
        self.assertLess(energy_after, energy_before,
                        "Energy after filtering white noise should be reduced")

    def test_high_band_with_high_frequency(self):
        """Kiểm tra bộ lọc băng tần cao với tín hiệu tần số cao"""
        test_freq = 10000  # Tần số nằm trong băng tần cao
        test_signal = np.sin(2 * np.pi * test_freq * self.t)
        processed_signal = self.equalizer.process_audio(test_signal)
        input_energy = np.sum(test_signal**2)
        output_energy = np.sum(processed_signal**2)
        self.assertAlmostEqual(input_energy, output_energy, delta=0.1 * input_energy,
                               msg="High band filter did not retain energy for high frequency signal")


if __name__ == '__main__':
    unittest.main()
