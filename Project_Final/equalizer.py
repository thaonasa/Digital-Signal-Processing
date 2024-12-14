import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QSlider, QLabel, QComboBox, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt
from pydub import AudioSegment
import soundfile as sf


class Equalizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Digital Equalizer")
        self.setGeometry(100, 100, 400, 300)

        layout = QVBoxLayout()

        # Slider for gain control of low, mid, high bands
        self.low_gain = QSlider(Qt.Horizontal)
        self.low_gain.setMinimum(-10)
        self.low_gain.setMaximum(10)
        self.low_gain.setValue(0)
        layout.addWidget(QLabel("Low Band Gain"))
        layout.addWidget(self.low_gain)

        self.mid_gain = QSlider(Qt.Horizontal)
        self.mid_gain.setMinimum(-10)
        self.mid_gain.setMaximum(10)
        self.mid_gain.setValue(0)
        layout.addWidget(QLabel("Mid Band Gain"))
        layout.addWidget(self.mid_gain)

        self.high_gain = QSlider(Qt.Horizontal)
        self.high_gain.setMinimum(-10)
        self.high_gain.setMaximum(10)
        self.high_gain.setValue(0)
        layout.addWidget(QLabel("High Band Gain"))
        layout.addWidget(self.high_gain)

        # Dropdown for selecting filter type
        self.filter_type = QComboBox()
        self.filter_type.addItem("FIR")
        self.filter_type.addItem("IIR")
        self.filter_type.addItem("FFT")
        layout.addWidget(QLabel("Select Filter Type"))
        layout.addWidget(self.filter_type)

        # Button to load audio file
        self.load_button = QLabel("Select Audio File")
        self.load_button.mousePressEvent = self.load_audio
        layout.addWidget(self.load_button)

        # Main widget
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_audio(self, event):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Audio File", "", "Audio Files (*.wav *.mp3)", options=options)
        if file_name:
            self.process_audio(file_name)

    def process_audio(self, file_name):
        # Read the audio file
        audio = AudioSegment.from_file(file_name)
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        samplerate = audio.frame_rate

        # Apply selected filter
        if self.filter_type.currentText() == "FIR":
            filtered = self.apply_fir_filter(samples, samplerate)
        elif self.filter_type.currentText() == "IIR":
            filtered = self.apply_iir_filter(samples, samplerate)
        elif self.filter_type.currentText() == "FFT":
            filtered = self.apply_fft_filter(samples, samplerate)

        # Save filtered audio
        sf.write("filtered_output.wav", filtered, samplerate)

        # Visualize the result
        self.visualize(samples, filtered)

    def apply_fir_filter(self, data, samplerate):
        # Define FIR filter (low-pass example)
        numtaps = 51  # Filter order
        bands = [0, 250, 4000, 20000]
        gains = [self.low_gain.value(), self.mid_gain.value(),
                 self.high_gain.value()]

        taps = signal.firwin2(numtaps, bands, gains, fs=samplerate)
        filtered = signal.lfilter(taps, 1.0, data)
        return filtered

    def apply_iir_filter(self, data, samplerate):
        # Define IIR filter (Butterworth bandpass filter example)
        lowcut = 250
        highcut = 4000
        b, a = signal.butter(4, [lowcut, highcut], btype='band', fs=samplerate)
        filtered = signal.lfilter(b, a, data)
        return filtered

    def apply_fft_filter(self, data, samplerate):
        # Apply FFT filter
        fft_data = np.fft.fft(data)
        freqs = np.fft.fftfreq(len(data), 1/samplerate)

        # Attenuate or amplify specific frequency bands
        fft_data[(freqs < 250)] *= self.low_gain.value()
        fft_data[(freqs >= 250) & (freqs < 4000)] *= self.mid_gain.value()
        fft_data[(freqs >= 4000)] *= self.high_gain.value()

        filtered = np.fft.ifft(fft_data).real
        return filtered

    def visualize(self, original, filtered):
        # Plot the original and filtered signals
        plt.figure(figsize=(10, 5))

        plt.subplot(2, 1, 1)
        plt.plot(original, color='blue')
        plt.title('Original Signal')

        plt.subplot(2, 1, 2)
        plt.plot(filtered, color='red')
        plt.title('Filtered Signal')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    app = QApplication([])
    window = Equalizer()
    window.show()
    app.exec_()
