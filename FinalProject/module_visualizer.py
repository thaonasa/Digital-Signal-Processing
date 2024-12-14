import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk

class AudioVisualizer:
    def __init__(self, frame, sample_rate):
        self.sample_rate = sample_rate
        self.fig = Figure(figsize=(8, 4))
        self.ax_wave = self.fig.add_subplot(211)
        self.ax_wave.set_title('Waveform')
        self.ax_wave.set_ylim(-1, 1)
        self.ax_wave.set_xlim(0, 1024)
        self.wave_line, = self.ax_wave.plot([], [], 'b-', lw=1)
        self.ax_spectrum = self.fig.add_subplot(212)
        self.ax_spectrum.set_title('Frequency Spectrum')
        self.ax_spectrum.set_ylim(-60, 60)
        self.ax_spectrum.set_xlim(20, 20000)
        self.ax_spectrum.set_xscale('log')
        self.spectrum_line, = self.ax_spectrum.plot([], [], 'g-', lw=1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.fig.tight_layout()
        
    def update(self, audio_data):
        self.wave_line.set_data(np.arange(len(audio_data)), audio_data)
        spectrum = np.fft.rfft(audio_data)
        freq = np.fft.rfftfreq(len(audio_data), 1/self.sample_rate)
        spectrum_db = 20 * np.log10(np.abs(spectrum) + 1e-10)
        self.spectrum_line.set_data(freq, spectrum_db)
        self.canvas.draw()