import numpy as np
from scipy import signal
import soundfile as sf
import sounddevice as sd
import queue


class AudioFile:
    def __init__(self):
        self.audio_data = None
        self.sample_rate = None
        self.current_position = 0
        self.duration = 0
        self.is_playing = False
        self.filename = None

    def load_file(self, filename):
        try:
            audio_data, sample_rate = sf.read(filename, dtype='float32')
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1, dtype='float32')
            self.audio_data = audio_data
            self.sample_rate = sample_rate
            self.current_position = 0
            self.duration = len(self.audio_data) / sample_rate
            self.filename = filename
            return True
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return False

    def get_next_chunk(self, chunk_size):
        if self.current_position >= len(self.audio_data):
            return None
        end_pos = min(self.current_position + chunk_size, len(self.audio_data))
        chunk = self.audio_data[self.current_position:end_pos]
        self.current_position = end_pos
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        return chunk

    def seek(self, position):
        self.current_position = int(position * self.sample_rate)


class AudioPlayer:
    def __init__(self, sample_rate, frame_size):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.stream = None
        self.audio_file = AudioFile()

    def play(self, callback):
        if self.stream is None or not self.stream.active:
            self.stream = sd.OutputStream(
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.frame_size,
                callback=callback
            )
            self.stream.start()
            self.audio_file.is_playing = True

    def stop(self):
        if self.stream is not None and self.stream.active:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            self.audio_file.is_playing = False

    def load_file(self, filename):
        return self.audio_file.load_file(filename)


class ThreeBandEqualizer:
    def __init__(self):
        self.sample_rate = 44100
        self.frame_size = 1024
        self.audio_queue = queue.Queue(maxsize=10)
        self.nyquist = self.sample_rate / 2
        self.low_cutoff = [20 / self.nyquist, 200 / self.nyquist]
        self.mid_cutoff = [200 / self.nyquist, 2000 / self.nyquist]
        self.high_cutoff = [2000 / self.nyquist, 20000 / self.nyquist]
        self.gains = np.ones(3)
        self.num_taps = 101
        # self.num_taps = 64
        self.filters = self._design_filters()
        self.player = AudioPlayer(self.sample_rate, self.frame_size)

    def _design_filters(self):
        filters = []
        filters.append(signal.firwin(
            self.num_taps, self.low_cutoff, pass_zero=False, window='hamming'))
        filters.append(signal.firwin(
            self.num_taps, self.mid_cutoff, pass_zero=False, window='hamming'))
        filters.append(signal.firwin(
            self.num_taps, self.high_cutoff, pass_zero=False, window='hamming'))
        return filters

    def process_audio(self, audio_data):
        output = np.zeros_like(audio_data)
        for i in range(3):
            filtered = signal.lfilter(self.filters[i], [1.0], audio_data)
            output += filtered * self.gains[i]
        max_val = np.max(np.abs(output))
        if max_val > 1.0:
            output = output / max_val
        return output

    def process_full_audio(self):
        if self.player.audio_file.audio_data is None:
            return None
        return self.process_audio(self.player.audio_file.audio_data)

    def audio_callback(self, outdata, frames, time, status):
        if status:
            print(status)
        chunk = self.player.audio_file.get_next_chunk(frames)
        if chunk is None:
            self.player.stop()
            self.player.audio_file.current_position = 0
            raise sd.CallbackStop()
        processed = self.process_audio(chunk)
        outdata[:] = processed.reshape(-1, 1)
        try:
            self.audio_queue.put_nowait(processed)
        except queue.Full:
            pass

    def set_gain(self, band_idx, gain_db):
        self.gains[band_idx] = 10 ** (gain_db / 20)

    def save_processed_audio(self, output_filename):
        if self.player.audio_file.audio_data is None:
            return False
        try:
            processed_audio = self.process_full_audio()
            sf.write(output_filename, processed_audio, self.sample_rate)
            return True
        except Exception as e:
            print(f"Error saving audio file: {e}")
            return False
