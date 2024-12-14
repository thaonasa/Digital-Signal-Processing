import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD
import threading
import queue
import time
import os
from module_filter import ThreeBandEqualizer
from module_visualizer import AudioVisualizer
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.dialogs import Messagebox

class ThreeBandEqualizerGUI:
    def __init__(self, equalizer):
        self.root = TkinterDnD.Tk()
        self.root.title("Three-Band Equalizer")
        self.style = ttk.Style()
        self.style.theme_use("cosmo")
        self.equalizer = equalizer
        self.audio_duration = 0
        self.running = False
        main_frame = ttk.Frame(self.root)
        main_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        viz_frame = ttk.Frame(main_frame)
        viz_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.visualizer = AudioVisualizer(viz_frame, equalizer.sample_rate)
        controls_frame = ttk.Frame(main_frame)
        controls_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        controls_frame.grid_rowconfigure(0, weight=0)
        controls_frame.grid_rowconfigure(1, weight=0)
        controls_frame.grid_rowconfigure(2, weight=1)
        controls_frame.grid_rowconfigure(3, weight=0)
        controls_frame.grid_rowconfigure(4, weight=0)
        style = ttk.Style()
        style.configure('Dropzone.TFrame', borderwidth=2, relief='solid')
        self.drop_frame = ttk.Frame(controls_frame, style='Dropzone.TFrame')
        self.drop_frame.grid(row=0, column=0, sticky="ew", pady=5)
        self.drop_label = ttk.Label(
            self.drop_frame,
            text="Drag and drop audio files here or click 'Open File'",
            padding=10
        )
        self.drop_label.pack(pady=5)
        self.drop_frame.drop_target_register(DND_FILES)
        self.drop_frame.dnd_bind('<<Drop>>', self.handle_drop)
        self.drop_frame.dnd_bind('<<DragEnter>>', self.on_drag_enter)
        self.drop_frame.dnd_bind('<<DragLeave>>', self.on_drag_leave)
        file_frame = ttk.Frame(controls_frame)
        file_frame.grid(row=1, column=0, sticky="ew", pady=5)
        ttk.Button(file_frame, text="Open File", command=self.open_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="Save Processed", command=self.save_file).pack(side=tk.LEFT, padx=5)
        self.file_label = ttk.Label(file_frame, text="No file loaded")
        self.file_label.pack(side=tk.LEFT, padx=5)
        progress_frame = ttk.Frame(controls_frame)
        progress_frame.grid(row=3, column=0, sticky="ew", pady=10)
        self.progress = ttk.Scale(
            progress_frame, from_=0, to=100, orient=tk.HORIZONTAL,
            command=self.seek_audio
        )
        self.progress.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=10)
        self.current_time_label = ttk.Label(progress_frame, text="00:00")
        self.current_time_label.pack(side=tk.LEFT)
        self.duration_label = ttk.Label(progress_frame, text="00:00")
        self.duration_label.pack(side=tk.RIGHT)
        control_frame = ttk.Frame(controls_frame)
        control_frame.grid(row=4, column=0, sticky="ew", pady=10)
        slider_frame = ttk.Frame(control_frame)
        slider_frame.pack(side=tk.TOP, fill=tk.X)
        self.slider_values = [tk.StringVar(value="0") for _ in range(3)]
        labels = [("Bass", "20 - 200 Hz"), ("Mid", "200 - 4000 Hz"), ("Treble", "4000 - 20000 Hz")]
        for i, (label, freq_range) in enumerate(labels):
            frame = ttk.Frame(slider_frame)
            frame.pack(side=tk.LEFT, padx=10)
            slider = ttk.Scale(
                frame, from_=12, to=-12, length=150, 
                orient=tk.VERTICAL,
                command=lambda x, idx=i: self.update_gain(idx, x)
            )
            slider.set(0)
            slider.pack()
            ttk.Label(frame, text=f"{label} ({freq_range})").pack()
            ttk.Label(frame, textvariable=self.slider_values[i]).pack()
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(side=tk.BOTTOM, pady=15)
        ttk.Button(button_frame, text="Play", command=self.play_audio).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Stop", command=self.stop_audio).pack(side=tk.LEFT, padx=5)
        self.running = True
        self.update_thread = threading.Thread(target=self.update_visualization)
        self.update_thread.daemon = True
        self.update_thread.start()
        self.progress_thread = threading.Thread(target=self.update_progress_bar)
        self.progress_thread.daemon = True
        self.progress_thread.start()

    def on_drag_enter(self, event):
        self.drop_label.configure(text="Release to load audio file")
        # style = ttk.Style()
        self.style.configure('Dropzone.TFrame', background='lightblue')
        self.drop_frame.configure(style='Dropzone.TFrame')

    def on_drag_leave(self, event):
        self.drop_label.configure(text="Drag and drop audio files here or click 'Open File'")
        # style = ttk.Style()
        self.style.configure('Dropzone.TFrame', background='white')
        self.drop_frame.configure(style='Dropzone.TFrame')

    def handle_drop(self, event):
        file_path = event.data
        file_path = file_path.strip('{}')
        self.on_drag_leave(event)
        if file_path.lower().endswith(('.wav', '.mp3', '.ogg')):
            if self.equalizer.player.load_file(file_path):
                self.file_label.config(text=os.path.basename(file_path))
                self.audio_duration = self.equalizer.player.audio_file.duration
                self.duration_label.config(text=self.format_time(self.audio_duration))
                self.progress.config(to=self.audio_duration)
                messagebox.showinfo("Success", "Audio file loaded successfully!")
            else:
                messagebox.showerror("Error", "Failed to load audio file")
        else:
            messagebox.showerror("Error", "Unsupported file format. Please use .wav, .mp3, or .ogg files")

    def open_file(self):
        filetypes = [
            ("Audio files", "*.wav;*.mp3;*.ogg"),
            ("WAV files", "*.wav"),
            ("MP3 files", "*.mp3"),
            ("OGG files", "*.ogg"),
            ("All files", "*.*")
        ]
        filename = filedialog.askopenfilename(filetypes=filetypes)
        if filename:
            if self.equalizer.player.load_file(filename):
                self.file_label.config(text=os.path.basename(filename))
                self.audio_duration = self.equalizer.player.audio_file.duration
                self.duration_label.config(text=self.format_time(self.audio_duration))
                self.progress.config(to=self.audio_duration)

    def save_file(self):
        if self.equalizer.player.audio_file.audio_data is None:
            tk.messagebox.showerror("Error", "No audio file loaded")
            return
        original_ext = os.path.splitext(self.equalizer.player.audio_file.filename)[1]
        filetypes = [("WAV files", "*.wav")]
        default_name = f"processed_audio{original_ext}"
        filename = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=filetypes,
            initialfile=default_name
        )
        if filename:
            if self.equalizer.save_processed_audio(filename):
                tk.messagebox.showinfo("Success", "Processed audio saved successfully!")
            else:
                tk.messagebox.showerror("Error", "Failed to save processed audio")

    def play_audio(self):
        if hasattr(self.equalizer.player.audio_file, 'audio_data') and self.equalizer.player.audio_file.audio_data is not None:
            self.equalizer.player.play(self.equalizer.audio_callback)
            self.update_progress_bar()

    def stop_audio(self):
        self.equalizer.player.stop()
        if hasattr(self.equalizer.player.audio_file, 'current_position'):
            self.equalizer.player.audio_file.current_position = 0
        self.progress.set(0)
        self.current_time_label.config(text="00:00")

    def seek_audio(self, value):
        if (hasattr(self.equalizer.player.audio_file, 'audio_data') and 
            self.equalizer.player.audio_file.audio_data is not None):
            try:
                position = float(value)
                self.equalizer.player.audio_file.seek(position)
                self.current_time_label.config(text=self.format_time(position))
            except ValueError as e:
                print(f"Invalid seek position: {e}")

    def update_visualization(self):
        QUEUE_TIMEOUT = 0.1
        while self.running:
            try:
                audio_data = self.equalizer.audio_queue.get(timeout=QUEUE_TIMEOUT)
                if audio_data is not None:
                    self.visualizer.update(audio_data)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Visualization error: {e}")
                time.sleep(0.1)
        self.visualizer.clear()

    def update_gain(self, band_idx, value):
        try:
            gain_value = float(value)
            self.equalizer.set_gain(band_idx, gain_value)
            self.slider_values[band_idx].set(f"{gain_value:.1f}")
        except ValueError:
            print(f"Invalid gain value: {value}")

    def format_time(self, seconds):
        try:
            minutes = int(float(seconds) // 60)
            seconds = int(float(seconds) % 60)
            return f"{minutes:02}:{seconds:02}"
        except (ValueError, TypeError):
            return "00:00"

    def cleanup(self):
        self.running = False
        self.equalizer.player.stop()
        if hasattr(self.equalizer, 'audio_queue'):
            while not self.equalizer.audio_queue.empty():
                try:
                    self.equalizer.audio_queue.get_nowait()
                except queue.Empty:
                    break

    def update_progress_bar(self):
        def update():
            last_position = 0
            start_time = None
            while self.running:
                if self.equalizer.player.audio_file.is_playing:
                    if start_time is None:
                        start_time = time.time()
                    elapsed_time = time.time() - start_time
                    current_position = int(elapsed_time * self.equalizer.player.sample_rate)
                    if current_position != last_position:
                        self.current_time = elapsed_time
                        self.progress.set(self.current_time)
                        self.current_time_label.config(text=self.format_time(self.current_time))
                        last_position = current_position
                    if current_position >= len(self.equalizer.player.audio_file.audio_data):
                        self.stop_audio()
                        break
                else:
                    start_time = None
                time.sleep(0.05)
            if not self.equalizer.player.audio_file.is_playing:
                self.progress.set(0)
                self.current_time_label.config(text="00:00")
        if not hasattr(self, "_progress_thread") or not self._progress_thread.is_alive():
            self._progress_thread = threading.Thread(target=update)
            self._progress_thread.daemon = True
            self._progress_thread.start()

    def format_time(self, seconds):
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02}:{seconds:02}"

    def run(self):
        try:
            self.root.mainloop()
        finally:
            self.running = False
            self.equalizer.player.stop()