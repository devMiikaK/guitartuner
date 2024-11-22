import numpy as np
import scipy.signal as signal
import scipy.fftpack
import sounddevice as sd
import threading
import tkinter as tk
from tkinter import ttk

# Constants
SAMPLE_RATE = 48000  # Hz
WINDOW_SIZE = 48000  # näytteiden määrä per ikkuna
WINDOW_STEP = 12000  # "liukuvan" ikkunan stepin koko
NUM_HPS = 5          # HPS = Harmonic product spectrum
POWER_THRESH = 1e-6 
WHITE_NOISE_THRESH = 0.2  #kohinan/äänenvaimennus
CONCERT_PITCH = 440  # Pitch/taajuus

#kitaran taajuusalue
MIN_FREQ = 70    # alin E kieli
MAX_FREQ = 1000  #korkea e kieli

# kynnys +/-, jolloin kieli on "vireessä"
TUNING_THRESHOLD = 0.5  # Hz
HANN_WINDOW = np.hanning(WINDOW_SIZE)#tällä vähennetään spektrivuotoja

ALL_NOTES = ["C", "C#", "D", "D#", "E", "F",  #kromaattisen asteikon kaikki nuotit.
             "F#", "G", "G#", "A", "A#", "B"]

def find_closest_note_and_deviation(freq): #funktio etsii lähimmän sävelen annetulle taajuudelle ja sit laskee poikkeaman
    if freq == 0:
        return None, None, None
    num_half_steps = 12 * np.log2(freq / CONCERT_PITCH)
    nearest_half_steps = int(round(num_half_steps))
    closest_pitch = CONCERT_PITCH * 2 ** (nearest_half_steps / 12)
    deviation = freq - closest_pitch  #poikkeama hertseinä
    note_index = (nearest_half_steps + 9) % 12  # 9 vastaa A säveltä
    octave = 4 + ((nearest_half_steps + 9) // 12)
    closest_note = ALL_NOTES[note_index] + str(octave)
    return closest_note, closest_pitch, deviation

def hps_analysis(magnitude_spectrum): # perustaajuus löydetään HPS analyysillä (harmonic product spectrum)
    hps_spec = magnitude_spectrum.copy()
    for h in range(2, NUM_HPS + 1):
        downsampled = magnitude_spectrum[::h]
        if len(downsampled) == 0:
            continue
        hps_spec[:len(downsampled)] *= downsampled
    max_index = np.argmax(hps_spec)

    if 1 <= max_index <= len(hps_spec) - 2:
        alpha = hps_spec[max_index - 1]
        beta = hps_spec[max_index]
        gamma = hps_spec[max_index + 1]
        numerator = alpha - gamma
        denominator = 2 * (alpha - 2 * beta + gamma)
        if denominator != 0:
            delta = numerator / denominator
        else:
            delta = 0
    else:
        delta = 0
    
    corrected_max_index = max_index + delta
    return corrected_max_index, hps_spec


current_freq = 0.0
closest_note = ''
closest_pitch = 0.0
frequency_deviation = 0.0 #poikkeama
stop_audio = False

def audio_processing():
    global current_freq, closest_note, closest_pitch, frequency_deviation, stop_audio
    if not hasattr(audio_processing, "window_samples"):
        audio_processing.window_samples = np.zeros(WINDOW_SIZE)

    def callback(indata, frames, time, status):
        global current_freq, closest_note, closest_pitch, frequency_deviation
        if status:
            print(f"Status: {status}")
            return

        # slider
        audio_processing.window_samples = np.roll(audio_processing.window_samples, -frames)
        audio_processing.window_samples[-frames:] = indata[:, 0]

        #signaalin teho
        signal_power = np.linalg.norm(audio_processing.window_samples, ord=2) ** 2 / len(audio_processing.window_samples)
        if signal_power < POWER_THRESH:
            return

        #nopea fourier muunnos
        windowed_samples = audio_processing.window_samples * HANN_WINDOW
        magnitude_spectrum = np.abs(scipy.fftpack.fft(windowed_samples)[:WINDOW_SIZE // 2])

        #kohinan vähennys
        avg_energy = np.mean(magnitude_spectrum)
        magnitude_spectrum[magnitude_spectrum < WHITE_NOISE_THRESH * avg_energy] = 0

        #käytetään HPS tunnistetun taajuuden löytämiseen
        corrected_max_index, hps_spec = hps_analysis(magnitude_spectrum)
        freq = corrected_max_index * (SAMPLE_RATE / WINDOW_SIZE)

        # tässä rajoitetaan taajuus vain kitaran taajuusalueeseen, joka asetettiin koodin alussa
        if freq < MIN_FREQ or freq > MAX_FREQ:
            return

        
        note, pitch, dev = find_closest_note_and_deviation(freq) #lähin sävel ja poikkeama

        # jos sävel oli kitaran taajuusalueella, muuttujat päivitetään.
        if note is not None:
            current_freq = freq
            closest_note = note
            closest_pitch = pitch
            frequency_deviation = dev

    try:
        with sd.InputStream(
            device=None,  # input laitteen ID, tän voi halutessa asettaa manuaalisesti haluamaan äänilaitteeseen. muuten oletus äänilaite
            channels=1,
            callback=callback,
            blocksize=WINDOW_STEP,
            samplerate=SAMPLE_RATE,
        ):
            while not stop_audio:
                sd.sleep(100)
    except Exception as e:
        print(f"An error occurred in audio processing: {e}")
        stop_audio = True


def main(): # UI
    global current_freq, closest_note, closest_pitch, frequency_deviation, stop_audio
    audio_thread = threading.Thread(target=audio_processing)
    audio_thread.daemon = True
    audio_thread.start()

    root = tk.Tk()
    root.title("Kitaran viritys")

    #labelit
    current_freq_label = ttk.Label(root, text="Taajuus:", font=("Helvetica", 16))
    current_freq_label.pack(pady=5)
    current_freq_value = ttk.Label(root, text="0.00 Hz", font=("Helvetica", 24))
    current_freq_value.pack(pady=5)

    closest_note_label = ttk.Label(root, text="Lähin sävel:", font=("Helvetica", 16))
    closest_note_label.pack(pady=5)
    closest_note_value = ttk.Label(root, text="---", font=("Helvetica", 24))
    closest_note_value.pack(pady=5)

    tuning_direction_label = ttk.Label(root, text="", font=("Helvetica", 24))
    tuning_direction_label.pack(pady=10)

    # "slider" jossa näkyy sävelet
    slider_frame = ttk.Frame(root)
    slider_frame.pack(pady=20)

    slider_canvas = tk.Canvas(slider_frame, width=600, height=100) #koko
    slider_canvas.pack()

    def update_gui():
        # tätä funktiota tarvitaan että labelit päivittyy vastaamaan nykyistä säveltä ja taajuutta
        if current_freq > 0:
            current_freq_value.config(text=f"{current_freq:.2f} Hz")
            closest_note_value.config(text=f"{closest_note} ({closest_pitch:.2f} Hz)")

            #tässä määritellään tuleeko kieltä kiristää vai löysätä
            if abs(frequency_deviation) < TUNING_THRESHOLD:
                direction = "✔" #"vireessä"
                tuning_direction_label.config(text=direction, foreground="green")
            elif frequency_deviation > 0:
                direction = "Löysää kieltä"
                tuning_direction_label.config(text=direction, foreground="red")
            else:
                direction = "Kiristä kieltä"
                tuning_direction_label.config(text=direction, foreground="red")
            slider_canvas.delete("all")

            #slideri
            slider_canvas.create_line(50, 50, 550, 50, fill="black", width=4)
            num_ticks = 13  # 1 oktaavi + 1 sävel = 13
            tick_positions = np.linspace(50, 550, num_ticks)

            base_num_half_steps = int(round(12 * np.log2(closest_pitch / CONCERT_PITCH)))
            start_half_steps = base_num_half_steps - 6  #6 half-steppiä

            #sävelet slideriin
            for i, x in enumerate(tick_positions):
                half_steps = start_half_steps + i
                note_index = (half_steps + 9) % 12
                octave = 4 + ((half_steps + 9) // 12)
                note_name = ALL_NOTES[note_index]
                full_note = note_name + str(octave)
                slider_canvas.create_line(x, 45, x, 55, fill="black", width=2)
                slider_canvas.create_text(x, 65, text=full_note, font=("Helvetica", 10))

            #taajuus logaritmi asteikolle
            start_pitch = closest_pitch / (2 ** (6 / 12))  # 6 half steppiä lähimmän sävelen alapuolelle
            end_pitch = closest_pitch * (2 ** (6 / 12))    #-||- yläpuolelle

            if current_freq <= 0 or start_pitch <= 0 or end_pitch <= 0:
                position = 300  #nykyinen taajuus merkataan keskelle slideriä
            else:
                position = 50 + ((np.log2(current_freq) - np.log2(start_pitch)) / (np.log2(end_pitch) - np.log2(start_pitch))) * 500
                position = max(50, min(550, position))

            slider_canvas.create_line(position, 30, position, 70, fill="blue", width=4)
            slider_canvas.create_text(position, 20, text=f"{current_freq:.2f} Hz", font=("Helvetica", 10), fill="blue")

        else:
            current_freq_value.config(text="0.00 Hz")
            closest_note_value.config(text="---")
            tuning_direction_label.config(text="", foreground="black")
            slider_canvas.delete("all")

        # ui päivittyy 50ms välein
        root.after(50, update_gui)
    root.after(50, update_gui)

    # tätä tarvitaan ettei ääntä vastaanoteta sovelluksen sulun jälkeen
    def on_closing():
        global stop_audio
        stop_audio = True
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
