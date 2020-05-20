import librosa
import librosa.display

SAMPLING_RATE = 16000

# ...
path_to_file = "data/audios/swxx_20142B42_1548728378904.wav"
wave, _ = librosa.load(path_to_file, sr=SAMPLING_RATE)

librosa.display.waveplot(wave, sr=SAMPLING_RATE)
