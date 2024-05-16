import pyaudio
import numpy as np
import librosa

# Function to continuously record audio from microphone
def record_audio():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16  # Keep the original format as Int16
    CHANNELS = 1
    RATE = 44100

    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    print("Recording...")

    while True:
        data = stream.read(CHUNK)
        audio_data = np.frombuffer(data, dtype=np.int16)

        # Normalize audio data to the range [-1, 1]
        normalized_audio = audio_data / 32768.0  # 32768 is the maximum value for int16

        # Perform real-time audio analysis
        # For example, extract MFCC features
        mfccs = librosa.feature.mfcc(y=normalized_audio, sr=RATE, n_mfcc=40)
        # Perform further analysis or processing as needed

    stream.stop_stream()
    stream.close()
    audio.terminate()

record_audio()
