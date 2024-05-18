import numpy as np
import pyaudio
import time
import librosa
import joblib

class AudioHandler(object):
    def __init__(self, model_path):
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = int(2 * self.RATE)
        self.p = None
        self.stream = None
        self.loaded_model = joblib.load(model_path)
        self.classes = ['Background', 'Drone']
        self.drone_detected = False

    def start(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.FORMAT,
                                  channels=self.CHANNELS,
                                  rate=self.RATE,
                                  input=True,
                                  output=False,
                                  stream_callback=self.callback,
                                  frames_per_buffer=self.CHUNK)
        self.stream.start_stream()

    def stop(self):
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        if self.p is not None:
            self.p.terminate()
        return True

    def predict(self, preprocessed_data):
        prediction = self.loaded_model.predict(preprocessed_data)
        return prediction

    def predict_class(self, preprocessed_data):
        prediction = self.predict(preprocessed_data)
        print("Predicted class:", prediction)

        if prediction[0][1] > 0.5:
            return True

        for i, class_label in enumerate(self.classes):
            print(f"Probability of {class_label}: {prediction[0][i]}")

    def callback(self, in_data, frame_count, time_info, flag):
        if self.drone_detected:
            return (in_data, pyaudio.paComplete)  # Stop the stream

        numpy_array = np.frombuffer(in_data, dtype=np.float32)
        mfccs = librosa.feature.mfcc(y=numpy_array, sr=self.RATE, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        features = mfccs_scaled.reshape(1, 1, -1)
        prediction = self.predict_class(features)

        if prediction:
            print("Drone detected.")
            self.drone_detected = True

        return (in_data, pyaudio.paContinue)

    def mainloop(self):
        try:
            while True:
                if self.drone_detected or not self.stream.is_active():
                    break
                time.sleep(2.0)
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            self.stop()

if __name__ == "__main__":
    model_path = './saved_model/soundmodelv1.pkl'
    audio = AudioHandler(model_path)
    audio.start()
    audio.mainloop()
