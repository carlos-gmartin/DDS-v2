import numpy as np
import pyaudio
import time
import librosa
import joblib
import cv2
from pydub import AudioSegment
from pydub.playback import play

def play_music():
    try:
        # Load the audio file
        sound = AudioSegment.from_file('./sound/US.wav')
        
        # Trim the audio to the first 2 seconds
        sound = sound[:2000]  # 2000 ms = 2 seconds
        
        # Play the audio file
        play(sound)
    except Exception as e:
        print(f"Error playing music: {e}")

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
        self.probabilities = [0.0, 0.0]  # To store probabilities of Background and Drone
        
        # Initialize threshold value
        self.threshold = 0.7  # Default threshold value

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
        self.showWaitingMessage()  # Display waiting message while waiting for audio input

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

    def predict_class(self, preprocessed_data, threshold):
        prediction = self.predict(preprocessed_data)
        self.probabilities = prediction[0]

        if prediction[0][1] > threshold:
            return True
        return False

    def callback(self, in_data, frame_count, time_info, flag):
        if self.drone_detected:
            return (in_data, pyaudio.paComplete)  # Stop the stream

        numpy_array = np.frombuffer(in_data, dtype=np.float32)
        mfccs = librosa.feature.mfcc(y=numpy_array, sr=self.RATE, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        features = mfccs_scaled.reshape(1, 1, -1)

        # Use dynamically adjusted threshold
        prediction = self.predict_class(features, self.threshold)

        if prediction:
            print("Drone detected.")
            self.drone_detected = True

        return (in_data, pyaudio.paContinue)

    def showWaitingMessage(self):
        text = "Searching for drone..."
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Create a blank image with a specific background color
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # Add a semi-transparent overlay for better text visibility
        overlay = frame.copy()
        alpha = 0.6  # Transparency factor
        cv2.rectangle(overlay, (0, 0), (1920, 200), (0, 255, 0), -1)  # Green rectangle
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Threshold line color
        threshold_color = (0, 255, 255)  # Yellow

        # Callback function for trackbar
        def onTrackbarChange(value):
            self.threshold = value / 100  # Convert slider value to threshold

        # Create window with trackbar
        cv2.namedWindow('Drone Detection Waiting')
        cv2.createTrackbar('Threshold', 'Drone Detection Waiting', int(self.threshold * 100), 100, onTrackbarChange)

        while not self.drone_detected and self.stream.is_active():
            # Clear the frame before updating it
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (1920, 200), (0, 255, 0), -1)  # Green rectangle
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # Add title
            cv2.putText(frame, text, (50, 100), font, 2, (255, 255, 255), 4)

            # Update text to show latest probabilities
            text_prob = "Probabilities:      " + f"Drone: {self.probabilities[1]*100:.2f}%" + "                               " + f"Background: {self.probabilities[0]*100:.2f}%"
            cv2.putText(frame, text_prob, (100, 900), font, 1, (0, 255, 0), 2)

            # Add sound bars for drone and background
            drone_bar_height = int(self.probabilities[1] * 500)  # Scale the probability to adjust bar height
            background_bar_height = int(self.probabilities[0] * 500)  # Scale the probability to adjust bar height
            cv2.rectangle(frame, (400, 800 - drone_bar_height), (600, 800), (0, 0, 255), -1)  # Red drone bar
            cv2.rectangle(frame, (1100, 800 - background_bar_height), (1300, 800), (0, 255, 0), -1)  # Blue background bar

            # Add static numbers on the side of the rectangles
            for i in range(0, 110, 10):
                cv2.putText(frame, str(i), (350, 805 - int(i * 500 / 100)), font, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, str(i), (1315, 805 - int(i * 500 / 100)), font, 0.5, (255, 255, 255), 1)

            # Add threshold line using slider position
            threshold_slider_pos = int(self.threshold * 100)
            cv2.line(frame, (400, 800 - threshold_slider_pos * 5), (600, 800 - threshold_slider_pos * 5), threshold_color, 2)

            # Add threshold text
            cv2.putText(frame, f"Threshold: {self.threshold:.2f}", (50, 150), font, 1, (255, 255, 255), 2)

            # Pass the frame to showDroneWarning if drone is detected
            if self.drone_detected:
                try:
                    # Create a banner with text
                    banner_text = "Drone Detected!"
                    banner_font_scale = 2
                    banner_font_color = (0, 0, 255)  # Red text
                    banner_font_thickness = 4
                    banner_text_size = cv2.getTextSize(banner_text, font, banner_font_scale, banner_font_thickness)[0]
                    banner_text_x = (frame.shape[1] - banner_text_size[0]) // 2
                    banner_text_y = (frame.shape[0] + banner_text_size[1]) // 2
                    cv2.putText(frame, banner_text, (banner_text_x, banner_text_y), font, banner_font_scale, banner_font_color, banner_font_thickness)
                except Exception as e:
                    print(f"Error in show Drone Warning: {e}")

            # Display the image in a window
            cv2.imshow('Drone Detection Waiting', frame)

            # Handle events and refresh window
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        # Delay before closing the window after drone detected
        play_music()
        cv2.waitKey(3000)
        cv2.destroyAllWindows()  # Close the window after drone detected

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

