import numpy as np
import librosa
import os

def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        print(f"Error encountered while parsing file: {file_name}")
        print(f"Exception details: {str(e)}")
        return None

# # Path to your dataset
# data_path = "../DroneAudioDataset/"
# labels = ["Drone", "Background"]

# # Path to save the .npy files
# save_path = "../mfccDataset/"

# # Create save_path directory if it doesn't exist
# os.makedirs(save_path, exist_ok=True)
# for label in labels:
#     dir_path = os.path.join(data_path, label)
#     for file_name in os.listdir(dir_path):
#         file_path = os.path.join(dir_path, file_name)
#         feature = extract_features(file_path)
#         if feature is not None:
#             # Construct the filename with label included
#             file_save_path = os.path.join(save_path, f"{label}_{file_name[:-4]}.npy")
#             np.save(file_save_path, feature)
