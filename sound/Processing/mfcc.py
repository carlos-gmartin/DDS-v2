import numpy as np
import os
import librosa as lib
import matplotlib.pyplot as plt

def extract_file_name(path):
    """
    Extract the base file name from a given path.
    """
    return os.path.basename(path)

def process(audio_file, save_path):
    """
    Process an audio file and save the extracted MFCC features.
    """
    # Load audio file
    y, sr = lib.load(audio_file, sr=None)

    # Extract MFCC features
    mfcc = lib.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Save feature list into npy
    mfcc_path = os.path.splitext(audio_file)[0] + '.npy'
    mfcc_path = extract_file_name(mfcc_path)
    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, mfcc_path), mfcc)

    return mfcc_path

def process_directory(directory_path, save_path):
    """
    Process all .wav files in a directory and save MFCC features.
    """
    num_processed = 0
    total_files = len([f for f in os.listdir(directory_path) if f.endswith('.wav')])
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.wav'):
                audio_file_path = os.path.join(root, file)
                _ = process(audio_file_path, save_path)
                num_processed += 1
                print(f"Total Processed: {num_processed}/{total_files} Current File: {audio_file_path}")

if __name__ == '__main__':
    directory_path = '../DroneAudioDataset/background'
    save_path = '../mfccDataset'
    process_directory(directory_path, save_path)
