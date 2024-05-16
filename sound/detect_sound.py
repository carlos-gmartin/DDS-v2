import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_dataset(directory):
    data = []
    for file in os.listdir(directory):
        if file.endswith('.npy'):
            file_path = os.path.join(directory, file)
            mfcc = np.load(file_path)
            category = 'Drone' if 'Drone' in file else 'Background'
            data.append((mfcc, category))
    return data

def split_data(data, test_size=0.15, val_size=0.15, random_state=42):
    X = [mfcc for mfcc, _ in data]
    y = [category for _, category in data]

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size/(test_size+val_size), random_state=random_state)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def build_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(64, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

def train_model(X_train, y_train, X_val, y_val, X_test, y_test, epochs=20, batch_size=32):
    # Find the maximum length of samples
    max_length = max(max(len(sample) for sample in X_train),
                     max(len(sample) for sample in X_val),
                     max(len(sample) for sample in X_test))

    # Pad/truncate samples to the maximum length
    X_train_padded = pad_sequences(X_train, maxlen=max_length, padding='post', truncating='post')
    X_val_padded = pad_sequences(X_val, maxlen=max_length, padding='post', truncating='post')
    X_test_padded = pad_sequences(X_test, maxlen=max_length, padding='post', truncating='post')

    input_shape = X_train_padded[0].shape
    model = build_model(input_shape)

    history = model.fit(np.array(X_train_padded), np.array(y_train),
                        validation_data=(np.array(X_val_padded), np.array(y_val)),
                        epochs=epochs, batch_size=batch_size, verbose=1)
    
    return model, history


def evaluate_model(model, X_test, y_test):
    test_loss, test_acc = model.evaluate(np.array(X_test), np.array(y_test))
    print("Test accuracy:", test_acc)

    y_pred = model.predict(np.array(X_test))
    y_pred_labels = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_test, y_pred_labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

def save_model(model, model_path):
    model.save(model_path)
    print("Model saved successfully.")

if __name__ == '__main__':
    # Set directory paths and other parameters
    directory_path = './mfccDataset'
    save_path = './saved_model'
    test_size = 0.2
    val_size = 0.2
    random_state = 42
    epochs = 20
    batch_size = 32

    # Step 1: Load dataset
    data = load_dataset(directory_path)

    # Step 2: Split data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(data, test_size, val_size, random_state)

    # Step 3: Train model
    model, history = train_model(X_train, y_train, X_val, y_val, epochs, batch_size)

    # Step 4: Evaluate model
    evaluate_model(model, X_test, y_test)

    # Step 5: Save model
    model_path = os.path.join(save_path, 'sound_classification_model.h5')
    save_model(model, model_path)
