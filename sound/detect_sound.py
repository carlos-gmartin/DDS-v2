import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Path to the directory containing MFCC files
data_path = "./mfccDataset/"

# List to store features and labels
features = []
labels = []

# Loop through each file in the directory
for file_name in os.listdir(data_path):
    if file_name.endswith('.npy'):
        # Load the MFCC feature
        feature = np.load(os.path.join(data_path, file_name))
        
        # Extract label from the file name (using the first part before the underscore)
        label = file_name.split('_')[0]  # Extract the label part before the underscore
       
        # Append the feature and label to the lists
        features.append(feature)
        labels.append(label)

# Convert lists to numpy arrays
X = np.array(features)
y = np.array(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape the data for input into the LSTM (assuming MFCCs are 2D)
# The shape should be (number of samples, number of time steps, number of features)
# In this case, number of time steps = 1 since we're using MFCCs averaged over time
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Convert string labels to numerical labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Define the LSTM model
model = models.Sequential([
    layers.LSTM(64, input_shape=(1, X_train.shape[2]), return_sequences=False),
    layers.Dense(len(label_encoder.classes_), activation='softmax')  # Use softmax for multi-class classification
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train_encoded, epochs=10, batch_size=32, validation_data=(X_test, y_test_encoded))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test_encoded)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

joblib.dump(model, './saved_model/soundmodelv1.pkl')

# Predict labels for test set
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Create confusion matrix
cm = confusion_matrix(y_test_encoded, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('./saved_model/soundmodelv1-matrix.jpg')  # Saving the plot
plt.show()  # Displaying the plot
