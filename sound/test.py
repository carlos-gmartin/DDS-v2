import joblib
import numpy as np
import librosa  # Assuming you used librosa for audio processing during training
import Processing.mfcc as processing

# Define the path to the individual audio file for testing
new_audio_file = './DroneAudioDataset/Background/Background1.wav'

# Call the extract_features function to extract MFCC features
features = processing.extract_features(new_audio_file)

# Load the saved model
loaded_model = joblib.load('./saved_model/soundmodelv1.pkl')

# Example function to predict the class
def predict_class(model, preprocessed_data):
    prediction = model.predict(preprocessed_data)
    return prediction

print(features.shape)

# Reshape the features to match the expected input shape of the model
features = features.reshape(1, 1, -1)

# Assuming your classes are represented as ['Class A', 'Class B']
classes = ['Background', 'Drone']


# Call the predict method of the model
prediction = predict_class(loaded_model, features)
print("Predicted class:", prediction)

# Output the predicted probabilities for each class
for i, class_label in enumerate(classes):
    print(f"Probability of {class_label}: {prediction[0][i]}")

