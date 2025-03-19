import kaggle
import zipfile
import os

def download_ravdess():
    """
    Downloads and extracts the RAVDESS dataset from Kaggle.
    """
    dataset_name = "uwrfkaggler/ravdess-emotional-speech-audio"
    output_path = "ravdess-emotional-speech-audio.zip"

    if not os.path.exists("dataset/ravdess"):
        print("Downloading RAVDESS dataset...")
        kaggle.api.dataset_download_files(dataset_name, path="./", unzip=False)

        print("Extracting dataset...")
        with zipfile.ZipFile(output_path, 'r') as zip_ref:
            zip_ref.extractall("dataset/ravdess")

        print("✅ RAVDESS dataset extracted successfully!")
    else:
        print("Dataset already exists.")

# Run the function
download_ravdess()

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# Define Emotion Mapping (RAVDESS Labels)
emotion_dict = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
}

def extract_features(audio_path):
    """
    Extract MFCCs, pitch, and energy from an audio file.
    """
    y, sr = librosa.load(audio_path, sr=16000)

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_mean = np.mean(mfccs.T, axis=0)

    # Extract Pitch
    pitch = librosa.yin(y, fmin=50, fmax=300)
    pitch_mean = np.mean(pitch)

    # Extract Energy
    energy = np.mean(librosa.feature.rms(y=y))

    return np.concatenate((mfccs_mean, [pitch_mean, energy]))

# Process Dataset
data, labels = [], []
dataset_path = "dataset/ravdess/"
for subdir, _, files in os.walk(dataset_path):
    for file in tqdm(files):
        if file.endswith(".wav"):
            file_path = os.path.join(subdir, file)
            features = extract_features(file_path)
            
            # Extract emotion label from filename
            emotion_label = file.split("-")[2]  # RAVDESS filename pattern
            emotion_name = emotion_dict[emotion_label]

            data.append(features)
            labels.append(emotion_name)

# Convert to DataFrame
df = pd.DataFrame(data)
df["emotion"] = labels
df.to_csv("ravdess_features.csv", index=False)

print("✅ Feature Extraction Complete!")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Load Extracted Features
df = pd.read_csv("ravdess_features.csv")

# Encode Emotion Labels
label_encoder = LabelEncoder()
df["emotion"] = label_encoder.fit_transform(df["emotion"])

# Split Data
X = np.array(df.drop(columns=["emotion"]))
y = to_categorical(df["emotion"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training Samples: {X_train.shape}, Testing Samples: {X_test.shape}")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, Flatten, BatchNormalization

# Define Speech Emotion Recognition Model
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(64, return_sequences=False),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(y_train.shape[1], activation='softmax')  # Number of emotion classes
])

# Compile the Model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
# Reshape for LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Train Model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Save Model
model.save("speech_emotion_model.h5")
print("✅ Model Training Complete!")
