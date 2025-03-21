import kaggle
import zipfile
import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim




# ✅ Download & Extract Dataset
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

        print(" RAVDESS dataset extracted successfully!")
    else:
        print("Dataset already exists.")

# Run the function
download_ravdess()

# Emotion Mapping (RAVDESS Labels)
emotion_dict = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
}

def extract_features(audio_path):
    """Extract MFCCs, pitch, and energy from an audio file."""
    try:
        y, sr = librosa.load(audio_path, sr=16000)

        # ✅ Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)

        return mfccs_mean  # ✅ Ensure only 40 features

    except Exception as e:
        print(f"❌ Error processing {audio_path}: {e}")
        return None


# Process Dataset
data, labels = [], []
dataset_path = "dataset/ravdess/"

for subdir, _, files in os.walk(dataset_path):
    for file in tqdm(files):
        if file.endswith(".wav"):
            file_path = os.path.join(subdir, file)
            features = extract_features(file_path)

            if features is not None:
                emotion_label = file.split("-")[2]  # Extract emotion from filename
                emotion_name = emotion_dict[emotion_label]

                data.append(features)
                labels.append(emotion_name)

# Convert to DataFrame
df = pd.DataFrame(data)
df["emotion"] = labels
df.to_csv("ravdess_features.csv", index=False)

print("✅ Feature Extraction Complete!")


# Load Extracted Features
df = pd.read_csv("ravdess_features.csv")

# Encode Emotion Labels
label_encoder = LabelEncoder()
df["emotion"] = label_encoder.fit_transform(df["emotion"])

# Split Data
X = np.array(df.drop(columns=["emotion"]))
y = np.array(df["emotion"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch Tensors
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # ✅ Fix: Add channel dim for CNN
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

print(f"Training Samples: {X_train.shape}, Testing Samples: {X_test.shape}")


class SpeechEmotionModel(nn.Module):
    def __init__(self, input_size=40, num_classes=8):
        super(SpeechEmotionModel, self).__init__()

        # ✅ 1D CNN for feature extraction
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.3)

        # ✅ BiLSTM for sequence modeling
        self.lstm = nn.LSTM(64, 128, batch_first=True, bidirectional=True)

        # ✅ Fully connected layer
        self.fc = nn.Linear(128 * 2, num_classes)  # 2x128 because BiLSTM is bidirectional

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.bn1(x)
        x = torch.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.dropout(x)

        x = x.permute(0, 2, 1)  # ✅ Fix shape for LSTM (batch, seq_len, features)
        x, _ = self.lstm(x)

        x = self.fc(x[:, -1, :])  # Use last LSTM output for classification
        return x


# Define Model
num_classes = len(label_encoder.classes_)
emotion_model = SpeechEmotionModel(input_size=40, num_classes=num_classes)

# ✅ Training Configuration
optimizer = optim.Adam(emotion_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
num_epochs = 30

for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = emotion_model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")

# ✅ Save Trained Model
torch.save(emotion_model.state_dict(), "speech_emotion_model.pth")
print("✅ Model retrained and saved successfully!")




import torch

# Load Model
def load_emotion_model(model_path="speech_emotion_model.pth"):
    model = SpeechEmotionModel(input_size=40, num_classes=8)  # ✅ Ensure input_size=40
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()  # ✅ Set to evaluation mode
    return model

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_emotion_model().to(device)

# Ensure test data is on the same device
X_test = X_test.to(device)
y_test = y_test.to(device)

# Fix X_test shape: Remove extra dimension
X_test = X_test.squeeze(2)  # ✅ Removes the extra dimension (1) at index 2

# Ensure correct shape for CNN+LSTM
print(f"🔍 Fixed X_test shape: {X_test.shape}")  # Should be (batch_size, 1, 40)

# Make Predictions
with torch.no_grad():
    outputs = model(X_test)  # ✅ Pass fixed input to the model
    predictions = torch.argmax(outputs, dim=1)

# Make Predictions
with torch.no_grad():
    outputs = model(X_test)  # ✅ Pass input through model
    predictions = torch.argmax(outputs, dim=1)

# Compute Accuracy
accuracy = (predictions == y_test).float().mean().item() * 100
print(f"✅ Test Accuracy: {accuracy:.2f}%")

