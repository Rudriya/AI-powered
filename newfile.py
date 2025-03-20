import librosa
import whisper
import ffmpeg
import soundfile as sf
import numpy as np
from pydub import AudioSegment
import noisereduce as nr
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import torch
from speechbrain.utils.fetching import fetch, LocalStrategy
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
import os

# ✅ Change Fetch Strategy to Copy
fetch.strategy = LocalStrategy.COPY  

# ✅ Load Models
# Load Whisper ASR model
whisper_model = whisper.load_model("base")

from model_train import load_emotion_model  # ✅ Import the model function

# ✅ Load the Model with Required Parameters
emotion_model = load_emotion_model("speech_emotion_model.pth", input_size=64, num_classes=8)

print("✅ Model Loaded Successfully!")
scaler = StandardScaler()


# ✅ Set speechbrain cache directory
os.environ["SPEECHBRAIN_CACHE"] = "C:/Users/Asus/OneDrive/Desktop/New project/project"

# ✅ Change Fetch Strategy to Copy
fetch.strategy = LocalStrategy.COPY  

# ✅ Load Speaker Verification Model
embedding_model = PretrainedSpeakerEmbedding.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=os.environ["SPEECHBRAIN_CACHE"]
)

print("✅ Speaker Embedding Model Loaded Successfully!")



# -------------------------- 1️⃣ Extract Audio from Video --------------------------
def extract_audio(video_path, output_audio_path):
    """
    Extract audio from a video file and save it as a WAV file.
    """
    AudioSegment.from_file(video_path).export(output_audio_path, format="wav")

# -------------------------- 2️⃣ Speech-to-Text using Whisper --------------------------
def transcribe_audio(audio_path):
    result = whisper_model.transcribe(audio_path)  # Use original audio
    return result["text"]

# -------------------------- 3️⃣ Prosody Analysis (Pitch, Tone, Speech Rate) --------------------------
def analyze_prosody(audio_path):
    """
    Extract pitch and speech rate from audio.
    """
    y, sr = librosa.load(audio_path, sr=16000)
    pitch = librosa.yin(y, fmin=50, fmax=300)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return np.mean(pitch), tempo

# -------------------------- 4️⃣ Noise Detection (Background Analysis) --------------------------
def detect_noise(audio_path):
    """
    Detect background noise in audio and reduce it using noisereduce.
    """
    y, sr = librosa.load(audio_path, sr=16000)
    reduced_noise = nr.reduce_noise(y=y, sr=sr)
    noise_level = np.mean(np.abs(y - reduced_noise))
    return noise_level > 0.01  # ✅ Flag if noise level is above threshold

# -------------------------- 5️⃣ Speaker Verification (Detect Multiple Voices) --------------------------
def verify_speaker(audio_path, reference_embedding):
    """
    Verify if another speaker is present by comparing embeddings.
    """
    y, sr = librosa.load(audio_path, sr=16000)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    embedding = embedding_model(y_tensor)

    # Ensure tensor format compatibility
    similarity = torch.nn.functional.cosine_similarity(reference_embedding, embedding, dim=0)

    return similarity.item() < 0.7  # ✅ If below threshold, another speaker is detected

# -------------------------- 6️⃣ Emotion Detection from Speech --------------------------
def predict_emotion(audio_path):
    """
    Predict emotions from speech using a pre-trained CNN model.
    """
    y, sr = librosa.load(audio_path, sr=16000, duration=3)  # ✅ Use first 3 seconds for analysis
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs = np.mean(mfccs.T, axis=0)

    # Standardize input (Prevent refitting scaler)
    mfccs_scaled = scaler.transform(mfccs.reshape(-1, 1))  # ✅ Use transform instead of fit_transform
    
    prediction = emotion_model.predict(mfccs_scaled)
    
    return np.argmax(prediction)  # ✅ Return emotion class index

# -------------------------- ✅ Example Usage --------------------------
if __name__ == "__main__":
    extract_audio("input_video.mp4", "output_audio.wav")
    text = transcribe_audio("output_audio.wav")
    pitch, tempo = analyze_prosody("output_audio.wav")
    noise_flag = detect_noise("output_audio.wav")
    emotion = predict_emotion("output_audio.wav")

    print("Transcribed Text:", text)
    print("Speech Pitch:", pitch, "Speech Rate:", tempo)
    print("Background Noise Detected:", noise_flag)
    print("Emotion Detected:", emotion)
