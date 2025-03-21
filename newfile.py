import librosa
import whisper
import soundfile as sf
import numpy as np
from pydub import AudioSegment
import noisereduce as nr
from sklearn.preprocessing import StandardScaler
import torch
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from huggingface_hub import login
from model_train import load_emotion_model  # Ensure this function is correctly defined

# ✅ Authenticate with Hugging Face
login(token="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")

# ✅ Load Speaker Embedding Model
model = PretrainedSpeakerEmbedding("pyannote/embedding", device="cpu")  # Change to "cuda" if using GPU
print("✅ Speaker Embedding Model Loaded Successfully!")

# ✅ Load Whisper Model
whisper_model = whisper.load_model("base")

# ✅ Load Emotion Model
emotion_model = load_emotion_model("speech_emotion_model.pth")
emotion_model.eval()  # ✅ Ensure model is in evaluation mode
scaler = StandardScaler()

# -------------------------- 1️⃣ Extract Audio from Video --------------------------
def extract_audio(video_path, output_audio_path):
    AudioSegment.from_file(video_path).export(output_audio_path, format="wav")

# -------------------------- 2️⃣ Speech-to-Text using Whisper --------------------------
def transcribe_audio(audio_path):
    result = whisper_model.transcribe(audio_path)
    return result["text"]

# -------------------------- 3️⃣ Prosody Analysis (Pitch, Tone, Speech Rate) --------------------------
def analyze_prosody(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    pitch = librosa.yin(y, fmin=50, fmax=300)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return np.mean(pitch), tempo

# -------------------------- 4️⃣ Noise Detection (Background Analysis) --------------------------
def detect_noise(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    reduced_noise = nr.reduce_noise(y=y, sr=sr)
    noise_level = np.mean(np.abs(y - reduced_noise))
    return noise_level > 0.01  # Flag if noise level is above threshold

# -------------------------- 5️⃣ Speaker Verification (Detect Multiple Voices) --------------------------
def verify_speaker(audio_path, reference_embedding):
    y, sr = librosa.load(audio_path, sr=16000)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    embedding = model(y_tensor)
    similarity = torch.nn.functional.cosine_similarity(reference_embedding, embedding, dim=0)
    return similarity.item() < 0.7  # If below threshold, another speaker is detected

# -------------------------- 6️⃣ Emotion Detection from Speech --------------------------
def predict_emotion(audio_path):
    """Predict emotions using CNN + BiLSTM model."""
    try:
        y, sr = librosa.load(audio_path, sr=16000, duration=3)  # ✅ Use first 3 seconds
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs = np.mean(mfccs.T, axis=0)

        # ✅ Normalize features
        mfccs_scaled = scaler.fit_transform(mfccs.reshape(1, -1))

        # ✅ Fix Input Shape for CNN + LSTM (batch, channels, features)
        mfccs_tensor = torch.tensor(mfccs_scaled, dtype=torch.float32).unsqueeze(0)  # (1, 40)
        mfccs_tensor = mfccs_tensor.squeeze(2)  # ✅ Shape (1, 1, 40)

        print(f"🔍 Fixed Input Shape to Model: {mfccs_tensor.shape}")

        # ✅ Make prediction
        with torch.no_grad():
            output = emotion_model(mfccs_tensor)
            emotion_index = torch.argmax(output, dim=1).item()

        return emotion_index  # ✅ Return predicted emotion index

    except Exception as e:
        print(f"❌ Error in Emotion Prediction: {e}")
        return None  # Return None in case of an error

# -------------------------- ✅ Example Usage --------------------------
if __name__ == "__main__":
    extract_audio("input_video.mp4", "output_audio.wav")
    text = transcribe_audio("output_audio.wav")
    pitch, tempo = analyze_prosody("output_audio.wav")
    noise_flag = detect_noise("output_audio.wav")
    emotion = predict_emotion("output_audio.wav")

    print("✅ Transcribed Text:", text)
    print("✅ Speech Pitch:", pitch, "Speech Rate:", tempo)
    print("✅ Background Noise Detected:", noise_flag)
    print("✅ Emotion Detected:", emotion if emotion is not None else "Error in prediction")
