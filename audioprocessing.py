import librosa
import whisper
import ffmpeg
import soundfile as sf
import numpy as np
from pydub import AudioSegment
import noisereduce as nr
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import torch
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

# Load Whisper ASR model
whisper_model = whisper.load_model("base")

# Load Emotion Detection Model (Pre-trained on RAVDESS)
emotion_model = load_model("speech_emotion_model.h5")
scaler = StandardScaler()

# Load Speaker Verification Model
embedding_model = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb")

# Extract Audio from Video
def extract_audio(video_path, output_audio_path):
    AudioSegment.from_file(video_path).export(output_audio_path, format="wav")

# Speech-to-Text using Whisper
def transcribe_audio(audio_path):
    audio, sr = librosa.load(audio_path, sr=16000)
    sf.write("processed_audio.wav", audio, sr)
    result = whisper_model.transcribe("processed_audio.wav")
    return result["text"]

# Prosody Analysis (Pitch, Tone, Speech Rate)
def analyze_prosody(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    pitch = librosa.yin(y, fmin=50, fmax=300)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return np.mean(pitch), tempo

# Noise Detection (Background Analysis)
def detect_noise(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    reduced_noise = nr.reduce_noise(y=y, sr=sr)
    noise_level = np.mean(np.abs(y - reduced_noise))
    return noise_level > 0.01  # Flag if noise level is above threshold

# Speaker Verification (Detect Multiple Voices)
def verify_speaker(audio_path, reference_embedding):
    y, sr = librosa.load(audio_path, sr=16000)
    embedding = embedding_model(torch.tensor(y).unsqueeze(0))
    similarity = torch.nn.functional.cosine_similarity(reference_embedding, embedding)
    return similarity.item() < 0.7  # If below threshold, another speaker is detected

# Emotion Detection from Speech
def predict_emotion(audio_path):
    y, sr = librosa.load(audio_path, sr=16000, duration=3)  # Use first 3 seconds
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs = np.mean(mfccs.T, axis=0)
    mfccs = scaler.fit_transform(mfccs.reshape(1, -1))
    prediction = emotion_model.predict(mfccs)
    return np.argmax(prediction)  # Return emotion class index

# Example Usage
# extract_audio("input_video.mp4", "output_audio.wav")
# text = transcribe_audio("output_audio.wav")
# pitch, tempo = analyze_prosody("output_audio.wav")
# noise_flag = detect_noise("output_audio.wav")
# emotion = predict_emotion("output_audio.wav")
# print("Transcribed Text:", text)
# print("Speech Pitch:", pitch, "Speech Rate:", tempo)
# print("Background Noise Detected:", noise_flag)
# print("Emotion Detected:", emotion)
