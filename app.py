from fastapi import FastAPI, File, UploadFile, Form
import cv2
import numpy as np
import torch
import whisper
import librosa
import openai
import bigO
import ast
from deepface import DeepFace
from pydub import AudioSegment
from PIL import Image
import torchvision.transforms as transforms

app = FastAPI()

# Load AI Models
whisper_model = whisper.load_model("base")  # Whisper model for speech-to-text
emotion_model = torch.load("cnn_lstm_model.pth")  # CNN-LSTM for facial behavior
emotion_model.eval()

# Object Detection Model (YOLO)
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# OpenAI API Key for GPT-based code review
openai.api_key = "your-api-key"

# Image transformation for CNN-LSTM
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# --------------------- 1️⃣ VIDEO PROCESSING (Face, Gaze, Object Detection) ---------------------
@app.post("/video_analysis/")
async def analyze_video(file: UploadFile = File(...)):
    nparr = np.frombuffer(await file.read(), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Face Detection
    faces = DeepFace.detectFace(frame, detector_backend="mtcnn")

    # Object Detection
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    objects_detected = []
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                objects_detected.append({"object": class_id, "confidence": float(confidence)})

    return {"faces_detected": len(faces), "objects_detected": objects_detected}


# --------------------- 2️⃣ AUDIO PROCESSING (Speech-to-Text, Emotion, Prosody) ---------------------
@app.post("/audio_analysis/")
async def analyze_audio(file: UploadFile = File(...)):
    # Convert to WAV format
    audio = AudioSegment.from_file(file.file)
    audio.export("audio.wav", format="wav")

    # Speech-to-text using Whisper
    transcript = whisper_model.transcribe("audio.wav")["text"]

    # Extract speech features
    y, sr = librosa.load("audio.wav", sr=16000)
    pitch = librosa.yin(y, 80, 400, sr=sr)
    energy = np.mean(y ** 2)

    return {"transcript": transcript, "pitch": np.mean(pitch), "energy": energy}


# --------------------- 3️⃣ FACIAL & BEHAVIORAL ANALYSIS (Microexpressions, Engagement) ---------------------
@app.post("/facial_behavior_analysis/")
async def analyze_behavior(file: UploadFile = File(...)):
    img = Image.open(file.file)
    img_tensor = transform(img).unsqueeze(0)

    # Predict emotions
    with torch.no_grad():
        output = emotion_model(img_tensor)
        predicted_emotion = torch.argmax(output, dim=1).item()

    return {"emotion": predicted_emotion}


# --------------------- 4️⃣ CODE EVALUATION (AST Parsing, Big-O, AI Review) ---------------------
@app.post("/evaluate_code/")
async def evaluate_code(code: str = Form(...)):
    # Analyze Code Structure (AST Parsing)
    tree = ast.parse(code)
    num_functions = sum(isinstance(node, ast.FunctionDef) for node in ast.walk(tree))

    # Algorithm Complexity (Big-O)
    exec_namespace = {}
    exec(code, exec_namespace)
    test_function = list(exec_namespace.values())[-1]
    bigo = bigO.BigO()
    complexity = bigo.test(test_function, lambda n: (n,))

    # GPT-4 Review
    review = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "Analyze the code quality and logic."},
                  {"role": "user", "content": code}]
    )["choices"][0]["message"]["content"]

    return {"num_functions": num_functions, "complexity": str(complexity), "ai_feedback": review}


# --------------------- 5️⃣ FINAL CANDIDATE ASSESSMENT ---------------------
@app.post("/final_assessment/")
async def final_assessment(
    face_analysis: dict, audio_analysis: dict, coding_analysis: dict
):
    """
    Combines all AI results into a final candidate report.
    """
    final_score = (
        (face_analysis["engagement"] * 0.3) +
        (audio_analysis["energy"] * 0.2) +
        (coding_analysis["num_functions"] * 0.5)
    )

    return {
        "final_score": final_score,
        "feedback": "Improve clarity in explanations" if final_score < 60 else "Good performance!"
    }


# --------------------- RUN FASTAPI SERVER ---------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
