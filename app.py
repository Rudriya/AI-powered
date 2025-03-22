from fastapi import FastAPI, File, UploadFile, Form, HTTPException
import cv2
import numpy as np
import torch
import whisper
import librosa
import openai
import ast
import json
from deepface import DeepFace
from pydub import AudioSegment
from PIL import Image
import torchvision.transforms as transforms
from starlette.responses import JSONResponse

app = FastAPI()

# ✅ Load AI Models
whisper_model = whisper.load_model("base")  # Whisper model for speech-to-text
emotion_model = torch.load("speech_emotion_model.pth")  # CNN-LSTM model for speech emotion
emotion_model.eval()

# ✅ Object Detection Model (YOLO)
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# ✅ OpenAI API Key for GPT-based code review
openai.api_key = "your-api-key"

# ✅ Image transformation for CNN-LSTM model
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ✅ Store registered user images
registered_faces = {}


# --------------------- 1️⃣ USER LOGIN & FACE REGISTRATION ---------------------
@app.post("/login/")
async def register_user(username: str = Form(...), image: UploadFile = File(...)):
    """
    User uploads their image for registration.
    """
    img = Image.open(image.file)
    registered_faces[username] = np.array(img)
    return {"message": "✅ User registered successfully!", "username": username}


@app.post("/verify_face/")
async def verify_face(username: str = Form(...), captured_image: UploadFile = File(...)):
    """
    Compare the registered face with the live face.
    """
    if username not in registered_faces:
        raise HTTPException(status_code=400, detail="User not registered!")

    # Convert uploaded image
    live_img = np.array(Image.open(captured_image.file))

    # Face verification using DeepFace
    result = DeepFace.verify(live_img, registered_faces[username], model_name="Facenet")

    if result["verified"]:
        return {"verified": True, "message": "✅ Face Matched! Access Granted."}
    else:
        return {"verified": False, "message": "❌ Face Mismatch! Access Denied."}


# --------------------- 2️⃣ VIDEO PROCESSING (Face, Gaze, Object Detection) ---------------------
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


# --------------------- 3️⃣ AUDIO PROCESSING (Speech-to-Text, Emotion, Prosody) ---------------------
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


# --------------------- 4️⃣ FACIAL & BEHAVIORAL ANALYSIS (Microexpressions, Engagement) ---------------------
@app.post("/facial_behavior_analysis/")
async def analyze_behavior(file: UploadFile = File(...)):
    img = Image.open(file.file)
    img_tensor = transform(img).unsqueeze(0)

    # Predict emotions
    with torch.no_grad():
        output = emotion_model(img_tensor)
        predicted_emotion = torch.argmax(output, dim=1).item()

    return {"emotion": predicted_emotion}


# --------------------- 5️⃣ CODE EVALUATION (AST Parsing, Big-O, Test Cases, AI Review) --------------------- #
import bigO
from pylint import epylint as lint

@app.post("/evaluate_code/")
async def evaluate_code(code: str = Form(...), test_cases: str = Form(default="[]")):
    """
    Evaluates candidate code based on:
    - Code Structure (AST)
    - Algorithm Complexity (Big-O)
    - Test Cases Execution
    - Code Quality Analysis (Pylint)
    - AI Rule-Based Feedback
    """

    # ✅ Code Structure Analysis (AST Parsing)
    try:
        tree = ast.parse(code)
        num_functions = sum(isinstance(node, ast.FunctionDef) for node in ast.walk(tree))
        num_loops = sum(isinstance(node, (ast.For, ast.While)) for node in ast.walk(tree))
        num_conditions = sum(isinstance(node, ast.If) for node in ast.walk(tree))

        structure_analysis = {
            "num_functions": num_functions,
            "num_loops": num_loops,
            "num_conditions": num_conditions
        }
    except SyntaxError as e:
        raise HTTPException(status_code=400, detail=f"Syntax Error in Code: {str(e)}")

    # ✅ Algorithm Complexity Analysis (Big-O)
    exec_namespace = {}
    try:
        exec(code, exec_namespace)
        test_function = [f for f in exec_namespace.values() if callable(f)][-1]  # Get the last defined function
        bigo = bigO.BigO()
        complexity = bigo.test(test_function, "random")
        complexity_str = str(complexity)
    except Exception as e:
        complexity_str = f"Error: {str(e)}"

    # ✅ Test Case Execution
    try:
        test_cases = json.loads(test_cases)  # Convert string input to list of tuples
        test_results = []
        for inp, expected in test_cases:
            try:
                output = test_function(*inp)
                test_results.append({
                    "input": inp,
                    "expected": expected,
                    "output": output,
                    "status": "Pass" if output == expected else "Fail"
                })
            except Exception as e:
                test_results.append({"input": inp, "error": str(e), "status": "Error"})
    except Exception as e:
        test_results = [{"error": f"Invalid test case format: {str(e)}"}]

    # ✅ Code Quality Analysis (Pylint)
    with open("temp_code.py", "w") as f:
        f.write(code)

    pylint_stdout, pylint_stderr = lint.py_run("temp_code.py", return_std=True)
    code_quality_report = pylint_stdout.getvalue()

    # ✅ AI-Generated Feedback (Predefined Rule-Based)
    feedback = []
    if num_functions > 3:
        feedback.append("Your code contains multiple functions. Ensure they follow modularity best practices.")
    if "O(N^2)" in complexity_str:
        feedback.append("Your algorithm has quadratic complexity (O(N^2)). Consider optimizing with better algorithms.")
    if any(res["status"] == "Fail" for res in test_results):
        feedback.append("Some test cases failed. Check edge cases and logic errors.")

    if not feedback:
        feedback.append("Code looks well-structured and optimized.")

    return {
        "structure_analysis": structure_analysis,
        "algorithm_complexity": complexity_str,
        "test_case_results": test_results,
        "code_quality_report": code_quality_report,
        "feedback": "\n".join(feedback)
    }


# --------------------- 6️⃣ FINAL CANDIDATE ASSESSMENT ---------------------
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
