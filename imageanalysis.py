import cv2
import torch
import dlib
import numpy as np
from deepface import DeepFace
from imutils import face_utils
from ultralytics import YOLO
from mtcnn import MTCNN

# Load YOLO model for object detection
yolo_model = YOLO("yolov8n.pt")  # Use a smaller YOLO model for efficiency

# Load Dlib facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize MTCNN for multi-person detection
mtcnn_detector = MTCNN()

def verify_identity(candidate_img, registered_img):
    """
    Compare candidate's face with registered profile using DeepFace (FaceNet).
    """
    result = DeepFace.verify(img1_path=candidate_img, img2_path=registered_img, model_name='Facenet')
    return result["verified"], result["distance"]

def detect_multiple_faces(frame):
    """
    Detect the number of faces using MTCNN.
    """
    faces = mtcnn_detector.detect_faces(frame)
    return len(faces)

def track_eye_gaze(frame):
    """
    Track eye gaze to check if candidate maintains screen focus.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)
        
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        
        left_ratio = (np.linalg.norm(left_eye[1] - left_eye[5]) + np.linalg.norm(left_eye[2] - left_eye[4])) / (2.0 * np.linalg.norm(left_eye[0] - left_eye[3]))
        right_ratio = (np.linalg.norm(right_eye[1] - right_eye[5]) + np.linalg.norm(right_eye[2] - right_eye[4])) / (2.0 * np.linalg.norm(right_eye[0] - right_eye[3]))

        avg_ratio = (left_ratio + right_ratio) / 2.0
        
        # Heuristic threshold for normal gaze
        if avg_ratio < 0.2:
            return "Looking Down"
        elif avg_ratio > 0.35:
            return "Looking Up"
        else:
            return "Looking at Screen"

def detect_objects(frame):
    """
    Detect objects (phone, books, extra devices) using YOLOv8.
    """
    results = yolo_model(frame)
    detected_objects = []
    
    for result in results.xyxy[0]:  # Bounding boxes format: (x1, y1, x2, y2, confidence, class)
        x1, y1, x2, y2, confidence, class_id = result
        class_id = int(class_id)

        # Define classes of interest (YOLO class labels)
        object_classes = {0: "Person", 67: "Cell Phone", 63: "Laptop", 73: "Book"}

        if class_id in object_classes and confidence > 0.5:
            detected_objects.append(object_classes[class_id])

    return detected_objects

def analyze_frame(candidate_img, registered_img):
    """
    Analyze a single frame to validate identity, fairness, and distractions.
    """
    frame = cv2.imread(candidate_img)

    # 1. Verify Identity
    identity_verified, face_distance = verify_identity(candidate_img, registered_img)

    # 2. Check for multiple people
    num_faces = detect_multiple_faces(frame)

    # 3. Track eye gaze
    gaze_status = track_eye_gaze(frame)

    # 4. Detect objects (external help)
    detected_objects = detect_objects(frame)

    # Generate Final Assessment Report
    fairness_status = "Fair Interview" if identity_verified and num_faces == 1 and not detected_objects else "Potential Cheating"

    report = {
        "identity_verified": identity_verified,
        "face_distance": face_distance,
        "multiple_people_detected": num_faces > 1,
        "eye_gaze_status": gaze_status,
        "detected_objects": detected_objects,
        "fairness_status": fairness_status
    }

    return report

# Example Usage
if __name__ == "__main__":
    candidate_img = "candidate_frame.jpg"  # Replace with actual frame
    registered_img = "registered_profile.jpg"  # Replace with actual registered image

    result = analyze_frame(candidate_img, registered_img)
    print(result)
