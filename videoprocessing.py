import cv2
import numpy as np
import dlib
import mediapipe as mp
import time

# Initialize models
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
mp_pose = mp.solutions.pose.Pose()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download required

# Video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_mesh = mp_face_mesh.process(frame_rgb)
    results_pose = mp_pose.process(frame_rgb)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    cheating_flags = {
        "multiple_faces": False,
        "looking_away": False,
        "phone_detected": False,
        "background_movement": False
    }
    
    # Multiple Face Detection
    if len(faces) > 1:
        cheating_flags["multiple_faces"] = True
    
    for face in faces:
        landmarks = predictor(gray, face)
        face_crop = frame[face.top():face.bottom(), face.left():face.right()]
        
        # Extract Lip movement (MediaPipe FaceMesh)
        if results_mesh.multi_face_landmarks:
            for face_landmarks in results_mesh.multi_face_landmarks:
                for idx, landmark in enumerate(face_landmarks.landmark):
                    h, w, _ = frame.shape
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    if idx in range(61, 68):  # Lip landmarks
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
    
    # Eye Gaze Detection (Detect Looking Away)
    for face in faces:
        landmarks = predictor(gray, face)
        left_eye_x = (landmarks.part(36).x + landmarks.part(39).x) // 2
        right_eye_x = (landmarks.part(42).x + landmarks.part(45).x) // 2
        nose_x = landmarks.part(30).x
        if abs(left_eye_x - nose_x) > 40 or abs(right_eye_x - nose_x) > 40:
            cheating_flags["looking_away"] = True
    
    # Background Movement Detection
    fgmask = cv2.createBackgroundSubtractorMOG2().apply(frame)
    if np.sum(fgmask) > 50000:
        cheating_flags["background_movement"] = True
    
    # Posture Analysis (Detect Slouching/Fidgeting)
    if results_pose.pose_landmarks:
        for landmark in results_pose.pose_landmarks.landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
    
    # Display Cheating Warnings
    y_offset = 30
    for key, value in cheating_flags.items():
        if value:
            cv2.putText(frame, f"Alert: {key.replace('_', ' ').title()} Detected", (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 30
    
    cv2.imshow("Video Processing", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
