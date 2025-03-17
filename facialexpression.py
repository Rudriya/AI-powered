import cv2

import numpy as np

import dlib

import time

import torch

import mediapipe as mp

from deepface import DeepFace  # Face recognition

from fer import FER  # Facial Emotion Recognition

from openface_model import OpenFaceAU  # Action Unit Analysis (requires OpenFace model)



# Load Models

face_detector = dlib.get_frontal_face_detector()

landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download required

emotion_model = FER(mtcnn=True)

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)



# OpenFace Action Unit Model (For Microexpression & Deception Detection)

openface_au = OpenFaceAU()



# Video Capture

cap = cv2.VideoCapture(0)



emotion_trend = []

cheating_flags = {

    "looking_away": False,

    "head_turns": False,

    "lip_sync_mismatch": False,

    "multiple_faces": False,

    "identity_mismatch": False,

    "microexpressions_detected": False

}



# Function to estimate head pose

def get_head_pose(landmarks, frame):

    image_pts = np.float32([

        (landmarks.part(30).x, landmarks.part(30).y),  # Nose tip

        (landmarks.part(8).x, landmarks.part(8).y),    # Chin

        (landmarks.part(36).x, landmarks.part(36).y),  # Left eye left corner

        (landmarks.part(45).x, landmarks.part(45).y),  # Right eye right corner

        (landmarks.part(48).x, landmarks.part(48).y),  # Left Mouth corner

        (landmarks.part(54).x, landmarks.part(54).y)   # Right mouth corner

    ])



    model_pts = np.float32([

        (0.0, 0.0, 0.0),  # Nose tip

        (0.0, -330.0, -65.0),  # Chin

        (-225.0, 170.0, -135.0),  # Left eye left corner

        (225.0, 170.0, -135.0),  # Right eye right corner

        (-150.0, -150.0, -125.0),  # Left mouth corner

        (150.0, -150.0, -125.0)  # Right mouth corner

    ])



    focal_length = frame.shape[1]

    camera_matrix = np.array([

        [focal_length, 0, frame.shape[1] / 2],

        [0, focal_length, frame.shape[0] / 2],

        [0, 0, 1]

    ], dtype="double")



    _, rotation_vec, _ = cv2.solvePnP(model_pts, image_pts, camera_matrix, None)

    angles = cv2.Rodrigues(rotation_vec)[0]



    return angles[0][0], angles[1][0], angles[2][0]  # Yaw, Pitch, Roll



# Function for face recognition

def verify_identity(face_crop, reference_image="reference.jpg"):

    result = DeepFace.verify(face_crop, reference_image, model_name='Facenet', enforce_detection=False)

    return result['verified']



# Process Video Stream

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:

        break



    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector(gray)



    # Multiple Face Detection (Detect Unauthorized Help)

    if len(faces) > 1:

        cheating_flags["multiple_faces"] = True



    for face in faces:

        landmarks = landmark_predictor(gray, face)

        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        face_crop = frame[y:y+h, x:x+w]



        # *Head Pose Estimation*

        yaw, pitch, roll = get_head_pose(landmarks, frame)

        if abs(yaw) > 20 or abs(pitch) > 20:  # Large deviations indicate looking away

            cheating_flags["head_turns"] = True



        # *Eye Gaze Tracking*

        left_eye_x = (landmarks.part(36).x + landmarks.part(39).x) // 2

        right_eye_x = (landmarks.part(42).x + landmarks.part(45).x) // 2

        nose_x = landmarks.part(30).x

        if abs(left_eye_x - nose_x) > 40 or abs(right_eye_x - nose_x) > 40:

            cheating_flags["looking_away"] = True



        # *Lip Sync Analysis (Detect Mismatched Audio & Video)*

        if mp_face_mesh.process(frame_rgb).multi_face_landmarks:

            for face_landmarks in mp_face_mesh.process(frame_rgb).multi_face_landmarks:

                for idx, landmark in enumerate(face_landmarks.landmark):

                    if idx in range(61, 68):  # Lip landmarks

                        cv2.circle(frame, (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])), 2, (0, 255, 0), -1)

        # (TODO: Sync with Speech-to-Text later)



        # *Emotion Recognition (CNN-based FER)*

        emotion, score = emotion_model.top_emotion(face_crop)

        emotion_trend.append(emotion)



        # *Microexpression & Deception Analysis using OpenFace*

        action_units = openface_au.detect_action_units(face_crop)

        if "AU12" in action_units and "AU45" in action_units:

            cheating_flags["microexpressions_detected"] = True



        # *Identity Verification (Ensure Candidate is Same)*

        if not verify_identity(face_crop):

            cheating_flags["identity_mismatch"] = True



    # *Display Cheating Warnings*

    y_offset = 30

    for key, value in cheating_flags.items():

        if value:

            cv2.putText(frame, f"Alert: {key.replace('_', ' ').title()} Detected", (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            y_offset += 30



    cv2.imshow("Advanced Facial Behavior Analysis", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):

        break



cap.release()

cv2.destroyAllWindows()