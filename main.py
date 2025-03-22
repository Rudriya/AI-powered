import streamlit as st
import requests
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from PIL import Image
import io

# Backend API URLs
BASE_URL = "http://localhost:8000"
LOGIN_API = f"{BASE_URL}/login/"
FACE_VERIFICATION_API = f"{BASE_URL}/verify_face/"
VIDEO_API = f"{BASE_URL}/video_analysis/"
AUDIO_API = f"{BASE_URL}/audio_analysis/"
CODE_API = f"{BASE_URL}/evaluate_code/"
FINAL_ASSESSMENT_API = f"{BASE_URL}/final_assessment/"

# Session State for User Authentication
if "user_authenticated" not in st.session_state:
    st.session_state.user_authenticated = False
if "registered_image" not in st.session_state:
    st.session_state.registered_image = None

# ---------------------------- Streamlit UI ---------------------------- #
def main():
    st.set_page_config(page_title="AI Interview System", layout="wide")
    st.title("🎤 AI-Powered Technical Interview System")

    # Show Login Page First
    if not st.session_state.user_authenticated:
        login_page()
    else:
        # Show Main Menu after Login
        st.sidebar.title("Navigation")
        option = st.sidebar.radio("Go to", ["Live Interview", "Upload File", "Code Evaluation", "Final Report"])
        
        if option == "Live Interview":
            live_interview()
        elif option == "Upload File":
            upload_file_analysis()
        elif option == "Code Evaluation":
            code_evaluation()
        elif option == "Final Report":
            final_assessment()


# ---------------------------- 🔹 Login & Face Registration ---------------------------- #
def login_page():
    st.header("🔑 Login to AI Interview System")

    username = st.text_input("Enter your Username:")
    uploaded_file = st.file_uploader("Upload your Registered Face Image", type=["jpg", "png"])

    if uploaded_file:
        st.session_state.registered_image = uploaded_file  # Store uploaded image in session
        st.image(uploaded_file, caption="Registered Face", width=250)

    if st.button("Register & Login"):
        if username and uploaded_file:
            st.session_state.user_authenticated = True
            st.success("✅ Login Successful! Proceed to Face Verification.")
            st.experimental_rerun()
        else:
            st.error("❌ Please enter a username and upload an image!")


# ---------------------------- 🔹 Face Verification Before Interview ---------------------------- #
def verify_user_face(uploaded_frame):
    if st.session_state.registered_image is None:
        return False, "❌ No registered image found. Please login first."

    # Convert images to bytes for API request
    registered_image_bytes = st.session_state.registered_image.getvalue()
    captured_frame_bytes = uploaded_frame.getvalue()

    files = {
        "registered_image": ("registered.jpg", registered_image_bytes, "image/jpeg"),
        "captured_frame": ("captured.jpg", captured_frame_bytes, "image/jpeg")
    }
    
    response = requests.post(FACE_VERIFICATION_API, files=files)
    result = response.json()
    
    return result["verified"], result["message"]


# ---------------------------- 🔹 Live Interview (Webcam & Face Verification) ---------------------------- #
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.text = "Initializing webcam..."
        self.verified = False

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        image = frame.to_ndarray(format="bgr24")
        
        # Convert OpenCV image to PIL for verification
        _, buffer = cv2.imencode(".jpg", image)
        frame_bytes = io.BytesIO(buffer)

        if not self.verified:
            self.verified, message = verify_user_face(frame_bytes)
            self.text = message

        return image

def live_interview():
    st.header("🎥 Live Video & Audio Interview")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📹 Face Verification & Recording")
        ctx = webrtc_streamer(key="video_feed", video_processor_factory=VideoProcessor)

        if ctx.video_processor:
            st.warning(ctx.video_processor.text)

    with col2:
        st.subheader("🎤 Audio Recording")
        st.write("🚀 *Feature Coming Soon!* (Use Upload for Now)")

    st.subheader("Start Analysis:")
    if st.button("Run AI Models on Live Feed"):
        st.warning("🚀 *Live processing feature will be integrated soon!*")
    

# ---------------------------- 🔹 Upload File for Analysis ---------------------------- #
def upload_file_analysis():
    st.header("📂 Upload Interview File for AI Analysis")

    col1, col2 = st.columns(2)
    video_file = col1.file_uploader("🎥 Upload Video File", type=["mp4", "avi", "mov"])
    audio_file = col2.file_uploader("🎤 Upload Audio File", type=["wav", "mp3"])

    if st.button("Analyze Interview"):
        if video_file:
            files = {"file": video_file.getvalue()}
            response = requests.post(VIDEO_API, files=files)
            st.subheader("📊 Video Analysis Results")
            st.json(response.json())

        if audio_file:
            files = {"file": audio_file.getvalue()}
            response = requests.post(AUDIO_API, files=files)
            st.subheader("🎵 Audio Analysis Results")
            st.json(response.json())


# ---------------------------- 🔹 Code Assessment ---------------------------- #
# ---------------------------- 🔹 Code Assessment ---------------------------- #
def code_evaluation():
    st.header("💻 Code Evaluation & AI Feedback")
    
    # Code Input
    code = st.text_area("📜 Paste Your Code for Evaluation", height=200)

    # Test Case Input
    st.subheader("🧪 Add Test Cases")
    test_cases_str = st.text_area("Enter test cases in format: [(input1, expected1), (input2, expected2)]", height=100)

    if st.button("Evaluate Code"):
        try:
            test_cases = eval(test_cases_str)  # Convert input string to Python list
        except Exception as e:
            st.error(f"❌ Invalid Test Cases Format: {e}")
            return
        
        # Send request to FastAPI backend
        response = requests.post(CODE_API, json={"code": code, "test_cases": test_cases})
        
        if response.status_code == 200:
            st.subheader("📊 AI Code Review")
            result = response.json()
            
            # Display Structured Analysis
            st.write("🔍 **Structure Analysis:**")
            st.json(result["structure_analysis"])

            st.write("⚡ **Algorithm Complexity:**")
            st.json(result["algorithm_complexity"])

            st.write("🧪 **Test Case Results:**")
            st.json(result["test_case_results"])

            st.write("🛠 **Code Quality Report:**")
            st.json(result["code_quality_report"])

            st.write("📋 **AI Feedback:**")
            st.success(result["feedback"])
        
        else:
            st.error("❌ Error in Code Evaluation. Please check your input.")



# ---------------------------- 🔹 Final Interview Report ---------------------------- #
def final_assessment():
    st.header("📜 Final Candidate Assessment")

    face_data = st.text_area("👤 Enter Face Analysis Data (JSON)", height=100)
    audio_data = st.text_area("🔊 Enter Audio Analysis Data (JSON)", height=100)
    code_data = st.text_area("💻 Enter Code Evaluation Data (JSON)", height=100)

    if st.button("Generate Final Report"):
        response = requests.post(FINAL_ASSESSMENT_API, json={
            "face_analysis": eval(face_data),
            "audio_analysis": eval(audio_data),
            "coding_analysis": eval(code_data)
        })
        st.subheader("📋 AI-Generated Final Report")
        st.json(response.json())


# ---------------------------- Run the Streamlit App ---------------------------- #
if __name__ == "__main__":
    main()
