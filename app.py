import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import json
import time
from collections import deque
from pathlib import Path

# Import our custom architecture
from core.vision import VisionEngine
from core.geometry import PhysioGeometry
from core.matcher import DTWJudge
from core.config import DATA_DIR

# Add this mapping outside the class at the top of app.py
# Pairs of joint indices to connect with lines
SKELETON_CONNECTIONS = [
    (11, 13), (13, 15), # Left Arm
    (12, 14), (14, 16), # Right Arm
    (11, 12), (23, 24), (11, 23), (12, 24), # Torso Box
    (23, 25), (25, 27), # Left Leg
    (24, 26), (26, 28)  # Right Leg
]

# -------------------------------------------------------------------
# 1. THE WEBRTC THREAD (Camera & Scoring)
# -------------------------------------------------------------------
class PhysioProcessor(VideoTransformerBase):
    def __init__(self):
        self.engine = VisionEngine()
        self.math_calc = PhysioGeometry()
        self.judge = DTWJudge()
        
        self.live_buffer = deque(maxlen=40) 
        self.current_score = 0.0
        self.reference_data = [] 

    def draw_skeleton(self, img, joints):
        """Draws the tracking skeleton on the live video feed."""
        h, w = img.shape[:2]
        
        # 1. Draw the connecting lines (Bones)
        for p1, p2 in SKELETON_CONNECTIONS:
            # Only draw if both joints have good visibility
            if joints[p1].visibility > 0.5 and joints[p2].visibility > 0.5:
                pt1 = (int(joints[p1].x * w), int(joints[p1].y * h))
                pt2 = (int(joints[p2].x * w), int(joints[p2].y * h))
                cv2.line(img, pt1, pt2, (255, 255, 255), 3) # White Bones

        # 2. Draw the tracking points (Joints)
        for idx in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
            if joints[idx].visibility > 0.5:
                cx = int(joints[idx].x * w)
                cy = int(joints[idx].y * h)
                cv2.circle(img, (cx, cy), 6, (0, 255, 0), -1) # Green Joints

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        joints = self.engine.process_frame(img)
        
        if joints:
            live_angles = self.math_calc.get_standard_angles(joints)
            self.live_buffer.append({"angles": live_angles})
            
            if len(self.live_buffer) > 15 and self.reference_data:
                self.current_score = self.judge.calculate_similarity(
                    self.reference_data, 
                    list(self.live_buffer)
                )
            
            # --- Call our new Full-Body Drawing Method ---
            self.draw_skeleton(img, joints)
            
        return img

# -------------------------------------------------------------------
# 2. THE MAIN STREAMLIT UI
# -------------------------------------------------------------------
def load_ground_truth(json_filename):
    path = DATA_DIR / json_filename
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return []

def main():
    st.set_page_config(page_title="PhysioVision Pro", layout="wide")
    st.title("PhysioVision Pro: Real-Time Movement Analysis")

    st.sidebar.header("Session Settings")
    exercise_choice = st.sidebar.selectbox(
        "Select Exercise",
        ["vid1.json", "vid2.json", "vid3.json", "vid4.json"]
    )

    reference_data = load_ground_truth(exercise_choice)
    if reference_data:
        st.sidebar.success(f"Loaded Reference: {exercise_choice}")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Live Tracking")
        webrtc_ctx = webrtc_streamer(
            key="physio-tracker",
            video_processor_factory=PhysioProcessor,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            async_processing=True
        )

    with col2:
        st.markdown("### DTW Similarity Score")
        score_placeholder = st.empty()
        
        # --- THE MISSING UI LOOP ---
        # This keeps pulling the score from the background thread to the frontend
        if webrtc_ctx.state.playing:
            if webrtc_ctx.video_processor:
                # Inject the JSON into the camera thread
                webrtc_ctx.video_processor.reference_data = reference_data
                
                while True:
                    score = webrtc_ctx.video_processor.current_score
                    
                    # Color code the score
                    color = "green" if score > 75 else "orange" if score > 50 else "red"
                    score_placeholder.markdown(
                        f"<h1 style='text-align: center; color: {color}; font-size: 80px;'>{score:.1f}%</h1>", 
                        unsafe_allow_html=True
                    )
                    time.sleep(0.3) # Refresh UI every 300ms

if __name__ == "__main__":
    main()