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

# Make sure to import this at the top if you haven't:
# from core.config import VIDEOS_DIR

def main():
    st.set_page_config(page_title="PhysioVision Pro", layout="wide")
    st.title("PhysioVision Pro: Real-Time Movement Analysis")

    st.sidebar.header("Session Settings")
    
    # We select by the video name now to make pathing easier
    exercise_choice = st.sidebar.selectbox(
        "Select Exercise",
        ["vid1", "vid2", "vid3", "vid4"]
    )

    json_filename = f"{exercise_choice}.json"
    mp4_filename = f"{exercise_choice}.mp4"
    
    # Load data and paths
    reference_data = load_ground_truth(json_filename)
    video_path = Path("videos") / mp4_filename

    if reference_data:
        st.sidebar.success(f"Loaded Reference: {json_filename}")

    # --- NEW LAYOUT: 3 Columns ---
    # Col 1: Reference Video | Col 2: Live Camera | Col 3: Score
    col1, col2, col3 = st.columns([1.5, 1.5, 1])

    with col1:
        st.markdown("### 1. Watch Ideal Form")
        if video_path.exists():
            # autoplay and loop keep the video playing endlessly
            st.video(str(video_path), loop=True, autoplay=True, muted=True)
        else:
            st.error(f"Video file not found: {video_path}")

    with col2:
        st.markdown("### 2. Live Tracking")
        webrtc_ctx = webrtc_streamer(
            key="physio-tracker",
            video_processor_factory=PhysioProcessor,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            async_processing=True
        )

    with col3:
        st.markdown("### 3. Live Score")
        score_placeholder = st.empty()
        
        # UI Loop for the Score
        if webrtc_ctx.state.playing:
            if webrtc_ctx.video_processor:
                webrtc_ctx.video_processor.reference_data = reference_data
                
                while True:
                    score = webrtc_ctx.video_processor.current_score
                    color = "green" if score > 75 else "orange" if score > 50 else "red"
                    
                    # Ensure we don't show 0.0 when they are just standing still
                    display_text = f"{score:.1f}%" if score > 0 else "Move..."
                    
                    score_placeholder.markdown(
                        f"""
                        <div style='display: flex; justify-content: center; align-items: center; height: 100%;'>
                            <h1 style='color: {color}; font-size: 80px; margin-top: 50px;'>{display_text}</h1>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    time.sleep(0.3)

if __name__ == "__main__":
    main()