import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import time
import json
from pathlib import Path

# Import our new 2D/3D Architecture
from core.vision import VisionEngine
from core.geometry import PhysioGeometry
from core.rom_judge import ROMJudge
from core.config import VIDEOS_DIR

# Define ROOT_DIR if not imported from config
ROOT_DIR = Path(__file__).parent

# Skeleton drawing logic
SKELETON_CONNECTIONS = [
    (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 12), (23, 24), (11, 23), (12, 24),
    (23, 25), (25, 27), (24, 26), (26, 28)
]

# -------------------------------------------------------------------
# 1. THE WEBRTC THREAD
# -------------------------------------------------------------------
class PhysioProcessor(VideoTransformerBase):
    def __init__(self):
        self.engine = VisionEngine()
        self.math_calc = PhysioGeometry()
        
        # These will be injected by the UI
        self.judge = None
        self.joint_name = None
        
        # NEW: Updated real-time metrics for the UI
        self.metrics = {"reps": 0, "state": "INITIALIZING", "progress": 0.0, "rep_score": 0.0}

    def draw_skeleton(self, img, joints):
        h, w = img.shape[:2]
        for p1, p2 in SKELETON_CONNECTIONS:
            if joints[p1].visibility > 0.5 and joints[p2].visibility > 0.5:
                pt1 = (int(joints[p1].x * w), int(joints[p1].y * h))
                pt2 = (int(joints[p2].x * w), int(joints[p2].y * h))
                cv2.line(img, pt1, pt2, (255, 255, 255), 3)

        for idx in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
            if joints[idx].visibility > 0.5:
                cx, cy = int(joints[idx].x * w), int(joints[idx].y * h)
                cv2.circle(img, (cx, cy), 6, (0, 255, 0), -1)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        joints = self.engine.process_frame(img)
        
        if joints:
            self.draw_skeleton(img, joints)
            
            # Run the ROM Judge if we have a calibrated state machine
            if self.judge and self.joint_name:
                # Calculate ALL angles
                angles = self.math_calc.get_standard_angles(joints)
                
                # NEW: Pass the ENTIRE dictionary of angles to the judge for form checking
                self.metrics = self.judge.process_frame(angles)
                    
        return img

# -------------------------------------------------------------------
# 2. THE MAIN STREAMLIT UI
# -------------------------------------------------------------------
def load_master_registry():
    """Instantly loads the full precomputed AI registry."""
    cache_path = ROOT_DIR / "data" / "master_registry.json"
    if cache_path.exists():
        with open(cache_path, "r") as f:
            return json.load(f)
    return {}

def main():
    st.set_page_config(page_title="PhysioVision Pro", layout="wide")
    st.title("PhysioVision Pro: ROM & Rep Tracking")

    # --- LOAD PRECOMPUTED DATA ---
    registry = load_master_registry()
    
    if not registry:
        st.error("No precomputed data found. Please run `python scripts/precompute_all.py` first!")
        return

    # --- SIDEBAR ---
    st.sidebar.header("Session Settings")
    
    # Map nice display names back to their dictionary keys
    display_to_key = {data["display_name"]: key for key, data in registry.items()}
    selected_display = st.sidebar.selectbox("Select Exercise", list(display_to_key.keys()))
    
    exercise_key = display_to_key[selected_display]
    active_data = registry[exercise_key]
    
    video_path = VIDEOS_DIR / active_data["video_file"]
    physics = active_data["physics"]
    
    primary_joint = physics["primary_joint"]
    joint_limits = physics["all_joints"][primary_joint]

    # Package the config perfectly for the existing ROMJudge
    judge_config = {
        "primary_joint": primary_joint,
        "min_angle": joint_limits["min_angle"],
        "max_angle": joint_limits["max_angle"],
        "tolerance": physics["tolerance"],
        "all_joints": physics["all_joints"] # Included for form-correction!
    }

    with st.sidebar:
        st.success("Precomputed Registry Loaded ⚡")
        st.markdown("### AI Elected Mover:")
        st.info(f"🦾 **{primary_joint.replace('_', ' ').title()}**")
        st.write(f"**Target Peak:** `{judge_config['max_angle']:.1f}°`")
        st.write(f"**Target Rest:** `{judge_config['min_angle']:.1f}°`")

    # --- MAIN LAYOUT ---
    col1, col2, col3 = st.columns([1.5, 1.5, 1])

    with col1:
        st.markdown("### 1. Watch Ideal Form")
        if video_path.exists():
            st.video(str(video_path), loop=True, autoplay=True, muted=True)

    with col2:
        st.markdown("### 2. Live Tracking")
        webrtc_ctx = webrtc_streamer(
            key="physio-tracker",
            video_processor_factory=PhysioProcessor,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            async_processing=True
        )

    with col3:
        st.markdown("### 3. Live Metrics")
        rep_ui = st.empty()
        score_ui = st.empty()
        state_ui = st.empty()
        
        # --- THE UI PULL LOOP ---
        if webrtc_ctx.state.playing and webrtc_ctx.video_processor:
            webrtc_ctx.video_processor.joint_name = primary_joint
            
            # Inject the precomputed config into the State Machine
            if not webrtc_ctx.video_processor.judge:
                webrtc_ctx.video_processor.judge = ROMJudge(judge_config)
            
            while True:
                metrics = webrtc_ctx.video_processor.metrics
                
                # NEW: Separate UI logic for Form Score vs Progress
                score_color = "green" if metrics["rep_score"] > 80 else "orange" if metrics["rep_score"] > 40 else "red"
                state_color = "blue" if metrics["state"] in ["LIFTING", "LOWERING"] else "green" if metrics["state"] == "PEAK_REACHED" else "gray"

                rep_ui.markdown(f"<h1 style='text-align: center; font-size: 60px;'>Reps: {metrics['reps']}</h1>", unsafe_allow_html=True)
                
                score_ui.markdown(f"""
                    <div style='text-align: center;'>
                        <h3 style='margin-bottom: 0;'>Last Rep Quality</h3>
                        <h1 style='color: {score_color}; font-size: 80px; margin-top: -10px;'>{metrics['rep_score']}%</h1>
                        <p style='color: gray; font-size: 20px;'>Current Motion: {metrics['progress']}%</p>
                    </div>
                """, unsafe_allow_html=True)
                
                state_ui.markdown(f"<h3 style='text-align: center; color: {state_color};'>{metrics['state']}</h3>", unsafe_allow_html=True)
                
                time.sleep(0.1)

if __name__ == "__main__":
    main()