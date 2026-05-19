import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import time
import json
import av
from pathlib import Path
import pandas as pd
import plotly.express as px
from datetime import datetime

# Import our custom architecture
from core.vision import VisionEngine
from core.geometry import PhysioGeometry
from core.rom_judge import ROMJudge
from core.config import VIDEOS_DIR
from core.database import init_db, save_workout, get_analytics_df

# Initialize the DB immediately when the app boots
init_db()

ROOT_DIR = Path(__file__).parent

SKELETON_CONNECTIONS = [
    (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 12), (23, 24), (11, 23), (12, 24),
    (23, 25), (25, 27), (24, 26), (26, 28)
]

# -------------------------------------------------------------------
# 1. THE WEBRTC THREAD
# -------------------------------------------------------------------
class PhysioProcessor(VideoProcessorBase):
    def __init__(self):
        self.engine = VisionEngine()
        self.math_calc = PhysioGeometry()
        
        self.judge = None
        self.joint_name = None
        
        self.metrics = {"reps": 0, "state": "INITIALIZING", "progress": 0.0, "last_score": 0.0, "avg_score": 0.0}

        self.joint_dependencies = {
            "left_elbow": [11, 13, 15], "right_elbow": [12, 14, 16],
            "left_shoulder": [23, 11, 13], "right_shoulder": [24, 12, 14],
            "left_knee": [23, 25, 27], "right_knee": [24, 26, 28]
        }

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

    def recv(self, frame): 
        img = frame.to_ndarray(format="bgr24")
        joints = self.engine.process_frame(img)
        
        if joints:
            self.draw_skeleton(img, joints)
            
            if self.judge and self.joint_name:
                raw_angles = self.math_calc.get_standard_angles(joints)
                clean_angles = {}
                
                # Z-Depth Detection for Sideways Exercises
                left_hip_z = joints[23].z
                right_hip_z = joints[24].z
                is_side_profile = abs(left_hip_z - right_hip_z) > 0.15 
                closer_side = "left" if left_hip_z < right_hip_z else "right"

                for j_name, angle in raw_angles.items():
                    if j_name in self.joint_dependencies:
                        is_visible = all(joints[idx].visibility > 0.4 for idx in self.joint_dependencies[j_name])
                        
                        if is_side_profile:
                            if closer_side == "left" and "right" in j_name:
                                is_visible = False
                            elif closer_side == "right" and "left" in j_name:
                                is_visible = False

                        clean_angles[j_name] = angle if is_visible else None
                    else:
                        clean_angles[j_name] = angle
                
                self.metrics = self.judge.process_frame(clean_angles)
                    
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# -------------------------------------------------------------------
# 2. UI HELPERS
# -------------------------------------------------------------------
def load_master_registry():
    cache_path = ROOT_DIR / "data" / "master_registry.json"
    if cache_path.exists():
        with open(cache_path, "r") as f:
            return json.load(f)
    return {}

def render_dashboard_view():
    st.title("📊 Recovery Analytics Dashboard")
    
    df = get_analytics_df()
    
    if df.empty:
        st.info("No workout history found. Complete a session in the Live Tracker to see your data!")
        return

    # Calculate Top-Level KPIs
    total_workouts = len(df)
    total_reps_ever = int(df['total_reps'].sum())
    overall_avg_score = df['average_score'].mean()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sessions", total_workouts)
    col2.metric("Lifetime Reps", total_reps_ever)
    col3.metric("Overall Form Score", f"{overall_avg_score:.1f}%")
    
    st.markdown("---")
    st.subheader("Form Progression Over Time")
    
    fig = px.line(
        df, 
        x="timestamp", 
        y="average_score", 
        color="exercise_name",
        markers=True,
        title="Average Form Quality per Session",
        labels={"average_score": "Form Score (%)", "timestamp": "Date", "exercise_name": "Exercise"}
    )
    fig.update_layout(yaxis_range=[0, 105], hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Session Ledger")
    display_df = df.sort_values("timestamp", ascending=False).copy()
    display_df['average_score'] = display_df['average_score'].apply(lambda x: f"{x:.1f}%")
    display_df['duration_seconds'] = display_df['duration_seconds'].apply(lambda x: f"{x}s")
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)

# -------------------------------------------------------------------
# 3. MAIN APP ROUTING
# -------------------------------------------------------------------
def main():
    st.set_page_config(page_title="PhysioVision Pro", layout="wide")

    registry = load_master_registry()
    if not registry:
        st.error("No precomputed data found. Please run `python scripts/precompute_all.py` first!")
        return

    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Go to:", ["Live Tracker", "Analytics Dashboard"])

    # === PAGE 1: THE DASHBOARD ===
    if app_mode == "Analytics Dashboard":
        render_dashboard_view()

    # === PAGE 2: THE LIVE TRACKER ===
    elif app_mode == "Live Tracker":
        st.title("PhysioVision Pro: ROM & Rep Tracking")

        # Start a workout timer in session state
        if "workout_start" not in st.session_state:
            st.session_state.workout_start = datetime.now()

        st.sidebar.header("Session Settings")
        display_to_key = {data["display_name"]: key for key, data in registry.items()}
        selected_display = st.sidebar.selectbox("Select Exercise", list(display_to_key.keys()))
        
        exercise_key = display_to_key[selected_display]
        active_data = registry[exercise_key]
        video_path = VIDEOS_DIR / active_data["video_file"]
        physics = active_data["physics"]
        
        primary_joint = physics["primary_joint"]
        joint_limits = physics["all_joints"][primary_joint]

        judge_config = {
            "primary_joint": primary_joint,
            "min_angle": joint_limits["min_angle"],
            "max_angle": joint_limits["max_angle"],
            "tolerance": physics["tolerance"],
            "all_joints": physics["all_joints"] 
        }

        with st.sidebar:
            st.success("Precomputed Registry Loaded ⚡")
            st.markdown("### AI Elected Mover:")
            st.info(f"🦾 **{primary_joint.replace('_', ' ').title()}**")
            st.write(f"**Target Peak:** `{judge_config['max_angle']:.1f}°`")
            st.write(f"**Target Rest:** `{judge_config['min_angle']:.1f}°`")

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
            
            if webrtc_ctx.state.playing and webrtc_ctx.video_processor:
                webrtc_ctx.video_processor.joint_name = primary_joint
                
                if not webrtc_ctx.video_processor.judge:
                    webrtc_ctx.video_processor.judge = ROMJudge(judge_config)
                
                # Update UI continuously while camera is running
                while webrtc_ctx.state.playing:
                    metrics = webrtc_ctx.video_processor.metrics
                    
                    score_color = "green" if metrics["avg_score"] > 80 else "orange" if metrics["avg_score"] > 40 else "red"
                    state_color = "blue" if metrics["state"] in ["LIFTING", "LOWERING"] else "green" if metrics["state"] == "PEAK_REACHED" else "gray"

                    rep_ui.markdown(f"<h1 style='text-align: center; font-size: 60px;'>Reps: {metrics['reps']}</h1>", unsafe_allow_html=True)
                    
                    score_ui.markdown(f"""
                        <div style='text-align: center;'>
                            <h3 style='margin-bottom: 0;'>Session Average</h3>
                            <h1 style='color: {score_color}; font-size: 80px; margin-top: -10px;'>{metrics['avg_score']}%</h1>
                            <p style='color: gray; font-size: 20px;'>Last Rep: {metrics['last_score']}% | Motion: {metrics['progress']}%</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    state_ui.markdown(f"<h3 style='text-align: center; color: {state_color};'>{metrics['state']}</h3>", unsafe_allow_html=True)
                    
                    time.sleep(0.05) 
                    
            # --- THE SAVE BUTTON ---
            st.markdown("---")
            if st.button("💾 Finish & Save Workout", use_container_width=True, type="primary"):
                if webrtc_ctx.video_processor and webrtc_ctx.video_processor.metrics["reps"] > 0:
                    final_metrics = webrtc_ctx.video_processor.metrics
                    duration = int((datetime.now() - st.session_state.workout_start).total_seconds())
                    
                    success = save_workout(
                        exercise_name=active_data["display_name"],
                        target_joint=primary_joint,
                        reps=final_metrics["reps"],
                        avg_score=final_metrics["avg_score"],
                        duration=duration
                    )
                    
                    if success:
                        st.success("Workout safely logged to database!")
                        st.session_state.workout_start = datetime.now()
                else:
                    st.warning("Complete at least 1 full rep before saving!")

if __name__ == "__main__":
    main()