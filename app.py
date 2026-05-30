"""
Application Entry Point (Streamlit UI & WebRTC Orchestrator)

Manages the main visual user interface and coordinates the asynchronous execution 
of the real-time computer vision pipeline. Bridges browser-side operations 
(WebRTC video, native Web Speech TTS execution) with server-side processing modules 
and the permanent data persistence layer.
"""
import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import time
import json
import av
from pathlib import Path
import pandas as pd
import plotly.express as px
from datetime import datetime
import subprocess
import sys

from core.vision import VisionEngine
from core.geometry import PhysioGeometry
from core.rom_judge import ROMJudge
from core.database import init_db, save_workout, get_analytics_df

# Initialize structural schema settings within our local SQLite container
init_db()

ROOT_DIR = Path(__file__).parent

# High-visibility landmarks array index map for painting skeletal overlay lines
SKELETON_CONNECTIONS = [
    (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 12), (23, 24), (11, 23), (12, 24),
    (23, 25), (25, 27), (24, 26), (26, 28)
]

class PhysioProcessor(VideoProcessorBase):
    """
    Asynchronous frame processing class inherited from streamlit-webrtc.
    Acts as a non-blocking background thread worker that handles camera frame capture, 
    routes image bytes through computer vision and mathematical modules, paints 
    the annotated skeletal feedback matrix, and exposes a shared memory dictionary 
    for UI rendering.
    """

    def __init__(self):
        """Initializes internal structural wrappers and sets default metrics structures."""
        self.engine = VisionEngine()
        self.math_calc = PhysioGeometry()
        
        self.judge = None
        self.joint_names = []
        
        self.metrics = {
            "reps": 0, "state": "INITIALIZING", "progress": 0.0, 
            "last_score": 0.0, "avg_score": 0.0, "rep_scores": [], 
            "feedback": "Ready...", "tts_feedback": ""
        }

        # MediaPipe landmark indices mapping to ensure primary/secondary visibility tracking
        self.joint_dependencies = {
            "left_elbow": [11, 13, 15], "right_elbow": [12, 14, 16],
            "left_shoulder": [23, 11, 13], "right_shoulder": [24, 12, 14],
            "left_knee": [23, 25, 27], "right_knee": [24, 26, 28],
            "left_shoulder_plane": [11, 12, 13], "right_shoulder_plane": [11, 12, 14],
            "left_hip_plane": [23, 24, 25], "right_hip_plane": [23, 24, 26]
        }

    def draw_skeleton(self, img, joints):
        """
        Paints the high-visibility visual feedback skeleton directly onto the video array matrix.
        Args:
            img (np.ndarray): The target raw BGR image array.
            joints (list): Collection of active landmarks containing screen space mapping coordinates.
        """
        h, w = img.shape[:2]
        for p1, p2 in SKELETON_CONNECTIONS:
            if joints[p1].visibility > 0.5 and joints[p2].visibility > 0.5:
                pt1 = (int(joints[p1].screen_x * w), int(joints[p1].screen_y * h))
                pt2 = (int(joints[p2].screen_x * w), int(joints[p2].screen_y * h))
                cv2.line(img, pt1, pt2, (255, 255, 255), 3)

        for idx in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
            if joints[idx].visibility > 0.5:
                cx, cy = int(joints[idx].screen_x * w), int(joints[idx].screen_y * h)
                cv2.circle(img, (cx, cy), 6, (0, 255, 0), -1)

    def recv(self, frame) -> av.VideoFrame:
        """
        Asynchronous frame processing callback executed for every incoming WebRTC packet.
        Extracts the uncompressed image matrix, isolates skeletal landmarks, extracts 
        geometric coordinates while checking visibility bounds to ignore occluded fields, 
        and updates the shared reference metrics structure.
        Args:
            frame (av.VideoFrame): Immediate raw video frame from the client web browser.
        Returns:
            av.VideoFrame: The annotated frame object packed back into WebRTC format specifications.
        """
        img = frame.to_ndarray(format="bgr24")
        joints = self.engine.process_frame(img)
        
        if joints:
            self.draw_skeleton(img, joints)
            
            if self.judge and self.joint_names:
                raw_angles = self.math_calc.get_standard_angles(joints)
                clean_angles = {}
                
                # Verify confidence tracking bounds across localized functional chains
                for j_name, angle in raw_angles.items():
                    if j_name in self.joint_dependencies:
                        is_visible = all(joints[idx].visibility > 0.65 for idx in self.joint_dependencies[j_name])
                        clean_angles[j_name] = angle if is_visible else None
                    else:
                        clean_angles[j_name] = angle
                
                self.metrics = self.judge.process_frame(clean_angles)
                  
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def load_master_registry() -> dict:
    """Loads the precomputed unified deployment manifest from data cache directories."""
    cache_path = ROOT_DIR / "data" / "master_registry.json"
    if cache_path.exists():
        with open(cache_path, "r") as f:
            return json.load(f)
    return {}

def render_dashboard_view():
    """Compiles and renders the historical recovery metrics dashboard using interactive Plotly timelines."""
    st.title("📊 Recovery Analytics Dashboard")
    df = get_analytics_df()
    
    if df.empty:
        st.info("No workout history found. Complete a session in the Live Tracker to see your data!")
        return

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
        df, x="timestamp", y="average_score", color="exercise_name", markers=True,
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

def render_upload_view():
    """Renders the UI for uploading and calibrating new custom exercises."""
    st.title("📤 Train New Exercise")
    st.markdown("Upload a single-rep reference video to train the CV model on a new movement trajectory.")

    # 1. Capture user inputs
    exercise_name = st.text_input(
        "Exercise Name (e.g., overhead_press):", 
        help="Use lowercase letters and underscores."
    ).strip().lower().replace(" ", "_")
    
    uploaded_file = st.file_uploader("Choose a reference video", type=["mp4"])

    if st.button("Process & Compile Registry", type="primary", use_container_width=True):
        if not exercise_name or not uploaded_file:
            st.error("⚠️ Please provide both a valid name and a video file.")
            return

        # 2. Save the video to the target assets directory
        video_filename = f"{exercise_name}.mp4"
        save_path = ROOT_DIR / "assets" / "videos" / video_filename

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"💾 Video successfully saved to {save_path.relative_to(ROOT_DIR)}")

        # 3. Trigger the Extractor Script
        with st.spinner("🧠 Step 1/2: Extracting spatial 3D trajectories (this may take a moment)..."):
            extractor_script = str(ROOT_DIR / "scripts" / "extractor.py")
            ext_result = subprocess.run(
                [sys.executable, "-X", "utf8", extractor_script], # Added -X utf8
                capture_output=True, text=True, encoding="utf-8"  # Added encoding
            )
            
            if ext_result.returncode != 0:
                st.error(f"❌ Extractor failed:\n{ext_result.stderr}")
                return
            st.success("✅ Spatial trajectory extraction complete!")

        # 4. Trigger the Precompute/Registry Compiler Script
        with st.spinner("⚙️ Step 2/2: Compiling mathematical registry and DTW baselines..."):
            precompute_script = str(ROOT_DIR / "scripts" / "precompute_all.py")
            pre_result = subprocess.run(
                [sys.executable, "-X", "utf8", precompute_script], # Added -X utf8
                capture_output=True, text=True, encoding="utf-8"   # Added encoding
            )
            
            if pre_result.returncode != 0:
                st.error(f"❌ Precomputation failed:\n{pre_result.stderr}")
                return
            st.success("✅ Master registry updated successfully!")

        # 5. Clear caches and prompt restart
        st.balloons()
        st.info("🎉 Training complete! The system has mapped the new physics limits. Refresh the page to load the new registry in the Live Tracker.")

def main():
    """Main frontend orchestration wrapper initializing application UI view trees and loops."""
    st.set_page_config(page_title="PhysioVision Pro", layout="wide")

    registry = load_master_registry()
    if not registry:
        st.error("No precomputed data found. Please run `python scripts/precompute_all.py` first!")
        return

    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Go to:", ["Live Tracker", "Analytics Dashboard", "Upload Custom Exercise"])

    if app_mode == "Upload Custom Exercise":
        render_upload_view()

    elif app_mode == "Analytics Dashboard":
        render_dashboard_view()

    elif app_mode == "Live Tracker":
        st.title("PhysioVision Pro: ROM & Rep Tracking")

        if "workout_start" not in st.session_state:
            st.session_state.workout_start = datetime.now()
        if "backup_metrics" not in st.session_state:
            st.session_state.backup_metrics = {"reps": 0, "avg_score": 0.0, "rep_scores": [], "feedback": "Ready...", "tts_feedback": ""}
        if "last_speech" not in st.session_state:
            st.session_state.last_speech = ""
            st.session_state.last_speech_time = 0.0

        st.sidebar.header("Session Settings")
        display_to_key = {data["display_name"]: key for key, data in registry.items()}
        selected_display = st.sidebar.selectbox("Select Exercise", list(display_to_key.keys()))
        
        st.sidebar.markdown("### Guide Mode")
        guide_mode = st.sidebar.radio(
            "Select visual reference type:", 
            ["Target Images (Step-by-Step)", "Reference Video (Continuous)"]
        )
        
        exercise_key = display_to_key[selected_display]
        active_data = registry[exercise_key]
        physics = active_data["physics"]
        
        video_path = ROOT_DIR / "assets" / "videos" / active_data["video_file"]
        
        primary_joints = physics.get("primary_joints", ["Unknown"])
        display_joint_text = ", ".join(primary_joints).replace("_", " ").title()

        judge_config = {
            "primary_joints": primary_joints,
            "tolerance": physics.get("tolerance", 15.0),
            "all_joints": physics["all_joints"],
            "golden_trajectory": physics.get("golden_trajectory", [])
        }

        with st.sidebar:
            st.success("Precomputed Registry Loaded 🟢")
            st.markdown("### AI Elected Mover:")
            st.info(f"🦾 **{display_joint_text}**")

        step_images = []
        steps_dir = ROOT_DIR / "assets" / "images" / exercise_key
        
        if steps_dir.exists():
            image_paths = []
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                image_paths.extend(list(steps_dir.glob(ext)))
                
            valid_paths = []
            for p in image_paths:
                if p.stem.isdigit():
                    valid_paths.append((int(p.stem), p))
                    
            valid_paths.sort(key=lambda x: x[0])
            target_size = None
            
            for _, path in valid_paths:
                img = cv2.imread(str(path))
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    if target_size is None:
                        target_size = (img_rgb.shape[1], img_rgb.shape[0])
                    else:
                        img_rgb = cv2.resize(img_rgb, target_size)
                    step_images.append(img_rgb)

        col1, col2, col3 = st.columns([1.5, 1.5, 1])

        with col1:
            if guide_mode == "Reference Video (Continuous)":
                st.markdown("### 1. Reference Video")
                if video_path.exists():
                    st.video(str(video_path), loop=True, autoplay=True, muted=True)
                else:
                    st.info(f"Reference video missing at: assets/videos/{active_data['video_file']}")
            else:
                st.markdown("### 1. Target Form Guide")
                guide_ui = st.empty() 

        with col2:
            st.markdown("### 2. Live Tracking")
            # --- AUDIO FEEDBACK FIX ---
            # media_stream_constraints are set to disable the microphone ("audio": False)
            # This prevents the browser from picking up the TTS output and creating an echo loop.
            webrtc_ctx = webrtc_streamer(
                key="physio-tracker",
                video_processor_factory=PhysioProcessor,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True
            )

        with col3:
            st.markdown("### 3. Live Metrics")
            audio_ui = st.empty()
            feedback_ui = st.empty()
            rep_ui = st.empty()
            score_ui = st.empty()
            state_ui = st.empty()
             
            if webrtc_ctx.state.playing and webrtc_ctx.video_processor:
                webrtc_ctx.video_processor.joint_names = primary_joints
                
                if not webrtc_ctx.video_processor.judge:
                    webrtc_ctx.video_processor.judge = ROMJudge(judge_config)
                 
                # Active background polling loop driving visual component syncing
                while webrtc_ctx.state.playing:
                    metrics = webrtc_ctx.video_processor.metrics
                    st.session_state.backup_metrics = metrics
                    
                    if guide_mode == "Target Images (Step-by-Step)":
                        num_images = len(step_images)
                        if num_images > 0:
                            if "ui_target_idx" not in st.session_state:
                                st.session_state.ui_target_idx = 0     
                                st.session_state.ui_direction = "UP"
                                st.session_state.last_transition_time = time.time() 
                                st.session_state.prev_target_idx = 0
                                st.session_state.fade_start_time = 0.0
                             
                            current_progress = metrics['progress'] 
                            target_threshold = (st.session_state.ui_target_idx / (num_images - 1)) * 100.0 if num_images > 1 else 0.0
                            hit_target = False
                            
                            if st.session_state.ui_direction == "UP":
                                if current_progress >= (target_threshold - 15.0):
                                    hit_target = True
                            else: 
                                if current_progress <= (target_threshold + 15.0):
                                    hit_target = True

                            current_time = time.time()
                            time_since_last = current_time - st.session_state.last_transition_time
                            min_wait = 2.0 if st.session_state.ui_target_idx == 0 else 0.5

                            if hit_target and time_since_last > min_wait and num_images > 1:
                                st.session_state.last_transition_time = current_time
                                st.session_state.prev_target_idx = st.session_state.ui_target_idx
                                st.session_state.fade_start_time = current_time
                                
                                if st.session_state.ui_direction == "UP":
                                    if st.session_state.ui_target_idx < num_images - 1:
                                        st.session_state.ui_target_idx += 1
                                    else:
                                        st.session_state.ui_direction = "DOWN"
                                        st.session_state.ui_target_idx = num_images - 2
                                else:
                                    if st.session_state.ui_target_idx > 0:
                                        st.session_state.ui_target_idx -= 1
                                    else:
                                        st.session_state.ui_direction = "UP"
                                        st.session_state.ui_target_idx = 1
                            
                            safe_current_idx = max(0, min(num_images - 1, st.session_state.ui_target_idx))
                            safe_prev_idx = max(0, min(num_images - 1, st.session_state.prev_target_idx))

                            current_img = step_images[safe_current_idx]
                            prev_img = step_images[safe_prev_idx]

                            fade_duration = 0.45 
                            time_in_fade = current_time - st.session_state.fade_start_time

                            # Execute dynamic linear blend to create seamless visual image transitions
                            if time_in_fade < fade_duration and safe_current_idx != safe_prev_idx:
                                alpha = max(0.0, min(1.0, time_in_fade / fade_duration))
                                blended = cv2.addWeighted(prev_img, 1.0 - alpha, current_img, alpha, 0)
                                guide_ui.image(blended, channels="RGB", use_container_width=True)
                            else:
                                guide_ui.image(current_img, channels="RGB", use_container_width=True)
                        else:
                            guide_ui.info(f"Add images (1.jpg, 2.jpg) to assets/images/{exercise_key}/ to enable the guide.")

                    # --- THE SPEECH ENGINE (Asynchronous Browser Audio Interface) ---
                    tts_text = metrics.get("tts_feedback", "")
                    current_time = time.time()
                    time_since_last_speech = current_time - st.session_state.last_speech_time
                    
                    if tts_text and metrics["state"] != "RESTING" and time_since_last_speech > 3.0:
                        is_correction = "Great form" not in tts_text
                        should_speak = False
                        
                        if tts_text != st.session_state.last_speech or is_correction:
                            should_speak = True
                            
                        if should_speak:
                            st.session_state.last_speech = tts_text
                            st.session_state.last_speech_time = current_time
                            
                            # Inject non-blocking JavaScript SpeechSynthesis hooks
                            js_code = f"""
                                <script>
                                    var synth = window.parent.speechSynthesis;
                                    if (!synth.speaking) {{
                                        var msg = new SpeechSynthesisUtterance("{tts_text}");
                                        msg.rate = 1.05;
                                        synth.speak(msg);
                                    }}
                                </script>
                            """
                            audio_ui.empty() 
                            with audio_ui:
                                components.html(js_code, height=0, width=0)

                    # --- UI RENDERING ---
                    feedback_text = metrics.get("feedback", "Ready...")
                    f_bg_color = "#1e4620" if "Flawless form!" in feedback_text else "#4a1911" if "Ready" not in feedback_text else "#2b2b2b"
                    f_text_color = "#a5d6a7" if "Flawless form!" in feedback_text else "#ffab91" if "Ready" not in feedback_text else "gray"
                    
                    feedback_ui.markdown(f"""
                        <div style='background-color: {f_bg_color}; padding: 10px; border-radius: 8px; text-align: center; margin-bottom: 10px; border: 1px solid {f_text_color};'>
                            <h4 style='color: {f_text_color}; margin: 0;'>{feedback_text.upper()}</h4>
                        </div>
                    """, unsafe_allow_html=True)

                    score_color = "green" if metrics["avg_score"] > 80 else "orange" if metrics["avg_score"] > 40 else "red"
                    state_color = "blue" if metrics["state"] in ["LIFTING", "LOWERING", "PEAK"] else "green" if metrics["state"] == "RESTING" else "gray"

                    rep_ui.markdown(f"<h1 style='text-align: center; font-size: 60px;'>Reps: {metrics['reps']}</h1>", unsafe_allow_html=True)
                    score_ui.markdown(f"""
                        <div style='text-align: center;'>
                            <h3 style='margin-bottom: 0;'>Session Average</h3>
                            <h1 style='color: {score_color}; font-size: 80px; margin-top: -10px;'>{metrics['avg_score']}%</h1>
                            <p style='color: gray; font-size: 20px;'>Last Rep: {metrics['last_score']}% | Motion: {metrics['progress']}%</p>
                        </div>
                    """, unsafe_allow_html=True)
                    state_ui.markdown(f"<h3 style='text-align: center; color: {state_color};'>{metrics['state'].replace('_', ' ')}</h3>", unsafe_allow_html=True)
                    
                    time.sleep(0.05) 
                    
            st.markdown("---")
            if st.button("💾 Finish & Save Workout", use_container_width=True, type="primary"):
                final_metrics = st.session_state.backup_metrics
                if final_metrics["reps"] > 0:
                    duration = int((datetime.now() - st.session_state.workout_start).total_seconds())
                    success = save_workout(
                        exercise_name=active_data["display_name"],
                        target_joint=display_joint_text,
                        reps=final_metrics["reps"],
                        avg_score=final_metrics["avg_score"],
                        duration=duration,
                        rep_scores=final_metrics.get("rep_scores", [])
                    )
                    if success:
                        st.success(f"Successfully saved {final_metrics['reps']} reps to the database!")
                        st.session_state.workout_start = datetime.now()
                        st.session_state.backup_metrics = {"reps": 0, "avg_score": 0.0, "rep_scores": [], "feedback": "Ready...", "tts_feedback": ""}
                else:
                    st.warning("Complete at least 1 full rep before saving!")

if __name__ == "__main__":
    main()