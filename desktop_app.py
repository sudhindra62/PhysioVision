import sys
import cv2
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

# PyQt UI Imports
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QStackedWidget,
                             QComboBox, QRadioButton, QButtonGroup, QTextEdit, 
                             QTableWidget, QTableWidgetItem, QHeaderView, QFrame)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap

# Native Matplotlib Integration for PyQt6
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Native Text-to-Speech
try:
    from PyQt6.QtTextToSpeech import QTextToSpeech
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

# Import your core PhysioVision modules 
from core.vision import VisionEngine
from core.geometry import PhysioGeometry
from core.rom_judge import ROMJudge
from core.database import get_analytics_df, save_workout

ROOT_DIR = Path(__file__).parent

# ==========================================
# CUSTOM STYLESHEETS (QSS)
# ==========================================
DARK_THEME = """
QWidget { background-color: #121212; color: #ffffff; font-family: 'Segoe UI', sans-serif; }
QFrame#Sidebar { background-color: #1e1e1e; border-right: 1px solid #333; }
QPushButton { background-color: #2b2b2b; border-radius: 12px; padding: 12px 20px; font-weight: bold; border: 1px solid #444; font-size: 14px;}
QPushButton:hover { background-color: #3a3a3a; }
QPushButton:checked { background-color: #0d6efd; color: white; border: none;}
QLabel#MediaBox { background-color: #000000; border: 2px solid #333; border-radius: 16px; }
QLabel#Title { font-size: 28px; font-weight: bold; margin-bottom: 10px;}
QLabel#MetricCard { background-color: #1e1e1e; border-radius: 12px; border: 1px solid #333; padding: 15px; font-size: 16px; font-weight: bold;}
QPushButton#ActionBlue { background-color: #0d6efd; color: white; border: none; }
QPushButton#ActionBlue:hover { background-color: #0b5ed7; }
QPushButton#ActionGreen { background-color: #198754; color: white; border: none; }
QPushButton#ActionGreen:hover { background-color: #157347; }
QPushButton#ActionRed { background-color: #dc3545; color: white; border: none; }
QPushButton#ActionRed:hover { background-color: #bb2d3b; }
QComboBox, QTextEdit { background-color: #1e1e1e; border: 1px solid #444; border-radius: 10px; padding: 10px; font-size: 14px;}
QTableWidget { background-color: #1e1e1e; border-radius: 12px; border: 1px solid #333; gridline-color: #333;}
QHeaderView::section { background-color: #2b2b2b; padding: 8px; font-weight: bold; border: none; border-right: 1px solid #333; border-bottom: 1px solid #333;}
QTableWidget::item { padding: 10px; }
"""

LIGHT_THEME = """
QWidget { background-color: #f8f9fa; color: #212529; font-family: 'Segoe UI', sans-serif; }
QFrame#Sidebar { background-color: #ffffff; border-right: 1px solid #dee2e6; }
QPushButton { background-color: #ffffff; border-radius: 12px; padding: 12px 20px; font-weight: bold; border: 1px solid #ced4da; color: #212529; font-size: 14px;}
QPushButton:hover { background-color: #e9ecef; }
QPushButton:checked { background-color: #0d6efd; color: white; border: none;}
QLabel#MediaBox { background-color: #e9ecef; border: 2px dashed #adb5bd; border-radius: 16px; }
QLabel#Title { font-size: 28px; font-weight: bold; margin-bottom: 10px;}
QLabel#MetricCard { background-color: #ffffff; border-radius: 12px; border: 1px solid #dee2e6; padding: 15px; font-size: 16px; font-weight: bold; color: #212529;}
QPushButton#ActionBlue { background-color: #0d6efd; color: white; border: none; }
QPushButton#ActionBlue:hover { background-color: #0b5ed7; }
QPushButton#ActionGreen { background-color: #198754; color: white; border: none; }
QPushButton#ActionGreen:hover { background-color: #157347; }
QPushButton#ActionRed { background-color: #dc3545; color: white; border: none; }
QPushButton#ActionRed:hover { background-color: #bb2d3b; }
QComboBox, QTextEdit { background-color: #ffffff; border: 1px solid #ced4da; border-radius: 10px; padding: 10px; font-size: 14px;}
QTableWidget { background-color: #ffffff; border-radius: 12px; border: 1px solid #dee2e6; gridline-color: #dee2e6;}
QHeaderView::section { background-color: #f8f9fa; padding: 8px; font-weight: bold; border: none; border-right: 1px solid #dee2e6; border-bottom: 1px solid #dee2e6;}
QTableWidget::item { padding: 10px; }
"""

# ==========================================
# THREADS (Background Processing)
# ==========================================
class ReferencePlayerThread(QThread):
    frame_ready = pyqtSignal(QImage)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self._run_flag = True

    def run(self):
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened(): return
        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = 1.0 / fps if fps > 0 else 1.0 / 30.0

        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            
            # MEMORY FIX: Added .copy() to detach memory from numpy garbage collector
            qt_image = QImage(rgb_image.data, w, h, rgb_image.strides[0], QImage.Format.Format_RGB888).copy()
            self.frame_ready.emit(qt_image)
            time.sleep(delay)
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    metrics_signal = pyqtSignal(dict)
    
    def __init__(self, exercise_config):
        super().__init__()
        self._run_flag = True
        self.config = exercise_config
        self.engine = VisionEngine()
        self.math_calc = PhysioGeometry()
        self.judge = ROMJudge(self.config.get("physics", {}))
        
        # SYNCHRONIZED SKELETON CONNECTIONS FROM APP.PY
        self.SKELETON_CONNECTIONS = [
            (11, 13), (13, 15), (12, 14), (14, 16), # Arms
            (11, 12), (23, 24), (11, 23), (12, 24), # Torso box
            (23, 25), (25, 27), (24, 26), (26, 28), # Legs
            (0, 11), (0, 12),                       # Neck/Head
            (15, 19), (16, 20),                     # Hands
            (27, 31), (28, 32)                      # Feet
        ]
        
        # SYNCHRONIZED JOINT DEPENDENCIES FROM APP.PY
        self.joint_dependencies = {
            "left_elbow": [11, 13, 15], "right_elbow": [12, 14, 16],
            "left_shoulder": [23, 11, 13], "right_shoulder": [24, 12, 14],
            "left_knee": [23, 25, 27], "right_knee": [24, 26, 28],
            "left_shoulder_plane": [11, 12, 13], "right_shoulder_plane": [11, 12, 14],
            "left_hip_plane": [23, 24, 25], "right_hip_plane": [23, 24, 26],
            # --- NEW DEPENDENCIES ---
            "neck": [0, 11, 12], 
            "left_hip": [11, 23, 25], "right_hip": [12, 24, 26],
            "left_ankle": [25, 27, 31], "right_ankle": [26, 28, 32],
            "left_wrist": [13, 15, 19], "right_wrist": [14, 16, 20]
        }

    def run(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
        if not cap.isOpened(): return

        while self._run_flag:
            ret, frame = cap.read()
            if not ret: continue

            frame = cv2.flip(frame, 1)
            joints = self.engine.process_frame(frame)
            
            if joints:
                self._draw_skeleton(frame, joints)
                raw_angles = self.math_calc.get_standard_angles(joints)
                
                clean_angles = {}
                for j_name, angle in raw_angles.items():
                    if j_name in self.joint_dependencies:
                        is_visible = all(joints[idx].visibility > 0.65 for idx in self.joint_dependencies[j_name])
                        clean_angles[j_name] = angle if is_visible else None
                    else:
                        clean_angles[j_name] = angle

                metrics = self.judge.process_frame(clean_angles)
                self.metrics_signal.emit(metrics)

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            
            # MEMORY FIX: Added .copy() to ensure safe threading
            qt_image = QImage(rgb_image.data, w, h, rgb_image.strides[0], QImage.Format.Format_RGB888).copy()
            self.change_pixmap_signal.emit(qt_image)
        cap.release()

    def _draw_skeleton(self, img, joints):
        h, w = img.shape[:2]
        for p1, p2 in self.SKELETON_CONNECTIONS:
            if joints[p1].visibility > 0.5 and joints[p2].visibility > 0.5:
                pt1 = (int(joints[p1].screen_x * w), int(joints[p1].screen_y * h))
                pt2 = (int(joints[p2].screen_x * w), int(joints[p2].screen_y * h))
                cv2.line(img, pt1, pt2, (255, 255, 255), 3)

        # SYNCHRONIZED VISIBLE JOINTS FROM APP.PY
        for idx in [0, 11, 12, 13, 14, 15, 16, 19, 20, 23, 24, 25, 26, 27, 28, 31, 32]:
            if joints[idx].visibility > 0.5:
                cx, cy = int(joints[idx].screen_x * w), int(joints[idx].screen_y * h)
                cv2.circle(img, (cx, cy), 6, (0, 255, 0), -1)

    def stop(self):
        self._run_flag = False
        self.wait()


# ==========================================
# MAIN APPLICATION UI
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self, registry):
        super().__init__()
        self.registry = registry
        self.current_exercise_config = None
        self.video_thread = None
        self.ref_thread = None
        self.is_dark_theme = True

        self.step_images = []
        self.ui_target_idx = 0
        self.prev_target_idx = 0
        self.ui_direction = "UP"
        self.last_transition_time = time.time()
        self.fade_start_time = 0.0
        
        self.image_blend_timer = QTimer()
        self.image_blend_timer.timeout.connect(self._process_image_blend)
        self.image_blend_timer.setInterval(33) 
        
        self.last_speech = ""
        self.last_speech_time = 0.0
        if TTS_AVAILABLE:
            self.tts = QTextToSpeech(self)
        else:
            self.tts = None

        self.setWindowTitle("PhysioVision Pro - Desktop")
        self.resize(1350, 900)
        
        self.init_ui()
        self.apply_theme()
        self.load_real_dashboard_data()
        self.update_ref_media()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.sidebar = QFrame()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(280)
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(25, 40, 25, 40)
        sidebar_layout.setSpacing(15)

        title_lbl = QLabel("PhysioVision")
        title_lbl.setObjectName("Title")
        sidebar_layout.addWidget(title_lbl)
        sidebar_layout.addSpacing(30)

        self.btn_group = QButtonGroup(self)
        self.btn_dashboard = self.create_nav_button("📊  Analytics Dashboard", 0)
        self.btn_exercise = self.create_nav_button("🏋️  Live Tracker", 1)
        self.btn_upload = self.create_nav_button("📤  Train New Exercise", 2)
        self.btn_dashboard.setChecked(True)

        sidebar_layout.addWidget(self.btn_dashboard)
        sidebar_layout.addWidget(self.btn_exercise)
        sidebar_layout.addWidget(self.btn_upload)
        sidebar_layout.addStretch()

        self.btn_theme = QPushButton("🌓  Toggle Theme")
        self.btn_theme.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_theme.clicked.connect(self.toggle_theme)
        sidebar_layout.addWidget(self.btn_theme)

        self.stacked_widget = QStackedWidget()
        self.stacked_widget.addWidget(self.build_dashboard_view())
        self.stacked_widget.addWidget(self.build_exercise_view())
        self.stacked_widget.addWidget(self.build_upload_view())

        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(self.stacked_widget)

    def create_nav_button(self, text, index):
        btn = QPushButton(text)
        btn.setCheckable(True)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(index))
        self.btn_group.addButton(btn)
        return btn

    # ----------------------------------------------------
    # VIEW 1: DASHBOARD (NOW WITH MATPLOTLIB CHART)
    # ----------------------------------------------------
    def build_dashboard_view(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(50, 50, 50, 50)
        layout.setSpacing(25)

        lbl = QLabel("Session Analytics")
        lbl.setObjectName("Title")
        layout.addWidget(lbl)

        # Matplotlib Graph Setup replacing the static Label
        self.graph_canvas = FigureCanvas(Figure(figsize=(8, 4)))
        self.graph_canvas.setMinimumHeight(300)
        layout.addWidget(self.graph_canvas)

        lbl_table = QLabel("Recent Workout Ledger")
        lbl_table.setStyleSheet("font-size: 18px; font-weight: bold; margin-top: 20px;")
        layout.addWidget(lbl_table)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Date", "Exercise", "Reps Completed", "Avg Score (%)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.table)
        return page

    def load_real_dashboard_data(self):
        """Loads SQLite data into both the Table and the Matplotlib Line Chart"""
        try:
            df = get_analytics_df()
            
            # --- Graph Rendering Logic ---
            ax = self.graph_canvas.figure.subplots() if not self.graph_canvas.figure.axes else self.graph_canvas.figure.axes[0]
            ax.clear()

            bg_color = '#1e1e1e' if self.is_dark_theme else '#f8f9fa'
            text_color = 'white' if self.is_dark_theme else 'black'
            
            self.graph_canvas.figure.patch.set_facecolor(bg_color)
            ax.set_facecolor(bg_color)
            ax.tick_params(colors=text_color)
            ax.xaxis.label.set_color(text_color)
            ax.yaxis.label.set_color(text_color)
            ax.title.set_color(text_color)
            for spine in ax.spines.values():
                spine.set_edgecolor(text_color)

            if df.empty:
                self.table.setRowCount(0)
                ax.text(0.5, 0.5, "No workout history found.", ha='center', va='center', color=text_color)
                self.graph_canvas.draw()
                return
            
            # Draw Data Lines
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            plot_df = df.sort_values(by='timestamp')
            for ex_name, group in plot_df.groupby('exercise_name'):
                label_name = str(ex_name).replace('_', ' ').title()
                ax.plot(group['timestamp'], group['average_score'], marker='o', linestyle='-', label=label_name)
                
            ax.set_title("Average Form Quality per Session")
            ax.set_ylabel("Form Score (%)")
            ax.set_ylim([0, 105])
            ax.legend()
            self.graph_canvas.figure.tight_layout()
            self.graph_canvas.draw()

            # --- Table Rendering Logic ---
            df_table = df.sort_values(by='timestamp', ascending=False).reset_index(drop=True)
            self.table.setRowCount(len(df_table))
            for row in range(len(df_table)):
                self.table.setItem(row, 0, QTableWidgetItem(df_table.loc[row, 'timestamp'].strftime("%Y-%m-%d %H:%M")))
                self.table.setItem(row, 1, QTableWidgetItem(str(df_table.loc[row, 'exercise_name']).replace('_', ' ').title()))
                self.table.setItem(row, 2, QTableWidgetItem(str(df_table.loc[row, 'total_reps'])))
                self.table.setItem(row, 3, QTableWidgetItem(f"{df_table.loc[row, 'average_score']:.1f}%"))
                
        except Exception as e:
            print(f"Database load error: {e}")

    # ----------------------------------------------------
    # VIEW 2: LIVE TRACKER
    # ----------------------------------------------------
    def build_exercise_view(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)

        top_bar = QHBoxLayout()
        top_bar.setSpacing(15)
        
        self.combo_exercise = QComboBox()
        self.combo_exercise.addItems([data["display_name"] for data in self.registry.values()])
        self.combo_exercise.setMinimumWidth(300)
        self.combo_exercise.currentIndexChanged.connect(self.update_ref_media) 
        
        self.radio_video = QRadioButton("Reference Video")
        self.radio_image = QRadioButton("Step-by-Step Images")
        self.radio_video.setChecked(True)
        self.radio_video.toggled.connect(self.update_ref_media) 

        self.btn_start_stop = QPushButton("▶ Start Tracker")
        self.btn_start_stop.setObjectName("ActionGreen")
        self.btn_start_stop.setCheckable(True)
        self.btn_start_stop.clicked.connect(self.toggle_tracking)

        top_bar.addWidget(QLabel("Select Target:"))
        top_bar.addWidget(self.combo_exercise)
        top_bar.addSpacing(20)
        top_bar.addWidget(self.radio_video)
        top_bar.addWidget(self.radio_image)
        top_bar.addStretch()
        top_bar.addWidget(self.btn_start_stop)
        layout.addLayout(top_bar)

        media_layout = QHBoxLayout()
        media_layout.setSpacing(30)
        
        self.ref_display = QLabel("Target Form Guide")
        self.ref_display.setObjectName("MediaBox")
        self.ref_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ref_display.setMinimumSize(480, 480)
        
        self.user_display = QLabel("Webcam Feed\n(Press Start Tracker)")
        self.user_display.setObjectName("MediaBox")
        self.user_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.user_display.setMinimumSize(480, 480)

        media_layout.addWidget(self.ref_display)
        media_layout.addWidget(self.user_display)
        layout.addLayout(media_layout, stretch=1)

        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(20)
        
        self.lbl_reps = QLabel("Reps: 0")
        self.lbl_reps.setObjectName("MetricCard")
        self.lbl_reps.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.lbl_score = QLabel("Score: 0.0%")
        self.lbl_score.setObjectName("MetricCard")
        self.lbl_score.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.lbl_state = QLabel("State: RESTING")
        self.lbl_state.setObjectName("MetricCard")
        self.lbl_state.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.lbl_feedback = QLabel("Feedback: Ready...")
        self.lbl_feedback.setObjectName("MetricCard")
        self.lbl_feedback.setAlignment(Qt.AlignmentFlag.AlignCenter)

        metrics_layout.addWidget(self.lbl_reps)
        metrics_layout.addWidget(self.lbl_score)
        metrics_layout.addWidget(self.lbl_state)
        metrics_layout.addWidget(self.lbl_feedback, stretch=2)
        layout.addLayout(metrics_layout)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(15)
        self.btn_delete = QPushButton("🗑️ Delete Session")
        self.btn_save = QPushButton("💾 Save Workout")
        self.btn_save.setObjectName("ActionBlue")
        self.btn_save.setMinimumWidth(250)
        self.btn_save.clicked.connect(self.save_real_session)
        
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_delete)
        btn_layout.addWidget(self.btn_save)
        layout.addLayout(btn_layout)

        return page

    # ----------------------------------------------------
    # MEDIA & ALPHA BLENDING LOGIC (FIXED)
    # ----------------------------------------------------
    def update_ref_media(self):
        selected_name = self.combo_exercise.currentText()
        exercise_key = next((k for k, v in self.registry.items() if v["display_name"] == selected_name), None)
        if not exercise_key: return

        if self.ref_thread:
            self.ref_thread.stop()
            self.ref_thread = None
            
        self.image_blend_timer.stop()

        if self.radio_video.isChecked():
            vid_path = ROOT_DIR / "assets" / "videos" / f"{exercise_key}.mp4"
            if vid_path.exists():
                self.ref_thread = ReferencePlayerThread(vid_path)
                self.ref_thread.frame_ready.connect(self.update_ref_frame)
                self.ref_thread.start()
            else:
                self.ref_display.setText(f"Video Not Found:\nassets/videos/{exercise_key}.mp4")
        else:
            self.step_images = []
            steps_dir = ROOT_DIR / "assets" / "images" / exercise_key
            
            if steps_dir.exists():
                image_paths = []
                for ext in ["*.jpg", "*.jpeg", "*.png"]:
                    image_paths.extend(list(steps_dir.glob(ext)))
                    
                valid_paths = [p for p in image_paths if p.stem.isdigit()]
                valid_paths.sort(key=lambda x: int(x.stem))
                
                target_size = None
                for path in valid_paths:
                    img = cv2.imread(str(path))
                    if img is not None:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        if target_size is None:
                            target_size = (img_rgb.shape[1], img_rgb.shape[0])
                        else:
                            img_rgb = cv2.resize(img_rgb, target_size)
                        self.step_images.append(img_rgb)

            if self.step_images:
                self.ui_target_idx = 0
                self.prev_target_idx = 0
                self.ui_direction = "UP"
                self.last_transition_time = time.time()
                self.fade_start_time = 0.0
                self.image_blend_timer.start()
            else:
                self.ref_display.setText(f"Images Not Found:\nassets/images/{exercise_key}/1.jpg")

    def _process_image_blend(self):
        if not self.step_images: return
        
        num_images = len(self.step_images)
        safe_current = max(0, min(num_images - 1, self.ui_target_idx))
        safe_prev = max(0, min(num_images - 1, self.prev_target_idx))

        current_img = self.step_images[safe_current]
        prev_img = self.step_images[safe_prev]

        fade_duration = 0.45 
        time_in_fade = time.time() - self.fade_start_time

        if time_in_fade < fade_duration and safe_current != safe_prev:
            alpha = max(0.0, min(1.0, time_in_fade / fade_duration))
            blended = cv2.addWeighted(prev_img, 1.0 - alpha, current_img, alpha, 0)
        else:
            blended = current_img

        h, w, ch = blended.shape
        # MEMORY FIX: Added .copy() to ensure safe rendering of the blended frame
        qt_image = QImage(blended.data, w, h, blended.strides[0], QImage.Format.Format_RGB888).copy()
        self.update_ref_frame(qt_image)

    def update_ref_frame(self, qt_img):
        scaled = QPixmap.fromImage(qt_img).scaled(self.ref_display.width(), self.ref_display.height(), Qt.AspectRatioMode.KeepAspectRatio)
        self.ref_display.setPixmap(scaled)

    # ----------------------------------------------------
    # VIEW 3: UPLOAD CUSTOM VIDEO
    # ----------------------------------------------------
    def build_upload_view(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(60, 60, 60, 60)
        layout.setSpacing(25)

        lbl = QLabel("Train New Custom Exercise")
        lbl.setObjectName("Title")
        layout.addWidget(lbl)
        
        self.btn_upload_vid = QPushButton("📥 Drop MP4 Video File Here")
        self.btn_upload_vid.setObjectName("MediaBox") 
        self.btn_upload_vid.setFixedHeight(200)
        layout.addWidget(self.btn_upload_vid)

        self.btn_extract = QPushButton("⚙️ Compile AI Registry")
        self.btn_extract.setObjectName("ActionBlue")
        self.btn_extract.setFixedHeight(60)
        layout.addWidget(self.btn_extract)

        self.terminal_log = QTextEdit()
        self.terminal_log.setReadOnly(True)
        self.terminal_log.setPlaceholderText("Extraction terminal output will appear here...")
        layout.addWidget(self.terminal_log)
        return page

    # ----------------------------------------------------
    # METRICS, STATE, AND TTS UPDATE LOGIC
    # ----------------------------------------------------
    def update_metrics_display(self, metrics):
        self.lbl_reps.setText(f"Reps: {metrics['reps']}")
        self.lbl_score.setText(f"Score: {metrics['avg_score']}%")
        self.lbl_state.setText(f"State: {metrics['state']}")
        self.lbl_feedback.setText(f"Feedback: {metrics['feedback']}")
        
        if metrics['state'] in ["LIFTING", "LOWERING"]: self.lbl_state.setStyleSheet("color: #0d6efd;") 
        elif metrics['state'] == "PEAK": self.lbl_state.setStyleSheet("color: #198754;") 
        else: self.lbl_state.setStyleSheet("") 
            
        if self.tts:
            tts_text = metrics.get("tts_feedback", "")
            current_time = time.time()
            time_since_last_speech = current_time - self.last_speech_time
            
            if tts_text and metrics["state"] != "RESTING" and time_since_last_speech > 3.0:
                is_correction = "Great form" not in tts_text
                should_speak = False
                
                if tts_text != self.last_speech or is_correction:
                    should_speak = True
                    
                if should_speak:
                    self.last_speech = tts_text
                    self.last_speech_time = current_time
                    self.tts.say(tts_text)

        if self.radio_image.isChecked() and self.step_images:
            num_images = len(self.step_images)
            if num_images > 1:
                current_progress = metrics.get('progress', 0.0)
                target_threshold = (self.ui_target_idx / (num_images - 1)) * 100.0
                
                hit_target = False
                if self.ui_direction == "UP":
                    if current_progress >= (target_threshold - 15.0): hit_target = True
                else: 
                    if current_progress <= (target_threshold + 15.0): hit_target = True

                current_time = time.time()
                time_since_last = current_time - self.last_transition_time
                min_wait = 2.0 if self.ui_target_idx == 0 else 0.5

                if hit_target and time_since_last > min_wait:
                    self.last_transition_time = current_time
                    self.prev_target_idx = self.ui_target_idx
                    self.fade_start_time = current_time
                    
                    if self.ui_direction == "UP":
                        if self.ui_target_idx < num_images - 1:
                            self.ui_target_idx += 1
                        else:
                            self.ui_direction = "DOWN"
                            self.ui_target_idx = num_images - 2
                    else:
                        if self.ui_target_idx > 0:
                            self.ui_target_idx -= 1
                        else:
                            self.ui_direction = "UP"
                            self.ui_target_idx = 1

    # ----------------------------------------------------
    # CORE CONTROLS & DB SAVING
    # ----------------------------------------------------
    def toggle_theme(self):
        self.is_dark_theme = not self.is_dark_theme
        self.apply_theme()
        self.load_real_dashboard_data() # Force graph refresh to match theme

    def apply_theme(self):
        if self.is_dark_theme: self.setStyleSheet(DARK_THEME)
        else: self.setStyleSheet(LIGHT_THEME)

    def toggle_tracking(self):
        if self.btn_start_stop.isChecked():
            self.btn_start_stop.setText("⏹ Stop Tracker")
            self.btn_start_stop.setObjectName("ActionRed") 
            self.apply_theme() 
            
            selected_name = self.combo_exercise.currentText()
            exercise_key = next((k for k, v in self.registry.items() if v["display_name"] == selected_name), None)
            
            if exercise_key:
                self.current_exercise_config = self.registry[exercise_key]
                self.start_camera_thread()
        else:
            self.btn_start_stop.setText("▶ Start Tracker")
            self.btn_start_stop.setObjectName("ActionGreen")
            self.apply_theme() 
            self.stop_camera_thread()

    def save_real_session(self):
        if not self.current_exercise_config: return
        try:
            current_reps = int(self.lbl_reps.text().split(": ")[1])
            current_score = float(self.lbl_score.text().split(": ")[1].replace("%", ""))
            
            if current_reps > 0:
                ex_name = self.current_exercise_config["display_name"]
                target = self.current_exercise_config["physics"]["primary_joints"][0]
                save_workout(ex_name, target, current_reps, current_score, duration=0)
                self.lbl_feedback.setText("Feedback: Workout Saved Successfully!")
                self.load_real_dashboard_data() 
            else:
                self.lbl_feedback.setText("Feedback: Cannot save. Complete at least 1 rep.")
        except Exception as e:
            print(f"Failed to save: {e}")

    def start_camera_thread(self):
        if self.video_thread is None:
            self.video_thread = VideoThread(self.current_exercise_config)
            self.video_thread.change_pixmap_signal.connect(self.update_video_frame)
            self.video_thread.metrics_signal.connect(self.update_metrics_display)
            self.video_thread.start()

    def stop_camera_thread(self):
        if self.video_thread is not None:
            self.video_thread.stop()
            self.video_thread = None
            self.user_display.clear()
            self.user_display.setText("Webcam Feed\n(Press Start Tracker)")
            self.lbl_state.setText("State: STOPPED")
            self.lbl_feedback.setText("Feedback: Tracker Offline")

    def update_video_frame(self, qt_img):
        scaled = QPixmap.fromImage(qt_img).scaled(self.user_display.width(), self.user_display.height(), Qt.AspectRatioMode.KeepAspectRatio)
        self.user_display.setPixmap(scaled)

    def closeEvent(self, event):
        self.stop_camera_thread()
        if self.ref_thread: self.ref_thread.stop()
        self.image_blend_timer.stop()
        event.accept()

# ==========================================
# APP LAUNCHER
# ==========================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion") 
    
    try:
        registry_path = ROOT_DIR / "data" / "master_registry.json"
        with open(registry_path, "r") as f:
            registry = json.load(f)
    except Exception:
        print("⚠️ Master registry not found in /data/. Please run precompute_all.py")
        registry = {}
        
    window = MainWindow(registry)
    window.show()
    sys.exit(app.exec())