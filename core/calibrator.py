"""
Auto-Calibrator Module

A utility module designed to autonomously convert raw reference videos into 
mathematical JSON rule sets. It eliminates the need to manually hardcode 
ranges of motion for new exercises, but allows for manual overrides when necessary.
"""
import cv2
import numpy as np
from core.vision import VisionEngine
from core.geometry import PhysioGeometry
from collections import defaultdict

class AutoCalibrator:
    """
    Scans physical reference videos to map anatomical ranges of motion, automatically 
    electing the primary mover and identifying bilateral symmetries for the registry.
    """

    def __init__(self):
        """
        Initializes the underlying Vision and Geometry engines required to 
        process the reference video.
        """
        self.engine = VisionEngine()
        self.math_calc = PhysioGeometry()
        
        # Visibility Dependency Map
        # Ensures the AI ignores noisy, off-screen body parts when picking the primary mover
        self.joint_dependencies = {
            "left_elbow": [11, 13, 15], "right_elbow": [12, 14, 16],
            "left_shoulder": [23, 11, 13], "right_shoulder": [24, 12, 14],
            "left_knee": [23, 25, 27], "right_knee": [24, 26, 28],
            "left_shoulder_plane": [11, 12, 13], "right_shoulder_plane": [11, 12, 14],
            "left_hip_plane": [23, 24, 25], "right_hip_plane": [23, 24, 26],
            "neck": [0, 11, 12],  # Removed hips to allow close-up camera framing
            "left_hip": [11, 23, 25], "right_hip": [12, 24, 26],
            "left_ankle": [25, 27, 31], "right_ankle": [26, 28, 32],
            "left_wrist": [13, 15, 19], "right_wrist": [14, 16, 20]
        }

    def scan_reference_video(self, video_path: str, manual_target_joint: str = None) -> dict | None:
        """
        Processes a video frame-by-frame to generate the base mathematical limits.
        
        Args:
            video_path (str): The absolute or relative path to the local MP4 file.
            manual_target_joint (str, optional): A specific joint to force as the primary mover.
                                                 If None, the AI will auto-detect the primary mover.
                                                 
        Returns:
            dict | None: A compiled physics rule set.
        """
        cap = cv2.VideoCapture(str(video_path))
        history = defaultdict(list)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            joints = self.engine.process_frame(frame)
            if joints:
                angles = self.math_calc.get_standard_angles(joints)
                
                for joint_name, angle in angles.items():
                    # Check if the joint is actually visible on screen
                    is_visible = True
                    if joint_name in self.joint_dependencies:
                        is_visible = all(joints[idx].visibility > 0.65 for idx in self.joint_dependencies[joint_name])
                    
                    # Only record the angle if the body part is visible and math is valid
                    if is_visible and angle is not None and angle > 0:
                        history[joint_name].append(angle)
                
        cap.release()

        # Build the full-body ROM profile
        full_body_rom = {}
        max_range = 0
        top_joint = None
        
        # Check the filename to see if we are targeting the neck or head
        filename = str(video_path).lower()
        is_neck_focus = "neck" in filename or "head" in filename
        
        for joint, angles in history.items():
            if not angles or len(angles) < 5:
                full_body_rom[joint] = {"min_angle": 0.0, "max_angle": 0.0, "range": 0.0}
                continue
            
            # Use percentiles to filter out 3D tracking anomalies/jitter
            true_min = float(np.percentile(angles, 5))
            true_max = float(np.percentile(angles, 95))
            joint_range = true_max - true_min
            
            full_body_rom[joint] = {
                "min_angle": round(true_min, 1),
                "max_angle": round(true_max, 1),
                "range": round(joint_range, 1)
            }

        # --- 1. USER OVERRIDE LOGIC ---
        if manual_target_joint and manual_target_joint in full_body_rom:
            top_joint = manual_target_joint
            max_range = full_body_rom[top_joint]["range"]
            print(f"🎯 Manual Override Accepted: Forcing '{top_joint}' as primary joint.")
            
        # --- 2. AI AUTO-DETECTION FALLBACK ---
        else:
            if manual_target_joint:
                print(f"⚠️ Warning: Requested joint '{manual_target_joint}' not found or visible enough. Falling back to AI auto-detection.")
                
            for joint, data in full_body_rom.items():
                # Prevent the AI from picking a 'plane' gatekeeper
                if joint.endswith("_plane"):
                    continue
                    
                # Ignore arms/shoulders if we know it's a neck exercise
                if is_neck_focus and joint in ["left_elbow", "right_elbow", "left_shoulder", "right_shoulder", "left_wrist", "right_wrist"]:
                    continue

                # Find the joint that moved the most
                if data["range"] > max_range:
                    max_range = data["range"]
                    top_joint = joint

        if not top_joint:
            return None

        # --- Bilateral Symmetry Detection ---
        primary_joints = [top_joint]
        
        # Map joints to their anatomical partners
        pairs = {
            "left_shoulder": "right_shoulder", "right_shoulder": "left_shoulder",
            "left_elbow": "right_elbow", "right_elbow": "left_elbow",
            "left_knee": "right_knee", "right_knee": "left_knee",
            "left_hip": "right_hip", "right_hip": "left_hip",
            "left_ankle": "right_ankle", "right_ankle": "left_ankle",
            "left_wrist": "right_wrist", "right_wrist": "left_wrist"
        }

        if top_joint in pairs:
            partner_joint = pairs[top_joint]
            # Ensure the partner joint actually exists in our scanned data
            if partner_joint in full_body_rom:
                partner_range = full_body_rom[partner_joint]["range"]
                
                # If the partner joint moves at least 70% as much as the primary, 
                # we classify this as a two-arm/two-leg exercise.
                if partner_range >= (max_range * 0.70):
                    primary_joints.append(partner_joint)

        return {
            "primary_joints": primary_joints,  
            "all_joints": full_body_rom,
            "tolerance": 15.0 
        }