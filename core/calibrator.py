"""
Auto-Calibrator Module

A utility module designed to autonomously convert raw reference videos into 
mathematical JSON rule sets. It eliminates the need to manually hardcode 
ranges of motion for new exercises.
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

    def scan_reference_video(self, video_path: str) -> dict | None:
        """
        Processes a video frame-by-frame to generate the base mathematical limits 
        for an exercise. 

        This algorithm uses the 5th and 95th percentiles of recorded angles to 
        safely ignore anomalous tracking spikes. It automatically determines the 
        target of the exercise by finding the joint with the largest Range of 
        Motion (ROM), and uses a 70% threshold to detect if the movement is 
        bilateral (e.g., a standard squat uses both knees).

        Args:
            video_path (str): The absolute or relative path to the local MP4 file.

        Returns:
            dict | None: A compiled physics rule set containing:
                         - 'primary_joints': List of the main active joints.
                         - 'all_joints': Dict containing min, max, and range for every joint.
                         - 'tolerance': Default strictness threshold (float).
                         Returns None if no joints are detected in the video.
        """
        cap = cv2.VideoCapture(str(video_path))
        
        # Use a dynamic dictionary so it never crashes on new angles
        history = defaultdict(list)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            joints = self.engine.process_frame(frame)
            if joints:
                angles = self.math_calc.get_standard_angles(joints)
                
                for joint_name, angle in angles.items():
                    if angle is not None and angle > 0:
                        history[joint_name].append(angle)
                
        cap.release()

        # Build the full-body ROM profile
        full_body_rom = {}
        max_range = 0
        top_joint = None
        
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

            # Prevent the AI from picking a 'plane' gatekeeper as the main exercise joint
            if joint_range > max_range and not joint.endswith("_plane"):
                max_range = joint_range
                top_joint = joint

        if not top_joint:
            return None

        # --- Bilateral Symmetry Detection ---
        primary_joints = [top_joint]
        
        # Map joints to their anatomical partners
        pairs = {
            "left_shoulder": "right_shoulder", "right_shoulder": "left_shoulder",
            "left_elbow": "right_elbow", "right_elbow": "left_elbow",
            "left_knee": "right_knee", "right_knee": "left_knee"
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