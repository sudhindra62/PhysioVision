import cv2
import numpy as np
from core.vision import VisionEngine
from core.geometry import PhysioGeometry

class AutoCalibrator:
    def __init__(self):
        self.engine = VisionEngine()
        self.math_calc = PhysioGeometry()

    def scan_reference_video(self, video_path: str):
        cap = cv2.VideoCapture(str(video_path))
        
        # Track all joints
        history = {
            "left_elbow": [], "right_elbow": [],
            "left_shoulder": [], "right_shoulder": [],
            "left_knee": [], "right_knee": []
        }

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            joints = self.engine.process_frame(frame)
            if joints:
                angles = self.math_calc.get_standard_angles(joints)
                
                # Check visibility for each joint calculation
                # (Assuming your get_standard_angles returns None or 0 for low visibility)
                for joint_name, angle in angles.items():
                    if angle is not None and angle > 0:
                        history[joint_name].append(angle)
                
        cap.release()

        # Build the full-body ROM profile
        full_body_rom = {}
        max_range = 0
        primary_joint = None
        
        for joint, angles in history.items():
            # If the joint was never seen (or seen for less than 5 frames), record as 0
            if not angles or len(angles) < 5:
                full_body_rom[joint] = {"min_angle": 0.0, "max_angle": 0.0, "range": 0.0}
                continue
            
            true_min = float(np.percentile(angles, 5))
            true_max = float(np.percentile(angles, 95))
            joint_range = true_max - true_min
            
            full_body_rom[joint] = {
                "min_angle": round(true_min, 1),
                "max_angle": round(true_max, 1),
                "range": round(joint_range, 1)
            }

            # Elect the rep-counting driver (the one that moves the most)
            if joint_range > max_range:
                max_range = joint_range
                primary_joint = joint

        if not primary_joint:
            return None

        return {
            "primary_joint": primary_joint,
            "all_joints": full_body_rom,
            "tolerance": 15.0 
        }