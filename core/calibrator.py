import cv2
import numpy as np
from core.vision import VisionEngine
from core.geometry import PhysioGeometry
from collections import defaultdict

class AutoCalibrator:
    def __init__(self):
        self.engine = VisionEngine()
        self.math_calc = PhysioGeometry()

    def scan_reference_video(self, video_path: str):
        cap = cv2.VideoCapture(str(video_path))
        
        # --- FIX 1: Use a dynamic dictionary so it never crashes on new angles ---
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
            
            true_min = float(np.percentile(angles, 5))
            true_max = float(np.percentile(angles, 95))
            joint_range = true_max - true_min
            
            full_body_rom[joint] = {
                "min_angle": round(true_min, 1),
                "max_angle": round(true_max, 1),
                "range": round(joint_range, 1)
            }

            # --- FIX 2: Prevent the AI from picking a 'plane' gatekeeper as the main exercise joint ---
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