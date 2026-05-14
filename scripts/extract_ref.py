import cv2
import mediapipe as mp
import json
import os
import numpy as np
from core.pose_estimator import PoseResult, Keypoint
from core.angle_calculator import AngleCalculator

# Initialize MediaPipe and MindDock Logic
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(static_image_mode=True, model_complexity=1)
angle_calc = AngleCalculator(use_3d=True)

def extract_exercise_angles(video_path, output_name):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    angle_history = []

    print(f"Processing: {video_path} at {fps} FPS...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB for MediaPipe [cite: 437]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(rgb_frame)

        if results.pose_landmarks:
            # 1. Convert landmarks to your PoseResult format [cite: 152]
            kps = []
            for i, lm in enumerate(results.pose_landmarks.landmark):
                kps.append(Keypoint(
                    name=str(i), # You can map these to JOINT_NAMES if preferred
                    x=lm.x, y=lm.y, z=lm.z, 
                    visibility=lm.visibility
                ))
            
            pose_res = PoseResult(keypoints=kps)

            # 2. Calculate all joint angles [cite: 71, 400]
            # Focus on REQ-5: Shoulder, Elbow, Hip, and Knee [cite: 506]
            current_angles = angle_calc.calculate_all_angles(pose_res)
            
            # 3. Clean data: Replace None with 0.0 for JSON stability
            clean_angles = {k: (v if v is not None else 0.0) for k, v in current_angles.items()}
            angle_history.append(clean_angles)

    cap.release()

    # Save to data folder [cite: 392]
    os.makedirs('data', exist_ok=True)
    with open(f'data/{output_name}.json', 'w') as f:
        json.dump(angle_history, f, indent=4)
    
    print(f"Successfully saved {len(angle_history)} frames to data/{output_name}.json")

if __name__ == "__main__":
    # Example usage for one of your 12 exercises
    extract_exercise_angles("videos/standard_squat.mp4", "squats")