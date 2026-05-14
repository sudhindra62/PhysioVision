import sys
import cv2
import json
from pathlib import Path

# Ensure Python can find our 'core' package
sys.path.append(str(Path(__file__).parent.parent))

from core.vision import VisionEngine
from core.geometry import PhysioGeometry
from core.config import VIDEOS_DIR, DATA_DIR, EXERCISE_MAP

def generate_ground_truth():
    print("🚀 Initializing Vision Engine (Heavy Model)...")
    engine = VisionEngine()
    math_calc = PhysioGeometry()

    for video_filename, exercise_name in EXERCISE_MAP.items():
        video_path = VIDEOS_DIR / video_filename
        
        if not video_path.exists():
            print(f"⚠️ Skipping {video_filename}: File not found at {video_path}")
            continue

        print(f"\nProcessing {video_filename} -> {exercise_name}.json")
        cap = cv2.VideoCapture(str(video_path))
        
        lead_data = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # 1. Vision Engine handles the vertical padding and 4K scaling automatically
            joints = engine.process_frame(frame)
            
            if joints:
                # 2. Geometry Engine handles the pure 3D vector math
                angles = math_calc.get_standard_angles(joints)
                
                # 3. Save as a serialized dictionary
                frame_data = {
                    "frame": frame_count,
                    "angles": angles
                }
                lead_data.append(frame_data)
                
                if frame_count % 30 == 0:
                    print(f"   Extracted {frame_count} frames...")
                    
        cap.release()
        
        # Save to data/ folder
        if lead_data:
            output_file = DATA_DIR / f"{exercise_name}.json"
            with open(output_file, "w") as f:
                json.dump(lead_data, f, indent=4)
            print(f"✅ Success: {len(lead_data)} frames saved to {output_file.name}")
        else:
            print(f"❌ Failed: No landmarks detected in {video_filename}")

if __name__ == "__main__":
    generate_ground_truth()