import sys
import json
import cv2
from pathlib import Path

# Setup paths
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from core.calibrator import AutoCalibrator
from core.config import VIDEOS_DIR, EXERCISE_MAP

def extract_metadata_and_thumbnail(video_path: Path, output_dir: Path):
    cap = cv2.VideoCapture(str(video_path))
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_sec = frame_count / fps if fps > 0 else 0
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 10)
    ret, frame = cap.read()
    
    thumbnail_filename = f"{video_path.stem}_thumb.jpg"
    thumbnail_path = output_dir / thumbnail_filename
    
    if ret:
        h, w = frame.shape[:2]
        new_w = 400
        new_h = int((new_w / w) * h)
        resized_frame = cv2.resize(frame, (new_w, new_h))
        cv2.imwrite(str(thumbnail_path), resized_frame)
    
    cap.release()
    
    return {
        "duration_sec": round(duration_sec, 1),
        "fps": round(fps, 2),
        "thumbnail": thumbnail_filename
    }

def get_golden_trajectory(video_id: str) -> list:
    exercise_name = EXERCISE_MAP.get(f"{video_id}.mp4", video_id)
    json_path = ROOT_DIR / "data" / f"{exercise_name}.json"
    
    if json_path.exists():
        with open(json_path, "r") as f:
            data = json.load(f)
            return [frame["angles"] for frame in data]
    return []

def main():
    print("🚀 Starting the Master Precomputation Pipeline...")
    calibrator = AutoCalibrator()
    
    data_dir = ROOT_DIR / "data"
    data_dir.mkdir(exist_ok=True)
    
    assets_dir = ROOT_DIR / "assets" 
    assets_dir.mkdir(exist_ok=True)
    
    master_registry = {}
    
    for video_path in VIDEOS_DIR.glob("*.mp4"):
        video_id = video_path.stem
        print(f"\n⚙️ Processing [{video_id}]...")
    
        print("   -> Extracting Metadata & Thumbnails...")
        meta_info = extract_metadata_and_thumbnail(video_path, assets_dir)
        
        print("   -> Running AI Biomechanical Scan...")
        physics_config = calibrator.scan_reference_video(str(video_path))
        
        if physics_config:
            print("   -> Baking Golden Trajectory for DTW Matching...")
            golden_seq = get_golden_trajectory(video_id)
            
            if not golden_seq:
                print(f"   ⚠️ Warning: No extracted JSON found for {video_id}. Run extractor.py first!")
            
            physics_config["golden_trajectory"] = golden_seq
            
            master_registry[video_id] = {
                "display_name": video_id.replace("_", " ").title(),
                "video_file": video_path.name,
                "meta": meta_info,
                "physics": physics_config
            }
            print(f"✅ Success! Elected Joint: {physics_config['primary_joint']} | Trajectory Frames: {len(golden_seq)}")
        else:
            print(f"❌ AI Failed to extract motion from {video_id}")

    registry_path = data_dir / "master_registry.json"
    with open(registry_path, "w") as f:
        json.dump(master_registry, f, indent=4)
        
    print(f"\n🎉 Pipeline Complete! Engine is ready. Data saved to {registry_path.name}")

if __name__ == "__main__":
    main()