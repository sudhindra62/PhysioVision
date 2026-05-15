import sys
import json
import cv2
from pathlib import Path

# Setup paths
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from core.calibrator import AutoCalibrator
from core.config import VIDEOS_DIR

def extract_metadata_and_thumbnail(video_path: Path, output_dir: Path):
    """Extracts duration, FPS, and saves a preview thumbnail."""
    cap = cv2.VideoCapture(str(video_path))
    
    # 1. Grab metadata
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_sec = frame_count / fps if fps > 0 else 0
    
    # 2. Extract Thumbnail (Grab the 10th frame to avoid black starting frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 10)
    ret, frame = cap.read()
    
    thumbnail_filename = f"{video_path.stem}_thumb.jpg"
    thumbnail_path = output_dir / thumbnail_filename
    
    if ret:
        # Resize thumbnail to save space (e.g., width 400px)
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

def main():
    print("🚀 Starting the Master Precomputation Pipeline...")
    calibrator = AutoCalibrator()
    
    # Output Directories
    data_dir = ROOT_DIR / "data"
    data_dir.mkdir(exist_ok=True)
    
    assets_dir = ROOT_DIR / "assets" # New folder for thumbnails
    assets_dir.mkdir(exist_ok=True)
    
    master_registry = {}
    
    # Process every video
    for video_path in VIDEOS_DIR.glob("*.mp4"):
        video_id = video_path.stem
        print(f"\n⚙️ Processing [{video_id}]...")
        
        # Step 1: Precompute Metadata & Assets
        print("   -> Extracting Metadata & Thumbnails...")
        meta_info = extract_metadata_and_thumbnail(video_path, assets_dir)
        
        # Step 2: Precompute AI Physics
        print("   -> Running AI Biomechanical Scan...")
        physics_config = calibrator.scan_reference_video(str(video_path))
        
        if physics_config:
            # Combine everything into one massive dictionary
            master_registry[video_id] = {
                "display_name": video_id.replace("_", " ").title(),
                "video_file": video_path.name,
                "meta": meta_info,
                "physics": physics_config
            }
            print(f"✅ Success! Elected Joint: {physics_config['primary_joint']}")
        else:
            print(f"❌ AI Failed to extract motion from {video_id}")

    # Save the Master Registry
    registry_path = data_dir / "master_registry.json"
    with open(registry_path, "w") as f:
        json.dump(master_registry, f, indent=4)
        
    print(f"\n🎉 Pipeline Complete! Engine is ready. Data saved to {registry_path.name}")

if __name__ == "__main__":
    main()