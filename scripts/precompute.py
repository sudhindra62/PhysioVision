import sys
import json
from pathlib import Path

# Ensure Python can find the core package
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from core.calibrator import AutoCalibrator
from core.config import VIDEOS_DIR

def main():
    print("Initializing AI Calibrator...")
    calibrator = AutoCalibrator()
    
    # We will save the results in a single cache file
    data_dir = ROOT_DIR / "data"
    data_dir.mkdir(exist_ok=True)
    cache_path = data_dir / "rom_cache.json"
    
    results = {}
    
    # Scan all mp4 files in the videos folder
    for video_path in VIDEOS_DIR.glob("*.mp4"):
        print(f"\nAnalyzing {video_path.name}...")
        config = calibrator.scan_reference_video(str(video_path))
        
        if config:
            results[video_path.stem] = config
            
            # The fixed print statements
            print(f"Elected Movers: {config.get('primary_joints')}")
            
            # Safely grab the range of the first elected joint for the console output
            first_joint = config.get('primary_joints', [None])[0]
            if first_joint:
                min_ang = config['all_joints'][first_joint]['min_angle']
                max_ang = config['all_joints'][first_joint]['max_angle']
                print(f"   Range ({first_joint}): {min_ang:.1f}° -> {max_ang:.1f}°")
        else:
            print(f"Failed to extract angles from {video_path.name}")

    # Save everything to the cache
    with open(cache_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nPrecomputation complete! Cache saved to {cache_path}")

if __name__ == "__main__":
    main()