"""
Range of Motion (ROM) Calibration Cache Script

Provides a lightweight precomputation workflow utility. It loops through raw
exercise video files, routes them through the AI calibration engine to determine 
active joints and anatomical angle limits, and exports a simplified configuration 
manifest (`rom_cache.json`) for quick diagnostic evaluations.
"""
import sys
import json
from pathlib import Path

# Setup system path constraints to allow runtime discovery of our internal 'core' modules
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from core.calibrator import AutoCalibrator
from core.config import VIDEOS_DIR

def main():
    """
    Executes a baseline video scan to compile isolated joint boundaries.
    
    Initializes the calibration framework, processes raw video assets 
    sequentially, extracts individual maximum and minimum angle constraints, 
    and writes the structural summary data into a lightweight JSON cache file.
    """
    print("Initializing AI Calibrator...")
    calibrator = AutoCalibrator()
    
    # Securely setup our data persistence directory tracking
    data_dir = ROOT_DIR / "data"
    data_dir.mkdir(exist_ok=True)
    cache_path = data_dir / "rom_cache.json"
    
    results = {}
    
    # Iterate and scan every video asset located within our assets directory
    for video_path in VIDEOS_DIR.glob("*.mp4"):
        print(f"\nAnalyzing {video_path.name}...")
        config = calibrator.scan_reference_video(str(video_path))
        
        if config:
            results[video_path.stem] = config
            
            # Print execution summaries to the system console for debugging
            print(f"Elected Movers: {config.get('primary_joints')}")
            
            # Safely isolate the target joint to fetch and display operational ranges
            first_joint = config.get('primary_joints', [None])[0]
            if first_joint:
                min_ang = config['all_joints'][first_joint]['min_angle']
                max_ang = config['all_joints'][first_joint]['max_angle']
                print(f"   Range ({first_joint}): {min_ang:.1f}° -> {max_ang:.1f}°")
        else:
            print(f"Failed to extract angles from {video_path.name}")

    # Serialize the summary metrics to a standalone operational cache file
    with open(cache_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nPrecomputation complete! Cache saved to {cache_path}")

if __name__ == "__main__":
    main()