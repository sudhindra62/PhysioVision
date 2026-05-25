"""
Configuration & Path Management Module

Centralizes the management of absolute file paths and dynamic asset mapping 
for the entire application. It automatically scans the local directory structure 
to ensure the extraction and calibration engines always have access to the 
latest reference videos without requiring hardcoded lists.
"""
from pathlib import Path

# --- Project Paths ---
# Dynamically resolves the root directory regardless of where the script is executed from
ROOT_DIR = Path(__file__).parent.parent
VIDEOS_DIR = ROOT_DIR / "assets" / "videos"
DATA_DIR = ROOT_DIR / "data"

# Dynamically generate the dictionary expected by the extraction script.
# Scans the folder and automatically creates mappings like: {"vid5.mp4": "vid5"}
EXERCISE_MAP = {f.name: f.stem for f in VIDEOS_DIR.glob("*.mp4")}

def get_available_videos() -> list[str]:
    """
    Retrieves a list of all exercise video names currently available in the assets folder.

    Scans the predefined video directory for all MP4 files and returns 
    their base names (stems) without file extensions. This is particularly 
    useful for dynamically populating Streamlit UI dropdowns.

    Returns:
        list[str]: A list of exercise string names (e.g., ['vid1', 'vid2', 'vid3']).
    """
    return [f.stem for f in VIDEOS_DIR.glob("*.mp4")]