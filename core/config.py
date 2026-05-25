from pathlib import Path

# --- Project Paths ---
ROOT_DIR = Path(__file__).parent.parent
VIDEOS_DIR = ROOT_DIR/ "assets" / "videos"
DATA_DIR = ROOT_DIR / "data"  # Adding this back in since your script imports it!

# Dynamically generate the dictionary expected by your extraction script
# This scans the folder and automatically creates: {"vid5.mp4": "vid5", ...}
EXERCISE_MAP = {f.name: f.stem for f in VIDEOS_DIR.glob("*.mp4")}

# You can keep your function if other parts of your app use it!
def get_available_videos():
    return [f.stem for f in VIDEOS_DIR.glob("*.mp4")]