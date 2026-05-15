from pathlib import Path

# --- Project Paths ---
ROOT_DIR = Path(__file__).parent.parent
VIDEOS_DIR = ROOT_DIR / "videos"

# The system will automatically detect available videos in the directory
def get_available_videos():
    return [f.stem for f in VIDEOS_DIR.glob("*.mp4")]