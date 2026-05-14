from pydantic import BaseModel
from typing import Dict
from pathlib import Path

class VideoMeta(BaseModel):
    w: int
    h: int
    fps: int

# Project Paths
ROOT_DIR = Path(__file__).parent.parent
VIDEOS_DIR = ROOT_DIR / "videos"
DATA_DIR = ROOT_DIR / "data"

# Ensure output directory exists
DATA_DIR.mkdir(exist_ok=True)

# Your exact video configurations
VIDEO_CONFIG: Dict[str, VideoMeta] = {
    "vid1": VideoMeta(w=2160, h=3840, fps=24), # 4K Vertical
    "vid2": VideoMeta(w=1080, h=1920, fps=25), # HD Vertical
    "vid3": VideoMeta(w=1080, h=1920, fps=25), # HD Vertical
    "vid4": VideoMeta(w=1080, h=1920, fps=20), # HD Vertical (Slow)
}

# Exercise mapping (Change these to match what vid1, vid2 actually are)
EXERCISE_MAP = {
    "vid1.mp4": "jumping_jacks",
    "vid2.mp4": "squats",
    "vid3.mp4": "lunges",
    "vid4.mp4": "bicep_curls"
}