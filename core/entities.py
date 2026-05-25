from pydantic import BaseModel
from typing import List, Dict, Optional

class Joint(BaseModel):
    # True 3D world coordinates for exact geometry math
    x: float
    y: float
    z: float
    visibility: float
    
    # Normalized 2D screen coordinates for drawing on the video feed
    screen_x: float = 0.0
    screen_y: float = 0.0

class PoseFrame(BaseModel):
    exercise_name: str
    frame_index: int
    angles: Dict[str, float]