from pydantic import BaseModel
from typing import List, Dict, Optional

class Joint(BaseModel):
    x: float
    y: float
    z: float
    visibility: float

class PoseFrame(BaseModel):
    exercise_name: str
    frame_index: int
    angles: Dict[str, float]