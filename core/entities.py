"""
Entities & Data Models Module

Defines the strict data structures used throughout the application. 
Utilizes Pydantic for runtime type checking and data validation to ensure 
that the neural network, math engine, and database communicate flawlessly.
"""
from pydantic import BaseModel
from typing import Dict

class Joint(BaseModel):
    """
    Represents a single anatomical landmark in space at a specific moment in time.
    
    This model deliberately separates True 3D World Coordinates (used for 
    accurate biomechanical math) from 2D Normalized Screen Coordinates 
    (used purely for UI rendering).
    
    Attributes:
        x (float): The 3D world X-coordinate (horizontal) in meters.
        y (float): The 3D world Y-coordinate (vertical) in meters.
        z (float): The 3D world Z-coordinate (depth) in meters.
        visibility (float): A confidence score (0.0 to 1.0) indicating how likely 
                            the joint is visible and not occluded.
        screen_x (float): The 2D normalized X-coordinate (0.0 to 1.0) for the UI overlay.
        screen_y (float): The 2D normalized Y-coordinate (0.0 to 1.0) for the UI overlay.
    """
    # True 3D world coordinates for exact geometry math
    x: float
    y: float
    z: float
    visibility: float
    
    # Normalized 2D screen coordinates for drawing on the video feed
    screen_x: float = 0.0
    screen_y: float = 0.0

class PoseFrame(BaseModel):
    """
    Represents a complete snapshot of a user's biomechanical state for a single frame.
    
    This model is primarily used for serialization, such as when extracting 
    ground-truth angles from a reference video to build the JSON master registry.
    
    Attributes:
        exercise_name (str): The identifier for the exercise being performed (e.g., 'bicep_curl').
        frame_index (int): The chronological index of this frame in the video sequence.
        angles (Dict[str, float]): A dictionary mapping anatomical names to their 
                                   calculated angles at this exact frame.
    """
    exercise_name: str
    frame_index: int
    angles: Dict[str, float]