import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from .entities import Joint 

class VisionEngine:
    # Changed from 'heavy' to 'full' to stop FPS drops
    def __init__(self, model_path: str = 'pose_landmarker_full.task'):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.6,
            min_pose_presence_confidence=0.6
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

    def process_frame(self, frame) -> list[Joint] | None:
        """
        Applies dynamic letterboxing to vertical videos, runs inference,
        and returns typed Joint objects.
        """
        h, w = frame.shape[:2]
        max_dim = max(h, w)
        
        pad_top = (max_dim - h) // 2
        pad_bottom = max_dim - h - pad_top
        pad_left = (max_dim - w) // 2
        pad_right = max_dim - w - pad_left
        
        padded_frame = cv2.copyMakeBorder(
            frame, pad_top, pad_bottom, pad_left, pad_right, 
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        
        rgb_frame = cv2.cvtColor(padded_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        result = self.detector.detect(mp_image)
        
        # Grab BOTH the true 3D space and the 2D screen space
        if not result.pose_world_landmarks or not result.pose_landmarks:
            return None
            
        raw_world = result.pose_world_landmarks[0]
        raw_screen = result.pose_landmarks[0]
        
        joints = [
            Joint(
                x=raw_world[i].x, 
                y=raw_world[i].y, 
                z=raw_world[i].z, 
                visibility=raw_world[i].visibility,
                screen_x=raw_screen[i].x,  # Added for UI drawing
                screen_y=raw_screen[i].y   # Added for UI drawing
            ) 
            for i in range(len(raw_world))
        ]
        
        return joints