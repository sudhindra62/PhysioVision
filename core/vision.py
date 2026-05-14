import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from .entities import Joint # From the entities.py we discussed

class VisionEngine:
    def __init__(self, model_path: str = 'pose_landmarker_heavy.task'):
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
        
        # Calculate dynamic padding
        pad_top = (max_dim - h) // 2
        pad_bottom = max_dim - h - pad_top
        pad_left = (max_dim - w) // 2
        pad_right = max_dim - w - pad_left
        
        # Apply black borders to create a perfect square
        padded_frame = cv2.copyMakeBorder(
            frame, pad_top, pad_bottom, pad_left, pad_right, 
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(padded_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        result = self.detector.detect(mp_image)
        
        if not result.pose_landmarks:
            return None
            
        # Map MediaPipe landmarks to our robust Pydantic entities
        raw_landmarks = result.pose_landmarks[0]
        joints = [
            Joint(x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility) 
            for lm in raw_landmarks
        ]
        
        return joints