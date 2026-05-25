"""
Vision Engine Module

Handles deep learning inference pipelines using the Google MediaPipe infrastructure.
Isolates neural network processing from downstream mathematical layers, applying 
dynamic letterboxing optimizations to standardize video aspect ratios prior to 
landmark parsing.
"""
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from .entities import Joint 

class VisionEngine:
    """
    An encapsulated interface for the MediaPipe PoseLandmarker neural network.
    
    Responsible for model lifecycle management, confidence configurations, 
    aspect-ratio adjustments, and splitting raw model inference into clean, 
    synchronized 3D world space coordinates and 2D screen coordinate payloads.
    """

    def __init__(self, model_path: str = 'pose_landmarker_full.task'):
        """
        Initializes the MediaPipe landmarker task utilizing a CPU-optimized runtime config.

        Args:
            model_path (str): File path pointing to the pre-compiled task graph model bundle.
                              Defaults to 'pose_landmarker_full.task' for optimized real-time FPS.
        """
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
        Processes an individual image frame to extract verified skeletal joint positions.

        Applies an aspect-aware padding matrix (letterboxing) to square the image layout, 
        guaranteeing that narrow or highly vertical video aspect ratios do not warp internal 
        spatial proportions. It explicitly matches true metrics-based metric coordinates 
        (`pose_world_landmarks`) with canvas pixel normalized values (`pose_landmarks`).

        Args:
            frame (np.ndarray): The raw multi-dimensional image array (BGR format) from the source feed.

        Returns:
            list[Joint] | None: A collection of 33 Pydantic-validated Joint structures 
                                mapping spatial nodes. Returns None if inference confidence boundaries 
                                are breached or no humans are present.
        """
        h, w = frame.shape[:2]
        max_dim = max(h, w)
        
        # Calculate padding parameters dynamically to preserve exact physical geometry proportions
        pad_top = (max_dim - h) // 2
        pad_bottom = max_dim - h - pad_top
        pad_left = (max_dim - w) // 2
        pad_right = max_dim - w - pad_left
        
        padded_frame = cv2.copyMakeBorder(
            frame, pad_top, pad_bottom, pad_left, pad_right, 
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        
        # Image matrix conversions required for MediaPipe's underlying SRGB tensor expectations
        rgb_frame = cv2.cvtColor(padded_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        result = self.detector.detect(mp_image)
        
        # Verify both data streams exist before extracting values to prevent indexing errors
        if not result.pose_world_landmarks or not result.pose_landmarks:
            return None
            
        raw_world = result.pose_world_landmarks[0]
        raw_screen = result.pose_landmarks[0]
        
        # Synchronize and align physical world metrics against UI rendering planes
        joints = [
            Joint(
                x=raw_world[i].x, 
                y=raw_world[i].y, 
                z=raw_world[i].z, 
                visibility=raw_world[i].visibility,
                screen_x=raw_screen[i].x,  # Leveraged directly for skeleton drawing vectors
                screen_y=raw_screen[i].y   # Leveraged directly for skeleton drawing vectors
            ) 
            for i in range(len(raw_world))
        ]
        
        return joints