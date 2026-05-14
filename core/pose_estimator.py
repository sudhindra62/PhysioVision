"""Abstract base class for pose estimation backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class Keypoint:
    """Represents a single keypoint detection.

    Attributes:
        name: Name of the keypoint (e.g., 'left_elbow')
        x: X coordinate (normalized 0-1 for image coordinates)
        y: Y coordinate (normalized 0-1 for image coordinates)
        z: Z coordinate (depth, unit depends on backend)
        visibility: Visibility score (0-1)
        presence: Presence score (0-1)
        world_x: World coordinate X in meters (optional)
        world_y: World coordinate Y in meters (optional)
        world_z: World coordinate Z in meters (optional)
    """
    name: str
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0
    presence: float = 1.0
    world_x: Optional[float] = None
    world_y: Optional[float] = None
    world_z: Optional[float] = None

    def to_image_coords(self, width: int, height: int) -> Tuple[int, int]:
        """Convert normalized coordinates to image pixel coordinates."""
        return (int(self.x * width), int(self.y * height))

    def world_coords(self) -> Optional[np.ndarray]:
        """Get world coordinates as numpy array."""
        if all(v is not None for v in [self.world_x, self.world_y, self.world_z]):
            return np.array([self.world_x, self.world_y, self.world_z])
        return None


@dataclass
class PoseResult:
    """Results from pose estimation.

    Attributes:
        keypoints: List of detected keypoints
        timestamp: Timestamp of the frame (milliseconds)
        confidence: Overall detection confidence
        image_width: Original image width
        image_height: Original image height
    """
    keypoints: List[Keypoint]
    timestamp: float = 0.0
    confidence: float = 1.0
    image_width: int = 0
    image_height: int = 0

    def get_keypoint(self, name: str) -> Optional[Keypoint]:
        """Get keypoint by name."""
        for kp in self.keypoints:
            if kp.name == name:
                return kp
        return None

    def get_keypoints_by_names(self, names: List[str]) -> List[Optional[Keypoint]]:
        """Get multiple keypoints by names."""
        return [self.get_keypoint(name) for name in names]

    def is_valid(self, min_confidence: float = 0.5) -> bool:
        """Check if pose detection is valid."""
        return self.confidence >= min_confidence and len(self.keypoints) > 0


class PoseEstimator(ABC):
    """Abstract base class for pose estimation backends.

    All pose estimation implementations should inherit from this class
    and implement the required methods.
    """

    def __init__(self, **kwargs):
        """Initialize pose estimator.

        Args:
            **kwargs: Backend-specific configuration
        """
        self.config = kwargs
        self.is_initialized = False

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the pose estimator.

        Returns:
            True if initialization successful, False otherwise
        """
        pass

    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> Optional[PoseResult]:
        """Process a single frame and detect pose.

        Args:
            frame: Input image as numpy array (BGR format)

        Returns:
            PoseResult object if detection successful, None otherwise
        """
        pass

    @abstractmethod
    def release(self):
        """Release resources and cleanup."""
        pass

    @abstractmethod
    def get_keypoint_names(self) -> List[str]:
        """Get list of all keypoint names supported by this backend.

        Returns:
            List of keypoint names
        """
        pass

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Get name of the backend.

        Returns:
            Backend name string
        """
        pass

    @property
    def num_keypoints(self) -> int:
        """Get number of keypoints detected by this backend.

        Returns:
            Number of keypoints
        """
        return len(self.get_keypoint_names())

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
