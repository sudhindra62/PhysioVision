"""Module for calculating joint angles from pose keypoints."""

from typing import Optional, Tuple, List
import numpy as np
from .pose_estimator import PoseResult, Keypoint


class AngleCalculator:
    """Calculate various body joint angles from pose keypoints."""

    # Define joint triplets: (point_a, vertex, point_c)
    JOINT_DEFINITIONS = {
        # Arms
        'left_elbow': ('left_shoulder', 'left_elbow', 'left_wrist'),
        'right_elbow': ('right_shoulder', 'right_elbow', 'right_wrist'),
        'left_shoulder': ('left_elbow', 'left_shoulder', 'left_hip'),
        'right_shoulder': ('right_elbow', 'right_shoulder', 'right_hip'),
        'left_wrist': ('left_elbow', 'left_wrist', 'left_index'),
        'right_wrist': ('right_elbow', 'right_wrist', 'right_index'),

        # Legs
        'left_hip': ('left_shoulder', 'left_hip', 'left_knee'),
        'right_hip': ('right_shoulder', 'right_hip', 'right_knee'),
        'left_knee': ('left_hip', 'left_knee', 'left_ankle'),
        'right_knee': ('right_hip', 'right_knee', 'right_ankle'),
        'left_ankle': ('left_knee', 'left_ankle', 'left_foot_index'),
        'right_ankle': ('right_knee', 'right_ankle', 'right_foot_index'),
    }

    def __init__(self, use_3d: bool = True):
        """Initialize angle calculator.

        Args:
            use_3d: Use 3D coordinates if available, otherwise use 2D
        """
        self.use_3d = use_3d

    @staticmethod
    def calculate_angle_3points(
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray
    ) -> float:
        """Calculate angle formed by three points (a-b-c, where b is vertex).

        Args:
            a: First point coordinates
            b: Vertex point coordinates
            c: Third point coordinates

        Returns:
            Angle in degrees (0-180)
        """
        # Vectors from vertex to other points
        ba = a - b
        bc = c - b

        # Calculate angle using dot product
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        # Clip to avoid numerical errors
        cosine = np.clip(cosine, -1.0, 1.0)
        angle = np.arccos(cosine)

        return np.degrees(angle)

    def get_keypoint_coords(
        self,
        keypoint: Optional[Keypoint],
        use_world: bool = True
    ) -> Optional[np.ndarray]:
        """Extract coordinates from keypoint.

        Args:
            keypoint: Keypoint object
            use_world: Use world coordinates if available

        Returns:
            Coordinates as numpy array or None if invalid
        """
        if keypoint is None or keypoint.visibility < 0.5:
            return None

        if self.use_3d and use_world:
            world_coords = keypoint.world_coords()
            if world_coords is not None:
                return world_coords

        # Fallback to image coordinates
        if self.use_3d:
            return np.array([keypoint.x, keypoint.y, keypoint.z])
        return np.array([keypoint.x, keypoint.y])

    def calculate_joint_angle(
        self,
        pose_result: PoseResult,
        joint: str,
        use_world: bool = True
    ) -> Optional[float]:
        """Calculate angle for a predefined joint.

        Args:
            pose_result: Pose detection result
            joint: Joint name (e.g., 'left_elbow', 'right_knee')
            use_world: Use world coordinates if available

        Returns:
            Angle in degrees or None if calculation fails
        """
        if joint not in self.JOINT_DEFINITIONS:
            raise ValueError(f"Unknown joint: {joint}. Available: {list(self.JOINT_DEFINITIONS.keys())}")

        point_names = self.JOINT_DEFINITIONS[joint]
        keypoints = pose_result.get_keypoints_by_names(point_names)

        return self.calculate_angle_from_keypoints(keypoints, use_world)

    def calculate_angle_from_keypoints(
        self,
        keypoints: Tuple[Optional[Keypoint], Optional[Keypoint], Optional[Keypoint]],
        use_world: bool = True
    ) -> Optional[float]:
        """Calculate angle from three keypoints.

        Args:
            keypoints: Tuple of (point_a, vertex, point_c)
            use_world: Use world coordinates if available

        Returns:
            Angle in degrees or None if any keypoint is invalid
        """
        coords = [self.get_keypoint_coords(kp, use_world) for kp in keypoints]

        if any(c is None for c in coords):
            return None

        return self.calculate_angle_3points(coords[0], coords[1], coords[2])

    def calculate_custom_angle(
        self,
        pose_result: PoseResult,
        point_a: str,
        vertex: str,
        point_c: str,
        use_world: bool = True
    ) -> Optional[float]:
        """Calculate angle from custom keypoint names.

        Args:
            pose_result: Pose detection result
            point_a: Name of first point
            vertex: Name of vertex point
            point_c: Name of third point
            use_world: Use world coordinates if available

        Returns:
            Angle in degrees or None if calculation fails
        """
        keypoints = pose_result.get_keypoints_by_names([point_a, vertex, point_c])
        return self.calculate_angle_from_keypoints(tuple(keypoints), use_world)

    def calculate_all_angles(
        self,
        pose_result: PoseResult,
        use_world: bool = True
    ) -> dict:
        """Calculate all predefined joint angles.

        Args:
            pose_result: Pose detection result
            use_world: Use world coordinates if available

        Returns:
            Dictionary mapping joint names to angles (None for failed calculations)
        """
        angles = {}
        for joint_name in self.JOINT_DEFINITIONS.keys():
            try:
                angles[joint_name] = self.calculate_joint_angle(
                    pose_result, joint_name, use_world
                )
            except Exception:
                angles[joint_name] = None

        return angles

    @staticmethod
    def get_midpoint(kp1: Keypoint, kp2: Keypoint) -> Optional[np.ndarray]:
        """Calculate midpoint between two keypoints.

        Args:
            kp1: First keypoint
            kp2: Second keypoint

        Returns:
            Midpoint coordinates or None
        """
        if not kp1 or not kp2:
            return None

        if kp1.world_coords() is not None and kp2.world_coords() is not None:
            return (kp1.world_coords() + kp2.world_coords()) / 2

        return np.array([(kp1.x + kp2.x) / 2, (kp1.y + kp2.y) / 2, (kp1.z + kp2.z) / 2])

    def calculate_head_tilt(self, pose_result: PoseResult) -> Optional[float]:
        """Calculate head tilt angle (side-to-side).

        Args:
            pose_result: Pose detection result

        Returns:
            Head tilt angle in degrees (0 = straight, positive = right tilt, negative = left tilt)
        """
        left_ear = pose_result.get_keypoint('left_ear')
        right_ear = pose_result.get_keypoint('right_ear')

        if not left_ear or not right_ear:
            # Fallback to eyes
            left_ear = pose_result.get_keypoint('left_eye')
            right_ear = pose_result.get_keypoint('right_eye')

        if left_ear and right_ear:
            dx = right_ear.x - left_ear.x
            dy = right_ear.y - left_ear.y
            angle = np.degrees(np.arctan2(dy, dx))
            # Normalize to -180 to 180, where 0 is straight
            return angle if abs(angle) < 90 else (angle - 180 if angle > 0 else angle + 180)

        return None

    def calculate_neck_angle(self, pose_result: PoseResult, use_world: bool = True) -> Optional[float]:
        """Calculate neck forward/backward angle.

        Args:
            pose_result: Pose detection result
            use_world: Use world coordinates if available

        Returns:
            Neck angle in degrees
        """
        nose = pose_result.get_keypoint('nose')
        left_shoulder = pose_result.get_keypoint('left_shoulder')
        right_shoulder = pose_result.get_keypoint('right_shoulder')
        left_ear = pose_result.get_keypoint('left_ear')
        right_ear = pose_result.get_keypoint('right_ear')

        if not all([left_shoulder, right_shoulder]):
            return None

        # Calculate shoulder midpoint
        shoulder_mid = self.get_midpoint(left_shoulder, right_shoulder)
        if shoulder_mid is None:
            return None

        # Use ear midpoint or nose
        if left_ear and right_ear:
            head_point = self.get_midpoint(left_ear, right_ear)
        elif nose:
            head_coords = self.get_keypoint_coords(nose, use_world)
            head_point = head_coords
        else:
            return None

        if head_point is None:
            return None

        # Calculate angle from vertical
        dx = head_point[0] - shoulder_mid[0]
        dy = head_point[1] - shoulder_mid[1]
        angle = np.degrees(np.arctan2(abs(dx), abs(dy)))

        return angle

    def calculate_body_lean(self, pose_result: PoseResult, use_world: bool = True) -> Optional[float]:
        """Calculate body forward/backward lean angle.

        Args:
            pose_result: Pose detection result
            use_world: Use world coordinates if available

        Returns:
            Body lean angle in degrees (0 = straight, positive = leaning forward)
        """
        left_shoulder = pose_result.get_keypoint('left_shoulder')
        right_shoulder = pose_result.get_keypoint('right_shoulder')
        left_hip = pose_result.get_keypoint('left_hip')
        right_hip = pose_result.get_keypoint('right_hip')

        if not all([left_shoulder, right_shoulder, left_hip, right_hip]):
            return None

        # Calculate midpoints
        shoulder_mid = self.get_midpoint(left_shoulder, right_shoulder)
        hip_mid = self.get_midpoint(left_hip, right_hip)

        if shoulder_mid is None or hip_mid is None:
            return None

        # Calculate angle from vertical
        dx = shoulder_mid[0] - hip_mid[0]
        dy = shoulder_mid[1] - hip_mid[1]

        # Angle from vertical (0 = straight, positive = forward lean)
        angle = np.degrees(np.arctan2(dx, dy))

        return angle

    def calculate_shoulder_tilt(self, pose_result: PoseResult) -> Optional[float]:
        """Calculate shoulder tilt (one shoulder higher than the other).

        Args:
            pose_result: Pose detection result

        Returns:
            Shoulder tilt angle in degrees (0 = level, positive = right higher, negative = left higher)
        """
        left_shoulder = pose_result.get_keypoint('left_shoulder')
        right_shoulder = pose_result.get_keypoint('right_shoulder')

        if left_shoulder and right_shoulder:
            dx = right_shoulder.x - left_shoulder.x
            dy = right_shoulder.y - left_shoulder.y
            angle = np.degrees(np.arctan2(dy, dx))
            return angle

        return None

    def calculate_hip_tilt(self, pose_result: PoseResult) -> Optional[float]:
        """Calculate hip tilt (one hip higher than the other).

        Args:
            pose_result: Pose detection result

        Returns:
            Hip tilt angle in degrees (0 = level)
        """
        left_hip = pose_result.get_keypoint('left_hip')
        right_hip = pose_result.get_keypoint('right_hip')

        if left_hip and right_hip:
            dx = right_hip.x - left_hip.x
            dy = right_hip.y - left_hip.y
            angle = np.degrees(np.arctan2(dy, dx))
            return angle

        return None

    def calculate_spine_curve(self, pose_result: PoseResult, use_world: bool = True) -> Optional[float]:
        """Calculate spine curvature (upper spine to lower spine angle).

        Args:
            pose_result: Pose detection result
            use_world: Use world coordinates if available

        Returns:
            Spine curve angle in degrees
        """
        left_shoulder = pose_result.get_keypoint('left_shoulder')
        right_shoulder = pose_result.get_keypoint('right_shoulder')
        left_hip = pose_result.get_keypoint('left_hip')
        right_hip = pose_result.get_keypoint('right_hip')

        if not all([left_shoulder, right_shoulder, left_hip, right_hip]):
            return None

        shoulder_mid = self.get_midpoint(left_shoulder, right_shoulder)
        hip_mid = self.get_midpoint(left_hip, right_hip)

        if shoulder_mid is None or hip_mid is None:
            return None

        # Calculate torso angle
        # Simplified: use shoulder-hip distance vs height
        dx = abs(shoulder_mid[0] - hip_mid[0])
        dy = abs(shoulder_mid[1] - hip_mid[1])

        if dy < 0.01:  # Avoid division by zero
            return None

        angle = np.degrees(np.arctan2(dx, dy))
        return angle

    def calculate_posture_metrics(
        self,
        pose_result: PoseResult,
        use_world: bool = True
    ) -> dict:
        """Calculate comprehensive posture metrics.

        Args:
            pose_result: Pose detection result
            use_world: Use world coordinates if available

        Returns:
            Dictionary with posture metrics
        """
        metrics = {}

        # Head and neck
        metrics['head_tilt'] = self.calculate_head_tilt(pose_result)
        metrics['neck_angle'] = self.calculate_neck_angle(pose_result, use_world)

        # Body alignment
        metrics['body_lean'] = self.calculate_body_lean(pose_result, use_world)
        metrics['shoulder_tilt'] = self.calculate_shoulder_tilt(pose_result)
        metrics['hip_tilt'] = self.calculate_hip_tilt(pose_result)
        metrics['spine_curve'] = self.calculate_spine_curve(pose_result, use_world)

        return metrics
