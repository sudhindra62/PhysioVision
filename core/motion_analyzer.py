"""Motion analysis and pattern recognition."""

from typing import List, Optional, Dict, Any
from collections import deque
import numpy as np
from .pose_estimator import PoseResult
from .angle_calculator import AngleCalculator


class MotionAnalyzer:
    """Analyze motion patterns and provide feedback."""

    def __init__(
        self,
        buffer_size: int = 30,
        smoothing_window: int = 5
    ):
        """Initialize motion analyzer.

        Args:
            buffer_size: Number of frames to keep in history
            smoothing_window: Window size for temporal smoothing
        """
        self.buffer_size = buffer_size
        self.smoothing_window = smoothing_window
        self.pose_history: deque = deque(maxlen=buffer_size)
        self.angle_history: Dict[str, deque] = {}
        self.calculator = AngleCalculator()

    def update(self, pose_result: PoseResult):
        """Update motion history with new pose.

        Args:
            pose_result: New pose detection result
        """
        self.pose_history.append(pose_result)

        # Calculate and store angles
        angles = self.calculator.calculate_all_angles(pose_result)
        for joint, angle in angles.items():
            if angle is not None:
                if joint not in self.angle_history:
                    self.angle_history[joint] = deque(maxlen=self.buffer_size)
                self.angle_history[joint].append(angle)

    def get_smoothed_angle(
        self,
        joint: str,
        method: str = 'moving_average'
    ) -> Optional[float]:
        """Get temporally smoothed angle for a joint.

        Args:
            joint: Joint name
            method: Smoothing method ('moving_average', 'exponential')

        Returns:
            Smoothed angle or None
        """
        if joint not in self.angle_history or len(self.angle_history[joint]) == 0:
            return None

        angles = list(self.angle_history[joint])

        if method == 'moving_average':
            window = min(self.smoothing_window, len(angles))
            return float(np.mean(angles[-window:]))

        elif method == 'exponential':
            # Exponential moving average
            alpha = 2.0 / (self.smoothing_window + 1)
            ema = angles[0]
            for angle in angles[1:]:
                ema = alpha * angle + (1 - alpha) * ema
            return float(ema)

        return angles[-1]

    def detect_rep_count(
        self,
        joint: str,
        threshold_low: float,
        threshold_high: float,
        min_frames: int = 10
    ) -> int:
        """Count repetitions based on joint angle thresholds.

        Args:
            joint: Joint to monitor
            threshold_low: Lower angle threshold
            threshold_high: Upper angle threshold
            min_frames: Minimum frames between reps

        Returns:
            Number of repetitions detected
        """
        if joint not in self.angle_history:
            return 0

        angles = list(self.angle_history[joint])
        if len(angles) < min_frames * 2:
            return 0

        reps = 0
        state = 'idle'
        frames_in_state = 0

        for angle in angles:
            if state == 'idle':
                if angle <= threshold_low:
                    state = 'low'
                    frames_in_state = 1

            elif state == 'low':
                if angle >= threshold_high:
                    if frames_in_state >= min_frames:
                        state = 'high'
                        frames_in_state = 1
                    else:
                        state = 'idle'
                        frames_in_state = 0
                else:
                    frames_in_state += 1

            elif state == 'high':
                if angle <= threshold_low:
                    if frames_in_state >= min_frames:
                        reps += 1
                        state = 'low'
                        frames_in_state = 1
                    else:
                        state = 'idle'
                        frames_in_state = 0
                else:
                    frames_in_state += 1

        return reps

    def get_angle_statistics(self, joint: str) -> Dict[str, float]:
        """Get statistical metrics for a joint angle.

        Args:
            joint: Joint name

        Returns:
            Dictionary with min, max, mean, std
        """
        if joint not in self.angle_history or len(self.angle_history[joint]) == 0:
            return {}

        angles = np.array(list(self.angle_history[joint]))

        return {
            'min': float(np.min(angles)),
            'max': float(np.max(angles)),
            'mean': float(np.mean(angles)),
            'std': float(np.std(angles)),
            'current': float(angles[-1]),
        }

    def check_posture(
        self,
        pose_result: PoseResult,
        rules: Dict[str, Dict[str, Any]]
    ) -> Dict[str, bool]:
        """Check if posture meets specified rules.

        Args:
            pose_result: Current pose
            rules: Dictionary of rules, e.g.:
                {
                    'back_straight': {
                        'joint': 'spine_upper',
                        'min': 160,
                        'max': 180
                    }
                }

        Returns:
            Dictionary mapping rule names to pass/fail
        """
        results = {}

        for rule_name, rule_spec in rules.items():
            joint = rule_spec.get('joint')
            min_angle = rule_spec.get('min', 0)
            max_angle = rule_spec.get('max', 180)

            angle = self.calculator.calculate_joint_angle(pose_result, joint)

            if angle is None:
                results[rule_name] = False
            else:
                results[rule_name] = min_angle <= angle <= max_angle

        return results

    def clear_history(self):
        """Clear all motion history."""
        self.pose_history.clear()
        self.angle_history.clear()
