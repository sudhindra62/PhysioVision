import numpy as np
from typing import List, Dict
from .entities import Joint  # <-- Added missing import

class PhysioGeometry:
    @staticmethod
    def get_angle(a: Joint, b: Joint, c: Joint) -> float:
        """
        Calculates the angle at vertex B using 3D vector dot products.
        Formula: theta = arccos( (ba . bc) / (|ba| * |bc|) )
        """
        # Create 3D vectors relative to joint B
        ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
        bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])

        # Calculate cosine using dot product / magnitudes
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        
        # arccos is sensitive; clip to [-1, 1] to prevent NaN
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return float(np.degrees(angle))

    def get_standard_angles(self, joints: List[Joint]) -> Dict[str, float]:
        """Maps specific MediaPipe indices to physiotherapeutic joint names."""
        # Mapping: Shoulder(11), Elbow(13), Wrist(15), Hip(23), Knee(25), Ankle(27)
        return {
            "left_elbow": self.get_angle(joints[11], joints[13], joints[15]),
            "right_elbow": self.get_angle(joints[12], joints[14], joints[16]),
            "left_shoulder": self.get_angle(joints[23], joints[11], joints[13]),
            "right_shoulder": self.get_angle(joints[24], joints[12], joints[14]),
            "left_knee": self.get_angle(joints[23], joints[25], joints[27]),
            "right_knee": self.get_angle(joints[24], joints[26], joints[28])
        }