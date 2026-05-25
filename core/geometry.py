import numpy as np
from typing import List, Dict
from .entities import Joint

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
        # Prevent division by zero
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        if norm_ba == 0 or norm_bc == 0:
            return 0.0

        cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        
        # arccos is sensitive; clip to [-1, 1] to prevent NaN
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return float(np.degrees(angle))

    def get_standard_angles(self, joints: List[Joint]) -> Dict[str, float]:
        """Maps specific MediaPipe indices to physiotherapeutic joint names and transverse planes."""
        angles = {
            "left_elbow": self.get_angle(joints[11], joints[13], joints[15]),
            "right_elbow": self.get_angle(joints[12], joints[14], joints[16]),
            "left_shoulder": self.get_angle(joints[23], joints[11], joints[13]),
            "right_shoulder": self.get_angle(joints[24], joints[12], joints[14]),
            "left_knee": self.get_angle(joints[23], joints[25], joints[27]),
            "right_knee": self.get_angle(joints[24], joints[26], joints[28])
        }

        # --- NEW: Virtual Joints for Plane Validation ---
        mid_shoulder = Joint(
            x=(joints[11].x + joints[12].x) / 2.0,
            y=(joints[11].y + joints[12].y) / 2.0,
            z=(joints[11].z + joints[12].z) / 2.0,
            visibility=min(joints[11].visibility, joints[12].visibility)
        )
        
        mid_hip = Joint(
            x=(joints[23].x + joints[24].x) / 2.0,
            y=(joints[23].y + joints[24].y) / 2.0,
            z=(joints[23].z + joints[24].z) / 2.0,
            visibility=min(joints[23].visibility, joints[24].visibility)
        )

        # --- NEW: Transverse Plane Extraction ---
        angles["left_shoulder_plane"] = self.get_angle(mid_shoulder, joints[11], joints[13])
        angles["right_shoulder_plane"] = self.get_angle(mid_shoulder, joints[12], joints[14])
        angles["left_hip_plane"] = self.get_angle(mid_hip, joints[23], joints[25])
        angles["right_hip_plane"] = self.get_angle(mid_hip, joints[24], joints[26])

        return angles