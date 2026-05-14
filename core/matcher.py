import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from typing import List, Dict

class DTWJudge:
    def __init__(self):
        # We define the keys we care about to ensure the NumPy arrays align perfectly
        self.joint_keys = [
            "left_elbow", "right_elbow", 
            "left_shoulder", "right_shoulder", 
            "left_knee", "right_knee"
        ]

    def _dict_to_array(self, frame_sequence: List[Dict]) -> np.ndarray:
        """Converts a list of angle dictionaries into a 2D NumPy array."""
        # Extracts angles in a strict order to prevent vector mismatch
        return np.array([
            [frame["angles"].get(k, 0.0) for k in self.joint_keys] 
            for frame in frame_sequence
        ])

    def calculate_similarity(self, lead_sequence: List[Dict], follow_sequence: List[Dict]) -> float:
        """
        Compares the Reference JSON (Lead) against the User's Live Buffer (Follow).
        Returns a score from 0.0 to 100.0.
        """
        if not lead_sequence or not follow_sequence:
            return 0.0
            
        lead_arr = self._dict_to_array(lead_sequence)
        follow_arr = self._dict_to_array(follow_sequence)
        
        # Calculate DTW distance
        distance, path = fastdtw(lead_arr, follow_arr, dist=euclidean)
        
        # Normalize the score
        # The normalization factor (e.g., 500) depends on how strict you want the judge to be.
        # A smaller number means a harsher penalty for slight deviations.
        max_acceptable_distance = len(follow_sequence) * 20.0 
        
        score = 100.0 * (1.0 - (distance / max_acceptable_distance))
        
        return float(np.clip(score, 0.0, 100.0))