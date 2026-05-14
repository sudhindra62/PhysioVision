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
        if not lead_sequence or not follow_sequence:
            return 0.0
            
        lead_arr = self._dict_to_array(lead_sequence)
        follow_arr = self._dict_to_array(follow_sequence)
        
        # Calculate DTW distance and the optimal path
        distance, path = fastdtw(lead_arr, follow_arr, dist=euclidean)
        
        # NEW LOGIC: Average distance per mapped frame
        avg_distance = distance / len(path)
        
        # If joints are perfectly aligned, avg_distance is 0. 
        # An avg_distance of ~60 means joints are off by an average of 10-15 degrees each.
        # We set 80 as our "Zero Score" threshold.
        score = 100.0 - (avg_distance * 1.25) 
        
        return float(np.clip(score, 0.0, 100.0))