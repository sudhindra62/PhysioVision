import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from typing import List, Dict

class DTWJudge:
    def __init__(self, active_joints: List[str]):
        # We strictly define the joint order to ensure vector dimensions align
        self.joint_keys = active_joints

    def _dict_to_array(self, frame_sequence: List[Dict]) -> np.ndarray:
        """Converts sequence of angle dictionaries into a robust 2D NumPy array."""
        return np.array([
            [frame.get(k, 0.0) for k in self.joint_keys] 
            for frame in frame_sequence
        ])

    def calculate_similarity(self, golden_sequence: List[Dict], user_sequence: List[Dict]) -> float:
        """
        Calculates time-invariant trajectory similarity. 
        Speed does not matter, only the path taken through space.
        """
        if len(golden_sequence) < 5 or len(user_sequence) < 5:
            return 0.0
            
        golden_arr = self._dict_to_array(golden_sequence)
        user_arr = self._dict_to_array(user_sequence)
        
        # O(N*M) Dynamic Time Warping alignment
        distance, path = fastdtw(golden_arr, user_arr, dist=euclidean)
        
        # Normalize distance by the path length to get average deviation per alignment frame
        avg_distance = distance / len(path)
        
        # Convert deviation to a 0-100 score. 
        # Tuning: A deviation of ~45 total angular degrees across all joints yields a score of 0.
        score = 100.0 - (avg_distance * 2.2) 
        
        return float(np.clip(score, 0.0, 100.0))
