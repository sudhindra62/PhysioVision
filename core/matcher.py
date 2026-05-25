"""
Dynamic Time Warping (DTW) Judge Module

Advanced sequence-matching engine. Uses multi-dimensional Dynamic Time Warping 
to evaluate the spatial accuracy of a user's movement trajectory independent of 
their execution speed.
"""
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from typing import List, Dict

class DTWJudge:
    """
    Evaluates biomechanical time-series data using spatial alignment.
    
    Unlike standard frame-by-frame comparison, DTW warps the time axis to find 
    the optimal geometric match between two sequences. This ensures users are graded 
    on their physical path through space (form), not their speed.
    """

    def __init__(self, active_joints: List[str]):
        """
        Initializes the DTW Judge with a strict schema of joint names.

        Args:
            active_joints (List[str]): A strictly ordered list of joint names 
                                       (e.g., ['left_elbow', 'right_elbow']). 
                                       This guarantees that the N-dimensional 
                                       NumPy arrays are built consistently.
        """
        # We strictly define the joint order to ensure vector dimensions align
        self.joint_keys = active_joints

    def _dict_to_array(self, frame_sequence: List[Dict]) -> np.ndarray:
        """
        Converts a sequence of angle dictionaries into a robust 2D NumPy array.

        Args:
            frame_sequence (List[Dict]): A chronological list where each element is a 
                                         dictionary of joint angles for that frame.

        Returns:
            np.ndarray: A 2D array of shape (num_frames, num_active_joints). 
                        Missing joints safely default to 0.0 to prevent matrix shape errors.
        """
        return np.array([
            [frame.get(k, 0.0) for k in self.joint_keys] 
            for frame in frame_sequence
        ])

    def calculate_similarity(self, golden_sequence: List[Dict], user_sequence: List[Dict]) -> float:
        """
        Calculates the time-invariant trajectory similarity between a user's 
        repetition and the master reference repetition.

        Uses the fastdtw approximation with an Euclidean distance metric to find 
        the optimal non-linear alignment path. The raw spatial distance is then 
        normalized by path length and inverted into a 0-100 percentage score.

        Args:
            golden_sequence (List[Dict]): The target geometric trajectory.
            user_sequence (List[Dict]): The trajectory recorded from the user.

        Returns:
            float: A form accuracy score from 0.0 (total failure) to 100.0 (perfect match).
                   Returns 0.0 if either sequence is too short to establish a path.
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