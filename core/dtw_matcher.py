import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from collections import deque
from typing import List, Dict

class DTWMatcher:
    def __init__(self, window_size: int = 45):
        self.window_size = window_size
        self.client_history = deque(maxlen=window_size)
        self.therapist_history = deque(maxlen=window_size)
        
    def add_frames(self, client_angles: Dict[str, float], therapist_angles: Dict[str, float]):
        """Buffers the current frame's angles[cite: 96, 97]."""
        self.client_history.append(client_angles)
        self.therapist_history.append(therapist_angles)

    def calculate_similarity_score(self, all_joints: List[str]) -> float:
        if len(self.client_history) < self.window_size // 2:
            return 100.0 

        # 1. DYNAMIC SELECTION: Only score joints that are visible AND moving
        active_joints = []
        for joint in all_joints:
            # Check if therapist has data for this joint (not None/0)
            t_values = [f.get(joint) for f in self.therapist_history if f.get(joint) is not None]
            c_values = [f.get(joint) for f in self.client_history if f.get(joint) is not None]
            
            # Criteria: Mutually visible in at least 70% of the buffer
            if len(t_values) > self.window_size * 0.7 and len(c_values) > self.window_size * 0.7:
                # Check for "Motion Intent": Is the therapist actually moving this joint?
                # If the angle variance is very low, the therapist is likely keeping it still.
                if np.var(t_values) > 5.0: # 5 degrees variance threshold
                    active_joints.append(joint)

        # Fallback: If no joint is "moving", use all mutually visible joints
        if not active_joints:
            active_joints = [j for j in all_joints if self.therapist_history[-1].get(j) and self.client_history[-1].get(j)]

        if not active_joints:
            return 100.0

        client_seq = []
        therapist_seq = []

        for c_frame, t_frame in zip(self.client_history, self.therapist_history):
            # Fill missing points with the last known angle or 0 to keep vector length consistent
            c_vec = [c_frame.get(j, 0.0) or 0.0 for j in active_joints]
            t_vec = [t_frame.get(j, 0.0) or 0.0 for j in active_joints]
            client_seq.append(c_vec)
            therapist_seq.append(t_vec)

        try:
            distance, path = fastdtw(client_seq, therapist_seq, dist=euclidean)
            # Normalize by path length and number of active joints [cite: 101, 102]
            avg_deviation = distance / (len(path) * len(active_joints))
            
            # Scoring: 0 deviation = 100%. 30 deg avg deviation = 0%[cite: 103].
            score = max(0.0, 100.0 - (avg_deviation * 3.33)) 
            return round(score, 1)
        except Exception:
            return 0.0

    def clear(self):
        self.client_history.clear()
        self.therapist_history.clear()