import time
from typing import Dict, List

class ROMJudge:
    def __init__(self, config: dict):
        self.config = config
        self.state = "RESTING"
        self.reps = 0
        
        self.last_rep_score = 0.0  
        self.total_score_accumulator = 0.0 
        self.average_score = 0.0 
        
        self.smoothed_angles = {}
        # Goldilocks zone for WebRTC EMA Smoothing
        self.alpha = 0.35         

        self.primary_joint = self.config["primary_joint"]
        self.active_joints = list(self.config["all_joints"].keys())
        self.golden_trajectory = self.config.get("golden_trajectory", [])

        self.rom_range = abs(self.config["all_joints"][self.primary_joint]["max_angle"] - self.config["all_joints"][self.primary_joint]["min_angle"])
        if self.rom_range == 0: self.rom_range = 1 
        
        self.max_achieved_pct = 0.0
        self.min_achieved_pct = 1.0
        self.phase_errors = []

    def get_completion_percentage(self, current_angle: float) -> float:
        moved = abs(current_angle - self.config["all_joints"][self.primary_joint]["min_angle"])
        pct = moved / self.rom_range
        return max(0.0, min(1.0, pct))

    def _calculate_frame_accuracy(self, user_angles: dict, current_pct: float) -> float:
        if not self.golden_trajectory: return 100.0

        target_angle = user_angles.get(self.primary_joint, 0.0)
        best_frame = min(self.golden_trajectory, key=lambda f: abs(f.get(self.primary_joint, 0.0) - target_angle))

        error_sum = 0.0
        joints_compared = 0

        for joint in self.active_joints:
            if joint == self.primary_joint: continue 
            
            u_ang = user_angles.get(joint)
            g_ang = best_frame.get(joint)

            if u_ang is not None and g_ang is not None:
                error_sum += abs(u_ang - g_ang)
                joints_compared += 1

        if joints_compared == 0: return 100.0
        
        avg_error = error_sum / joints_compared
        
        # The Tolerance Buffer: 15 degrees of human variance is perfectly fine
        if avg_error <= 15.0:
            return 100.0
        else:
            return max(0.0, 100.0 - ((avg_error - 15.0) * 3.33))

    def process_frame(self, raw_angles: dict) -> dict:
        clean_angles = {}
        for joint in self.active_joints:
            angle = raw_angles.get(joint)
            if angle is not None:
                if joint not in self.smoothed_angles:
                    self.smoothed_angles[joint] = angle
                else:
                    self.smoothed_angles[joint] = (self.alpha * angle) + ((1.0 - self.alpha) * self.smoothed_angles[joint])
            clean_angles[joint] = self.smoothed_angles.get(joint, 0.0)

        primary_angle = clean_angles.get(self.primary_joint, self.config["all_joints"][self.primary_joint]["min_angle"])
        pct = self.get_completion_percentage(primary_angle)

        if self.state == "RESTING":
            if pct > 0.20: 
                self.state = "LIFTING"
                self.max_achieved_pct = pct
                self.min_achieved_pct = 1.0 
                self.phase_errors = []

        elif self.state == "LIFTING":
            self.max_achieved_pct = max(self.max_achieved_pct, pct)
            self.phase_errors.append(self._calculate_frame_accuracy(clean_angles, pct))
            
            if pct < self.max_achieved_pct - 0.10 and self.max_achieved_pct > 0.50:
                self.state = "LOWERING"
                self.min_achieved_pct = pct

        elif self.state == "LOWERING":
            self.min_achieved_pct = min(self.min_achieved_pct, pct)
            self.phase_errors.append(self._calculate_frame_accuracy(clean_angles, pct))
            
            is_resting = pct < 0.20
            is_bouncing = pct > self.min_achieved_pct + 0.15 and self.min_achieved_pct < 0.50

            if is_resting or is_bouncing:
                total_displacement = self.max_achieved_pct - self.min_achieved_pct
                
                # TRASH FILTER 1: Must cover at least 40% ROM
                if total_displacement >= 0.40:
                    
                    base_score = min(100.0, (total_displacement / 0.85) * 100.0)
                    form_accuracy = sum(self.phase_errors) / len(self.phase_errors) if self.phase_errors else 100.0
                    
                    calculated_rep_score = round((base_score * 0.7) + (form_accuracy * 0.3), 1)
                    
                    # TRASH FILTER 2: Rep must score at least 50% to be counted
                    if calculated_rep_score >= 50.0:
                        self.reps += 1
                        self.last_rep_score = calculated_rep_score
                        self.total_score_accumulator += self.last_rep_score
                        self.average_score = self.total_score_accumulator / self.reps

                # Reset state cleanly
                if is_resting:
                    self.state = "RESTING"
                else:
                    self.state = "LIFTING"
                
                self.max_achieved_pct = pct
                self.min_achieved_pct = 1.0
                self.phase_errors = []

        return {
            "reps": self.reps,
            "state": self.state,
            "progress": round(pct * 100, 1), 
            "last_score": self.last_rep_score,
            "avg_score": round(self.average_score, 1)
        }