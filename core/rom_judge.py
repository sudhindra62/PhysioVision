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
        self.alpha = 0.35        

        # --- DUAL JOINT UPGRADE ---
        # Fallback to a list if they still use the old "primary_joint" key
        self.primary_joints = self.config.get("primary_joints", [self.config.get("primary_joint")])
        self.active_joints = list(self.config["all_joints"].keys())
        self.golden_trajectory = self.config.get("golden_trajectory", [])

        self.max_achieved_pct = 0.0
        self.min_achieved_pct = 1.0
        self.phase_errors = []
        
        # --- STRICT FSM GATES ---
        self.BOTTOM_THRESHOLD = 0.25  # Must return below 25% ROM to finish
        self.TOP_THRESHOLD = 0.80     # BOTH arms must hit 80% ROM to unlock a valid rep

    def get_completion_percentage(self, clean_angles: dict) -> float:
        """Calculates percentage based on the WEAKEST/LOWEST arm to strictly enforce form."""
        pcts = []
        for joint in self.primary_joints:
            # Safely grab min and max for this specific joint
            min_ang = self.config["all_joints"][joint]["min_angle"]
            max_ang = self.config["all_joints"][joint]["max_angle"]
            rom_range = abs(max_ang - min_ang) if abs(max_ang - min_ang) > 0 else 1
            
            current_angle = clean_angles.get(joint, min_ang)
            moved = abs(current_angle - min_ang)
            pct = moved / rom_range
            pcts.append(max(0.0, min(1.0, pct)))
            
        # Return the minimum percentage. BOTH arms must clear the threshold.
        return min(pcts) if pcts else 0.0

    def _calculate_frame_accuracy(self, user_angles: dict) -> float:
        if not self.golden_trajectory: return 100.0

        # Find the best frame by averaging the primary joints to match the trajectory timing
        def get_avg_primary(frame):
            vals = [frame.get(j, 0.0) for j in self.primary_joints if j in frame]
            return sum(vals) / len(vals) if vals else 0.0

        user_target = get_avg_primary(user_angles)
        best_frame = min(self.golden_trajectory, key=lambda f: abs(get_avg_primary(f) - user_target))

        error_sum = 0.0
        joints_compared = 0

        for joint in self.active_joints:
            # Score ALL active joints, including both primary shoulders
            u_ang = user_angles.get(joint)
            g_ang = best_frame.get(joint)

            if u_ang is not None and g_ang is not None:
                error_sum += abs(u_ang - g_ang)
                joints_compared += 1

        if joints_compared == 0: return 100.0
        
        avg_error = error_sum / joints_compared
        
        # 15 degrees of human variance tolerance
        if avg_error <= 15.0:
            return 100.0
        else:
            return max(0.0, 100.0 - ((avg_error - 15.0) * 3.33))

    def process_frame(self, raw_angles: dict) -> dict:
        # 1. Smooth the incoming angles
        clean_angles = {}
        for joint in self.active_joints:
            angle = raw_angles.get(joint)
            if angle is not None:
                if joint not in self.smoothed_angles:
                    self.smoothed_angles[joint] = angle
                else:
                    self.smoothed_angles[joint] = (self.alpha * angle) + ((1.0 - self.alpha) * self.smoothed_angles[joint])
            clean_angles[joint] = self.smoothed_angles.get(joint, 0.0)

        # 2. Get the strictest completion percentage (lowest arm)
        pct = self.get_completion_percentage(clean_angles)

        # ---------------------------------------------------------
        # FINITE STATE MACHINE LOGIC
        # ---------------------------------------------------------
        
        if self.state == "RESTING":
            if pct > self.BOTTOM_THRESHOLD:
                self.state = "LIFTING"
                self.max_achieved_pct = pct
                self.min_achieved_pct = 1.0
                self.phase_errors = []

        elif self.state == "LIFTING":
            self.max_achieved_pct = max(self.max_achieved_pct, pct)
            self.phase_errors.append(self._calculate_frame_accuracy(clean_angles))

            if pct >= self.TOP_THRESHOLD:
                self.state = "PEAK"

            elif pct < self.max_achieved_pct - 0.15:
                # Direction changed downward before BOTH arms hit 80%. Half-rep detected.
                self.state = "LOWERING_INVALID"
                self.min_achieved_pct = pct

        elif self.state == "PEAK":
            self.max_achieved_pct = max(self.max_achieved_pct, pct)
            self.phase_errors.append(self._calculate_frame_accuracy(clean_angles))

            if pct < self.max_achieved_pct - 0.10:
                self.state = "LOWERING_VALID"
                self.min_achieved_pct = pct

        elif self.state in ["LOWERING_VALID", "LOWERING_INVALID"]:
            self.min_achieved_pct = min(self.min_achieved_pct, pct)
            self.phase_errors.append(self._calculate_frame_accuracy(clean_angles))

            is_resting = pct <= self.BOTTOM_THRESHOLD
            is_bouncing = pct > self.min_achieved_pct + 0.15

            if is_resting or is_bouncing:
                # ONLY score and increment if coming down from a valid PEAK
                if self.state == "LOWERING_VALID" and is_resting:
                    base_score = min(100.0, ((self.max_achieved_pct - self.min_achieved_pct) / 0.80) * 100.0)
                    form_accuracy = sum(self.phase_errors) / len(self.phase_errors) if self.phase_errors else 100.0
                    
                    calculated_rep_score = round((base_score * 0.7) + (form_accuracy * 0.3), 1)
                    
                    if calculated_rep_score >= 50.0:
                        self.reps += 1
                        self.last_rep_score = calculated_rep_score
                        self.total_score_accumulator += self.last_rep_score
                        self.average_score = self.total_score_accumulator / self.reps

                # Reset state safely
                if is_resting:
                    self.state = "RESTING"
                else: 
                    self.state = "LIFTING"
                    self.max_achieved_pct = pct

                self.phase_errors = []

        return {
            "reps": self.reps,
            "state": self.state,
            "progress": round(pct * 100, 1), 
            "last_score": self.last_rep_score,
            "avg_score": round(self.average_score, 1)
        }