import time
from typing import Dict, List

class ROMJudge:
    def __init__(self, config: dict):
        self.config = config
        self.state = "IDLE"
        self.reps = 0
        
        # Session Metrics
        self.last_rep_score = 0.0  
        self.total_score_accumulator = 0.0 
        self.average_score = 0.0 
        
        # Kinematics
        self.last_pct = 0.0
        self.smoothed_angles = {}
        self.velocities = {}      
        self.last_time = time.time()
        self.alpha = 0.40         

        # Configuration
        self.primary_joint = self.config["primary_joint"]
        self.active_joints = list(self.config["all_joints"].keys())
        self.golden_trajectory = self.config.get("golden_trajectory", [])

        self.rom_range = abs(self.config["all_joints"][self.primary_joint]["max_angle"] - self.config["all_joints"][self.primary_joint]["min_angle"])
        if self.rom_range == 0: self.rom_range = 1 
        
        # Phase Tracking Variables
        self.phase_errors = []
        self.max_achieved_pct = 0.0
        self.current_valley_pct = 1.0

    def _process_kinematics(self, raw_angles: dict) -> dict:
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0: dt = 0.01 
        
        clean_angles = {}

        for joint in self.active_joints:
            raw_angle = raw_angles.get(joint)

            if joint not in self.smoothed_angles:
                if raw_angle is not None:
                    self.smoothed_angles[joint] = raw_angle
                    self.velocities[joint] = 0.0
                continue 

            if raw_angle is not None:
                if abs(raw_angle - self.smoothed_angles[joint]) > 45.0:
                    raw_angle = None 

            if raw_angle is not None:
                smoothed = (self.alpha * raw_angle) + ((1.0 - self.alpha) * self.smoothed_angles[joint])
                omega = (smoothed - self.smoothed_angles[joint]) / dt 
                self.smoothed_angles[joint] = smoothed
                self.velocities[joint] = (self.alpha * omega) + ((1.0 - self.alpha) * self.velocities.get(joint, 0.0))
                clean_angles[joint] = smoothed
            else:
                last_angle = self.smoothed_angles[joint]
                velocity = self.velocities.get(joint, 0.0)
                damped_velocity = velocity * 0.9 
                predicted_angle = last_angle + (damped_velocity * dt)
                
                self.smoothed_angles[joint] = predicted_angle
                self.velocities[joint] = damped_velocity
                clean_angles[joint] = predicted_angle

        self.last_time = current_time
        return clean_angles

    def get_completion_percentage(self, current_angle: float) -> float:
        moved = abs(current_angle - self.config["all_joints"][self.primary_joint]["min_angle"])
        pct = moved / self.rom_range
        return max(0.0, min(1.0, pct))

    def _calculate_frame_accuracy(self, user_angles: dict, current_pct: float) -> float:
        """
        Takes the user's current progress (e.g. 50% lifted) and finds the frame in the 
        Golden Trajectory where the expert was also at 50% lifted. 
        Then, grades the user's overall posture against the expert's posture.
        """
        if not self.golden_trajectory: return 100.0

        target_angle = user_angles.get(self.primary_joint, 0.0)

        # 1. Find the closest trajectory frame based on the Primary Joint's angle
        best_frame = min(self.golden_trajectory, key=lambda f: abs(f.get(self.primary_joint, 0.0) - target_angle))

        # 2. Check the posture of ALL OTHER joints
        error_sum = 0.0
        joints_compared = 0

        for joint in self.active_joints:
            if joint == self.primary_joint: continue # We already aligned via primary
            
            u_ang = user_angles.get(joint)
            g_ang = best_frame.get(joint)

            if u_ang is not None and g_ang is not None:
                error_sum += abs(u_ang - g_ang)
                joints_compared += 1

        if joints_compared == 0: return 100.0

        avg_joint_error_degrees = error_sum / joints_compared
        
        # Scoring Math: 0 degrees off = 100%. 30+ degrees off = 0%.
        frame_score = max(0.0, 100.0 - (avg_joint_error_degrees * 3.33))
        return frame_score

    def process_frame(self, raw_angles: dict) -> dict:
        clean_angles = self._process_kinematics(raw_angles)
        
        primary_angle = clean_angles.get(self.primary_joint, self.config["all_joints"][self.primary_joint]["min_angle"])
        pct = self.get_completion_percentage(primary_angle)
        
        # Strict Checkpoint Gates
        target_peak_pct = 0.85  # Must pass 85% to unlock peak phase
        target_rest_pct = 0.15  # Must drop below 15% to finish rep

        # -------------------------------------------------------------------
        # STRICT PHASE-GATED STATE MACHINE
        # -------------------------------------------------------------------
        
        if self.state == "IDLE":
            if pct < target_rest_pct:
                self.state = "READY"

        elif self.state == "READY":
            if pct > target_rest_pct:
                self.state = "LIFTING"
                self.phase_errors = []
                self.max_achieved_pct = pct

        elif self.state == "LIFTING":
            self.max_achieved_pct = max(self.max_achieved_pct, pct)
            self.phase_errors.append(self._calculate_frame_accuracy(clean_angles, pct))

            if pct >= target_peak_pct:
                # Checkpoint 1 Unlocked
                self.state = "PEAK_REACHED"
            elif pct < target_rest_pct:
                # ABORTED REP: They dropped back down without hitting the peak.
                self.state = "READY" 

        elif self.state == "PEAK_REACHED":
            self.max_achieved_pct = max(self.max_achieved_pct, pct)
            if pct < target_peak_pct - 0.05:
                self.state = "LOWERING"
                self.current_valley_pct = pct

        elif self.state == "LOWERING":
            self.current_valley_pct = min(self.current_valley_pct, pct)
            self.phase_errors.append(self._calculate_frame_accuracy(clean_angles, pct))

            if pct <= target_rest_pct:
                # Checkpoint 2 Unlocked: PERFECT REP
                self.state = "FINISH"
                
            elif pct > self.current_valley_pct + 0.15:
                # BYPASS ATTEMPT DETECTED: They started lifting again without hitting the bottom
                self.state = "FINISH" 

        # --- SCORING & RESET BLOCK ---
        if self.state == "FINISH":
            self.reps += 1

            # 1. Base Posture Score (Average of all frame accuracies)
            base_accuracy = sum(self.phase_errors) / len(self.phase_errors) if self.phase_errors else 0.0

            # 2. ROM Bypass Penalties
            rom_penalty = 0.0
            
            if pct > target_rest_pct + 0.1: 
                # Short-repped the bottom (e.g. stopped at 40%, penalty is 40 points)
                rom_penalty += (pct * 100) 

            if self.max_achieved_pct < target_peak_pct: 
                # Short-repped the top
                rom_penalty += ((1.0 - self.max_achieved_pct) * 100)

            # Final Calculation
            final_score = round(max(0.0, base_accuracy - rom_penalty), 1)

            self.last_rep_score = final_score
            self.total_score_accumulator += self.last_rep_score
            self.average_score = self.total_score_accumulator / self.reps

            # Reset for the next repetition
            self.state = "READY"
            self.phase_errors = []
            self.max_achieved_pct = 0.0

        self.last_pct = pct

        return {
            "reps": self.reps,
            "state": self.state,
            "progress": round(pct * 100, 1), 
            "last_score": self.last_rep_score,
            "avg_score": round(self.average_score, 1)
        }
