# import time

# class ROMJudge:
#     def __init__(self, config):
#         self.config = config
#         self.state = "IDLE"
#         self.reps = 0
#         self.current_peak_pct = 0.0
        
#         self.current_rep_sequence = []
#         self.last_rep_score = 0.0  
#         self.total_score_accumulator = 0.0 
#         self.average_score = 0.0 
        
#         # --- NEW: Kinematics & Smoothing Engine ---
#         self.last_pct = 0.0
#         self.smoothed_angles = {}
#         self.velocities = {}      # Stores angular velocity (degrees per second)
#         self.last_time = time.time()
#         self.alpha = 0.25         # EMA Smoothing Factor

#         self.rom_range = abs(self.config["max_angle"] - self.config["min_angle"])
#         if self.rom_range == 0: self.rom_range = 1 
#         self.tolerance_pct = self.config["tolerance"] / self.rom_range

#     def _process_kinematics(self, raw_angles: dict) -> dict:
#         """Applies EMA smoothing and uses physics to predict missing joints."""
#         current_time = time.time()
#         dt = current_time - self.last_time
#         if dt <= 0: dt = 0.01 # Prevent division by zero
        
#         clean_angles = {}

#         for joint in self.config["all_joints"].keys():
#             raw_angle = raw_angles.get(joint)

#             # Initialize if this is the first frame we've seen this joint
#             if joint not in self.smoothed_angles:
#                 if raw_angle is not None:
#                     self.smoothed_angles[joint] = raw_angle
#                     self.velocities[joint] = 0.0
#                 continue 

#             # PATH A: The joint is VISIBLE (Apply EMA and calculate velocity)
#             if raw_angle is not None:
#                 # Glitch rejection: If angle jumps wildly in 1 frame (AI hallucination), reject it
#                 if abs(raw_angle - self.smoothed_angles[joint]) > 45.0:
#                     raw_angle = None # Force it to use the physics prediction instead

#             if raw_angle is not None:
#                 smoothed = (self.alpha * raw_angle) + ((1.0 - self.alpha) * self.smoothed_angles[joint])
#                 omega = (smoothed - self.smoothed_angles[joint]) / dt # Angular Velocity
                
#                 self.smoothed_angles[joint] = smoothed
#                 self.velocities[joint] = (self.alpha * omega) + ((1.0 - self.alpha) * self.velocities.get(joint, 0.0))
#                 clean_angles[joint] = smoothed

#             # PATH B: The joint is HIDDEN (Use Kinematics to predict the future)
#             else:
#                 last_angle = self.smoothed_angles[joint]
#                 velocity = self.velocities.get(joint, 0.0)
                
#                 # We dampen the velocity so the "ghost arm" slowly comes to a stop 
#                 # instead of spinning forever if the camera is blocked for 10 seconds.
#                 damped_velocity = velocity * 0.9 
#                 predicted_angle = last_angle + (damped_velocity * dt)
                
#                 self.smoothed_angles[joint] = predicted_angle
#                 self.velocities[joint] = damped_velocity
#                 clean_angles[joint] = predicted_angle

#         self.last_time = current_time
#         return clean_angles

#     def get_completion_percentage(self, current_angle: float) -> float:
#         moved = abs(current_angle - self.config["min_angle"])
#         pct = moved / self.rom_range
#         return max(0.0, min(1.0, pct))

#     def _evaluate_rep_sequence(self):
#         if not self.current_rep_sequence: return 0.0

#         penalty = 0.0
#         primary = self.config["primary_joint"]

#         for frame_angles in self.current_rep_sequence:
#             for joint, limits in self.config["all_joints"].items():
#                 if joint == primary: continue 
                
#                 angle = frame_angles.get(joint)
#                 if angle and angle > 10.0: # Ignore completely zeroed out joints
#                     if angle > limits["max_angle"] + 15.0 or angle < limits["min_angle"] - 15.0:
#                         penalty += 0.4 # Accumulate form penalty
        
#         final_score = 100.0 - penalty
#         return round(max(0.0, min(100.0, final_score)), 1)

#     def process_frame(self, raw_angles: dict) -> dict:
#         # 1. Run the physics engine to get perfect, unbroken tracking
#         clean_angles = self._process_kinematics(raw_angles)
        
#         primary_angle = clean_angles.get(self.config["primary_joint"], self.config["min_angle"])
#         pct = self.get_completion_percentage(primary_angle)
        
#         target_peak_pct = 1.0 - self.tolerance_pct
#         target_rest_pct = 0.0 + (self.tolerance_pct * 1.5) 

#         # --- DYNAMIC STATE MACHINE ---
#         if self.state in ["LIFTING", "PEAK_REACHED", "LOWERING"]:
#             self.current_rep_sequence.append(clean_angles)

#         if self.state == "IDLE" and pct > target_rest_pct:
#             self.state = "LIFTING"
#             self.current_peak_pct = pct
#             self.current_rep_sequence = [clean_angles] 

#         elif self.state == "LIFTING":
#             if pct > self.current_peak_pct: self.current_peak_pct = pct
#             if pct >= target_peak_pct: self.state = "PEAK_REACHED"

#         elif self.state == "PEAK_REACHED":
#             if pct < target_peak_pct - 0.05: self.state = "LOWERING"

#         elif self.state == "LOWERING":
#             # REVERSAL FIX: Rep completes if they hit bottom OR suddenly start lifting again
#             is_at_bottom = pct < target_rest_pct
#             is_lifting_again = pct > self.last_pct + 0.08 
            
#             if is_at_bottom or is_lifting_again:
#                 if len(self.current_rep_sequence) > 10: # Debouncer to kill jitter fake-reps
#                     self.reps += 1
#                     self.last_rep_score = self._evaluate_rep_sequence()
                    
#                     self.total_score_accumulator += self.last_rep_score
#                     self.average_score = self.total_score_accumulator / self.reps
                
#                 # Seamlessly transition into the next rep if they reversed direction early
#                 self.state = "LIFTING" if is_lifting_again else "IDLE"
#                 self.current_peak_pct = pct if is_lifting_again else 0.0
#                 self.current_rep_sequence = [clean_angles] if is_lifting_again else []

#         self.last_pct = pct

#         return {
#             "reps": self.reps,
#             "state": self.state,
#             "progress": round(pct * 100, 1), 
#             "last_score": self.last_rep_score,
#             "avg_score": round(self.average_score, 1)
#         }

import time

class ROMJudge:
    def __init__(self, config):
        self.config = config
        self.state = "IDLE"
        self.reps = 0
        
        self.current_peak_pct = 0.0
        self.current_valley_pct = 1.0 # NEW: Tracks the bottom of a fast rep
        
        self.current_rep_sequence = []
        self.last_rep_score = 0.0  
        self.total_score_accumulator = 0.0 
        self.average_score = 0.0 
        
        self.last_pct = 0.0
        self.smoothed_angles = {}
        self.velocities = {}      
        self.last_time = time.time()
        
        # FIX 1: Increased Alpha from 0.25 to 0.40. 
        # Fast exercises need less smoothing so we don't chop off the peaks!
        self.alpha = 0.40         

        self.rom_range = abs(self.config["max_angle"] - self.config["min_angle"])
        if self.rom_range == 0: self.rom_range = 1 
        self.tolerance_pct = self.config["tolerance"] / self.rom_range

    def _process_kinematics(self, raw_angles: dict) -> dict:
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0: dt = 0.01 
        
        clean_angles = {}

        for joint in self.config["all_joints"].keys():
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
        moved = abs(current_angle - self.config["min_angle"])
        pct = moved / self.rom_range
        return max(0.0, min(1.0, pct))

    def _evaluate_rep_sequence(self):
        if not self.current_rep_sequence: return 0.0

        penalty = 0.0
        primary = self.config["primary_joint"]

        for frame_angles in self.current_rep_sequence:
            for joint, limits in self.config["all_joints"].items():
                if joint == primary: continue 
                
                angle = frame_angles.get(joint)
                if angle and angle > 10.0: 
                    if angle > limits["max_angle"] + 15.0 or angle < limits["min_angle"] - 15.0:
                        penalty += 0.4 
        
        final_score = 100.0 - penalty
        return round(max(0.0, min(100.0, final_score)), 1)

    def process_frame(self, raw_angles: dict) -> dict:
        clean_angles = self._process_kinematics(raw_angles)
        
        primary_angle = clean_angles.get(self.config["primary_joint"], self.config["min_angle"])
        pct = self.get_completion_percentage(primary_angle)
        
        target_peak_pct = 1.0 - self.tolerance_pct
        target_rest_pct = 0.0 + (self.tolerance_pct * 1.5) 

        # --- THE CONTINUOUS CYCLIC STATE MACHINE ---
        if self.state in ["LIFTING", "PEAK_REACHED", "LOWERING"]:
            self.current_rep_sequence.append(clean_angles)

        if self.state == "IDLE" and pct > target_rest_pct:
            self.state = "LIFTING"
            self.current_peak_pct = pct
            self.current_rep_sequence = [clean_angles] 

        elif self.state == "LIFTING":
            if pct > self.current_peak_pct: 
                self.current_peak_pct = pct
            if pct >= target_peak_pct: 
                self.state = "PEAK_REACHED"

        elif self.state == "PEAK_REACHED":
            if pct < target_peak_pct - 0.05: 
                self.state = "LOWERING"
                self.current_valley_pct = pct # Initialize the valley tracker as we start dropping

        elif self.state == "LOWERING":
            # Track the absolute lowest point of this drop
            if pct < self.current_valley_pct:
                self.current_valley_pct = pct

            is_at_bottom = pct < target_rest_pct
            
            # FIX 2: Valley Detection. Are we 5% higher than the deepest point we just hit?
            is_lifting_again = pct > self.current_valley_pct + 0.05 
            
            if is_at_bottom or is_lifting_again:
                # FIX 3: Lowered debounce from 10 to 5 to catch extremely fast movements
                if len(self.current_rep_sequence) > 5: 
                    self.reps += 1
                    self.last_rep_score = self._evaluate_rep_sequence()
                    
                    self.total_score_accumulator += self.last_rep_score
                    self.average_score = self.total_score_accumulator / self.reps
                
                # Loop straight back into lifting!
                self.state = "LIFTING" if is_lifting_again else "IDLE"
                self.current_peak_pct = pct if is_lifting_again else 0.0
                self.current_rep_sequence = [clean_angles] if is_lifting_again else []

        self.last_pct = pct

        return {
            "reps": self.reps,
            "state": self.state,
            "progress": round(pct * 100, 1), 
            "last_score": self.last_rep_score,
            "avg_score": round(self.average_score, 1)
        }