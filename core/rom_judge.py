class ROMJudge:
    def __init__(self, config):
        self.config = config
        self.state = "IDLE"
        self.reps = 0
        self.current_peak_pct = 0.0
        
        self.current_rep_sequence = []
        self.last_rep_score = 0.0  
        
        # NEW: Session Averaging Math
        self.total_score_accumulator = 0.0 
        self.average_score = 0.0 

        self.rom_range = abs(self.config["max_angle"] - self.config["min_angle"])
        if self.rom_range == 0: self.rom_range = 1 
        self.tolerance_pct = self.config["tolerance"] / self.rom_range

    def get_completion_percentage(self, current_angle: float) -> float:
        moved = abs(current_angle - self.config["min_angle"])
        pct = moved / self.rom_range
        return max(0.0, min(1.0, pct))

    def _evaluate_rep_sequence(self):
        if not self.current_rep_sequence:
            return 0.0

        penalty = 0.0
        primary = self.config["primary_joint"]

        for frame_angles in self.current_rep_sequence:
            for joint, limits in self.config["all_joints"].items():
                if joint == primary:
                    continue 
                
                angle = frame_angles.get(joint)
                if angle is not None and angle > 0:
                    if angle > limits["max_angle"] + 15.0 or angle < limits["min_angle"] - 15.0:
                        penalty += 0.5 
        
        final_score = 100.0 - penalty
        return round(max(0.0, min(100.0, final_score)), 1)

    def process_frame(self, all_angles: dict) -> dict:
        primary_angle = all_angles.get(self.config["primary_joint"], self.config["min_angle"])
        pct = self.get_completion_percentage(primary_angle)
        
        target_peak_pct = 1.0 - self.tolerance_pct
        
        # FIX 1: Widen the rest tolerance so the rep closes faster and eliminates UI lag
        target_rest_pct = 0.0 + (self.tolerance_pct * 1.5) 

        if self.state in ["LIFTING", "PEAK_REACHED", "LOWERING"]:
            self.current_rep_sequence.append(all_angles)

        if self.state == "IDLE" and pct > target_rest_pct:
            self.state = "LIFTING"
            self.current_peak_pct = pct
            self.current_rep_sequence = [all_angles] 

        elif self.state == "LIFTING":
            if pct > self.current_peak_pct:
                self.current_peak_pct = pct
            if pct >= target_peak_pct:
                self.state = "PEAK_REACHED"

        elif self.state == "PEAK_REACHED":
            if pct < target_peak_pct - 0.1: 
                self.state = "LOWERING"

        elif self.state == "LOWERING":
            if pct < target_rest_pct:
                
                # FIX 2: The Anti-Jitter Debouncer (Must be longer than 10 frames to be a real human rep)
                if len(self.current_rep_sequence) > 10:
                    self.reps += 1
                    self.last_rep_score = self._evaluate_rep_sequence()
                    
                    # FIX 3: Calculate the true Session Average
                    self.total_score_accumulator += self.last_rep_score
                    self.average_score = self.total_score_accumulator / self.reps
                
                # Always reset state, whether it was a valid rep or just AI noise
                self.state = "IDLE"
                self.current_peak_pct = 0.0
                self.current_rep_sequence = []

        return {
            "reps": self.reps,
            "state": self.state,
            "progress": round(pct * 100, 1), 
            "last_score": self.last_rep_score,
            "avg_score": round(self.average_score, 1) # Send the new average to the UI
        }