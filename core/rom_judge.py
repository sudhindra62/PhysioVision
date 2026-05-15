class ROMJudge:
    def __init__(self, config):
        self.config = config
        self.state = "IDLE"
        self.reps = 0
        self.current_peak_pct = 0.0
        
        # --- NEW: Sequence Tracking ---
        self.current_rep_sequence = []
        self.last_rep_score = 0.0  # Starts at 0 until first rep is done

        self.rom_range = abs(self.config["max_angle"] - self.config["min_angle"])
        if self.rom_range == 0: self.rom_range = 1 # Prevent division by zero
        self.tolerance_pct = self.config["tolerance"] / self.rom_range

    def get_completion_percentage(self, current_angle: float) -> float:
        moved = abs(current_angle - self.config["min_angle"])
        pct = moved / self.rom_range
        return max(0.0, min(1.0, pct))

    def _evaluate_rep_sequence(self):
        """Analyzes the entire array of frames recorded during the rep."""
        if not self.current_rep_sequence:
            return 0.0

        penalty = 0.0
        primary = self.config["primary_joint"]

        # Form Police: Check every frame of the rep
        for frame_angles in self.current_rep_sequence:
            for joint, limits in self.config["all_joints"].items():
                if joint == primary:
                    continue # We don't penalize the primary joint moving
                
                angle = frame_angles.get(joint)
                if angle is not None and angle > 0:
                    # If a secondary joint moves more than 15 degrees outside its precomputed 
                    # natural range, we consider it "cheating" or "bad form"
                    if angle > limits["max_angle"] + 15.0 or angle < limits["min_angle"] - 15.0:
                        penalty += 0.5 # Accumulate penalty for every bad frame
        
        # Calculate final score based on penalties
        final_score = 100.0 - penalty
        return round(max(0.0, min(100.0, final_score)), 1)

    def process_frame(self, all_angles: dict) -> dict:
        """Now accepts the full dictionary of all 6 joint angles."""
        
        primary_angle = all_angles.get(self.config["primary_joint"], self.config["min_angle"])
        pct = self.get_completion_percentage(primary_angle)
        
        target_peak_pct = 1.0 - self.tolerance_pct
        target_rest_pct = 0.0 + self.tolerance_pct

        # Always record the frame if we are actively in motion
        if self.state in ["LIFTING", "PEAK_REACHED", "LOWERING"]:
            self.current_rep_sequence.append(all_angles)

        # State 0: IDLE
        if self.state == "IDLE" and pct > target_rest_pct:
            self.state = "LIFTING"
            self.current_peak_pct = pct
            self.current_rep_sequence = [all_angles] # Start recording a new sequence!

        # State 1: LIFTING
        elif self.state == "LIFTING":
            if pct > self.current_peak_pct:
                self.current_peak_pct = pct
            if pct >= target_peak_pct:
                self.state = "PEAK_REACHED"

        # State 2: PEAK (Holding the top)
        elif self.state == "PEAK_REACHED":
            if pct < target_peak_pct - 0.1: # Moved slightly down from peak
                self.state = "LOWERING"

        # State 3: LOWERING (Returning to rest)
        elif self.state == "LOWERING":
            if pct < target_rest_pct:
                # REP COMPLETED!
                self.reps += 1
                self.state = "IDLE"
                
                self.last_rep_score = self._evaluate_rep_sequence()
                
                # Clear buffer for the next rep
                self.current_peak_pct = 0.0
                self.current_rep_sequence = []

        return {
            "reps": self.reps,
            "state": self.state,
            "progress": round(pct * 100, 1), # Visual indicator (0-100%)
            "rep_score": self.last_rep_score # Actual quality of the movement
        }