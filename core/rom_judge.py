class ROMJudge:
    def __init__(self, config): # Assuming config is your ROMConfig or from AutoCalibrator
        self.config = config
        self.state = "IDLE"
        self.reps = 0
        self.current_peak_pct = 0.0
        
        # Calculate the total range of motion
        self.rom_range = abs(self.config["max_angle"] - self.config["min_angle"])
        
        # Convert the raw angle tolerance into a percentage (e.g., 15 deg / 70 deg range = ~21%)
        self.tolerance_pct = self.config["tolerance"] / self.rom_range

    def get_completion_percentage(self, current_angle: float) -> float:
        """Maps any angle to a 0.0 (rest) to 1.0 (peak) scale, regardless of direction."""
        # How far have we moved from the resting position?
        moved = abs(current_angle - self.config["min_angle"])
        pct = moved / self.rom_range
        return max(0.0, min(1.0, pct)) # Clamp between 0.0 and 1.0

    def process_frame(self, current_angle: float) -> dict:
        # 1. Normalize the math first
        pct = self.get_completion_percentage(current_angle)
        
        # 2. Universal thresholds (e.g., Peak is > 85%, Rest is < 15%)
        target_peak_pct = 1.0 - self.tolerance_pct
        target_rest_pct = 0.0 + self.tolerance_pct

        # State 0: Waiting to start
        if self.state == "IDLE" and pct > target_rest_pct:
            self.state = "LIFTING"
            self.current_peak_pct = pct

        # State 1: Moving towards the target
        elif self.state == "LIFTING":
            if pct > self.current_peak_pct:
                self.current_peak_pct = pct
                
            if pct >= target_peak_pct:
                self.state = "PEAK_REACHED"

        # State 2: Hit the target, now returning to rest
        elif self.state == "PEAK_REACHED":
            if pct < target_rest_pct:
                self.reps += 1
                self.state = "IDLE"
                self.current_peak_pct = 0.0

        return {
            "reps": self.reps,
            "state": self.state,
            "live_score": round(pct * 100, 1), # Automatically a perfect 0-100 score
            "raw_angle": round(current_angle, 1) # Good for UI debugging
        }