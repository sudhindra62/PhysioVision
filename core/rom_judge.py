"""
Range of Motion (ROM) Judge Module

The core stateful evaluation engine of the application. It applies mathematical 
filtering to raw inputs, enforces strict structural cycle locks to validate 
repetitions, tracks historic session metrics, and generates dual-targeted 
feedback payloads (high-density UI text and natural language TTS).
"""
import time
from typing import Dict, List

class ROMJudge:
    """
    Manages the real-time lifecycle of physical therapy and fitness repetitions.
    Operates as a deterministic finite state machine (RESTING -> LIFTING -> PEAK -> LOWERING)
    backed by hysteresis thresholds to eliminate false positives caused by camera 
    shaking, spatial jitter, or micro-bounces at structural boundaries.
    """

    def __init__(self, config: dict):
        """
        Initializes the state machine, smoothing profiles, and historic accumulators.
        Performs dynamic baseline calibration to determine whether an exercise start 
        state is at a minimum extension or maximum flexion angle, allowing the system 
        to natively support both pushing and pulling anatomical patterns.
        
        Args:
            config (dict): Configuration dictionary loaded from the master registry.
        """
        self.config = config
        self.state = "RESTING"
        self.reps = 0
        
        self.last_rep_score = 0.0  
        self.total_score_accumulator = 0.0 
        self.average_score = 0.0 
        self.rep_scores = [] 
        
        # Exponential moving average (EMA) cache for noise suppression
        self.smoothed_angles = {}
        self.alpha = 0.35        

        self.primary_joints = self.config.get("primary_joints", [self.config.get("primary_joint")])
        self.active_joints = list(self.config["all_joints"].keys())
        self.golden_trajectory = self.config.get("golden_trajectory", [])

        self.max_achieved_pct = 0.0
        self.min_achieved_pct = 1.0
        
        # --- STRICT CYCLE LOCKS (UPDATED FOR 2D FORESHORTENING) ---
        # Thresholds adapted to account for 2D camera Z-axis compression
        self.BOTTOM_THRESHOLD = 0.20  # Allows resting at 20% of textbook limit
        self.TOP_THRESHOLD = 0.45     # Registers peak at 45% (compensates for missing depth)
        self.HYSTERESIS = 0.10        # Required drop before changing direction
        
        self.valid_peak_reached = False 
        self.current_rep_errors = []

        # --- We prepare a blank slate for live posture calibration ---
        self.live_baselines = None

    def get_completion_percentage(self, clean_angles: dict) -> float:
        """Calculates the normalized progress of the active repetition."""
        pcts = []
        for joint in self.primary_joints:
            min_ang = self.config["all_joints"][joint]["min_angle"]
            max_ang = self.config["all_joints"][joint]["max_angle"]
            
            # Use the live posture captured when you started the tracker
            base_angle = self.live_baselines.get(joint, min_ang) if self.live_baselines else min_ang
            current_angle = clean_angles.get(joint, base_angle)
            
            # Calculate max possible movement from where you are sitting
            max_possible_movement = max(abs(max_ang - base_angle), abs(min_ang - base_angle))
            if max_possible_movement < 10.0: # Safeguard against tiny ranges
                max_possible_movement = 10.0
                
            moved = abs(current_angle - base_angle)
            
            pct = moved / max_possible_movement
            pcts.append(max(0.0, min(1.0, pct)))
            
        return min(pcts) if pcts else 0.0

    def _calculate_frame_accuracy(self, user_angles: dict) -> float:
        """Evaluates immediate posture precision against the reference model."""
        def get_avg_primary(frame):
            vals = [frame.get(j, 0.0) for j in self.primary_joints if j in frame]
            return sum(vals) / len(vals) if vals else 0.0

        user_target = get_avg_primary(user_angles)
        best_frame = min(self.golden_trajectory, key=lambda f: abs(get_avg_primary(f) - user_target))

        error_sum = 0.0
        joints_compared = 0

        for joint in self.active_joints:
            u_ang = user_angles.get(joint)
            g_ang = best_frame.get(joint)

            if u_ang is not None and g_ang is not None:
                error_sum += abs(u_ang - g_ang)
                joints_compared += 1

        if joints_compared == 0: return 100.0
        
        avg_error = error_sum / joints_compared
        
        if avg_error <= 5.0:
            return 100.0
        else:
            return max(0.0, 100.0 - ((avg_error - 5.0) * 4.0))

    def _generate_live_feedback(self, user_angles: dict) -> dict:
        """Generates dual-output structural adjustments for the user."""
        if not self.golden_trajectory or self.state == "RESTING":
            return {"ui": "Ready...", "tts": ""}

        def get_avg_primary(frame):
            vals = [frame.get(j, 0.0) for j in self.primary_joints if j in frame]
            return sum(vals) / len(vals) if vals else 0.0

        user_target = get_avg_primary(user_angles)
        best_frame = min(self.golden_trajectory, key=lambda f: abs(get_avg_primary(f) - user_target))

        ui_list = []
        tts_list = []

        for joint in self.primary_joints:
            u_ang = user_angles.get(joint)
            g_ang = best_frame.get(joint)

            if u_ang is not None and g_ang is not None:
                diff = g_ang - u_ang
                
                if abs(diff) > 5.0:
                    clean_name = joint.replace("_", " ").title()
                    offset = round(abs(diff), 1)
                    
                    direction_ui = "Extend/Open" if diff > 0 else "Bend/Close"
                    ui_list.append(f"{direction_ui} {clean_name} ({offset}°)")
                    
                    direction_tts = "extend" if diff > 0 else "bend"
                    tts_list.append(f"Please {direction_tts} your {clean_name}")

        return {
            "ui": " | ".join(ui_list) if ui_list else "Flawless form!",
            "tts": " and ".join(tts_list) if tts_list else "Flawless form!"
        }

    def process_frame(self, raw_angles: dict) -> dict:
        """The master evaluation cycle triggered for every incoming video frame."""
        clean_angles = {}
        for joint, angle in raw_angles.items():
            if angle is not None:
                if joint not in self.smoothed_angles:
                    self.smoothed_angles[joint] = angle
                else:
                    self.smoothed_angles[joint] = (self.alpha * angle) + ((1.0 - self.alpha) * self.smoothed_angles[joint])
            clean_angles[joint] = self.smoothed_angles.get(joint, 0.0)

        # --- LIVE POSTURE CALIBRATION ---
        # Captures your exact resting posture on the first valid frame
        if self.live_baselines is None:
            if all(clean_angles.get(j, 0.0) > 0 for j in self.primary_joints):
                self.live_baselines = {j: clean_angles[j] for j in self.primary_joints}
            else:
                # Wait for the camera to find you before starting
                return {
                    "reps": self.reps, "state": "CALIBRATING", "progress": 0.0, 
                    "last_score": 0.0, "avg_score": 0.0, "rep_scores": [], 
                    "feedback": "Get in starting position...", "tts_feedback": ""
                }

        pct = self.get_completion_percentage(clean_angles)
        frame_acc = self._calculate_frame_accuracy(clean_angles)

        # --- DYNAMIC PLANE GATEKEEPER ---
        plane_is_valid = True
        plane_error_message = ""

        if self.golden_trajectory and self.primary_joints:
            def get_avg_primary(frame):
                vals = [frame.get(j, 0.0) for j in self.primary_joints if j in frame]
                return sum(vals) / len(vals) if vals else 0.0

            user_target = get_avg_primary(clean_angles)
            best_frame = min(self.golden_trajectory, key=lambda f: abs(get_avg_primary(f) - user_target))

            primary = self.primary_joints[0]
            base_joint = primary.replace("_elbow", "_shoulder").replace("_knee", "_hip")
            plane_key = f"{base_joint}_plane"

            user_plane = clean_angles.get(plane_key)
            golden_plane = best_frame.get(plane_key)

            if user_plane is not None and golden_plane is not None:
                plane_diff = user_plane - golden_plane
                # Increased to 40.0 to account for 2D camera spatial fluctuation
                if abs(plane_diff) > 40.0: 
                    plane_is_valid = False
                    action = "FORWARD" if plane_diff > 0 else "SIDEWAYS"
                    part = "ARM" if "shoulder" in base_joint else "LEG"
                    plane_error_message = f"BRING {part} MORE {action} ({round(abs(plane_diff), 1)}°)"

        # ---------------------------------------------------------
        # THE STRICT LOOP: Start -> Top -> Start
        # ---------------------------------------------------------
        
        if self.state == "RESTING":
            if pct > self.BOTTOM_THRESHOLD and plane_is_valid:
                self.state = "LIFTING"
                self.max_achieved_pct = pct
                self.min_achieved_pct = 1.0
                self.current_rep_errors = [frame_acc]
                self.valid_peak_reached = False 

        elif self.state == "LIFTING":
            self.max_achieved_pct = max(self.max_achieved_pct, pct)
            self.current_rep_errors.append(frame_acc)

            if pct >= self.TOP_THRESHOLD:
                if plane_is_valid:
                    self.state = "PEAK"
                    self.valid_peak_reached = True 

            elif pct < self.max_achieved_pct - self.HYSTERESIS:
                self.state = "LOWERING"
                self.min_achieved_pct = pct

        elif self.state == "PEAK":
            self.max_achieved_pct = max(self.max_achieved_pct, pct)
            self.current_rep_errors.append(frame_acc)

            if pct < self.max_achieved_pct - self.HYSTERESIS:
                self.state = "LOWERING"
                self.min_achieved_pct = pct

        elif self.state == "LOWERING":
            self.min_achieved_pct = min(self.min_achieved_pct, pct)
            self.current_rep_errors.append(frame_acc)

            is_resting = pct <= self.BOTTOM_THRESHOLD
            is_bouncing = pct > self.min_achieved_pct + self.HYSTERESIS

            if is_resting:
                if self.valid_peak_reached:
                    # Dynamically scale score against the Top Threshold to account for 2D bounds
                    base_score = min(100.0, ((self.max_achieved_pct - self.min_achieved_pct) / self.TOP_THRESHOLD) * 100.0)
                    form_accuracy = sum(self.current_rep_errors) / len(self.current_rep_errors) if self.current_rep_errors else 100.0
                    
                    calculated_rep_score = round((base_score * 0.7) + (form_accuracy * 0.3), 1)
                    
                    if calculated_rep_score >= 50.0:
                        self.reps += 1
                        self.last_rep_score = calculated_rep_score
                        self.total_score_accumulator += self.last_rep_score
                        self.average_score = self.total_score_accumulator / self.reps
                        self.rep_scores.append(calculated_rep_score)

                self.state = "RESTING"
                self.valid_peak_reached = False
                self.current_rep_errors = []

            elif is_bouncing:
                self.state = "LIFTING"
                self.max_achieved_pct = pct

        feedback_data = self._generate_live_feedback(clean_angles)
        current_ui_feedback = feedback_data["ui"]
        current_tts_feedback = feedback_data["tts"]

        if not plane_is_valid and pct > self.BOTTOM_THRESHOLD:
            current_ui_feedback = plane_error_message
            action_tts = "forward" if "FORWARD" in plane_error_message else "to the side"
            part_tts = "arm" if "ARM" in plane_error_message else "leg"
            current_tts_feedback = f"Bring your {part_tts} {action_tts}"

        # Scale UI progress so it fills to 100% visually when hitting the 2D TOP_THRESHOLD
        ui_progress = min(100.0, (pct / self.TOP_THRESHOLD) * 100.0)

        return {
            "reps": self.reps,
            "state": self.state,
            "progress": round(ui_progress, 1), 
            "last_score": self.last_rep_score,
            "avg_score": round(self.average_score, 1),
            "rep_scores": self.rep_scores,
            "feedback": current_ui_feedback,
            "tts_feedback": current_tts_feedback
        }