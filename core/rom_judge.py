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
            config (dict): Configuration dictionary loaded from the master registry, 
                           containing 'primary_joints', 'all_joints' specifications, 
                           and the 'golden_trajectory' array.
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
        
        # --- STRICT CYCLE LOCKS ---
        # Hysteresis bounds to prevent state flickering at boundary limits
        self.BOTTOM_THRESHOLD = 0.25 
        self.TOP_THRESHOLD = 0.80    
        self.valid_peak_reached = False 
        self.current_rep_errors = []

        # --- DYNAMIC BASELINE CALIBRATION ---
        self.joint_baselines = {}
        for joint in self.primary_joints:
            min_ang = self.config["all_joints"][joint]["min_angle"]
            max_ang = self.config["all_joints"][joint]["max_angle"]
            
            base_angle = min_ang
            if self.golden_trajectory:
                start_angle = self.golden_trajectory[0].get(joint, min_ang)
                # If the initial frame is closer to max_angle, assume reverse motion (pulling)
                if abs(start_angle - max_ang) < abs(start_angle - min_ang):
                    base_angle = max_ang
                    
            self.joint_baselines[joint] = base_angle

    def get_completion_percentage(self, clean_angles: dict) -> float:
        """
        Calculates the normalized progress of the active repetition.

        Maps the current geometric coordinates against the pre-calibrated baseline range 
        of motion, bounding the final calculation mathematically between 0.0 and 1.0.

        Args:
            clean_angles (dict): Smoothed dictionary mapping joint names to float degrees.

        Returns:
            float: A normalized value (0.0 to 1.0) indicating execution completion.
        """
        pcts = []
        for joint in self.primary_joints:
            min_ang = self.config["all_joints"][joint]["min_angle"]
            max_ang = self.config["all_joints"][joint]["max_angle"]
            rom_range = abs(max_ang - min_ang) if abs(max_ang - min_ang) > 0 else 1
            
            base_angle = self.joint_baselines.get(joint, min_ang)
            current_angle = clean_angles.get(joint, base_angle)
            
            moved = abs(current_angle - base_angle)
            pct = moved / rom_range
            pcts.append(max(0.0, min(1.0, pct)))
            
        return min(pcts) if pcts else 0.0

    def _calculate_frame_accuracy(self, user_angles: dict) -> float:
        """
        Evaluates immediate posture precision against the reference model.

        Employs a nearest-neighbor spatial search to find the frame within the 
        golden trajectory that closest matches the user's current progress percentage. 
        It then calculates the average joint error deviation across all active tracking nodes.

        Args:
            user_angles (dict): Current smoothed angles of the user.

        Returns:
            float: An immediate accuracy score bounded between 0.0 and 100.0.
        """
        if not self.golden_trajectory: return 100.0

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
        
        # Form scoring scale: Errors under 15° are considered passing. 
        # Progressive penalties apply linearly beyond that baseline.
        if avg_error <= 15.0:
            return 100.0
        else:
            return max(0.0, 100.0 - ((avg_error - 15.0) * 3.33))

    def _generate_live_feedback(self, user_angles: dict) -> dict:
        """
        Generates dual-output structural adjustments for the user.

        Compares targeted trajectories to isolate deviations. If a primary joint 
        slips out of tolerance by more than 15°, it constructs explicit directional 
        instructions.

        Args:
            user_angles (dict): Current smoothed angles of the user.

        Returns:
            dict: A payload containing two distinct string properties:
                  - 'ui': Technical string with angles (e.g., 'Extend Elbow (18°)')
                  - 'tts': Conversational string (e.g., 'Please extend your left elbow')
        """
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
                
                if abs(diff) > 15.0:
                    clean_name = joint.replace("_", " ").title()
                    offset = round(abs(diff), 1)
                    
                    # 1. UI String (Data-heavy)
                    direction_ui = "Extend/Open" if diff > 0 else "Bend/Close"
                    ui_list.append(f"{direction_ui} {clean_name} ({offset}°)")
                    
                    # 2. TTS String (Natural phrasing)
                    direction_tts = "extend" if diff > 0 else "bend"
                    tts_list.append(f"Please {direction_tts} your {clean_name}")

        return {
            "ui": " | ".join(ui_list) if ui_list else "Great form!",
            "tts": " and ".join(tts_list) if tts_list else "Great form!"
        }

    def process_frame(self, raw_angles: dict) -> dict:
        """
        The master evaluation cycle triggered for every incoming video frame.
        
        Performs four serial processes:
        1. Noise Filtering: Applies Exponential Moving Average (EMA) smoothing to input angles.
        2. Plane Verification: Evaluates the Transverse Plane Angle to ensure alignment.
        3. Finite State Machine Update: Drives the structural lifecycle loop.
        4. Performance Valuation: scores completed repetitions using a weighted blend 
           of extension depth (70%) and multi-joint accuracy profiles (30%).

        Args:
            raw_angles (dict): Unfiltered geometric dictionary from the math layer.

        Returns:
            dict: High-density state payload required by frontend rendering layers.
        """
        clean_angles = {}
        for joint, angle in raw_angles.items():
            if angle is not None:
                if joint not in self.smoothed_angles:
                    self.smoothed_angles[joint] = angle
                else:
                    # Exponential Moving Average filter execution
                    self.smoothed_angles[joint] = (self.alpha * angle) + ((1.0 - self.alpha) * self.smoothed_angles[joint])
            clean_angles[joint] = self.smoothed_angles.get(joint, 0.0)

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
                # 25 degree operational window allowed before gatekeeper activates
                if abs(plane_diff) > 25.0: 
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
                # Absolute validation lock: Can only achieve PEAK if moving inside the valid 3D plane
                if plane_is_valid:
                    self.state = "PEAK"
                    self.valid_peak_reached = True 

            elif pct < self.max_achieved_pct - 0.15:
                # Hysteresis breach indicates premature descent
                self.state = "LOWERING"
                self.min_achieved_pct = pct

        elif self.state == "PEAK":
            self.max_achieved_pct = max(self.max_achieved_pct, pct)
            self.current_rep_errors.append(frame_acc)

            if pct < self.max_achieved_pct - 0.15:
                self.state = "LOWERING"
                self.min_achieved_pct = pct

        elif self.state == "LOWERING":
            self.min_achieved_pct = min(self.min_achieved_pct, pct)
            self.current_rep_errors.append(frame_acc)

            is_resting = pct <= self.BOTTOM_THRESHOLD
            is_bouncing = pct > self.min_achieved_pct + 0.15 # Hysteresis recovery flag

            if is_resting:
                # Repetition complete boundary reached
                if self.valid_peak_reached:
                    # Weighting: 70% Extension Completeness, 30% Continuous Path Precision
                    base_score = min(100.0, ((self.max_achieved_pct - self.min_achieved_pct) / 0.80) * 100.0)
                    form_accuracy = sum(self.current_rep_errors) / len(self.current_rep_errors) if self.current_rep_errors else 100.0
                    
                    calculated_rep_score = round((base_score * 0.7) + (form_accuracy * 0.3), 1)
                    
                    # Filter out low-grade partial movements from the permanent storage layer
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

        # Generate dual-output feedback configurations
        feedback_data = self._generate_live_feedback(clean_angles)
        current_ui_feedback = feedback_data["ui"]
        current_tts_feedback = feedback_data["tts"]

        # Override system strings with priority alerts if the 3D plane gatekeeper fires
        if not plane_is_valid and pct > self.BOTTOM_THRESHOLD:
            current_ui_feedback = plane_error_message
            action_tts = "forward" if "FORWARD" in plane_error_message else "to the side"
            part_tts = "arm" if "ARM" in plane_error_message else "leg"
            current_tts_feedback = f"Bring your {part_tts} {action_tts}"

        return {
            "reps": self.reps,
            "state": self.state,
            "progress": round(pct * 100, 1), 
            "last_score": self.last_rep_score,
            "avg_score": round(self.average_score, 1),
            "rep_scores": self.rep_scores,
            "feedback": current_ui_feedback,
            "tts_feedback": current_tts_feedback
        }