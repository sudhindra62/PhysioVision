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
        self.rep_scores = [] 
        
        self.smoothed_angles = {}
        self.alpha = 0.35        

        self.primary_joints = self.config.get("primary_joints", [self.config.get("primary_joint")])
        self.active_joints = list(self.config["all_joints"].keys())
        self.golden_trajectory = self.config.get("golden_trajectory", [])

        self.max_achieved_pct = 0.0
        self.min_achieved_pct = 1.0
        
        # --- STRICT CYCLE LOCKS ---
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
                if abs(start_angle - max_ang) < abs(start_angle - min_ang):
                    base_angle = max_ang
                    
            self.joint_baselines[joint] = base_angle

    def get_completion_percentage(self, clean_angles: dict) -> float:
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
        
        if avg_error <= 15.0:
            return 100.0
        else:
            return max(0.0, 100.0 - ((avg_error - 15.0) * 3.33))

    def _generate_live_feedback(self, user_angles: dict) -> str:
        if not self.golden_trajectory or self.state == "RESTING":
            return "Ready..."

        def get_avg_primary(frame):
            vals = [frame.get(j, 0.0) for j in self.primary_joints if j in frame]
            return sum(vals) / len(vals) if vals else 0.0

        user_target = get_avg_primary(user_angles)
        best_frame = min(self.golden_trajectory, key=lambda f: abs(get_avg_primary(f) - user_target))

        feedback_list = []
        for joint in self.primary_joints:
            u_ang = user_angles.get(joint)
            g_ang = best_frame.get(joint)

            if u_ang is not None and g_ang is not None:
                diff = g_ang - u_ang
                
                if abs(diff) > 15.0:
                    direction = "Extend/Open" if diff > 0 else "Bend/Close"
                    clean_name = joint.replace("_", " ").title()
                    offset = round(abs(diff), 1)
                    feedback_list.append(f"{direction} {clean_name} ({offset}°)")

        return " | ".join(feedback_list) if feedback_list else "Great form!"

    def process_frame(self, raw_angles: dict) -> dict:
        clean_angles = {}
        # Changed to process ALL angles, including the new plane angles
        for joint, angle in raw_angles.items():
            if angle is not None:
                if joint not in self.smoothed_angles:
                    self.smoothed_angles[joint] = angle
                else:
                    self.smoothed_angles[joint] = (self.alpha * angle) + ((1.0 - self.alpha) * self.smoothed_angles[joint])
            clean_angles[joint] = self.smoothed_angles.get(joint, 0.0)

        pct = self.get_completion_percentage(clean_angles)
        frame_acc = self._calculate_frame_accuracy(clean_angles)

        # --- NEW: DYNAMIC PLANE GATEKEEPER ---
        plane_is_valid = True
        plane_error_message = ""

        if self.golden_trajectory and self.primary_joints:
            def get_avg_primary(frame):
                vals = [frame.get(j, 0.0) for j in self.primary_joints if j in frame]
                return sum(vals) / len(vals) if vals else 0.0

            user_target = get_avg_primary(clean_angles)
            best_frame = min(self.golden_trajectory, key=lambda f: abs(get_avg_primary(f) - user_target))

            primary = self.primary_joints[0]
            # Route elbow/knee logic to their root joint's plane
            base_joint = primary.replace("_elbow", "_shoulder").replace("_knee", "_hip")
            plane_key = f"{base_joint}_plane"

            user_plane = clean_angles.get(plane_key)
            golden_plane = best_frame.get(plane_key)

            # Validate the user's transverse plane against the golden record
            if user_plane is not None and golden_plane is not None:
                plane_diff = user_plane - golden_plane
                if abs(plane_diff) > 25.0: # 25 degree strict plane tolerance
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
                # ONLY grant PEAK if the plane geometry is correct
                if plane_is_valid:
                    self.state = "PEAK"
                    self.valid_peak_reached = True 

            elif pct < self.max_achieved_pct - 0.15:
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
            is_bouncing = pct > self.min_achieved_pct + 0.15

            if is_resting:
                if self.valid_peak_reached:
                    base_score = min(100.0, ((self.max_achieved_pct - self.min_achieved_pct) / 0.80) * 100.0)
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

        current_feedback = self._generate_live_feedback(clean_angles)

        # --- Override Feedback if Gatekeeper activated ---
        if not plane_is_valid and pct > self.BOTTOM_THRESHOLD:
            current_feedback = plane_error_message

        return {
            "reps": self.reps,
            "state": self.state,
            "progress": round(pct * 100, 1), 
            "last_score": self.last_rep_score,
            "avg_score": round(self.average_score, 1),
            "rep_scores": self.rep_scores,
            "feedback": current_feedback
        }