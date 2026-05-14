# This file defines the reward functions based on Section A.1.2 and Table S6.

class RewardFunctions:
    def __init__(self, robot_type='anymal_d'):
        self.robot_type = robot_type.lower()
        self.weights = self._get_reward_weights()

    def _get_reward_weights(self):
        # Table S6: Reward weights
        if self.robot_type == 'anymal_d':
            return {
                'wvxy': 1.0,
                'wωz': 0.5,
                'wvz': -2.0,
                'wωxy': -0.05,
                'wqτ': -2.5e-5,
                'wq': -2.5e-7, # Note: paper has wq twice, assuming one is wq (joint pos) and one is wqd (joint dev)
                'wa': -0.01,
                'wfa': 0.5,
                'wc': -1.0,
                'wg': -5.0,
                'wfc': 0.0,
                'wqd': 0.0, # From Table S6, 'Wqd' for ANYmal D is 0.0
            }
        elif self.robot_type == 'unitree_g1':
            return {
                'wvxy': 1.0,
                'wωz': 0.5,
                'wvz': -2.0,
                'wωxy': -0.05,
                'wqt': -2.5e-5,
                'wq': -2.5e-7,
                'wa': -0.05,
                'wfa': 0.0,
                'wc': -1.0,
                'wg': -5.0,
                'wfc': 1.0,
                'wqd': -1.0,
            }
        else:
            raise ValueError(f"Unknown robot type: {self.robot_type}")

    def _exp_quadratic_loss(self, diff, sigma):
        # Helper for r_v_xy and r_omega_z
        # return self.weights[weight_key] * math.exp(-(diff**2) / (sigma**2))
        return -(diff**2) / (sigma**2) # Simplified for static representation

    def _quadratic_loss(self, value):
        # Helper for r_v_z, r_omega_xy, r_q_tau, r_ddot_q
        # return self.weights[weight_key] * (value**2)
        return (value**2) # Simplified

    def _abs_loss(self, value):
        # Helper for r_q_d
        # return self.weights[weight_key] * abs(value)
        return abs(value) # Simplified

    def r_linear_velocity_tracking_xy(self, commanded_v_xy, current_v_xy):
        # Equation 240: r_v_xy = w_v_xy * e^(-||c_xy - v_xy||_2^2 / sigma_v_xy^2)
        # sigma_v_xy = 0.25
        diff = [c - v for c, v in zip(commanded_v_xy, current_v_xy)] # Conceptual element-wise diff
        diff_norm_sq = sum(x*x for x in diff)
        sigma_v_xy = 0.25
        exponent = -diff_norm_sq / (sigma_v_xy**2)
        # In a real implementation, would use math.exp(exponent)
        return self.weights['wvxy'] * exponent # Simplified: exponent directly represents log of exp part

    def r_angular_velocity_tracking_z(self, commanded_omega_z, current_omega_z):
        # Equation 256: r_omega_z = w_omega_z * e^(-||c_z - omega_z||_2^2 / sigma_omega_z^2)
        # sigma_omega_z = 0.25
        diff_norm_sq = (commanded_omega_z - current_omega_z)**2 # Conceptual scalar diff
        sigma_omega_z = 0.25
        exponent = -diff_norm_sq / (sigma_omega_z**2)
        return self.weights['wωz'] * exponent # Simplified

    def r_linear_velocity_z(self, v_z):
        # Equation 264: r_v_z = w_v_z * ||v_z||_2^2
        return self.weights['wvz'] * self._quadratic_loss(v_z)

    def r_angular_velocity_xy(self, omega_xy):
        # Equation 272: r_omega_xy = w_omega_xy * ||omega_xy||_2^2
        omega_xy_norm_sq = sum(x*x for x in omega_xy) # Conceptual norm sq
        return self.weights['wωxy'] * omega_xy_norm_sq # Using directly the norm sq

    def r_joint_torque(self, tau):
        # Equation 280: r_q_tau = w_q_tau * ||tau||_2^2
        tau_norm_sq = sum(x*x for x in tau) # Conceptual norm sq
        return self.weights['wqτ'] * tau_norm_sq

    def r_joint_acceleration(self, ddot_q):
        # Equation 288: r_ddot_q = w_ddot_q * ||ddot_q||_2^2
        ddot_q_norm_sq = sum(x*x for x in ddot_q) # Conceptual norm sq
        # The paper's Table S6 uses 'wq' for one of these, which is confusing.
        # I'll use 'wq' for joint acceleration as it's the second 'wq' entry in the table and distinct from joint torque.
        # If 'Wqτ' is for joint torque, then 'wq' is for joint acceleration.
        # The value is -2.5e-7 in Table S6, I'll use that.
        return self.weights['wq'] * ddot_q_norm_sq

    def r_action_rate(self, prev_action, current_action):
        # Equation 296: r_dot_a = w_dot_a * ||a' - a||_2^2
        diff = [pa - ca for pa, ca in zip(prev_action, current_action)] # Conceptual element-wise diff
        diff_norm_sq = sum(x*x for x in diff)
        # The paper uses 'Wa' and 'Wà' for ANYmal D and Unitree G1 respectively.
        # I'll map 'Wa' to 'wa' for ANYmal D and 'Wà' to 'wa' for Unitree G1 in weights.
        return self.weights['wa'] * diff_norm_sq

    def r_feet_air_time(self, t_fa):
        # Equation 307: r_f_a = w_f_a * t_f_a
        return self.weights['wfa'] * t_fa

    def r_undesired_contacts(self, c_u):
        # Equation 315: r_c = w_c * c_u
        return self.weights['wc'] * c_u

    def r_flat_orientation(self, g_xy):
        # Equation 323: r_g = w_g * g_xy^2
        g_xy_norm_sq = sum(x*x for x in g_xy) # Conceptual norm sq
        return self.weights['wg'] * g_xy_norm_sq

    def r_foot_clearance(self, h_fc):
        # Equation 331: r_f_c = w_f_c * h_f_c
        return self.weights['wfc'] * h_fc

    def r_joint_deviation(self, q, q_0):
        # Equation 339: r_q_d = w_q_d * ||q - q_0||_1
        diff = [q_val - q0_val for q_val, q0_val in zip(q, q_0)] # Conceptual element-wise diff
        diff_norm_1 = sum(abs(x) for x in diff)
        return self.weights['wqd'] * diff_norm_1

    def calculate_total_reward(self, **kwargs):
        # This function would take all necessary state and action components
        # and compute the sum of weighted rewards.
        # For static representation, return a dummy sum.
        total_reward = 0.0

        # Example of how reward terms would be called
        # if 'commanded_v_xy' in kwargs and 'current_v_xy' in kwargs:
        #     total_reward += self.r_linear_velocity_tracking_xy(kwargs['commanded_v_xy'], kwargs['current_v_xy'])
        # ... and so on for all reward components.

        return total_reward # Placeholder for actual calculation

