# This file defines the observation, action, and privileged information spaces for ANYmal D and Unitree G1.
# Based on Tables S2, S3, S4, and S5.

# Helper function to parse dimension strings like '0:3'
def parse_dimensions(dim_str):
    if ':' in dim_str:
        start, end = map(int, dim_str.split(':'))
        return end - start
    return int(dim_str) if dim_str.isdigit() else 0 # Return 0 for non-numeric or empty dimensions

class ObservationSpaces:
    def __init__(self):
        # Table S2: World model observation space
        self.world_model_anymal_d = {
            'base_linear_velocity': parse_dimensions('0:3'), # U
            'base_angular_velocity': parse_dimensions('3:6'), # 3
            'projected_gravity': parse_dimensions('6:9'), # g
            'joint_positions': parse_dimensions('9:21'), # q
            'joint_velocities': parse_dimensions('21:33'), # q
            'joint_torques': parse_dimensions('33:45'), # τ
        }
        self.world_model_anymal_d_dim = sum(self.world_model_anymal_d.values())

        self.world_model_unitree_g1 = {
            'base_linear_velocity': parse_dimensions('0:3'), # U
            'base_angular_velocity': parse_dimensions('3:6'), # 3
            'projected_gravity': parse_dimensions('6:9'), # g
            'joint_positions': parse_dimensions('9:38'), # q
            'joint_velocities': parse_dimensions('38:67'), # q
            'joint_torques': parse_dimensions('67:96'), # τ
        }
        self.world_model_unitree_g1_dim = sum(self.world_model_unitree_g1.values())

        # Table S5: Policy observation space
        self.policy_anymal_d = {
            'base_linear_velocity': parse_dimensions('0:3'), # U
            'base_angular_velocity': parse_dimensions('3:6'), # 3
            'projected_gravity': parse_dimensions('6:9'), # g
            'velocity_command': parse_dimensions('9:12'), # c
            'joint_positions': parse_dimensions('12:24'), # q
            'joint_velocities': parse_dimensions('24:36'), # q
            'last_actions': parse_dimensions('36:48'), # a'
        }
        self.policy_anymal_d_dim = sum(self.policy_anymal_d.values())

        self.policy_unitree_g1 = {
            'base_linear_velocity': parse_dimensions('0:3'), # U
            'base_angular_velocity': parse_dimensions('3:6'), # 3
            'projected_gravity': parse_dimensions('6:9'), # g
            'velocity_command': parse_dimensions('9:12'), # c
            'joint_positions': parse_dimensions('12:41'), # q
            'joint_velocities': parse_dimensions('41:70'), # q
            'last_actions': parse_dimensions('70:99'), # a'
        }
        self.policy_unitree_g1_dim = sum(self.policy_unitree_g1.values())

class ActionSpaces:
    def __init__(self):
        # Table S4: Action space
        self.anymal_d = {
            'joint_position_targets': parse_dimensions('0:12'), # q*
        }
        self.anymal_d_dim = sum(self.anymal_d.values())

        self.unitree_g1 = {
            'joint_position_targets': parse_dimensions('0:29'), # q*
        }
        self.unitree_g1_dim = sum(self.unitree_g1.values())

class PrivilegedInfoSpaces:
    def __init__(self):
        # Table S3: World model privileged information space
        self.anymal_d = {
            'knee_contact': parse_dimensions('0:4'),
            'foot_contact': parse_dimensions('4:8'),
        }
        self.anymal_d_dim = sum(self.anymal_d.values())

        self.unitree_g1 = {
            'body_contact': parse_dimensions('0:26'),
            'foot_height': parse_dimensions('26:28'),
            'foot_velocity': parse_dimensions('28:30'),
        }
        self.unitree_g1_dim = sum(self.unitree_g1.values())

