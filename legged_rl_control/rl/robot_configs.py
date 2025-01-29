from dataclasses import dataclass
import os
from ament_index_python.packages import get_package_share_directory

@dataclass
class RobotConfig:
    model_path: str
    observation_exclusions: tuple = ()
    action_dim: int = None
    joint_speed_limits: dict = None

# Configuration for Unitree A1
A1_CONFIG = RobotConfig(
    model_path=os.path.join(
        get_package_share_directory('legged_rl_control'),
        'config/robots/a1.xml'
    ),
    observation_exclusions=('base_pos', 'base_orn', 'base_vel'),
    action_dim=12,
    joint_speed_limits={
        # Front Right leg
        'FR_hip_joint': 8.0,
        'FR_thigh_joint': 8.0,
        'FR_knee_joint': 8.0,
        'FR_ankle_joint': 8.0,
        # Front Left leg
        'FL_hip_joint': 8.0,
        'FL_thigh_joint': 8.0,
        'FL_knee_joint': 8.0,
        'FL_ankle_joint': 8.0,
        # Rear Right leg
        'RR_hip_joint': 8.0,
        'RR_thigh_joint': 8.0,
        'RR_knee_joint': 8.0,
        'RR_ankle_joint': 8.0,
        # Rear Left leg
        'RL_hip_joint': 8.0,
        'RL_thigh_joint': 8.0,
        'RL_knee_joint': 8.0,
        'RL_ankle_joint': 8.0,
    }
) 