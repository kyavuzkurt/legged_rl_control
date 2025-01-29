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
A1_CONFIG = {
    "model_path": os.path.join(
        get_package_share_directory('legged_rl_control'),
        'config/scene.xml'
    ),
    "control_dims": 12,          # Number of controllable joints
    "action_scale": 0.5,         # Scaling factor for actions
    "obs_dim": 48,               # Should match sensor data dimensions
    "num_actions": 12,           # Must match robot's actuated DOF
    "min_base_height": 0.2,      # Minimum base height before termination
    "reward_weights": {
        "forward_velocity": 1.0,
        "joint_torque": -0.01,
        "action_rate": -0.1,
        "foot_slip": -0.5
    },
    "sim_params": {
        "physics_engine": "MuJoCo",  # Or your simulator
        "control_freq": 100,     # Hz
        "decimation": 4,         # Policy runs at 25Hz
    },
    "observation_exclusions": ['base_pos', 'base_orn'],
} 