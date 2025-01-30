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
    "obs_dim": 36,               # Corrected observation dimension (19 qpos + 18 qvel + 6 imu)
    "num_actions": 12,           # Must match robot's actuated DOF
    "min_base_height": 0.2,      # Minimum base height before termination
    "target_base_height": 0.3,  # meters
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
        "max_episode_steps": 1000  # 10 seconds at 100Hz
    },
    "observation_exclusions": ['base_pos', 'base_orn'],
    "reward_components": {
        "orientation": 1.5,
        "base_height": 1.0,
        "velocity_penalty": -0.2,
        "action_penalty": -0.05,
        "symmetry": 0.3,
        "leg_activity": 0.2
    },
    "launch_viewer": False,  # Default for training
    "visual_evaluation": {
        "enable": True,
        "fps": 60,
        "window_size": (1920, 1080)
    },
    "termination_conditions": {
        "standing_still_threshold": 0.1,  # m/s
        "action_variance_threshold": 0.1,
        "action_buffer_size": 10,
        "max_standing_still_duration": 5.0  # seconds
    }
} 