import gymnasium as gym
import numpy as np
import mujoco
from legged_rl_control.nodes.mujoco_sim import MujocoSimulator
from legged_rl_control.utils.quat_to_euler import quat_to_euler
from gymnasium import spaces
from collections import deque, defaultdict
from legged_rl_control.rl.wrappers.normalize_observation import NormalizeObservation
from legged_rl_control.controllers.pid_controller import create_pid_controllers_from_config
from legged_rl_control.rl.wrappers.domain_randomization import DomainRandomizationWrapper
import rclpy

class LeggedEnv(gym.Env):
    def __init__(self, config, controller_config=None):
        # Add default values for critical parameters
        self.config = {
            "min_base_height": 0.15,
            "max_episode_length": 1000,
            **config  # User config overrides defaults
        }
        self.sim = MujocoSimulator(
            self.config["model_path"],
            self.config.get("render", False)
        )
        self.step_count = 0  # Initialize step counter here
        self._setup_spaces()
        self.stationary_steps = 0
        self.min_base_height = 0.2  # Minimum acceptable base height in meters
        # Initialize Mujoco model/data from simulator
        self.model = self.sim.model
        self.data = self.sim.data
        term_config = config.get("termination_conditions", {})
        self.action_buffer = deque(maxlen=term_config.get("action_buffer_size", 10))
        self.standing_still_threshold = term_config.get("standing_still_threshold", 0.1)
        self.action_variance_threshold = term_config.get("action_variance_threshold", 0.1)
        self.standing_still_timer = 0.0
        self.dt = 1.0 / config["sim_params"]["control_freq"]
        self._max_episode_steps = config.get("max_episode_steps", 1000)
        self.steps = 0  # Add step counter
        self.termination_reasons = defaultdict(int)  # Track reasons for termination
        
        # Initialize PID controllers
        self.pid_controllers = create_pid_controllers_from_config(
            controller_config=controller_config["pid"],
            num_joints=self.config["num_actions"],
            dt=self.dt
        )
        
        # Modify action space to be position targets
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.config["num_actions"],))

    def _setup_spaces(self):
        obs_size = len(self._get_obs())
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,))

    def _get_obs(self):
        # Collect raw sensor data
        qpos = self.sim.data.qpos.copy()
        qvel = self.sim.data.qvel.copy()
        imu_acc = self.sim.data.sensor('imu_acc').data.copy()
        imu_gyro = self.sim.data.sensor('imu_gyro').data.copy()
        
        raw_obs = np.concatenate([qpos, qvel, imu_acc, imu_gyro])
        return self._filter_obs(raw_obs)

    def _filter_obs(self, obs):
        # Implement filtered observations based on config
        start_idx = 0
        if "base_pos" in self.config.get("observation_exclusions", []):
            # Remove first 3 elements (base position)
            obs = obs[3:]
            start_idx += 3
        if "base_orn" in self.config.get("observation_exclusions", []):
            # Remove next 4 elements (base orientation quaternion)
            obs = np.concatenate([obs[:start_idx], obs[start_idx+4:]])
        
        return obs.astype(np.float32)

    def step(self, action):
        # Convert normalized actions to joint positions
        joint_pos_targets = self._action_to_joint_positions(action)
        
        # Compute PID outputs
        pid_outputs = np.zeros_like(action)
        for i, pid in enumerate(self.pid_controllers):
            current_pos = self.sim.data.qpos[7 + i]
            current_vel = self.sim.data.qvel[6 + i]
            pid_outputs[i] = pid.compute(
                setpoint=joint_pos_targets[i],
                measurement=current_pos,
                measurement_deriv=current_vel
            )
        
        # Apply PID outputs
        self.action_buffer.append(pid_outputs.copy())
        self.sim.data.ctrl[:] = pid_outputs
        self.sim.sim_step()
        
        self.step_count += 1  # Increment step counter
        reward = self._calculate_reward()
        terminated = self._check_done()
        truncated = self._get_truncated(self._get_obs(), action)
        info = {}
        
        return self._get_obs(), reward, terminated, truncated, info
    
    def _get_truncated(self, obs, action):
        truncated = False
        base_lin_vel = np.linalg.norm(self.sim.data.qvel[0:3])
        
        # Update standing still timer
        if base_lin_vel < self.standing_still_threshold:
            self.standing_still_timer += self.dt
        else:
            self.standing_still_timer = 0.0
            
        # Check time-based standing still with safety get()
        max_still_time = self.config["termination_conditions"].get(
            "max_standing_still_duration", 5.0
        )
        if self.standing_still_timer >= max_still_time:
            self.termination_reasons['prolonged_inactivity'] += 1
            truncated = True
            
        # Check hopping while stationary
        if len(self.action_buffer) == self.action_buffer.maxlen:
            action_var = np.var(np.array(self.action_buffer).flatten())
            if (base_lin_vel < self.standing_still_threshold and 
                action_var > self.action_variance_threshold):
                self.termination_reasons['excessive_leg_movement'] += 1
                truncated = True
                
        return truncated or self.steps >= self._max_episode_steps

    def reset(self, seed=None, options=None):
        # Update the reset method to handle seeding
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.sim.advance()  # Ensure simulation state is updated
        self.step_count = 0  # Reset step counter here
        self.action_buffer.clear()
        # Reset PID controllers
        for pid in self.pid_controllers:
            pid.reset()
        info = {"reset_reason": "initial"}
        return self._get_obs(), info

    def _calculate_reward(self):
        """Modular reward calculation for standing behavior with leg hops"""
        reward_components = self.config.get("reward_components", {
            "orientation": 0.8,      # Weight for upright orientation
            "base_height": 0.5,      # Weight for maintaining base height
            "velocity_penalty": -0.2, # Weight for velocity penalty
            "action_penalty": -0.05,  # Weight for action magnitude penalty
            "symmetry": 0.5,         # Weight for leg movement symmetry
            "leg_activity": 0.2      # Weight for leg movement encouragement
        })
        
        total_reward = 0.0
        for component, weight in reward_components.items():
            if hasattr(self, f"_{component}_reward"):
                component_value = getattr(self, f"_{component}_reward")()
                total_reward += weight * component_value
            
        return total_reward

    def _orientation_reward(self):
        """Reward for maintaining upright orientation"""
        orientation = self.sim.data.sensor('imu_orn').data
        roll, pitch, _ = quat_to_euler(orientation)
        return 1.0 - (abs(roll) + abs(pitch))/np.pi  # Max value 1 when upright

    def _base_height_reward(self):
        """Reward for maintaining target base height"""
        target_height = self.config.get("target_base_height", 0.3)
        current_height = self.sim.data.qpos[2]
        height_error = abs(current_height - target_height)
        return np.exp(-height_error**2/0.01)  # Gaussian-style reward

    def _velocity_penalty_reward(self):
        """Penalty for base movement"""
        base_lin_vel = self.sim.data.qvel[0:3]
        base_ang_vel = self.sim.data.qvel[3:6]
        return -(np.sum(np.square(base_lin_vel)) + np.sum(np.square(base_ang_vel)))

    def _action_penalty_reward(self):
        """Penalty for large actions to encourage efficiency"""
        return -np.sum(np.square(self.sim.data.ctrl))  # L2 penalty on actions

    def _symmetry_reward(self):
        """Reward for symmetric leg movements (multiple symmetry types)"""
        num_joints = self.config.get("num_joints_per_leg", 3)
        fl = self.sim.data.qpos[7:7+num_joints]          # Front left
        fr = self.sim.data.qpos[7+num_joints:7+2*num_joints]  # Front right
        hl = self.sim.data.qpos[7+2*num_joints:7+3*num_joints] # Hind left
        hr = self.sim.data.qpos[7+3*num_joints:7+4*num_joints] # Hind right
        
        front_diff = np.sum(np.square(fl - fr))          # Front pair symmetry
        hind_diff = np.sum(np.square(hl - hr))           # Hind pair symmetry
        left_diff = np.sum(np.square(fl - hl))           # Left side symmetry
        right_diff = np.sum(np.square(fr - hr))          # Right side symmetry
        diagonal1_diff = np.sum(np.square(fl - hr))      # Diagonal symmetry 1
        diagonal2_diff = np.sum(np.square(fr - hl))      # Diagonal symmetry 2
        
        # Combine with configurable weights from robot config
        symmetry_weights = self.config.get("symmetry_weights", {
            'front': 0.4,
            'hind': 0.4,
            'left_right': 0.1,
            'diagonal': 0.1
        })
        
        return -(
            symmetry_weights['front'] * (front_diff + hind_diff) +
            symmetry_weights['hind'] * (left_diff + right_diff) +
            symmetry_weights['diagonal'] * (diagonal1_diff + diagonal2_diff)
        )

    def _leg_activity_reward(self):
        """Encourage leg movement through joint velocity reward"""
        joint_velocities = self.sim.data.qvel[6:]  # Skip base velocities
        return np.mean(np.abs(joint_velocities))  # Encourage some movement

    def _check_done(self):
        # Add configuration validation
        required_keys = ['min_base_height', 'max_episode_length']
        for key in required_keys:
            if key not in self.config:
                raise KeyError(f"Missing required config key: {key}")

        base_height = self.sim.get_base_height()
        return (
            base_height < self.config["min_base_height"] or
            self.step_count >= self.config["max_episode_length"]
        )

    def _action_to_joint_positions(self, action):
        """Convert normalized actions to physical joint positions"""
        joint_pos_targets = np.zeros_like(action)
        for i in range(self.config["num_actions"]):
            joint_id = i  # Assuming actuator order matches joint order
            joint_range = self.model.jnt_range[joint_id]
            joint_pos_targets[i] = (
                (action[i] + 1) / 2 * (joint_range[1] - joint_range[0]) + joint_range[0]
            )
        return joint_pos_targets

    def close(self):
        """Clean up ROS resources"""
        if hasattr(self, 'sim'):
            self.sim.destroy_node()
            rclpy.shutdown()
        super().close()

    # Add seed method for compatibility
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

def make_env(config):
    """Factory function for creating wrapped environments"""
    env = LeggedEnv(config)
    
    # Apply observation normalization if enabled
    if config.get("observation_normalization", {}).get("enabled", False):
        env = NormalizeObservation(env,
            epsilon=config["observation_normalization"].get("epsilon", 1e-8),
            clip=config["observation_normalization"].get("clip", 10.0),
            update_stats=config["observation_normalization"].get("update_during_training", True)
        )
    
    # Apply domain randomization if enabled
    if config.get("domain_randomization", {}).get("enabled", False):
        randomization_config = config["domain_randomization"].get("parameters", {})
        env = DomainRandomizationWrapper(env, randomization_config)
    
    return env
