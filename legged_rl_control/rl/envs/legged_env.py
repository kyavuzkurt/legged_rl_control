import gymnasium as gym
import numpy as np
import mujoco
from legged_rl_control.nodes.mujoco_sim import MujocoSimulator
from legged_rl_control.rl.robot_configs import RobotConfig
from legged_rl_control.utils.quat_to_euler import quat_to_euler
from gymnasium import spaces
from collections import deque, defaultdict

class LeggedEnv(gym.Env):
    def __init__(self, config):
        self.config = config
        self.sim = MujocoSimulator(
            config["model_path"],
            launch_viewer=config.get("launch_viewer", False)
        )
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
        
    def _setup_spaces(self):
        obs_size = len(self._get_obs())
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,))
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.config["num_actions"],))

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
        self.action_buffer.append(action.copy())
        self.sim.data.ctrl[:] = action
        self.sim.sim_step()
        
        self.steps += 1  # Increment step counter
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
            
        # Check time-based standing still
        if self.standing_still_timer >= self.config["termination_conditions"]["max_standing_still_duration"]:
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
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.sim.advance()  # Ensure simulation state is updated
        self.steps = 0  # Reset step counter
        self.action_buffer.clear()
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
        """Reward for symmetric leg movements (front/hind pairs)"""
        # Assuming 3 joints per leg and base qpos at start
        num_joints = self.config.get("num_joints_per_leg", 3)
        fl = self.sim.data.qpos[7:7+num_joints]          # Front left
        fr = self.sim.data.qpos[7+num_joints:7+2*num_joints]  # Front right
        hl = self.sim.data.qpos[7+2*num_joints:7+3*num_joints] # Hind left
        hr = self.sim.data.qpos[7+3*num_joints:7+4*num_joints] # Hind right
        
        front_diff = np.sum(np.square(fl - fr))
        hind_diff = np.sum(np.square(hl - hr))
        return -0.1*(front_diff + hind_diff)  # Penalize asymmetry

    def _leg_activity_reward(self):
        """Encourage leg movement through joint velocity reward"""
        joint_velocities = self.sim.data.qvel[6:]  # Skip base velocities
        return np.mean(np.abs(joint_velocities))  # Encourage some movement

    def _check_done(self):
        # Get orientation from IMU (quaternion [w,x,y,z])
        orientation = self.sim.data.sensor('imu_orn').data
        
        # Convert quaternion to Euler angles (roll, pitch, yaw)
        roll, pitch, _ = quat_to_euler(orientation)
        
        # 1. Orientation check (fall condition)
        max_tilt = 0.5  # ~30 degrees in radians
        if abs(roll) > max_tilt or abs(pitch) > max_tilt:
            return True
            
        # 2. Base height check (fall condition)
        base_height = self.sim.data.qpos[2]  # Z position of base
        if base_height < self.config["min_base_height"]:
            return True
            
        # 3. Stationary timeout check (velocity magnitude)
        lin_vel = self.sim.data.qvel[0:3]
        vel_magnitude = np.linalg.norm(lin_vel)
        
        if vel_magnitude < 0.1:  # 0.1 m/s threshold
            self.stationary_steps += 1
            if self.stationary_steps > 200:  # 10 seconds at 20Hz
                return True
        else:
            self.stationary_steps = 0
            
        return False
