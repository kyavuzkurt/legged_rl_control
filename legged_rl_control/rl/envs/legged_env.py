import gymnasium as gym
import numpy as np
import mujoco
from legged_rl_control.nodes.mujoco_sim import MujocoSimulator
from legged_rl_control.rl.robot_configs import RobotConfig
from legged_rl_control.utils.quat_to_euler import quat_to_euler
from gymnasium import spaces

class LeggedEnv(gym.Env):
    def __init__(self, config):
        self.config = config
        self.sim = MujocoSimulator(config["model_path"])
        self._setup_spaces()
        self.stationary_steps = 0
        self.min_base_height = 0.2  # Minimum acceptable base height in meters
        # Initialize Mujoco model/data from simulator
        self.model = self.sim.model
        self.data = self.sim.data
        
    def _setup_spaces(self):
        obs_size = len(self._get_obs())
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,))
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.config["num_actions"],))

    def _get_obs(self):
        # Collect raw sensor data
        qpos = self.sim.data.qpos.copy()        # Joint positions
        qvel = self.sim.data.qvel.copy()        # Joint velocities
        imu_acc = self.sim.data.sensor('imu_acc').data.copy()
        imu_gyro = self.sim.data.sensor('imu_gyro').data.copy()
        
        # Create raw observation vector
        raw_obs = np.concatenate([
            qpos,
            qvel,
            imu_acc,
            imu_gyro
        ])
        
        # Apply filtering and return processed observation
        return self._filter_obs(raw_obs)

    def _filter_obs(self, obs):
        # Implement basic filtering - remove excluded observations
        # Example: Remove base position (first 3 elements) if configured
        if "base_pos" in self.config.get("observation_exclusions", []):
            obs = obs[3:]
        
        # Ensure we return a numpy array with proper dimensions
        return obs.astype(np.float32)

    def step(self, action):
        self.sim.data.ctrl[:] = action
        self.sim.sim_step()
        
        reward = self._calculate_reward()
        terminated = self._check_done()
        truncated = False  # Timeout handling can be added here
        info = {}
        
        return self._get_obs(), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.sim.advance()  # Ensure simulation state is updated
        observation = self._get_obs()
        info = {"reset_reason": "initial"}
        return observation, info

    def _calculate_reward(self):
        # Get base linear and angular velocities from simulation
        base_lin_vel = self.sim.data.qvel[0:3]  # [vx, vy, vz]
        base_ang_vel = self.sim.data.qvel[3:6]  # [wx, wy, wz]
        
        # Penalize linear velocity (encourage staying stationary)
        lin_vel_penalty = np.sum(np.square(base_lin_vel))
        
        # Penalize angular velocity (encourage upright orientation)
        ang_vel_penalty = np.sum(np.square(base_ang_vel))
        
        # Small survival bonus per timestep
        survival_bonus = 0.1
        
        reward = survival_bonus - (lin_vel_penalty + ang_vel_penalty)
        
        return reward

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
