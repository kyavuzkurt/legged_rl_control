"""
New file for domain randomization wrapper
"""
import numpy as np
from gymnasium import Wrapper
from collections.abc import Mapping
from collections import deque
class DomainRandomizationWrapper(Wrapper):
    """
    Domain randomization wrapper for legged robots
    Applies randomization at each reset before episode starts
    """
    def __init__(self, env, randomization_config):
        super().__init__(env)
        self.randomization_config = randomization_config
        self._parse_randomization_config()

    def _parse_randomization_config(self):
        # Set default ranges for parameters
        self.default_ranges = {
            'friction': (0.5, 1.5),
            'damping': (0.8, 1.2),
            'armature': (0.8, 1.2),
            'motor_strength': (0.9, 1.1),
            'body_mass': (0.8, 1.2),
            'sensor_noise': 0.02,
            'latency': (0, 4),  # steps of delay
            'gravity': (-1.0, 1.0),  # percentage perturbation
        }
        
        # Merge with user config
        if isinstance(self.randomization_config, Mapping):
            self.default_ranges.update(self.randomization_config)

    def reset(self, **kwargs):
        # Apply randomization before actual reset
        self._randomize_physics()
        self._randomize_sensors()
        self._randomize_latency()
        
        return super().reset(**kwargs)

    def _randomize_physics(self):
        model = self.env.sim.model
        
        # Randomize friction for all geoms
        if 'friction' in self.default_ranges:
            for geom_id in range(model.ngeom):
                low, high = self.default_ranges['friction']
                model.geom_friction[geom_id] = np.random.uniform(low, high, 3)

        # Randomize joint properties
        if 'damping' in self.default_ranges:
            for joint_id in range(model.njnt):
                low, high = self.default_ranges['damping']
                model.dof_damping[joint_id] *= np.random.uniform(low, high)

        if 'armature' in self.default_ranges:
            for joint_id in range(model.njnt):
                low, high = self.default_ranges['armature']
                model.dof_armature[joint_id] *= np.random.uniform(low, high)

        # Randomize body masses
        if 'body_mass' in self.default_ranges:
            for body_id in range(model.nbody):
                if body_id == 0:  # Skip world body
                    continue
                low, high = self.default_ranges['body_mass']
                model.body_mass[body_id] *= np.random.uniform(low, high)

        # Randomize actuator strength
        if 'motor_strength' in self.default_ranges:
            for motor_id in range(model.nu):
                low, high = self.default_ranges['motor_strength']
                gain = np.random.uniform(low, high)
                model.actuator_gainprm[motor_id, 0] *= gain

        # Randomize gravity
        if 'gravity' in self.default_ranges:
            low, high = self.default_ranges['gravity']
            perturbation = np.random.uniform(low, high, 3)
            model.opt.gravity[:3] += model.opt.gravity[:3] * perturbation

    def _randomize_sensors(self):
        # Store noise parameters for step()
        self.sensor_noise = {}
        if 'sensor_noise' in self.default_ranges:
            self.sensor_noise = {
                'imu_acc': np.random.uniform(0, self.default_ranges['sensor_noise']),
                'imu_gyro': np.random.uniform(0, self.default_ranges['sensor_noise']),
                'joint_pos': np.random.uniform(0, self.default_ranges['sensor_noise']),
                'joint_vel': np.random.uniform(0, self.default_ranges['sensor_noise'])
            }

    def _randomize_latency(self):
        if 'latency' in self.default_ranges:
            self.latency_steps = np.random.randint(*self.default_ranges['latency'])
            self.action_buffer = deque(maxlen=self.latency_steps)

    def step(self, action):
        if hasattr(self, 'latency_steps') and self.latency_steps > 0:
            self.action_buffer.append(action)
            if len(self.action_buffer) >= self.latency_steps:
                actual_action = self.action_buffer.popleft()
            else:
                actual_action = np.zeros_like(action)
        else:
            actual_action = action
            
        obs, reward, terminated, truncated, info = super().step(actual_action)
        
        # Add sensor noise
        if self.sensor_noise:
            obs = self._add_sensor_noise(obs)
            
        return obs, reward, terminated, truncated, info

    def _add_sensor_noise(self, obs):
        # Get observation components
        qpos_dim = self.env.sim.data.qpos.size
        qvel_dim = self.env.sim.data.qvel.size
        
        # Split observation into components
        qpos = obs[:qpos_dim]
        qvel = obs[qpos_dim:qpos_dim+qvel_dim]
        imu_acc = obs[qpos_dim+qvel_dim:qpos_dim+qvel_dim+3]
        imu_gyro = obs[qpos_dim+qvel_dim+3:qpos_dim+qvel_dim+6]
        
        # Add noise
        qpos += np.random.normal(0, self.sensor_noise['joint_pos'], qpos.shape)
        qvel += np.random.normal(0, self.sensor_noise['joint_vel'], qvel.shape)
        imu_acc += np.random.normal(0, self.sensor_noise['imu_acc'], imu_acc.shape)
        imu_gyro += np.random.normal(0, self.sensor_noise['imu_gyro'], imu_gyro.shape)
        
        # Rebuild observation
        return np.concatenate([qpos, qvel, imu_acc, imu_gyro]).astype(np.float32) 