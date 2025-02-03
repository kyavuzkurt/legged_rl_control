import numpy as np
from gymnasium import Wrapper
from gymnasium.spaces import Box
from collections import deque

class NormalizeObservation(Wrapper):
    """
    Modular observation normalization wrapper compatible with any Gymnasium environment.
    Maintains running statistics for observation normalization.
    """
    def __init__(self, env, epsilon=1e-8, clip=10.0, update_stats=True):
        super().__init__(env)
        
        # Initialize observation space checks
        assert isinstance(env.observation_space, Box), "Normalization only works with Box observation spaces"
        
        # Store configuration parameters
        self.epsilon = epsilon
        self.clip = clip
        self.update_stats = update_stats
        
        # Initialize running statistics
        self.mean = np.zeros(env.observation_space.shape, dtype=np.float32)
        self.var = np.ones(env.observation_space.shape, dtype=np.float32)
        self.count = 1e-4  # Avoid division by zero
        
        # Update observation space to reflect normalized range
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=env.observation_space.shape,
            dtype=np.float32
        )

    def _update_stats(self, obs):
        batch_mean = np.mean(obs, axis=0)
        batch_var = np.var(obs, axis=0)
        batch_count = obs.shape[0] if len(obs.shape) > 1 else 1
        
        # Update running statistics
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        # Update mean
        self.mean += delta * batch_count / total_count
        
        # Update variance
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        self.var = M2 / total_count
        
        self.count = min(total_count, 1e6)  # Prevent infinite growth

    def observation(self, obs):
        # Apply normalization
        obs = (obs - self.mean) / np.sqrt(self.var + self.epsilon)
        
        # Clip to prevent extreme values
        return np.clip(obs, -self.clip, self.clip).astype(np.float32)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if self.update_stats:
            self._update_stats(obs)
            
        return self.observation(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        if self.update_stats:
            self._update_stats(obs)
            
        return self.observation(obs), info

    def save_stats(self, path):
        np.savez(path, mean=self.mean, var=self.var, count=self.count)

    def load_stats(self, path):
        data = np.load(path)
        self.mean = data['mean']
        self.var = data['var']
        self.count = data['count'] 