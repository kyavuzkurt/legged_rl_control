from stable_baselines3 import SAC
from legged_rl_control.rl.envs import LeggedEnv

env = LeggedEnv()
model = SAC("MlpPolicy", env, verbose=1, device='cuda')
model.learn(total_timesteps=1_000_000)
model.save("legged_sac") 