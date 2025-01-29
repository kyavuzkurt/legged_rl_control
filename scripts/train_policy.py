import rclpy
from stable_baselines3 import PPO
from legged_rl_control.envs import LeggedRobotEnv

def main():
    rclpy.init()
    
    # Create environment
    env = LeggedRobotEnv(
        sim_src='isaac',
        policy_type='mlp',
        normalize_obs=True
    )
    
    # Initialize PPO
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./logs/",
        device='cuda'
    )
    
    # Start training
    model.learn(total_timesteps=1_000_000)
    
    # Save the trained model
    model.save("legged_policy_ppo")

if __name__ == '__main__':
    main() 