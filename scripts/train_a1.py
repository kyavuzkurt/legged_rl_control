import rclpy
from legged_rl_control.rl.envs.legged_env import LeggedEnv
from legged_rl_control.rl.robot_configs import A1_CONFIG
from stable_baselines3 import SAC

def main():
    rclpy.init()  # Initialize ROS2 context
    env = LeggedEnv(A1_CONFIG)
    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1_000_000)
    model.save("a1_policy")
    rclpy.shutdown()  # Cleanup ROS2 context

if __name__ == '__main__':
    main() 