import rclpy
from legged_rl_control.rl.envs.legged_env import LeggedEnv
from legged_rl_control.rl.robot_configs import A1_CONFIG
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

def main():
    rclpy.init()
    
    # Validate environment configuration
    assert A1_CONFIG["num_actions"] == 12, "A1 should have 12 actuators (3 joints per leg * 4 legs)"
    assert A1_CONFIG["obs_dim"] == 48, "Observation dimension should match robot sensors"
    
    # Create vectorized environment (required for SB3)
    env = DummyVecEnv([lambda: LeggedEnv(A1_CONFIG)])
    
    # Configure training parameters
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./a1_tensorboard/",
        buffer_size=1_000_000,
        learning_starts=10000,
        batch_size=256,
    )
    
    # Save checkpoints every 100k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path="./a1_checkpoints/",
        name_prefix="a1_policy"
    )
    
    try:
        model.learn(
            total_timesteps=2_000_000,
            callback=checkpoint_callback,
            tb_log_name="sac_a1"
        )
        model.save("a1_policy_final")
    finally:
        env.close()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 