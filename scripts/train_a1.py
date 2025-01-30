import rclpy
from legged_rl_control.rl.envs.legged_env import LeggedEnv
from legged_rl_control.rl.robot_configs import A1_CONFIG
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from legged_rl_control.utils.logger import TrainingLogger
import numpy as np

def main():
    rclpy.init()
    
    # Validate environment configuration
    assert A1_CONFIG["num_actions"] == 12, "A1 should have 12 actuators (3 joints per leg * 4 legs)"
    assert A1_CONFIG["obs_dim"] == 36, "Observation dimension should match robot sensors"
    
    # Create vectorized environment with Gymnasium compatibility
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
        device="auto",
    )
    
    # Save checkpoints every 100k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path="./a1_checkpoints/",
        name_prefix="a1_policy"
    )
    
    # Initialize logging
    logger = TrainingLogger()
    
    # Add custom callback
    from stable_baselines3.common.callbacks import BaseCallback
    class LoggingCallback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)
            self.episode_rewards = []
            self.episode_lengths = []

        def _on_step(self) -> bool:
            if "episode" in self.locals:
                self.episode_rewards.append(self.locals['episode']['r'])
                self.episode_lengths.append(self.locals['episode']['l'])
                
                if len(self.episode_rewards) % 10 == 0:  # Log every 10 episodes
                    logger.log_training({
                        'total_timesteps': self.num_timesteps,
                        'episode_reward_mean': np.mean(self.episode_rewards[-10:]),
                        'episode_len_mean': np.mean(self.episode_lengths[-10:]),
                        'value_loss': self.model.value_loss.item() if hasattr(self.model, 'value_loss') else 0,
                        'policy_loss': self.model.policy_loss.item() if hasattr(self.model, 'policy_loss') else 0,
                        'entropy_coeff': self.model.ent_coef_tensor.item() if hasattr(self.model, 'ent_coef_tensor') else 0,
                        'alpha': self.model.log_alpha.exp().item() if hasattr(self.model, 'log_alpha') else 0,
                        'grad_norm': self.model.grad_norm if hasattr(self.model, 'grad_norm') else 0
                    })
            return True

    try:
        model.learn(
            total_timesteps=3_000_000,
            callback=[checkpoint_callback, LoggingCallback()],  # Add our callback
            tb_log_name="sac_a1",
            progress_bar=True
        )
        model.save("a1_policy_final")
    finally:
        env.close()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 