import rclpy
import yaml
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from legged_rl_control.rl.envs.legged_env import LeggedEnv
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from legged_rl_control.utils.logger import TrainingLogger
import numpy as np
import torch

def load_config():
    pkg_path = get_package_share_directory('legged_rl_control')
    
    # Load robot config
    robot_config_path = Path(pkg_path) / 'config/robots/a1_config.yaml'
    with open(robot_config_path) as f:
        robot_config = yaml.safe_load(f)
    
    # Load training config
    training_config_path = Path(pkg_path) / 'config/training/a1_training.yaml'
    with open(training_config_path) as f:
        training_config = yaml.safe_load(f)
    
    # Construct full path to scene.xml
    robot_config["model_path"] = str(Path(pkg_path) / robot_config["model_path"])
    
    return robot_config, training_config

def main():
    rclpy.init()
    
    # Load configurations
    robot_config, training_config = load_config()
    
    # Validate environment configuration
    assert robot_config["num_actions"] == 12, "A1 should have 12 actuators"
    assert robot_config["obs_dim"] == 36, "Observation dimension mismatch"
    
    # Create environment
    env = DummyVecEnv([lambda: LeggedEnv(robot_config)])
    
    # Configure SAC parameters
    model = SAC(
        training_config["sac_params"]["policy"],
        env,
        verbose=1,
        **{k: v for k, v in training_config["sac_params"].items() if k != "policy"}
    )
    
    # Configure checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=training_config["checkpoint"]["save_freq"],
        save_path=training_config["checkpoint"]["save_path"],
        name_prefix=training_config["checkpoint"]["name_prefix"]
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

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")

    try:
        model.learn(
            total_timesteps=training_config["total_timesteps"],
            callback=[checkpoint_callback, LoggingCallback()],
            tb_log_name="sac_a1",
            progress_bar=True
        )
        model.save("a1_policy_final")
    finally:
        env.close()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 