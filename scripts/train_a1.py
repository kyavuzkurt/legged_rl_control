import rclpy
import yaml
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from legged_rl_control.rl.envs.legged_env import LeggedEnv
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from legged_rl_control.utils.logger import TrainingLogger
import numpy as np
import torch

# Configuration Management
def load_configs(pkg_name='legged_rl_control'):
    """Load configurations from YAML files"""
    pkg_path = get_package_share_directory(pkg_name)
    
    config_paths = {
        'robot': Path(pkg_path) / 'config/robots/a1_config.yaml',
        'training': Path(pkg_path) / 'config/training/a1_training.yaml',
        'controller': Path(pkg_path) / 'config/controllers/pid.yaml'
    }
    
    configs = {}
    for name, path in config_paths.items():
        with open(path) as f:
            configs[name] = yaml.safe_load(f)
    
    # Resolve model path
    configs['robot']["model_path"] = str(Path(pkg_path) / configs['robot']["model_path"])
    return configs['robot'], configs['training'], configs['controller']

def validate_environment_config(robot_config):
    """Validate critical environment parameters"""
    assert robot_config["num_actions"] == 12, "A1 should have 12 actuators"
    assert robot_config["obs_dim"] == 36, "Observation dimension mismatch"
    assert Path(robot_config["model_path"]).exists(), "Model file not found"

# Environment Setup
def make_training_env(robot_config, controller_config):
    """Create and return training environment with controller"""
    return DummyVecEnv([lambda: LeggedEnv(robot_config, controller_config)])

# Model Configuration
def build_sac_model(env, training_config, device='auto'):
    """Configure and return SAC model"""
    sac_params = training_config["sac_params"]
    return SAC(
        policy=sac_params["policy"],
        env=env,
        verbose=1,
        **{k: v for k, v in sac_params.items() if k != "policy"}
    )


# Callbacks & Logging
class TrainingMonitorCallback(BaseCallback):
    """Custom callback for logging training metrics"""
    def __init__(self, logger, window_size=10, verbose=0):
        super().__init__(verbose)
        self._logger = logger
        self.window_size = window_size
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        if "episode" in self.locals:
            self._update_metrics()
            if len(self.episode_rewards) % self.window_size == 0:
                self._log_metrics()
        return True

    def _update_metrics(self):
        self.episode_rewards.append(self.locals['episode']['r'])
        self.episode_lengths.append(self.locals['episode']['l'])

    def _log_metrics(self):
        metrics = {
            'total_timesteps': self.num_timesteps,
            'episode_reward_mean': np.mean(self.episode_rewards[-self.window_size:]),
            'episode_len_mean': np.mean(self.episode_lengths[-self.window_size:]),
            'value_loss': getattr(self.model, 'value_loss', 0.0).item(),
            'policy_loss': getattr(self.model, 'policy_loss', 0.0).item(),
            'entropy_coeff': getattr(self.model, 'ent_coef_tensor', 0.0).item(),
            'alpha': getattr(self.model.log_alpha, 'exp', lambda: 0.0)().item(),
            'grad_norm': getattr(self.model, 'grad_norm', 0.0)
        }
        self._logger.log_training(metrics)

def setup_callbacks(training_config, logger):
    """Create and return list of callbacks"""
    checkpoint_cb = CheckpointCallback(
        save_freq=training_config["checkpoint"]["save_freq"],
        save_path=training_config["checkpoint"]["save_path"],
        name_prefix=training_config["checkpoint"]["name_prefix"]
    )
    
    monitor_cb = TrainingMonitorCallback(logger=logger)
    return [checkpoint_cb, monitor_cb]

# Main Training Logic
def train_agent():
    """Main training workflow"""
    rclpy.init()
    
    # Load and validate configurations
    robot_config, training_config, controller_config = load_configs()
    validate_environment_config(robot_config)

    # Environment setup
    env = make_training_env(robot_config, controller_config)
    
    # Model initialization
    model = build_sac_model(env, training_config, 
                           device=training_config["sac_params"].get("device", "auto"))
    
    # Logging and callbacks
    logger = TrainingLogger()
    callbacks = setup_callbacks(training_config, logger)

    # GPU diagnostics
    if torch.cuda.is_available():
        print(f"Training on {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()

    # Training execution
    try:
        model.learn(
            total_timesteps=training_config["total_timesteps"],
            callback=callbacks,
            tb_log_name="sac_a1",
            progress_bar=True
        )
        model.save("a1_policy_final")
    finally:
        env.close()
        rclpy.shutdown()

if __name__ == '__main__':
    train_agent() 